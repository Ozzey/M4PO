from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import CHECKPOINT_SCHEMA_VERSION, IMPLEMENTATION_ID, M4POConfig
from m4po.common.evaluation import evaluate_policy
from m4po.common.language import load_task_contexts
from m4po.common.utils import JsonlLogger, ensure_dir, get_device, set_seed
from m4po.envs import make_vector_env
from m4po.m4po import M4POAgent
from m4po.trainer.online_trainer import OnlineTrainer


def _optional_metric(value: Any) -> float | None:
    """Keep undefined benchmark metrics undefined (in particular NaN success)."""

    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_success(value: Any) -> float | None:
    """Success is a native binary signal, never inferred from a reward or score."""

    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "biuf":
        raise ValueError("Native success must be a binary scalar or undefined")
    result = float(array)
    if not math.isfinite(result):
        return None
    if result not in (0.0, 1.0):
        raise ValueError("Native success must be binary (zero or one)")
    return result


def _complete_mean(values) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(values))


class _RollingMetrics:
    """Preserve honest episode windows across checkpoint and job boundaries."""

    window_size = 20
    fields = ("returns", "lengths", "successes", "scores")

    def __init__(self, task_names):
        self.task_names = list(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("Rolling metrics require unique task names")
        self.global_windows = self._windows()
        self.per_task = {
            name: {**self._windows(), "episodes_completed": 0}
            for name in self.task_names
        }
        self.history_start_completed_episodes = 0
        self._snapshot = None

    def _windows(self):
        return {field: deque(maxlen=self.window_size) for field in self.fields}

    def record(self, task, episode_return, length, success, score):
        values = (
            float(episode_return),
            int(length),
            _optional_success(success),
            _optional_metric(score),
        )
        for field, value in zip(self.fields, values):
            self.global_windows[field].append(value)
            self.per_task[task][field].append(value)
        self.per_task[task]["episodes_completed"] += 1
        self._snapshot = None

    def summary(self):
        successes = [
            (value, len(stats["successes"]))
            for stats in self.per_task.values()
            if (value := _complete_mean(stats["successes"])) is not None
        ]
        scores = [
            value
            for stats in self.per_task.values()
            if (value := _complete_mean(stats["scores"])) is not None
        ]
        return {
            "train_success_rate_supported_tasks_20": (
                float(np.mean([value for value, _ in successes])) if successes else None
            ),
            "train_success_tasks": len(successes),
            "train_success_task_coverage": len(successes) / len(self.task_names),
            "train_success_episodes_in_window": sum(count for _, count in successes),
            "train_score_macro_20": float(np.mean(scores)) if scores else None,
            "train_score_task_count": len(scores),
            "train_score_task_coverage": len(scores) / len(self.task_names),
            "train_score_full_coverage": len(scores) == len(self.task_names),
            "rolling_window_episodes": self.window_size,
            "rolling_metrics_history_start_episode": self.history_start_completed_episodes,
        }

    def state_dict(self):
        # Copy only after an episode completes. Most vector steps leave these
        # windows unchanged, and committed snapshots must not mutate in place.
        if self._snapshot is None:
            self._snapshot = {
                "schema_version": 1,
                "window_size": self.window_size,
                "task_names": list(self.task_names),
                "history_start_completed_episodes": self.history_start_completed_episodes,
                "global": {
                    field: list(values) for field, values in self.global_windows.items()
                },
                "per_task": {
                    name: {
                        **{field: list(stats[field]) for field in self.fields},
                        "episodes_completed": stats["episodes_completed"],
                    }
                    for name, stats in self.per_task.items()
                },
            }
        return self._snapshot

    def load_state_dict(self, snapshot, *, completed_episodes):
        if (
            not isinstance(snapshot, Mapping)
            or type(snapshot.get("schema_version")) is not int
            or snapshot.get("schema_version") != 1
            or type(snapshot.get("window_size")) is not int
            or snapshot.get("window_size") != self.window_size
            or snapshot.get("task_names") != self.task_names
        ):
            raise ValueError(
                "Rolling metrics checkpoint schema, tasks or window capacity differ"
            )
        history_start = snapshot.get("history_start_completed_episodes")
        if (
            type(history_start) is not int
            or not 0 <= history_start <= completed_episodes
        ):
            raise ValueError("Rolling metrics history start is invalid")
        tasks = snapshot.get("per_task")
        if not isinstance(tasks, Mapping) or set(tasks) != set(self.task_names):
            raise ValueError("Rolling metrics checkpoint task names differ")

        def validate_windows(windows, episodes, *, per_task=False):
            expected = set(self.fields) | (
                {"episodes_completed"} if per_task else set()
            )
            if not isinstance(windows, Mapping) or set(windows) != expected:
                raise ValueError("Rolling metrics checkpoint has invalid window fields")
            validated = self._windows()
            for field in self.fields:
                values = windows[field]
                if not isinstance(values, list) or len(values) != min(
                    self.window_size, episodes
                ):
                    raise ValueError(
                        "Rolling metrics window length disagrees with episode count"
                    )
                for value in values:
                    if field in {"successes", "scores"} and value is None:
                        validated[field].append(None)
                        continue
                    if field == "lengths":
                        valid = type(value) is int and value > 0
                    else:
                        valid = (
                            type(value) in (int, float)
                            and math.isfinite(value)
                            and (field != "successes" or value in (0, 1))
                        )
                    if not valid:
                        raise ValueError(
                            f"Rolling metrics checkpoint has invalid {field}"
                        )
                    validated[field].append(value)
            return validated

        per_task = {}
        total = 0
        for task in self.task_names:
            stats = tasks[task]
            count = (
                stats.get("episodes_completed") if isinstance(stats, Mapping) else None
            )
            if type(count) is not int or count < 0:
                raise ValueError("Rolling metrics episode counter is invalid")
            per_task[task] = {
                **validate_windows(stats, count, per_task=True),
                "episodes_completed": count,
            }
            total += count
        if total + history_start != completed_episodes:
            raise ValueError(
                "Rolling metrics counters disagree with completed episodes"
            )
        global_windows = validate_windows(snapshot.get("global"), total)
        self.global_windows, self.per_task = global_windows, per_task
        self.history_start_completed_episodes = history_start
        self._snapshot = None


class _WorkerEpisode:
    """Own one worker's partial episode until the environment completes it."""

    def __init__(self, observation, task_id: int, embodiment_id: int, action_mask):
        self.observations = {
            name: [np.asarray(value, dtype=np.float32).copy()]
            for name, value in observation.items()
        }
        self.task_id = int(task_id)
        self.embodiment_id = int(embodiment_id)
        self.action_mask = np.asarray(action_mask, dtype=np.float32).copy()
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminated: list[float] = []

    def __len__(self) -> int:
        return len(self.actions)

    def append(self, observation, action, reward: float, terminated: bool) -> None:
        for name, values in self.observations.items():
            values.append(np.asarray(observation[name], dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        self.terminated.append(float(terminated))

    def finalize(self) -> Episode:
        length = len(self)
        return Episode(
            observations={
                name: torch.from_numpy(np.stack(values))
                for name, values in self.observations.items()
            },
            actions=torch.from_numpy(np.stack(self.actions)),
            rewards=torch.tensor(self.rewards, dtype=torch.float32).view(length, 1),
            terminated=torch.tensor(self.terminated, dtype=torch.float32).view(
                length, 1
            ),
            task_ids=torch.full((length + 1,), self.task_id, dtype=torch.long),
            embodiment_ids=torch.full(
                (length + 1,), self.embodiment_id, dtype=torch.long
            ),
            action_masks=torch.from_numpy(
                np.repeat(self.action_mask[None], length, axis=0)
            ),
        )


class OffPolicyTrainer(OnlineTrainer):
    """Collect completed episodes and train with uniform episodic replay."""

    _RESUME_OVERRIDE_KEYS = OnlineTrainer._RESUME_OVERRIDE_KEYS | {
        "save_replay",
        "max_wall_time_seconds",
    }

    @staticmethod
    def _save_checkpoint(
        agent, path: Path, *, step: int, extra: dict[str, Any]
    ) -> None:
        """Keep the previous complete checkpoint if writing the new one fails."""

        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        try:
            agent.save(Path(temporary), step=step, extra=extra)
            os.replace(temporary, path)
            if path.name == "latest.pt":
                # Monitoring/requeue decisions must not deserialize multi-GB
                # replay. The stat fields detect a stale status after a crash
                # between replacing the checkpoint and publishing its status.
                checkpoint_stat = path.stat()
                status = {
                    "step": int(step),
                    **{
                        name: extra.get(name)
                        for name in (
                            "update_count",
                            "training_complete",
                            "replay_saved",
                            "stop_reason",
                            "pretrain_updates_completed",
                            "update_credit",
                            "checkpoint_phase",
                            "resumable",
                            "last_eval_step",
                            "pending_evaluation",
                        )
                    },
                    "checkpoint_size": checkpoint_stat.st_size,
                    "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                }
                status_path = path.parent / "latest_status.json"
                descriptor, status_temporary = tempfile.mkstemp(
                    prefix=".latest_status.", suffix=".tmp", dir=path.parent
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                        json.dump(status, file, allow_nan=False)
                        file.write("\n")
                    os.replace(status_temporary, status_path)
                finally:
                    if os.path.exists(status_temporary):
                        os.unlink(status_temporary)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _validate_off_policy_resume(
        cls, payload: Any, cfg: M4POConfig
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("Resume checkpoint payload must be a mapping")
        if payload.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError("Resume checkpoint has an incompatible implementation ID")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Off-policy resume requires a current-schema off-policy checkpoint"
            )
        saved_config = payload.get("cfg")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Resume checkpoint does not contain a valid configuration")
        if saved_config.get("learning_mode", "on_policy") != "off_policy":
            raise ValueError("Cannot resume on-policy weights as off-policy training")
        current_config = cfg.to_dict()
        saved_config = M4POConfig.from_checkpoint_dict(saved_config).to_dict()
        mismatches = {
            key: (saved_config.get(key), value)
            for key, value in current_config.items()
            if key not in cls._RESUME_OVERRIDE_KEYS and saved_config.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={old!r}, configured={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(
                "Resume checkpoint training configuration does not match: " + details
            )
        extra = payload.get("extra")
        if not isinstance(extra, Mapping):
            raise ValueError("Resume checkpoint extra metadata must be a mapping")
        if (
            extra.get("resumable") is not True
            or extra.get("checkpoint_phase") != "update_boundary"
        ):
            raise ValueError(
                "Resume checkpoint is diagnostic-only after a partial optimizer update"
            )
        result = {}
        for name in (
            "step",
            "update_count",
            "completed_episodes",
            "pretrain_updates_completed",
            "discarded_partial_transitions",
        ):
            value = payload.get(name) if name == "step" else extra.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(
                    f"Resume checkpoint {name} must be a non-negative integer"
                )
            result[name] = int(value)
        if result["step"] % cfg.num_envs or result["step"] > cfg.total_steps:
            raise ValueError(
                "Resume checkpoint step must be a completed vector-step boundary within total_steps"
            )
        if result["pretrain_updates_completed"] > cfg.pretrain_updates:
            raise ValueError(
                "Resume checkpoint pretraining counter exceeds configured pretraining"
            )
        if result["update_count"] < result["pretrain_updates_completed"]:
            raise ValueError(
                "Resume checkpoint update_count is less than completed pretraining"
            )
        credit = extra.get("update_credit")
        if (
            isinstance(credit, bool)
            or not isinstance(credit, (float, int))
            or not math.isfinite(credit)
            or credit < 0
        ):
            raise ValueError(
                "Resume checkpoint update_credit must be finite and non-negative"
            )
        result["update_credit"] = float(credit)
        pretrain_done = extra.get("pretrain_done")
        if not isinstance(pretrain_done, bool) or pretrain_done != (
            result["pretrain_updates_completed"] == cfg.pretrain_updates
        ):
            raise ValueError(
                "Resume checkpoint pretrain_done disagrees with the pretraining counter"
            )
        result["pretrain_done"] = pretrain_done
        initial_evaluation_completed = extra.get("initial_evaluation_completed", False)
        if not isinstance(initial_evaluation_completed, bool):
            raise ValueError("Resume checkpoint initial evaluation flag is invalid")
        result["initial_evaluation_completed"] = initial_evaluation_completed
        last_eval_step = extra.get("last_eval_step", result["step"])
        if (
            isinstance(last_eval_step, bool)
            or not isinstance(last_eval_step, int)
            or not -1 <= last_eval_step <= result["step"]
        ):
            raise ValueError("Resume checkpoint last_eval_step is invalid")
        result["last_eval_step"] = last_eval_step
        return result

    def train(self) -> Path:
        wall_start = time.monotonic()
        cfg = self.cfg
        cfg.validate()
        set_seed(cfg.seed, cfg.torch_deterministic)
        device = get_device(cfg.device)
        cfg.device = str(device)
        log_dir = ensure_dir(cfg.log_dir)
        checkpoint_dir = ensure_dir(log_dir / "checkpoints")
        logger = JsonlLogger(log_dir / "metrics.jsonl")
        try:
            envs = make_vector_env(cfg)
        except BaseException:
            logger.close()
            raise
        try:
            cfg.num_tasks = int(envs.num_tasks)
            cfg.num_embodiments = int(envs.num_embodiments)
            cfg.proprio_dim = int(envs.observation_spec.proprio_dim)
            cfg.state_dim = int(envs.observation_spec.state_dim)
            cfg.validate()
            observation_spec = envs.observation_spec
            if cfg.task_context_mode == "learned":
                task_contexts = None
            elif cfg.task_context_mode == "none":
                task_contexts = np.zeros((cfg.num_tasks, 1), dtype=np.float32)
            else:
                task_contexts = (
                    load_task_contexts(cfg.task_context_path, envs.task_names)
                    if cfg.task_context_path
                    else np.asarray(envs.task_contexts, dtype=np.float32)
                )
                if task_contexts.shape[0] != cfg.num_tasks:
                    raise ValueError("Environment task contexts do not match num_tasks")
            signature = self._environment_signature(
                envs, task_contexts, cfg.task_context_mode
            )
            replay = EpisodeReplayBuffer(
                cfg.replay_capacity,
                cfg.model_horizon,
                cfg.batch_size,
                device=device,
                seed=cfg.seed,
                observation_spec=observation_spec,
            )
            state = {
                "step": 0,
                "update_count": 0,
                "completed_episodes": 0,
                "pretrain_updates_completed": 0,
                "pretrain_done": cfg.pretrain_updates == 0,
                "discarded_partial_transitions": 0,
                "update_credit": 0.0,
                "initial_evaluation_completed": False,
                "last_eval_step": -1,
            }
            replay_restored = False
            restored_seed_steps = 0
            initialization = {"kind": "random"}
            rolling = _RollingMetrics(envs.task_names)
            rolling_metrics_restored = False
            rolling_metrics_reset_on_resume = False
            if cfg.resume_checkpoint:
                resume_path = Path(cfg.resume_checkpoint)
                payload = torch.load(
                    resume_path, map_location="cpu", weights_only=False
                )
                state = self._validate_off_policy_resume(payload, cfg)
                try:
                    checkpoint_spec = ObservationSpec(**payload["observation_spec"])
                    checkpoint_action_dim = int(payload["action_dim"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "Resume checkpoint has invalid observation/action metadata"
                    ) from exc
                if (
                    checkpoint_spec != observation_spec
                    or checkpoint_action_dim != envs.action_dim
                ):
                    raise ValueError(
                        "Resume checkpoint observation/action dimensions do not match the environment"
                    )
                extra = payload["extra"]
                if "rolling_metrics_state" in extra:
                    rolling.load_state_dict(
                        extra["rolling_metrics_state"],
                        completed_episodes=state["completed_episodes"],
                    )
                    rolling_metrics_restored = True
                else:
                    # Legacy checkpoints did not persist these windows. Make
                    # their shortened history explicit rather than fabricating it.
                    rolling.history_start_completed_episodes = state[
                        "completed_episodes"
                    ]
                    rolling_metrics_reset_on_resume = True
                initialization = extra.get("initialization", {"kind": "random"})
                if (
                    cfg.init_checkpoint
                    and initialization.get("kind") != "demonstration_pretraining"
                ):
                    raise ValueError(
                        "Resume checkpoint lost demonstration initialization provenance"
                    )
                if extra.get("environment_signature") != signature:
                    raise ValueError(
                        "Resume checkpoint task, embodiment, context, control-rate, or action-token semantics do not match the configured environment"
                    )
                saved_replay = extra.get("replay_saved", False)
                if not isinstance(saved_replay, bool):
                    raise ValueError("Resume checkpoint replay_saved must be boolean")
                if not saved_replay and "replay_state" in extra:
                    raise ValueError(
                        "Resume replay snapshot disagrees with replay_saved"
                    )
                if saved_replay:
                    replay.load_state_dict(extra.get("replay_state"))
                    if (
                        len(replay) > state["step"]
                        or replay.num_episodes > state["completed_episodes"]
                    ):
                        raise ValueError(
                            "Resume replay size disagrees with completed collection counters"
                        )
                    if any(
                        int(episode.task_ids[0]) >= cfg.num_tasks
                        or int(episode.embodiment_ids[0]) >= cfg.num_embodiments
                        for episode in replay._episodes
                    ):
                        raise ValueError(
                            "Resume replay contains task/embodiment IDs outside the environment"
                        )
                    if not torch.equal(
                        replay.rng_state_dict()["generator"],
                        extra["replay_rng_state"]["generator"].cpu(),
                    ):
                        raise ValueError("Resume replay RNG metadata is inconsistent")
                    restored_seed_steps = extra.get("seed_steps_collected")
                    if (
                        isinstance(restored_seed_steps, bool)
                        or not isinstance(restored_seed_steps, int)
                        or not 0 <= restored_seed_steps <= state["step"]
                    ):
                        raise ValueError(
                            "Resume replay seed_steps_collected is invalid"
                        )
                    replay_restored = True
                agent = M4POAgent.load(
                    resume_path,
                    device,
                    override_cfg=cfg,
                    restore_optimizers=True,
                    _payload=payload,
                )
                self._restore_environment_state(envs, extra.get("environment_state"))
                replay.load_rng_state_dict(extra["replay_rng_state"])
                self._restore_rng_state(extra.get("rng_state"))
                # Release the loaded multi-GB snapshot after replay owns its CPU copies.
                del payload, extra
            elif cfg.init_checkpoint:
                from m4po.common.initialization import load_demonstration_initialization

                agent, initialization = load_demonstration_initialization(
                    cfg.init_checkpoint,
                    cfg,
                    observation_spec,
                    envs.action_dim,
                    task_contexts,
                    device,
                )
            else:
                agent = M4POAgent(
                    observation_spec,
                    envs.action_dim,
                    cfg,
                    device,
                    task_contexts=task_contexts,
                )
            cfg.save_yaml(log_dir / "config_resolved.yaml")
            recent_returns = rolling.global_windows["returns"]
            recent_lengths = rolling.global_windows["lengths"]
            recent_successes = rolling.global_windows["successes"]
            recent_scores = rolling.global_windows["scores"]
            per_task = rolling.per_task
            latest_metrics: dict[str, float] = {}
            start_step = state["step"]
            start_time = time.time()
            session_steps = restored_seed_steps if replay_restored else 0
            stopped_for_wall_time = False
            checkpoint_phase = "update_boundary"
            intervals = {
                name: getattr(cfg, f"{name}_every") for name in ("log", "eval", "save")
            }
            next_event = {
                name: ((state["step"] // interval) + 1) * interval if interval else None
                for name, interval in intervals.items()
            }
            if intervals["eval"]:
                # Unlike collection/logging, a due frozen evaluation must not
                # disappear when a wall-time checkpoint lands on its boundary.
                next_event["eval"] = (
                    max(1, state["last_eval_step"] // intervals["eval"] + 1)
                    * intervals["eval"]
                )

            def runtime_state() -> dict[str, Any]:
                return {
                    "rng_state": self._capture_rng_state(),
                    "replay_rng_state": replay.rng_state_dict(),
                    "environment_state": self._capture_environment_state(envs),
                    "rolling_metrics_state": rolling.state_dict(),
                }

            committed_runtime_state = runtime_state()
            committed_state = dict(state)
        except BaseException:
            self._close_training_resources(envs, logger)
            raise

        def checkpoint_extra(
            *, resumable=True, runtime=None, saved_state=None, include_replay=False
        ) -> dict[str, Any]:
            counters = dict(state if saved_state is None else saved_state)
            # Checkpoint resume restarts simulator workers. Account for partial
            # episodes that are not included in completed-episode replay.
            if resumable:
                counters["discarded_partial_transitions"] += sum(
                    len(worker) for worker in workers if worker is not None
                )
            save_replay = (
                bool(getattr(cfg, "save_replay", False))
                and include_replay
                and resumable
            )
            pending_evaluation = bool(
                intervals["eval"]
                and counters["step"] >= intervals["eval"]
                and counters["last_eval_step"]
                < (counters["step"] // intervals["eval"]) * intervals["eval"]
            )
            result = {
                **{key: value for key, value in counters.items() if key != "step"},
                "environment_signature": signature,
                "initialization": initialization,
                "learning_mode": "off_policy",
                "resumable": resumable,
                "checkpoint_phase": "update_boundary"
                if resumable
                else "partial_optimizer_update",
                "replay_saved": save_replay,
                "resume_rebuilds_replay": not save_replay,
                "seed_steps_collected": session_steps,
                "pending_evaluation": pending_evaluation,
                "training_complete": (
                    counters["step"] >= cfg.total_steps
                    and not pending_evaluation
                    and counters["update_credit"] < 1.0 - 1e-10
                    and (
                        counters["pretrain_done"]
                        or session_steps < cfg.seed_steps
                        or not replay.can_sample
                    )
                ),
                "stop_reason": "wall_time_limit"
                if stopped_for_wall_time
                else "training_budget",
                **(runtime_state() if runtime is None else runtime),
            }
            if save_replay:
                result["replay_state"] = replay.state_dict()
            return result

        def update() -> None:
            nonlocal latest_metrics, checkpoint_phase
            checkpoint_phase = "update"
            latest_metrics = agent.update(
                replay,
                progress_fraction=min(state["step"] / max(cfg.total_steps, 1), 1.0),
            )
            state["update_count"] += 1

        def wall_time_reached() -> bool:
            limit = float(getattr(cfg, "max_wall_time_seconds", 0))
            return limit > 0 and time.monotonic() - wall_start >= limit

        def finish_pending_updates() -> None:
            nonlocal stopped_for_wall_time
            if session_steps < cfg.seed_steps or not replay.can_sample:
                return
            while not state["pretrain_done"]:
                update()
                state["pretrain_updates_completed"] += 1
                state["pretrain_done"] = (
                    state["pretrain_updates_completed"] == cfg.pretrain_updates
                )
                if wall_time_reached():
                    stopped_for_wall_time = True
                    return
            while state["update_credit"] >= 1.0 - 1e-10:
                update()
                state["update_credit"] = max(0.0, state["update_credit"] - 1.0)
                if wall_time_reached():
                    stopped_for_wall_time = True
                    return

        def task_metrics() -> dict[str, Any]:
            return {
                name: {
                    "success_rate_20": _complete_mean(stats["successes"]),
                    "score_mean_20": _complete_mean(stats["scores"]),
                    "return_mean_20": float(np.mean(stats["returns"]))
                    if stats["returns"]
                    else None,
                    "length_mean_20": float(np.mean(stats["lengths"]))
                    if stats["lengths"]
                    else None,
                    "episodes_completed": stats["episodes_completed"],
                }
                for name, stats in per_task.items()
            }

        def log_metrics() -> None:
            elapsed = max(time.time() - start_time, 1e-6)
            logger.write(
                {
                    "step": state["step"],
                    "phase": "off_policy_training",
                    "initialization_kind": initialization["kind"],
                    "demonstration_pretrain_updates": initialization.get(
                        "demonstration_pretraining", {}
                    ).get("completed_updates", 0),
                    "steps_per_second": (state["step"] - start_step) / elapsed,
                    "updates": state["update_count"],
                    "episodes_completed": state["completed_episodes"],
                    "pretrain_updates_completed": state["pretrain_updates_completed"],
                    "replay_transitions": len(replay),
                    "replay_episodes": replay.num_episodes,
                    "replay_restored": replay_restored,
                    "stopped_for_wall_time": stopped_for_wall_time,
                    "last_eval_step": state["last_eval_step"],
                    "discarded_partial_transitions": state[
                        "discarded_partial_transitions"
                    ],
                    "seed_collection": session_steps < cfg.seed_steps
                    or not replay.can_sample,
                    "train_return_mean_20": float(np.mean(recent_returns))
                    if recent_returns
                    else None,
                    "train_length_mean_20": float(np.mean(recent_lengths))
                    if recent_lengths
                    else None,
                    "train_success_rate_20": _complete_mean(recent_successes),
                    "train_score_mean_20": _complete_mean(recent_scores),
                    "train_per_task": task_metrics(),
                    "rolling_metrics_restored": rolling_metrics_restored,
                    "rolling_metrics_reset_on_resume": rolling_metrics_reset_on_resume,
                    **rolling.summary(),
                    **latest_metrics,
                }
            )

        def due(name: str) -> bool:
            return next_event[name] is not None and state["step"] >= next_event[name]

        def advance(name: str) -> None:
            while next_event[name] is not None and state["step"] >= next_event[name]:
                next_event[name] += intervals[name]

        def evaluate_due() -> None:
            if not due("eval"):
                return
            training_rng_state = self._capture_rng_state()
            try:
                evaluation = evaluate_policy(
                    agent,
                    cfg,
                    episodes=int(getattr(cfg, "eval_episodes", 3)),
                    deterministic=True,
                    expected_environment_signature=signature,
                )
            finally:
                self._restore_rng_state(training_rng_state)
            logger.write(
                {
                    "step": state["step"],
                    "eval": evaluation,
                    "seed": cfg.seed,
                    "initialization": initialization,
                }
            )
            state["last_eval_step"] = state["step"]
            advance("eval")

        workers: list[_WorkerEpisode | None] = [None] * envs.num_envs
        observation = None
        preserve_episodes = bool(
            getattr(envs, "preserve_episodes_between_rollouts", False)
        )
        try:
            if (
                state["step"] == 0
                and bool(getattr(cfg, "eval_at_start", False))
                and not state["initial_evaluation_completed"]
                and not wall_time_reached()
            ):
                training_rng_state = self._capture_rng_state()
                try:
                    evaluation = evaluate_policy(
                        agent,
                        cfg,
                        episodes=int(getattr(cfg, "eval_episodes", 3)),
                        deterministic=True,
                        expected_environment_signature=signature,
                    )
                finally:
                    self._restore_rng_state(training_rng_state)
                logger.write(
                    {
                        "step": 0,
                        "eval": evaluation,
                        "seed": cfg.seed,
                        "initialization": initialization,
                    }
                )
                state["initial_evaluation_completed"] = True
                state["last_eval_step"] = 0
            # A wall-time checkpoint may stop between pretraining/UTD updates.
            # Finish their saved counters/credit before executing another action.
            if not wall_time_reached():
                finish_pending_updates()
                committed_runtime_state = runtime_state()
                committed_state = dict(state)
                checkpoint_phase = "update_boundary"
                if not stopped_for_wall_time and not wall_time_reached():
                    evaluate_due()
                    committed_runtime_state = runtime_state()
                    committed_state = dict(state)
            if wall_time_reached():
                stopped_for_wall_time = True
            while state["step"] < cfg.total_steps and not stopped_for_wall_time:
                if wall_time_reached():
                    stopped_for_wall_time = True
                    break
                # A segment chooses the task/embodiment, just as M3PO chooses an
                # episode context. Simulator task switches force a reset, so only
                # already completed worker episodes enter replay.
                checkpoint_phase = "collection"
                if observation is None or not preserve_episodes:
                    state["discarded_partial_transitions"] += sum(
                        len(worker) for worker in workers if worker is not None
                    )
                    workers = [None] * envs.num_envs
                    observation = self._reset_rollout(envs)
                    observation_spec.validate(observation)
                for _ in range(
                    min(
                        cfg.rollout_steps,
                        (cfg.total_steps - state["step"]) // envs.num_envs,
                    )
                ):
                    checkpoint_phase = "collection"
                    task_ids, embodiment_ids, action_masks = self._metadata(envs)
                    task_ids, embodiment_ids, action_masks = (
                        task_ids.copy(),
                        embodiment_ids.copy(),
                        action_masks.copy(),
                    )
                    for index in range(envs.num_envs):
                        worker = workers[index]
                        if worker is None:
                            workers[index] = _WorkerEpisode(
                                {
                                    name: value[index]
                                    for name, value in observation.items()
                                },
                                task_ids[index],
                                embodiment_ids[index],
                                action_masks[index],
                            )
                        elif (
                            worker.task_id != task_ids[index]
                            or worker.embodiment_id != embodiment_ids[index]
                            or not np.array_equal(
                                worker.action_mask, action_masks[index]
                            )
                        ):
                            raise ValueError(
                                "Environment changed worker context without completing its episode"
                            )
                    seed_collection = (
                        session_steps < cfg.seed_steps or not replay.can_sample
                    )
                    if seed_collection:
                        action = (
                            np.random.uniform(
                                -1, 1, size=(envs.num_envs, envs.action_dim)
                            ).astype(np.float32)
                            * action_masks
                        )
                    else:
                        action = agent.act_np(
                            observation,
                            task_ids=task_ids,
                            embodiment_ids=embodiment_ids,
                            action_mask=action_masks,
                            deterministic=False,
                        ).action
                    checkpoint_phase = "collection"
                    next_observation, reward, done, infos = envs.step(action)
                    done = np.asarray(done, dtype=np.bool_).reshape(envs.num_envs)
                    reward = np.asarray(reward, dtype=np.float32).reshape(envs.num_envs)
                    if not np.isfinite(reward).all() or len(infos) != envs.num_envs:
                        raise ValueError(
                            "Environment returned invalid rewards or worker infos"
                        )
                    transition_next = self._transition_next_observation(
                        next_observation, done, infos
                    )
                    observation_spec.validate(transition_next)
                    for index, finished in enumerate(done):
                        worker = workers[index]
                        assert worker is not None
                        # Time limits finish replay episodes but retain Bellman
                        # bootstrapping from the true, pre-reset final observation.
                        truncated = infos[index].get(
                            "truncated", infos[index].get("timeout", False)
                        )
                        terminated = bool(
                            infos[index].get(
                                "terminated", bool(finished) and not truncated
                            )
                        )
                        if terminated and not finished:
                            raise ValueError(
                                "Environment reported termination without completing the worker episode"
                            )
                        worker.append(
                            {
                                name: value[index]
                                for name, value in transition_next.items()
                            },
                            action[index],
                            reward[index],
                            terminated,
                        )
                        if finished:
                            replay.add(worker.finalize())
                            episode_return, episode_length = (
                                float(sum(worker.rewards)),
                                len(worker),
                            )
                            success = _optional_success(infos[index].get("success"))
                            score = _optional_metric(infos[index].get("score"))
                            rolling.record(
                                envs.task_names[worker.task_id],
                                episode_return,
                                episode_length,
                                success,
                                score,
                            )
                            state["completed_episodes"] += 1
                            workers[index] = None
                    observation = next_observation
                    state["step"] += envs.num_envs
                    session_steps += envs.num_envs
                    if session_steps >= cfg.seed_steps and replay.can_sample:
                        # Seed data receive the one-time pretraining budget. The
                        # online ratio applies to subsequent collected transitions.
                        if not seed_collection:
                            state["update_credit"] += (
                                cfg.updates_per_step * envs.num_envs
                            )
                        finish_pending_updates()
                    committed_runtime_state = runtime_state()
                    committed_state = dict(state)
                    checkpoint_phase = "update_boundary"
                    if stopped_for_wall_time or wall_time_reached():
                        stopped_for_wall_time = True
                        break
                    if due("log"):
                        log_metrics()
                        advance("log")
                    evaluate_due()
                    committed_runtime_state = runtime_state()
                    committed_state = dict(state)
                    if wall_time_reached():
                        stopped_for_wall_time = True
                        break
                    if due("save"):
                        for filename in (f"step_{state['step']}.pt", "latest.pt"):
                            self._save_checkpoint(
                                agent,
                                checkpoint_dir / filename,
                                step=state["step"],
                                extra=checkpoint_extra(
                                    include_replay=filename == "latest.pt"
                                ),
                            )
                        advance("save")
            state["discarded_partial_transitions"] += sum(
                len(worker) for worker in workers if worker is not None
            )
            workers = [None] * envs.num_envs
            log_metrics()
            committed_runtime_state = runtime_state()
            committed_state = dict(state)
            self._save_checkpoint(
                agent,
                checkpoint_dir / "latest.pt",
                step=state["step"],
                extra=checkpoint_extra(
                    runtime=committed_runtime_state, include_replay=True
                ),
            )
        except (KeyboardInterrupt, Exception) as exc:
            filename = (
                "interrupted.pt"
                if isinstance(exc, KeyboardInterrupt)
                else "emergency.pt"
            )
            saved_state = state if checkpoint_phase == "update" else committed_state
            self._save_checkpoint(
                agent,
                checkpoint_dir / filename,
                step=saved_state["step"],
                extra=checkpoint_extra(
                    resumable=checkpoint_phase != "update",
                    runtime=committed_runtime_state,
                    saved_state=saved_state,
                    include_replay=checkpoint_phase == "update_boundary",
                ),
            )
            raise
        finally:
            self._close_training_resources(envs, logger)
        return checkpoint_dir / "latest.pt"
