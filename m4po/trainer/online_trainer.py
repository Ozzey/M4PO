from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from m4po.common.buffer import ObservationSpec, RolloutBuffer
from m4po.common.config import (
    CHECKPOINT_SCHEMA_VERSION,
    IMPLEMENTATION_ID,
    M4POConfig,
)
from m4po.common.environment import environment_signature
from m4po.common.evaluation import evaluate_policy
from m4po.common.language import load_task_contexts
from m4po.common.utils import JsonlLogger, ensure_dir, get_device, set_seed
from m4po.envs import make_vector_env
from m4po.m4po import M4POAgent

Observation = dict[str, np.ndarray]


class OnlineTrainer:
    """Run episodic replay learning or the explicit legacy on-policy trainer."""

    _RESUME_OVERRIDE_KEYS = frozenset(
        {
            "total_steps",
            "eval_every",
            "save_every",
            "log_every",
            "log_dir",
            "device",
            "quiet",
            "resume_checkpoint",
            "max_wall_time_seconds",
            "save_replay",
        }
    )

    def __init__(self, cfg: M4POConfig):
        self.cfg = cfg

    @staticmethod
    def _metadata(envs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        task_ids = np.asarray(envs.task_ids, dtype=np.int64).reshape(envs.num_envs)
        embodiment_ids = np.asarray(envs.embodiment_ids, dtype=np.int64).reshape(
            envs.num_envs
        )
        action_masks = np.asarray(envs.action_masks, dtype=np.float32).reshape(
            envs.num_envs, envs.action_dim
        )
        return task_ids, embodiment_ids, action_masks

    @staticmethod
    def _copy_observation(observation: Mapping[str, np.ndarray]) -> Observation:
        return {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in observation.items()
        }

    @staticmethod
    def _reset_rollout(envs: Any) -> Observation:
        reset_rollout = getattr(envs, "reset_rollout", None)
        observation = reset_rollout() if callable(reset_rollout) else envs.reset()
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in observation.items()
        }

    @classmethod
    def _transition_next_observation(
        cls,
        auto_reset_observation: Mapping[str, np.ndarray],
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> Observation:
        transition_observation = cls._copy_observation(auto_reset_observation)
        for index, finished in enumerate(done):
            if not bool(finished):
                continue
            terminal = infos[index].get("terminal_observation")
            if terminal is None:
                raise ValueError(
                    "Environment auto-reset a completed worker without providing "
                    "info['terminal_observation']"
                )
            for name in transition_observation:
                transition_observation[name][index] = np.asarray(
                    terminal[name], dtype=np.float32
                )
        return transition_observation

    @staticmethod
    def _environment_signature(
        envs: Any,
        task_contexts: np.ndarray | None,
        task_context_mode: str,
    ) -> dict[str, Any]:
        """Serialize semantics that cannot safely change across resume."""

        return environment_signature(envs, task_contexts, task_context_mode)

    @classmethod
    def _validate_resume_payload(
        cls,
        payload: Any,
        cfg: M4POConfig,
        rollout_batch_size: int,
    ) -> tuple[int, int, int]:
        """Validate that a checkpoint is a compatible completed-update boundary."""

        if not isinstance(payload, Mapping):
            raise ValueError("Resume checkpoint payload must be a mapping")
        if payload.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError("Resume checkpoint has an incompatible implementation ID")
        if payload.get("checkpoint_schema_version") not in (
            2,
            CHECKPOINT_SCHEMA_VERSION,
        ):
            raise ValueError(
                "Resume checkpoint has an incompatible checkpoint schema version"
            )
        saved_config = payload.get("cfg")
        if not isinstance(saved_config, Mapping):
            raise ValueError("Resume checkpoint does not contain a valid configuration")
        saved_config = M4POConfig.from_checkpoint_dict(saved_config).to_dict()
        current_config = cfg.to_dict()
        mismatches = {
            key: (saved_config.get(key), current_config[key])
            for key in current_config
            if key not in cls._RESUME_OVERRIDE_KEYS
            and saved_config.get(key) != current_config[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={saved!r}, configured={current!r}"
                for key, (saved, current) in sorted(mismatches.items())
            )
            raise ValueError(
                "Resume checkpoint training configuration does not match: " + details
            )

        raw_step = payload.get("step")
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, np.integer)):
            raise ValueError("Resume checkpoint step must be an integer")
        step = int(raw_step)
        if step < 0 or step % rollout_batch_size:
            raise ValueError(
                "Resume checkpoint step is not a completed fresh-rollout boundary"
            )
        if step > cfg.total_steps:
            raise ValueError(
                f"Resume checkpoint step {step} exceeds total_steps={cfg.total_steps}"
            )
        if (cfg.total_steps - step) % rollout_batch_size:
            raise ValueError(
                "Remaining resume budget must contain an integer number of fresh rollouts"
            )

        extra = payload.get("extra", {})
        if not isinstance(extra, Mapping):
            raise ValueError("Resume checkpoint extra metadata must be a mapping")
        if extra.get("resumable", True) is not True:
            raise ValueError(
                "Resume checkpoint was captured during a partial optimizer update and "
                "is diagnostic-only"
            )
        raw_update_count = extra.get("update_count", 0)
        raw_completed_episodes = extra.get("completed_episodes", 0)
        for name, value in (
            ("update_count", raw_update_count),
            ("completed_episodes", raw_completed_episodes),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Resume checkpoint {name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"Resume checkpoint {name} must be non-negative")
        update_count = int(raw_update_count)
        if update_count * rollout_batch_size != step:
            raise ValueError(
                "Resume checkpoint step/update_count metadata does not describe a "
                "completed update boundary"
            )
        return step, update_count, int(raw_completed_episodes)

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            ),
        }

    @staticmethod
    def _restore_rng_state(state: Any) -> None:
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise ValueError("Resume checkpoint RNG state must be a mapping")
        required = {"python", "numpy", "torch_cpu", "torch_cuda"}
        if set(state) != required:
            raise ValueError("Resume checkpoint RNG state has invalid fields")
        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_initialized() and state["torch_cuda"] is not None:
                cuda_states = state["torch_cuda"]
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError(
                        "Resume checkpoint CUDA RNG state has a different device count"
                    )
                torch.cuda.set_rng_state_all(cuda_states)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("Resume checkpoint contains invalid RNG state") from exc

    @staticmethod
    def _capture_environment_state(envs: Any) -> Any:
        state_dict = getattr(envs, "rollout_state_dict", None)
        if not callable(state_dict):
            state_dict = getattr(envs, "state_dict", None)
        return state_dict() if callable(state_dict) else None

    @staticmethod
    def _restore_environment_state(envs: Any, state: Any) -> None:
        if state is None:
            return
        load_state_dict = getattr(envs, "load_rollout_state_dict", None)
        if not callable(load_state_dict):
            load_state_dict = getattr(envs, "load_state_dict", None)
        if not callable(load_state_dict):
            raise ValueError(
                "Resume checkpoint contains rollout state but the environment adapter "
                "cannot restore it"
            )
        load_state_dict(state)

    @staticmethod
    def _close_training_resources(envs: Any, logger: JsonlLogger) -> None:
        """Close both resources even when simulator teardown itself fails."""

        try:
            envs.close()
        finally:
            logger.close()

    def train(self) -> Path:
        if self.cfg.learning_mode == "off_policy":
            from m4po.trainer.off_policy_trainer import OffPolicyTrainer

            return OffPolicyTrainer(self.cfg).train()
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
            observation_spec: ObservationSpec = envs.observation_spec
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
            environment_signature = self._environment_signature(
                envs, task_contexts, cfg.task_context_mode
            )

            global_step = 0
            update_count = 0
            completed_episodes = 0
            if cfg.resume_checkpoint:
                resume_path = Path(cfg.resume_checkpoint)
                if not resume_path.exists():
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {resume_path}"
                    )
                payload = torch.load(
                    resume_path, map_location="cpu", weights_only=False
                )
                global_step, update_count, completed_episodes = (
                    self._validate_resume_payload(
                        payload,
                        cfg,
                        cfg.rollout_batch_size,
                    )
                )
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
                saved_signature = payload.get("extra", {}).get("environment_signature")
                if saved_signature != environment_signature:
                    raise ValueError(
                        "Resume checkpoint task, embodiment, context, control-rate, or action-token "
                        "semantics do not match the configured environment"
                    )
                agent = M4POAgent.load(
                    resume_path,
                    device,
                    override_cfg=cfg,
                    restore_optimizers=True,
                )
                self._restore_environment_state(
                    envs,
                    payload.get("extra", {}).get("environment_state"),
                )
                # Agent construction consumes random numbers before loading the saved
                # parameters. Restore RNGs last so the next rollout is a true continuation.
                self._restore_rng_state(payload.get("extra", {}).get("rng_state"))
            else:
                agent = M4POAgent(
                    observation_spec,
                    envs.action_dim,
                    cfg,
                    device,
                    task_contexts=task_contexts,
                )
            cfg.save_yaml(log_dir / "config_resolved.yaml")

            running_returns = np.zeros(envs.num_envs, dtype=np.float64)
            running_lengths = np.zeros(envs.num_envs, dtype=np.int32)
            completed_returns: list[float] = []
            completed_lengths: list[int] = []
            completed_successes: list[float] = []
            latest_metrics: dict[str, float] = {}
            next_eval = (
                ((global_step // cfg.eval_every) + 1) * cfg.eval_every
                if cfg.eval_every
                else None
            )
            next_save = (
                ((global_step // cfg.save_every) + 1) * cfg.save_every
                if cfg.save_every
                else None
            )
            next_log = (
                ((global_step // cfg.log_every) + 1) * cfg.log_every
                if cfg.log_every
                else None
            )
            start_step = global_step
            start_time = time.time()

            checkpoint_phase = "update_boundary"

            def runtime_state() -> dict[str, Any]:
                return {
                    "rng_state": self._capture_rng_state(),
                    "environment_state": self._capture_environment_state(envs),
                }

            committed_runtime_state = runtime_state()
        except BaseException:
            self._close_training_resources(envs, logger)
            raise

        def checkpoint_extra(
            *,
            resumable: bool = True,
            saved_runtime_state: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            state = (
                runtime_state() if saved_runtime_state is None else saved_runtime_state
            )
            return {
                "update_count": update_count,
                "completed_episodes": completed_episodes,
                "environment_signature": environment_signature,
                "resumable": resumable,
                "checkpoint_phase": (
                    "update_boundary" if resumable else "partial_optimizer_update"
                ),
                "rng_state": state["rng_state"],
                "environment_state": state["environment_state"],
            }

        try:
            while global_step < cfg.total_steps:
                next_global_step = global_step + cfg.rollout_batch_size
                if next_global_step > cfg.total_steps:
                    raise RuntimeError(
                        "Training budget would end inside a fresh rollout; check resume metadata"
                    )
                # Algorithm 2 samples a task/embodiment and resets each worker
                # at the start of every fresh rollout. Simulator adapters may
                # implement `reset_rollout` to resample those contexts.
                checkpoint_phase = "collection"
                observation = self._reset_rollout(envs)
                running_returns.fill(0.0)
                running_lengths.fill(0)
                rollout_returns: list[float] = []
                rollout_lengths: list[int] = []
                rollout_successes: list[float] = []
                rollout = RolloutBuffer(observation_spec, envs.num_envs)
                for _ in range(cfg.rollout_steps):
                    task_ids, embodiment_ids, action_masks = self._metadata(envs)
                    output = agent.act_np(
                        observation,
                        task_ids=task_ids,
                        embodiment_ids=embodiment_ids,
                        action_mask=action_masks,
                        deterministic=False,
                    )
                    next_observation, reward, done, infos = envs.step(output.action)
                    transition_next = self._transition_next_observation(
                        next_observation, done, infos
                    )
                    rollout.append(
                        observation,
                        transition_next,
                        output.action,
                        output.pre_tanh_action,
                        output.candidate_noise,
                        output.log_prob,
                        reward,
                        done.astype(np.float32),
                        task_ids,
                        embodiment_ids,
                        action_masks,
                    )
                    running_returns += reward.astype(np.float64)
                    running_lengths += 1
                    for index, finished in enumerate(done):
                        if not bool(finished):
                            continue
                        rollout_returns.append(float(running_returns[index]))
                        rollout_lengths.append(int(running_lengths[index]))
                        rollout_successes.append(
                            float(bool(infos[index].get("success", False)))
                        )
                        running_returns[index] = 0.0
                        running_lengths[index] = 0
                    observation = next_observation

                fresh_batch = rollout.finalize()
                checkpoint_phase = "update"
                latest_metrics = agent.update(
                    fresh_batch,
                    progress_fraction=min(
                        next_global_step / max(cfg.total_steps, 1), 1.0
                    ),
                )
                global_step = next_global_step
                update_count += 1
                completed_returns.extend(rollout_returns)
                completed_lengths.extend(rollout_lengths)
                completed_successes.extend(rollout_successes)
                completed_episodes += len(rollout_returns)
                checkpoint_phase = "update_boundary"
                committed_runtime_state = runtime_state()

                if next_log is not None and global_step >= next_log:
                    elapsed = max(time.time() - start_time, 1e-6)
                    logger.write(
                        {
                            "step": global_step,
                            "phase": "on_policy_training",
                            "steps_per_second": (global_step - start_step) / elapsed,
                            "updates": update_count,
                            "episodes_completed": completed_episodes,
                            "train_return_mean_20": (
                                float(np.mean(completed_returns[-20:]))
                                if completed_returns
                                else None
                            ),
                            "train_length_mean_20": (
                                float(np.mean(completed_lengths[-20:]))
                                if completed_lengths
                                else None
                            ),
                            "train_success_rate_20": (
                                float(np.mean(completed_successes[-20:]))
                                if completed_successes
                                else None
                            ),
                            **latest_metrics,
                        }
                    )
                    while global_step >= next_log:
                        next_log += cfg.log_every

                if next_eval is not None and global_step >= next_eval:
                    training_rng_state = self._capture_rng_state()
                    try:
                        evaluation = evaluate_policy(
                            agent, cfg, episodes=3, deterministic=True
                        )
                    finally:
                        # Deterministic execution still samples MPPI candidates;
                        # evaluation must not perturb the next training rollout.
                        self._restore_rng_state(training_rng_state)
                    logger.write({"step": global_step, "eval": evaluation})
                    while global_step >= next_eval:
                        next_eval += cfg.eval_every

                if next_save is not None and global_step >= next_save:
                    agent.save(
                        checkpoint_dir / f"step_{global_step}.pt",
                        step=global_step,
                        extra=checkpoint_extra(),
                    )
                    agent.save(
                        checkpoint_dir / "latest.pt",
                        step=global_step,
                        extra=checkpoint_extra(),
                    )
                    while global_step >= next_save:
                        next_save += cfg.save_every
                committed_runtime_state = runtime_state()
        except KeyboardInterrupt:
            agent.save(
                checkpoint_dir / "interrupted.pt",
                step=global_step,
                extra=checkpoint_extra(
                    resumable=checkpoint_phase != "update",
                    saved_runtime_state=committed_runtime_state,
                ),
            )
            raise
        except Exception:
            agent.save(
                checkpoint_dir / "emergency.pt",
                step=global_step,
                extra=checkpoint_extra(
                    resumable=checkpoint_phase != "update",
                    saved_runtime_state=committed_runtime_state,
                ),
            )
            raise
        finally:
            self._close_training_resources(envs, logger)

        agent.save(
            checkpoint_dir / "latest.pt",
            step=global_step,
            extra=checkpoint_extra(saved_runtime_state=committed_runtime_state),
        )
        return checkpoint_dir / "latest.pt"
