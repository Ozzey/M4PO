from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from m4po.common.config import M4POConfig
from m4po.common.environment import environment_signature
from m4po.envs import make_vector_env


def _agent_task_contexts(agent: Any) -> np.ndarray | None:
    encoder = getattr(getattr(agent, "model", None), "encoder", None)
    features = getattr(encoder, "task_context_features", None)
    if features is None:
        return None
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    return np.asarray(features, dtype=np.float32)


def _metadata(
    env: Any,
    *,
    num_tasks: int,
    num_embodiments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate the metadata for the observations currently exposed."""

    num_envs = int(env.num_envs)
    action_dim = int(env.action_dim)
    task_ids = np.asarray(env.task_ids, dtype=np.int64)
    embodiment_ids = np.asarray(env.embodiment_ids, dtype=np.int64)
    action_masks = np.asarray(env.action_masks, dtype=np.float32)
    if task_ids.shape != (num_envs,) or embodiment_ids.shape != (num_envs,):
        raise ValueError(
            "Environment metadata must contain one task and embodiment ID per worker"
        )
    if action_masks.shape != (num_envs, action_dim):
        raise ValueError(
            "Environment action masks must have shape "
            f"{(num_envs, action_dim)}, got {tuple(action_masks.shape)}"
        )
    if np.any(task_ids < 0) or np.any(task_ids >= num_tasks):
        raise ValueError("Environment task IDs are out of range")
    if np.any(embodiment_ids < 0) or np.any(embodiment_ids >= num_embodiments):
        raise ValueError("Environment embodiment IDs are out of range")
    if not np.all(np.isfinite(action_masks)) or not np.all(
        (action_masks == 0.0) | (action_masks == 1.0)
    ):
        raise ValueError("Environment action masks must be finite binary values")
    tokenizer = getattr(env, "action_tokenizer", None)
    if tokenizer is not None:
        expected_masks = tokenizer.mask_for(embodiment_ids)
        if not np.array_equal(action_masks, expected_masks):
            raise ValueError("Environment action masks do not match embodiment IDs")
    return task_ids.copy(), embodiment_ids.copy(), action_masks.copy()


def _require_cartesian_coverage(
    task_ids: np.ndarray,
    embodiment_ids: np.ndarray,
    expected_pairs: list[tuple[int, int]],
) -> None:
    actual = Counter(
        zip(
            task_ids.astype(int).tolist(),
            embodiment_ids.astype(int).tolist(),
            strict=True,
        )
    )
    expected = Counter(expected_pairs)
    if actual != expected:
        missing = sorted((expected - actual).elements())
        duplicates = sorted((actual - expected).elements())
        raise ValueError(
            "Evaluation environment must expose every task/embodiment pair exactly once "
            f"after reset; missing={missing}, duplicates={duplicates}"
        )


def _require_selected_pair(
    task_ids: np.ndarray,
    embodiment_ids: np.ndarray,
    selected_pair: tuple[int, int],
) -> None:
    """Require every active worker to expose the selected sequential pair."""

    selected_task, selected_embodiment = selected_pair
    matches = (task_ids == selected_task) & (embodiment_ids == selected_embodiment)
    if not np.all(matches):
        actual = sorted(
            Counter(
                zip(
                    task_ids.astype(int).tolist(),
                    embodiment_ids.astype(int).tolist(),
                    strict=True,
                )
            ).items()
        )
        raise ValueError(
            "Sequential evaluation environment must expose only the selected "
            f"task/embodiment pair {selected_pair}; actual={actual}"
        )


def _evaluate_pairs_sequentially(
    agent: Any,
    env: Any,
    *,
    expected_pairs: list[tuple[int, int]],
    per_pair_returns: dict[tuple[int, int], list[float]],
    per_pair_lengths: dict[tuple[int, int], list[int]],
    per_pair_success: dict[tuple[int, int], list[float]],
    episodes: int,
    deterministic: bool,
    num_envs: int,
    num_tasks: int,
    num_embodiments: int,
) -> None:
    """Evaluate adapters that can host only one task/embodiment pair at a time."""

    select_pair = getattr(env, "set_evaluation_pair", None)
    if not callable(select_pair):
        raise TypeError(
            "An environment with sequential_evaluation=True must provide callable "
            "set_evaluation_pair(task_id, embodiment_id)"
        )

    for selected_pair in expected_pairs:
        select_pair(*selected_pair)
        if int(env.num_envs) != num_envs:
            raise ValueError(
                "Sequential evaluation must preserve the number of workers when "
                "switching task/embodiment pairs"
            )
        observation = env.reset()
        task_ids, embodiment_ids, action_masks = _metadata(
            env,
            num_tasks=num_tasks,
            num_embodiments=num_embodiments,
        )
        _require_selected_pair(task_ids, embodiment_ids, selected_pair)
        running_returns = np.zeros(num_envs, dtype=np.float64)
        running_lengths = np.zeros(num_envs, dtype=np.int32)

        while len(per_pair_returns[selected_pair]) < episodes:
            step_task_ids = task_ids.copy()
            step_embodiment_ids = embodiment_ids.copy()
            output = agent.act_np(
                observation,
                task_ids=step_task_ids,
                embodiment_ids=step_embodiment_ids,
                action_mask=action_masks,
                deterministic=deterministic,
            )
            observation, reward, done, infos = env.step(output.action)
            reward = np.asarray(reward, dtype=np.float64)
            done = np.asarray(done, dtype=np.bool_)
            if reward.shape != (num_envs,) or done.shape != (num_envs,):
                raise ValueError(
                    "Environment rewards and done flags must have shape [num_envs]"
                )
            if len(infos) != num_envs:
                raise ValueError(
                    "Environment infos must contain one mapping per worker"
                )
            running_returns += reward
            running_lengths += 1
            for index, finished in enumerate(done):
                if not bool(finished):
                    continue
                pair = (int(step_task_ids[index]), int(step_embodiment_ids[index]))
                if pair != selected_pair:
                    raise ValueError(
                        "Sequential evaluation worker changed task/embodiment pair "
                        "during its final step"
                    )
                info = infos[index]
                if "task_id" in info and int(info["task_id"]) != pair[0]:
                    raise ValueError(
                        "Completed episode task metadata changed during its final step"
                    )
                if "embodiment_id" in info and int(info["embodiment_id"]) != pair[1]:
                    raise ValueError(
                        "Completed episode embodiment metadata changed during its final step"
                    )
                if len(per_pair_returns[pair]) < episodes:
                    per_pair_returns[pair].append(float(running_returns[index]))
                    per_pair_lengths[pair].append(int(running_lengths[index]))
                    per_pair_success[pair].append(
                        float(bool(info.get("success", False)))
                    )
                running_returns[index] = 0.0
                running_lengths[index] = 0

            next_task_ids, next_embodiment_ids, next_action_masks = _metadata(
                env,
                num_tasks=num_tasks,
                num_embodiments=num_embodiments,
            )
            ongoing = ~done
            if np.any(next_task_ids[ongoing] != step_task_ids[ongoing]) or np.any(
                next_embodiment_ids[ongoing] != step_embodiment_ids[ongoing]
            ):
                raise ValueError(
                    "Environment changed task/embodiment metadata mid-episode"
                )
            _require_selected_pair(
                next_task_ids, next_embodiment_ids, selected_pair
            )
            task_ids = next_task_ids
            embodiment_ids = next_embodiment_ids
            action_masks = next_action_masks


def evaluate_policy(
    agent: Any,
    cfg: M4POConfig,
    episodes: int = 5,
    deterministic: bool = True,
    *,
    expected_environment_signature: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Evaluate every configured task/embodiment pair without learning."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    eval_cfg = deepcopy(cfg)
    eval_cfg.num_envs = max(1, int(cfg.num_tasks) * int(cfg.num_embodiments))
    env = make_vector_env(eval_cfg)
    try:
        num_envs = int(env.num_envs)
        num_tasks = int(env.num_tasks)
        num_embodiments = int(env.num_embodiments)
        task_names = list(env.task_names)
        embodiment_names = list(env.embodiment_names)
        if num_tasks != int(cfg.num_tasks) or num_embodiments != int(
            cfg.num_embodiments
        ):
            raise ValueError(
                "Evaluation environment task/embodiment counts do not match the agent config"
            )
        if len(task_names) != num_tasks or len(embodiment_names) != num_embodiments:
            raise ValueError(
                "Environment names do not match its task/embodiment counts"
            )
        if hasattr(agent, "action_dim") and int(agent.action_dim) != int(env.action_dim):
            raise ValueError("Evaluation environment action dimension does not match the agent")
        if (
            hasattr(agent, "observation_spec")
            and agent.observation_spec != env.observation_spec
        ):
            raise ValueError("Evaluation environment observation spec does not match the agent")

        if expected_environment_signature is not None:
            actual_signature = environment_signature(
                env,
                _agent_task_contexts(agent),
                cfg.task_context_mode,
            )
            if dict(expected_environment_signature) != actual_signature:
                raise ValueError(
                    "Evaluation environment task, embodiment, context, control-rate, or "
                    "action-token semantics do not match the checkpoint"
                )

        expected_pairs = [
            (task_id, embodiment_id)
            for task_id in range(num_tasks)
            for embodiment_id in range(num_embodiments)
        ]
        pair_labels = {
            pair: f"{task_names[pair[0]]}/{embodiment_names[pair[1]]}"
            for pair in expected_pairs
        }
        if len(set(pair_labels.values())) != len(pair_labels):
            raise ValueError(
                "Task and embodiment names produce ambiguous evaluation pair keys"
            )
        per_pair_returns = {pair: [] for pair in expected_pairs}
        per_pair_lengths = {pair: [] for pair in expected_pairs}
        per_pair_success = {pair: [] for pair in expected_pairs}
        if bool(getattr(env, "sequential_evaluation", False)):
            _evaluate_pairs_sequentially(
                agent,
                env,
                expected_pairs=expected_pairs,
                per_pair_returns=per_pair_returns,
                per_pair_lengths=per_pair_lengths,
                per_pair_success=per_pair_success,
                episodes=episodes,
                deterministic=deterministic,
                num_envs=num_envs,
                num_tasks=num_tasks,
                num_embodiments=num_embodiments,
            )
        else:
            # Metadata describes the observation returned by reset, so it must be
            # read afterwards. Some adapters assign contexts dynamically at reset.
            observation = env.reset()
            task_ids, embodiment_ids, action_masks = _metadata(
                env,
                num_tasks=num_tasks,
                num_embodiments=num_embodiments,
            )
            if num_envs != len(expected_pairs):
                raise ValueError(
                    "Evaluation requires one worker per task/embodiment pair, got "
                    f"{num_envs} workers for {len(expected_pairs)} pairs"
                )
            _require_cartesian_coverage(task_ids, embodiment_ids, expected_pairs)
            running_returns = np.zeros(num_envs, dtype=np.float64)
            running_lengths = np.zeros(num_envs, dtype=np.int32)

            while any(len(values) < episodes for values in per_pair_returns.values()):
                step_task_ids = task_ids.copy()
                step_embodiment_ids = embodiment_ids.copy()
                output = agent.act_np(
                    observation,
                    task_ids=step_task_ids,
                    embodiment_ids=step_embodiment_ids,
                    action_mask=action_masks,
                    deterministic=deterministic,
                )
                observation, reward, done, infos = env.step(output.action)
                reward = np.asarray(reward, dtype=np.float64)
                done = np.asarray(done, dtype=np.bool_)
                if reward.shape != (num_envs,) or done.shape != (num_envs,):
                    raise ValueError(
                        "Environment rewards and done flags must have shape [num_envs]"
                    )
                if len(infos) != num_envs:
                    raise ValueError(
                        "Environment infos must contain one mapping per worker"
                    )
                running_returns += reward
                running_lengths += 1
                for index, finished in enumerate(done):
                    if not bool(finished):
                        continue
                    pair = (
                        int(step_task_ids[index]),
                        int(step_embodiment_ids[index]),
                    )
                    info = infos[index]
                    if "task_id" in info and int(info["task_id"]) != pair[0]:
                        raise ValueError(
                            "Completed episode task metadata changed during its final step"
                        )
                    if (
                        "embodiment_id" in info
                        and int(info["embodiment_id"]) != pair[1]
                    ):
                        raise ValueError(
                            "Completed episode embodiment metadata changed during its "
                            "final step"
                        )
                    if len(per_pair_returns[pair]) < episodes:
                        per_pair_returns[pair].append(float(running_returns[index]))
                        per_pair_lengths[pair].append(int(running_lengths[index]))
                        per_pair_success[pair].append(
                            float(bool(info.get("success", False)))
                        )
                    running_returns[index] = 0.0
                    running_lengths[index] = 0

                next_task_ids, next_embodiment_ids, next_action_masks = _metadata(
                    env,
                    num_tasks=num_tasks,
                    num_embodiments=num_embodiments,
                )
                ongoing = ~done
                if np.any(next_task_ids[ongoing] != step_task_ids[ongoing]) or np.any(
                    next_embodiment_ids[ongoing] != step_embodiment_ids[ongoing]
                ):
                    raise ValueError(
                        "Environment changed task/embodiment metadata mid-episode"
                    )
                task_ids = next_task_ids
                embodiment_ids = next_embodiment_ids
                action_masks = next_action_masks

        per_pair: dict[str, dict[str, object]] = {}
        for pair in expected_pairs:
            key = pair_labels[pair]
            returns = per_pair_returns[pair]
            lengths = per_pair_lengths[pair]
            successes = per_pair_success[pair]
            per_pair[key] = {
                "task": task_names[pair[0]],
                "embodiment": embodiment_names[pair[1]],
                "episodes": len(returns),
                "returns": returns,
                "lengths": lengths,
                "successes": successes,
                "return_mean": float(np.mean(returns)) if returns else 0.0,
                "return_std": float(np.std(returns)) if returns else 0.0,
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "success_rate": float(np.mean(successes)) if successes else 0.0,
            }

        per_task: dict[str, dict[str, float]] = {}
        for task_name in task_names:
            records = [
                record for record in per_pair.values() if record["task"] == task_name
            ]
            per_task[task_name] = {
                "return_mean": float(
                    np.mean([record["return_mean"] for record in records])
                ),
                "success_rate": float(
                    np.mean([record["success_rate"] for record in records])
                ),
            }
        per_embodiment: dict[str, dict[str, float]] = {}
        for embodiment_name in embodiment_names:
            records = [
                record
                for record in per_pair.values()
                if record["embodiment"] == embodiment_name
            ]
            per_embodiment[embodiment_name] = {
                "return_mean": float(
                    np.mean([record["return_mean"] for record in records])
                ),
                "success_rate": float(
                    np.mean([record["success_rate"] for record in records])
                ),
            }

        all_returns = [
            value for values in per_pair_returns.values() for value in values
        ]
        all_lengths = [
            value for values in per_pair_lengths.values() for value in values
        ]
        all_successes = [
            value for values in per_pair_success.values() for value in values
        ]
        embodiment_success = [
            record["success_rate"] for record in per_embodiment.values()
        ]
        return {
            "episodes": len(all_returns),
            "episodes_per_pair": episodes,
            "returns": all_returns,
            "lengths": all_lengths,
            "successes": all_successes,
            "return_mean": float(np.mean(all_returns)) if all_returns else 0.0,
            "return_std": float(np.std(all_returns)) if all_returns else 0.0,
            "length_mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
            "success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
            "worst_embodiment_success": (
                float(min(embodiment_success)) if embodiment_success else 0.0
            ),
            "per_pair": per_pair,
            "per_task": per_task,
            "per_embodiment": per_embodiment,
        }
    finally:
        env.close()
