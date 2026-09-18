from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
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


def _evaluation_pairs(
    env: Any, num_tasks: int, num_embodiments: int, *, require_explicit: bool
) -> list[tuple[int, int]]:
    """Use the adapter's supported pairs instead of inventing missing robots."""

    configured = getattr(env, "evaluation_pairs", None)
    if configured is None:
        if require_explicit:
            raise ValueError("MMBench adapters must expose explicit evaluation_pairs")
        return [
            (task_id, embodiment_id)
            for task_id in range(num_tasks)
            for embodiment_id in range(num_embodiments)
        ]
    pairs: list[tuple[int, int]] = []
    for raw_pair in configured:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise ValueError(
                "evaluation_pairs entries must be (task_id, embodiment_id)"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_pair
        ):
            raise ValueError("evaluation_pairs IDs must be integers")
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        if not (0 <= pair[0] < num_tasks and 0 <= pair[1] < num_embodiments):
            raise ValueError("evaluation_pairs IDs are out of range")
        if pair in pairs:
            raise ValueError("evaluation_pairs must not contain duplicate pairs")
        pairs.append(pair)
    if not pairs:
        raise ValueError("evaluation_pairs must not be empty")
    if {pair[0] for pair in pairs} != set(range(num_tasks)) or {
        pair[1] for pair in pairs
    } != set(range(num_embodiments)):
        raise ValueError(
            "evaluation_pairs must cover every configured task and embodiment"
        )
    return pairs


def _episode_metric(info: Mapping[str, Any], name: str) -> float | None:
    """Keep native scores and represent undefined success as JSON null."""

    value = info.get(name)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"Episode {name} must be a scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        return None
    if name == "success" and scalar not in (0.0, 1.0):
        raise ValueError("Episode success must be boolean, zero, one, or undefined")
    return scalar


def _complete_mean(values: list[float | None]) -> float | None:
    """Do not disguise a partially defined benchmark metric as a complete one."""

    if not values or any(value is None for value in values):
        return None
    return float(np.mean(values))


def _require_supported_pairs(
    task_ids: np.ndarray,
    embodiment_ids: np.ndarray,
    expected_pairs: list[tuple[int, int]],
) -> None:
    actual = set(zip(task_ids.tolist(), embodiment_ids.tolist(), strict=True))
    unsupported = sorted(actual - set(expected_pairs))
    if unsupported:
        raise ValueError(
            f"Environment exposed unsupported evaluation pairs: {unsupported}"
        )


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
    per_pair_success: dict[tuple[int, int], list[float | None]],
    per_pair_scores: dict[tuple[int, int], list[float | None]],
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
                    per_pair_success[pair].append(_episode_metric(info, "success"))
                    per_pair_scores[pair].append(_episode_metric(info, "score"))
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
            _require_selected_pair(next_task_ids, next_embodiment_ids, selected_pair)
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
    is_mmbench = str(getattr(cfg, "env", "")).lower().strip() == "mmbench"
    if is_mmbench:
        eval_cfg._mmbench_evaluation = True
    # MMBench runs its sparse native task/robot assignments sequentially. Do
    # not create thousands of workers for a nonexistent Cartesian product.
    eval_cfg.num_envs = max(
        1,
        int(getattr(cfg, "mmbench_eval_num_envs", 1))
        if is_mmbench
        else int(cfg.num_tasks) * int(cfg.num_embodiments),
    )
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
        task_domains = getattr(env, "task_domains", None)
        if task_domains is not None:
            task_domains = list(task_domains)
            if len(task_domains) != num_tasks or any(
                not isinstance(domain, str) or not domain.strip()
                for domain in task_domains
            ):
                raise ValueError(
                    "Environment task_domains must name one domain per task"
                )
        if hasattr(agent, "action_dim") and int(agent.action_dim) != int(
            env.action_dim
        ):
            raise ValueError(
                "Evaluation environment action dimension does not match the agent"
            )
        if (
            hasattr(agent, "observation_spec")
            and agent.observation_spec != env.observation_spec
        ):
            raise ValueError(
                "Evaluation environment observation spec does not match the agent"
            )

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

        expected_pairs = _evaluation_pairs(
            env, num_tasks, num_embodiments, require_explicit=is_mmbench
        )
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
        per_pair_scores = {pair: [] for pair in expected_pairs}
        if bool(getattr(env, "sequential_evaluation", False)):
            _evaluate_pairs_sequentially(
                agent,
                env,
                expected_pairs=expected_pairs,
                per_pair_returns=per_pair_returns,
                per_pair_lengths=per_pair_lengths,
                per_pair_success=per_pair_success,
                per_pair_scores=per_pair_scores,
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
                        per_pair_success[pair].append(_episode_metric(info, "success"))
                        per_pair_scores[pair].append(_episode_metric(info, "score"))
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
                _require_supported_pairs(
                    next_task_ids, next_embodiment_ids, expected_pairs
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
            scores = per_pair_scores[pair]
            per_pair[key] = {
                "task": task_names[pair[0]],
                "embodiment": embodiment_names[pair[1]],
                "episodes": len(returns),
                "returns": returns,
                "lengths": lengths,
                "successes": successes,
                "scores": scores,
                "return_mean": float(np.mean(returns)) if returns else 0.0,
                "return_std": float(np.std(returns)) if returns else 0.0,
                "length_mean": float(np.mean(lengths)) if lengths else 0.0,
                "success_rate": _complete_mean(successes),
                "success_defined_episodes": sum(
                    value is not None for value in successes
                ),
                "score_mean": _complete_mean(scores),
                "score_defined_episodes": sum(value is not None for value in scores),
            }
            if task_domains is not None:
                per_pair[key]["domain"] = task_domains[pair[0]]

        per_task: dict[str, dict[str, float | None]] = {}
        for task_name in task_names:
            records = [
                record for record in per_pair.values() if record["task"] == task_name
            ]
            per_task[task_name] = {
                "return_mean": float(
                    np.mean([record["return_mean"] for record in records])
                ),
                "success_rate": _complete_mean(
                    [record["success_rate"] for record in records]
                ),
                "score_mean": _complete_mean(
                    [record["score_mean"] for record in records]
                ),
            }
        per_embodiment: dict[str, dict[str, float | None]] = {}
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
                "success_rate": _complete_mean(
                    [record["success_rate"] for record in records]
                ),
                "score_mean": _complete_mean(
                    [record["score_mean"] for record in records]
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
        all_scores = [value for values in per_pair_scores.values() for value in values]
        embodiment_success = [
            record["success_rate"] for record in per_embodiment.values()
        ]
        result = {
            "episodes": len(all_returns),
            "episodes_per_pair": episodes,
            "returns": all_returns,
            "lengths": all_lengths,
            "successes": all_successes,
            "scores": all_scores,
            "return_mean": float(np.mean(all_returns)) if all_returns else 0.0,
            "return_std": float(np.std(all_returns)) if all_returns else 0.0,
            "length_mean": float(np.mean(all_lengths)) if all_lengths else 0.0,
            "success_rate": _complete_mean(all_successes),
            "success_defined_episodes": sum(
                value is not None for value in all_successes
            ),
            "worst_embodiment_success": (
                float(min(embodiment_success))
                if embodiment_success
                and all(value is not None for value in embodiment_success)
                else None
            ),
            "score_mean": _complete_mean(
                [record["score_mean"] for record in per_task.values()]
            ),
            "score_defined_episodes": sum(value is not None for value in all_scores),
            "score_aggregation": "mean_of_task_mean_scores",
            "per_pair": per_pair,
            "per_task": per_task,
            "per_embodiment": per_embodiment,
        }
        if task_domains is not None:
            per_domain = {}
            for domain in dict.fromkeys(task_domains):
                records = [
                    per_task[name]
                    for name, task_domain in zip(task_names, task_domains, strict=True)
                    if task_domain == domain
                ]
                per_domain[domain] = {
                    "task_count": len(records),
                    "return_mean": _complete_mean(
                        [record["return_mean"] for record in records]
                    ),
                    "success_rate": _complete_mean(
                        [record["success_rate"] for record in records]
                    ),
                    "score_mean": _complete_mean(
                        [record["score_mean"] for record in records]
                    ),
                }
            result["per_domain"] = per_domain
        if is_mmbench:
            canonical_count = int(getattr(env, "benchmark_task_count", 200))
            full_selected = (
                bool(getattr(env, "benchmark_full_suite", False))
                and num_tasks == canonical_count == 200
            )
            score_task_count = sum(
                record["score_mean"] is not None for record in per_task.values()
            )
            result["benchmark_coverage"] = {
                "canonical_task_count": canonical_count,
                "selected_task_count": num_tasks,
                "evaluated_task_count": len(per_task),
                "score_task_count": score_task_count,
                "domain_count": len(set(task_domains))
                if task_domains is not None
                else None,
                "evaluated_pair_count": len(per_pair),
                "full_suite_selected": full_selected,
                "complete_score_coverage": score_task_count == num_tasks,
                "full_suite": full_selected and score_task_count == 200,
            }
        return result
    finally:
        env.close()
