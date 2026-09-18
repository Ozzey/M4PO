from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.envs.mmbench_env import (
    MAX_ACTION_DIM,
    MAX_STATE_DIM,
    _embodiments,
    _native_source_digest,
    _newt_root,
    load_mmbench_catalog,
)


DATASET_REPOSITORY = "nicklashansen/mmbench"
DATASET_REVISION = "a59d457df617400d3e45a5158c8deac8a52055b4"
DATASET_MANIFEST_SHA256 = (
    "0f6085f2a38446a1060074999c164ffba2eb19d8f4f58bb782986228338e3a85"
)
MANISKILL_EPISODES = 20


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def stream_sha256(file) -> str:
    """Hash incrementally, including on the supported Python 3.10 runtime."""

    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DemonstrationDataset:
    replay: EpisodeReplayBuffer
    observation_spec: ObservationSpec
    task_contexts: np.ndarray
    provenance: dict[str, Any]


def iter_demonstration_episodes(
    tensors: Mapping[str, torch.Tensor],
    *,
    task_id: int,
    embodiment_id: int,
    state_dim: int,
    action_dim: int,
    episode_horizon: int,
    max_episodes: int | None = None,
) -> Iterator[Episode]:
    """Convert NEWT's initial dummy row plus transition rows without shifting states.

    In each upstream episode, action/reward/terminated at row t+1 describe the
    transition from observation t to observation t+1. In particular, a timeout
    is not a terminal state for the bootstrapped value target.
    """

    fields = ("obs", "action", "reward", "terminated", "episode")
    if any(name not in tensors for name in fields):
        raise ValueError("Demonstration shard is missing required transition fields")
    values = {name: tensors[name] for name in fields}
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("Demonstration fields must be tensors")
    if any(value.device.type != "cpu" for value in values.values()):
        raise ValueError("Demonstration shards must be loaded onto the CPU")
    if not 0 < state_dim <= MAX_STATE_DIM or not 0 < action_dim <= MAX_ACTION_DIM:
        raise ValueError("Native demonstration dimensions are outside padded bounds")
    if episode_horizon <= 0 or max_episodes is not None and max_episodes <= 0:
        raise ValueError("Demonstration episode limits must be positive")
    obs, actions, rewards, terminated, episode_ids = (values[name] for name in fields)
    if episode_ids.ndim != 1 or episode_ids.numel() < 2:
        raise ValueError("Demonstration episode IDs must be a nonempty row vector")
    rows = episode_ids.numel()
    shapes = {
        "obs": (rows, MAX_STATE_DIM),
        "action": (rows, MAX_ACTION_DIM),
        "reward": (rows,),
        "terminated": (rows,),
        "episode": (rows,),
    }
    if any(tuple(value.shape) != shapes[name] for name, value in values.items()):
        raise ValueError("Demonstration tensors have incompatible row shapes")
    if episode_ids.dtype != torch.int64 or terminated.dtype != torch.bool:
        raise ValueError("Episode IDs must be int64 and termination flags bool")
    if any(not value.is_floating_point() for value in (obs, actions, rewards)):
        raise ValueError(
            "Demonstration observations, actions and rewards must be floats"
        )
    unique, counts = torch.unique_consecutive(episode_ids, return_counts=True)
    if not torch.equal(unique, torch.arange(len(unique), dtype=torch.int64)):
        raise ValueError("Episode IDs must be contiguous, ordered, and begin at zero")
    if not torch.isfinite(obs).all() or torch.any(obs[:, state_dim:] != 0):
        raise ValueError("Demonstration observations must be finite with zero padding")
    offset = 0
    for index, count in enumerate(counts.tolist()):
        end = offset + count
        if count < 2:
            raise ValueError("A demonstration episode must contain a transition")
        length = count - 1
        if length > episode_horizon:
            raise ValueError("Demonstration episode exceeds the native time limit")
        if not torch.isnan(actions[offset]).all() or not torch.isnan(rewards[offset]):
            raise ValueError("Initial demonstration action/reward must be dummy NaNs")
        if terminated[offset] or terminated[offset + 1 : end - 1].any():
            raise ValueError("Only the final demonstration transition may terminate")
        if length != episode_horizon and not terminated[end - 1]:
            raise ValueError("Nonterminal demonstration episode is incomplete")
        action = actions[offset + 1 : end]
        reward = rewards[offset + 1 : end]
        if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
            raise ValueError("Demonstration transitions must contain finite values")
        if torch.any(action.abs() > 1.0 + 1e-6):
            raise ValueError("Demonstration actions must lie in [-1, 1]")
        if torch.any(action[:, action_dim:] != 0):
            raise ValueError("Invalid demonstration action coordinates must be zero")
        if max_episodes is None or index < max_episodes:
            state_mask = torch.zeros(count, MAX_STATE_DIM)
            state_mask[:, :state_dim] = 1
            action_mask = torch.zeros(length, MAX_ACTION_DIM)
            action_mask[:, :action_dim] = 1
            yield Episode(
                observations={
                    "image": torch.empty(count, 0, 0, 0),
                    "proprio": torch.empty(count, 0),
                    "proprio_mask": torch.empty(count, 0),
                    "state": obs[offset:end],
                    "state_mask": state_mask,
                },
                actions=action,
                rewards=reward.unsqueeze(-1),
                terminated=terminated[offset + 1 : end].float().unsqueeze(-1),
                task_ids=torch.full((count,), task_id, dtype=torch.long),
                embodiment_ids=torch.full((count,), embodiment_id, dtype=torch.long),
                action_masks=action_mask,
            )
        offset = end
    if max_episodes is not None and len(counts) < max_episodes:
        raise ValueError(
            "Demonstration shard has fewer than the required capped episodes"
        )


def load_demonstrations(cfg, data_dir: str | Path, preflight_report: str | Path):
    """Verify pinned artifacts, then retain CPU episodes without FIFO eviction.

    Only one source shard (including any unused visual features) is deserialized
    at a time. Checkpoints save this immutable dataset's identity and sampler
    RNG, not another copy of the demonstrations.
    """

    cfg.validate()
    if cfg.env != "mmbench" or cfg.learning_mode != "off_policy":
        raise ValueError("Demonstration pretraining requires off-policy MMBench")
    metadata, task_sets, catalog_digest = load_mmbench_catalog(cfg.mmbench_root)
    tasks = list(task_sets["soup"])
    if len(tasks) != 200 or list(cfg.task_names) != tasks:
        raise ValueError(
            "Demonstration pretraining requires all 200 tasks in canonical order"
        )
    directory = Path(data_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = manifest.pop("dataset_manifest_sha256", None)
    if digest != canonical_digest(manifest) or digest != DATASET_MANIFEST_SHA256:
        raise ValueError("Demonstration manifest digest does not match its content")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repo_id") != DATASET_REPOSITORY
        or manifest.get("revision") != DATASET_REVISION
        or manifest.get("catalog_sha256") != catalog_digest
        or manifest.get("task_count") != 200
    ):
        raise ValueError("Demonstration manifest is not the pinned canonical dataset")
    files = manifest.get("files", [])
    if not isinstance(files, list) or [row.get("task") for row in files] != tasks:
        raise ValueError(
            "Demonstration manifest must list every canonical task exactly once"
        )
    if manifest.get("total_bytes") != sum(row.get("size", -1) for row in files):
        raise ValueError("Demonstration manifest byte count is inconsistent")
    report_bytes = Path(preflight_report).read_bytes()
    report = json.loads(report_bytes)
    native_digest = _native_source_digest(_newt_root(cfg.mmbench_root))
    if (
        report.get("full_suite_passed") is not True
        or report.get("complete") is not True
        or report.get("passed_task_count") != 200
        or report.get("failed_task_count") != 0
        or report.get("catalog_sha256") != catalog_digest
        or report.get("required_tasks") != tasks
        or set(report.get("per_task", {})) != set(tasks)
    ):
        raise ValueError("A successful full-suite native preflight is required")
    _, bodies = _embodiments(tasks, metadata)
    spec = ObservationSpec((0, 0, 0), 0, MAX_STATE_DIM)
    replay = EpisodeReplayBuffer(
        cfg.replay_capacity,
        cfg.model_horizon,
        cfg.batch_size,
        seed=cfg.seed,
        observation_spec=spec,
    )
    contexts = np.asarray(
        [metadata[name]["text_embedding"] for name in tasks], dtype=np.float32
    )
    if contexts.shape != (200, 512) or not np.isfinite(contexts).all():
        raise ValueError("Official task embeddings must be finite 512-vectors")
    coverage = {}
    for task_id, (task, record) in enumerate(zip(tasks, files)):
        native = report["per_task"][task]
        row = metadata[task]
        if (
            native.get("status") != "passed"
            or native.get("catalog_sha256") != catalog_digest
            or native.get("native_sources_sha256") != native_digest
            or native.get("native_action_dim") != row["action_dim"]
            or native.get("embodiment") != row["embodiment"]
            or native.get("metadata_horizon") != row["max_episode_steps"]
            or type(native.get("native_state_dim")) is not int
            or not 0 < native["native_state_dim"] <= MAX_STATE_DIM
        ):
            raise ValueError(f"Native preflight contract mismatch for {task}")
        path = directory / f"{task}.pt"
        if record.get("path") != path.name or path.resolve().parent != directory:
            raise ValueError("Demonstration shard path is not canonical")
        expected_size, expected_sha = record.get("size"), record.get("sha256")
        if (
            type(expected_size) is not int
            or expected_size <= 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise ValueError("Demonstration shard has invalid integrity metadata")
        with path.open("rb") as file:
            if path.stat().st_size != expected_size:
                raise ValueError(f"Demonstration shard size mismatch: {task}")
            actual = stream_sha256(file)
            if actual != expected_sha:
                raise ValueError(f"Demonstration shard SHA256 mismatch: {task}")
            file.seek(0)
            # Official shards serialize TensorDict; validate bytes before this
            # trusted, pinned deserialization and never load them onto CUDA.
            payload = torch.load(file, map_location="cpu", weights_only=False)
        tensors = {
            name: payload[name]
            for name in ("obs", "action", "reward", "terminated", "episode")
        }
        del payload
        cap = MANISKILL_EPISODES if task in task_sets.get("maniskill", []) else None
        episodes_before, transitions_before = replay.num_episodes, len(replay)
        for episode in iter_demonstration_episodes(
            tensors,
            task_id=task_id,
            embodiment_id=bodies.index(row["embodiment"]),
            state_dim=native["native_state_dim"],
            action_dim=row["action_dim"],
            episode_horizon=row["max_episode_steps"],
            max_episodes=cap,
        ):
            replay.capacity = max(replay.capacity, len(replay) + episode.length)
            replay.add(episode)
        del tensors
        coverage[task] = {
            "episodes": replay.num_episodes - episodes_before,
            "transitions": len(replay) - transitions_before,
            "native_state_dim": native["native_state_dim"],
        }
        if coverage[task]["episodes"] == 0:
            raise ValueError(f"No valid demonstration episodes for {task}")
    return DemonstrationDataset(
        replay=replay,
        observation_spec=spec,
        task_contexts=contexts,
        provenance={
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": digest,
            "catalog_sha256": catalog_digest,
            "native_sources_sha256": native_digest,
            "preflight_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "task_count": 200,
            "task_names": tasks,
            "full_suite": True,
            "per_task": coverage,
            "episodes": replay.num_episodes,
            "transitions": len(replay),
            "maniskill_episodes_per_task": MANISKILL_EPISODES,
            "row_conversion": "observations[0:T+1]; action,reward,terminated[1:T+1]",
            "sampling": "uniform_transition_starts_with_validity_masked_tails",
            "online_demo_mixing": False,
        },
    )
