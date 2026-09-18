from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from m4po.common.buffer import ObservationSpec
from m4po.common.config import M4POConfig
from m4po.envs.mmbench_env import (
    _native_source_digest,
    _newt_root,
    load_mmbench_catalog,
)
from m4po.m4po import M4POAgent


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_demonstration_initialization(
    path: str | Path,
    cfg: M4POConfig,
    observation_spec: ObservationSpec,
    action_dim: int,
    task_contexts: np.ndarray,
    device: torch.device,
) -> tuple[M4POAgent, dict[str, Any]]:
    """Load completed M4PO demo pretraining, never foreign NEWT parameters.

    The online trainer owns fresh counters and replay. Optimizers and learned
    parameters transfer, while all task/control/model semantics must match.
    """

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Initialization checkpoint must be a mapping")
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ValueError("Initialization checkpoint metadata must be a mapping")
    provenance = extra.get("demonstration_pretraining", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("Initialization demonstration provenance must be a mapping")
    updates = provenance.get("completed_updates")
    if (
        cfg.env != "mmbench"
        or cfg.learning_mode != "off_policy"
        or payload.get("step") != 0
        or extra.get("stage") != "demonstration_pretraining"
        or extra.get("training_complete") is not True
        or provenance.get("stage_completed") is not True
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates <= 0
        or type(provenance.get("target_updates")) is not int
        or type(extra.get("updates_completed")) is not int
        or updates != provenance.get("target_updates")
        or updates != extra.get("updates_completed")
        or provenance.get("actor_objective") != "masked_behavior_cloning_plus_entropy"
        or provenance.get("online_demo_mixing") is not False
    ):
        raise ValueError(
            "Initialization requires completed M4PO demonstration pretraining"
        )
    saved_cfg = M4POConfig.from_checkpoint_dict(payload["cfg"])
    runtime_keys = {
        "seed",
        "num_envs",
        "total_steps",
        "seed_steps",
        "pretrain_updates",
        "rollout_steps",
        "eval_every",
        "eval_at_start",
        "eval_episodes",
        "save_every",
        "log_every",
        "log_dir",
        "device",
        "quiet",
        "torch_deterministic",
        "resume_checkpoint",
        "init_checkpoint",
        "save_replay",
        "max_wall_time_seconds",
        "mmbench_root",
        "mmbench_eval_num_envs",
    }
    saved_values = saved_cfg.to_dict()
    mismatches = [
        key
        for key, value in cfg.to_dict().items()
        if key not in runtime_keys and value != saved_values[key]
    ]
    if mismatches:
        raise ValueError(
            f"Initialization configuration differs: {', '.join(mismatches)}"
        )
    _, _, catalog_hash = load_mmbench_catalog(cfg.mmbench_root)
    if (
        provenance.get("task_count") != cfg.num_tasks
        or provenance.get("task_names") != cfg.task_names
        or provenance.get("catalog_sha256") != catalog_hash
        or provenance.get("native_sources_sha256")
        != _native_source_digest(_newt_root(cfg.mmbench_root))
        or not isinstance(provenance.get("dataset_sha256"), str)
        or re.fullmatch(r"[a-f0-9]{64}", provenance["dataset_sha256"]) is None
        or ObservationSpec(**payload["observation_spec"]) != observation_spec
        or payload.get("action_dim") != action_dim
    ):
        raise ValueError(
            "Initialization task, dataset, or native environment semantics differ"
        )
    saved_contexts = payload.get("task_contexts")
    if saved_contexts is None or not np.array_equal(
        saved_contexts.cpu().numpy(), task_contexts
    ):
        raise ValueError("Initialization frozen task contexts differ")
    agent = M4POAgent(
        observation_spec, action_dim, cfg, device, task_contexts=task_contexts
    )
    agent._load_payload(payload, restore_optimizers=True)
    return agent, {
        "kind": "demonstration_pretraining",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "demonstration_pretraining": provenance,
    }
