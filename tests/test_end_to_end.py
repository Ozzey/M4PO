from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from m4po.common.config import M4POConfig
from m4po.common.evaluation import evaluate_policy
from m4po.m4po import M4POAgent
from m4po.trainer import OnlineTrainer


def _optimizer_step(payload: dict, name: str) -> int:
    steps = [
        int(state["step"])
        for state in payload[name]["state"].values()
        if "step" in state
    ]
    assert steps
    assert len(set(steps)) == 1
    return steps[0]


def test_mock_train_and_evaluate_pipeline(tmp_path: Path):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.learning_mode = "on_policy"
    cfg.total_steps = cfg.num_envs * cfg.rollout_steps
    cfg.save_every = 0
    cfg.log_every = cfg.total_steps
    cfg.log_dir = str(tmp_path / "run")
    checkpoint = OnlineTrainer(cfg).train()
    assert checkpoint.exists()
    assert (Path(cfg.log_dir) / "config_resolved.yaml").exists()
    lines = (Path(cfg.log_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines and json.loads(lines[-1])["phase"] == "on_policy_training"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == cfg.total_steps
    assert payload["extra"]["update_count"] == 1
    assert payload["extra"]["environment_signature"]["task_names"] == cfg.task_names

    agent = M4POAgent.load(checkpoint, torch.device("cpu"))
    actor_before = {
        name: value.detach().clone() for name, value in agent.actor.state_dict().items()
    }
    results = evaluate_policy(agent, agent.cfg, episodes=1)
    assert results["episodes"] == cfg.num_tasks * cfg.num_embodiments
    expected_pairs = {
        f"{task}/{embodiment}"
        for task in cfg.task_names
        for embodiment in cfg.embodiment_names
    }
    assert set(results["per_pair"]) == expected_pairs
    assert all(record["episodes"] == 1 for record in results["per_pair"].values())
    assert set(results["per_task"]) == set(cfg.task_names)
    assert set(results["per_embodiment"]) == set(cfg.embodiment_names)
    assert len(results["returns"]) == results["episodes"]
    assert len(results["lengths"]) == results["episodes"]
    assert len(results["successes"]) == results["episodes"]
    assert math.isclose(
        results["return_mean"],
        sum(results["returns"]) / results["episodes"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert results["worst_embodiment_success"] == min(
        record["success_rate"] for record in results["per_embodiment"].values()
    )
    assert 0.0 <= results["success_rate"] <= 1.0
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in agent.actor.state_dict().items()
    )


def test_resume_restores_optimizers_and_exact_progress(tmp_path: Path):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.learning_mode = "on_policy"
    cfg.total_steps = cfg.rollout_batch_size
    cfg.eval_every = 0
    cfg.save_every = 0
    cfg.log_every = 0
    cfg.log_dir = str(tmp_path / "first")
    first_checkpoint = OnlineTrainer(cfg).train()
    first = torch.load(first_checkpoint, map_location="cpu", weights_only=False)

    resumed_cfg = replace(
        cfg,
        total_steps=2 * cfg.rollout_batch_size,
        log_dir=str(tmp_path / "resumed"),
        resume_checkpoint=str(first_checkpoint),
    )
    resumed_checkpoint = OnlineTrainer(resumed_cfg).train()
    resumed = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)

    assert first["step"] == cfg.rollout_batch_size
    assert resumed["step"] == resumed_cfg.total_steps
    assert first["extra"]["update_count"] == 1
    assert resumed["extra"]["update_count"] == 2
    assert resumed["extra"]["environment_signature"] == first["extra"][
        "environment_signature"
    ]
    for optimizer in (
        "world_model_optimizer",
        "actor_optimizer",
        "critic_optimizer",
    ):
        assert _optimizer_step(resumed, optimizer) > _optimizer_step(first, optimizer)

    unaligned = deepcopy(first)
    unaligned["step"] = cfg.num_envs
    unaligned["extra"]["update_count"] = 0
    with pytest.raises(ValueError, match="completed fresh-rollout boundary"):
        OnlineTrainer._validate_resume_payload(
            unaligned,
            resumed_cfg,
            resumed_cfg.rollout_batch_size,
        )

    partial_update = deepcopy(first)
    partial_update["extra"]["resumable"] = False
    with pytest.raises(ValueError, match="partial optimizer update"):
        OnlineTrainer._validate_resume_payload(
            partial_update,
            resumed_cfg,
            resumed_cfg.rollout_batch_size,
        )
