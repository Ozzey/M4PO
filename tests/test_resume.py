from __future__ import annotations

from pathlib import Path

import pytest
import torch

import m4po.trainer.online_trainer as trainer_module
from m4po.common.config import (
    CHECKPOINT_SCHEMA_VERSION,
    IMPLEMENTATION_ID,
    M4POConfig,
)
from m4po.m4po import M4POAgent
from m4po.trainer import OnlineTrainer


def _assert_learned_components_equal(first: dict, second: dict) -> None:
    for component in ("model", "actor", "external_critic", "augmented_critic"):
        assert first[component].keys() == second[component].keys()
        for name, value in first[component].items():
            assert torch.equal(value, second[component][name]), (component, name)


def test_setup_failure_closes_environment_and_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.total_steps = cfg.rollout_batch_size
    cfg.log_dir = str(tmp_path / "setup_failure")
    environment = trainer_module.make_vector_env(cfg)
    original_close = environment.close
    close_calls = 0

    def track_environment_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    environment.close = track_environment_close

    loggers = []

    class TrackingLogger:
        def __init__(self, _path: Path):
            self.closed = False
            loggers.append(self)

        def close(self) -> None:
            self.closed = True

    def fail_agent_construction(*_args, **_kwargs):
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(trainer_module, "make_vector_env", lambda _cfg: environment)
    monkeypatch.setattr(trainer_module, "JsonlLogger", TrackingLogger)
    monkeypatch.setattr(trainer_module, "M4POAgent", fail_agent_construction)

    with pytest.raises(RuntimeError, match="injected setup failure"):
        OnlineTrainer(cfg).train()

    assert close_calls == 1
    assert len(loggers) == 1
    assert loggers[0].closed
    assert not (Path(cfg.log_dir) / "checkpoints" / "emergency.pt").exists()


def test_interrupted_partial_rollout_resumes_from_completed_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.total_steps = cfg.rollout_batch_size
    cfg.save_every = 0
    cfg.log_every = 0
    cfg.log_dir = str(tmp_path / "interrupted")

    original_act_np = M4POAgent.act_np
    action_calls = 0

    def interrupt_second_action(self, *args, **kwargs):
        nonlocal action_calls
        action_calls += 1
        if action_calls == 2:
            raise RuntimeError("injected collection failure")
        return original_act_np(self, *args, **kwargs)

    monkeypatch.setattr(M4POAgent, "act_np", interrupt_second_action)
    with pytest.raises(RuntimeError, match="injected collection failure"):
        OnlineTrainer(cfg).train()

    emergency = Path(cfg.log_dir) / "checkpoints" / "emergency.pt"
    payload = torch.load(emergency, map_location="cpu", weights_only=False)
    assert payload["step"] == 0
    assert payload["extra"]["update_count"] == 0
    assert payload["extra"]["resumable"] is True

    monkeypatch.setattr(M4POAgent, "act_np", original_act_np)
    resumed = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    resumed.total_steps = resumed.rollout_batch_size
    resumed.save_every = 0
    resumed.log_every = 0
    resumed.log_dir = str(tmp_path / "resumed")
    resumed.resume_checkpoint = str(emergency)
    checkpoint = OnlineTrainer(resumed).train()
    resumed_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed_payload["step"] == resumed.total_steps
    assert resumed_payload["extra"]["update_count"] == 1


def test_resume_payload_rejects_mismatch_and_unsafe_boundaries() -> None:
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.total_steps = cfg.rollout_batch_size
    payload = {
        "implementation_id": IMPLEMENTATION_ID,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "cfg": cfg.to_dict(),
        "step": 0,
        "extra": {"update_count": 0, "completed_episodes": 0, "resumable": True},
    }

    incompatible = M4POConfig(**cfg.to_dict())
    incompatible.planner_samples += 1
    with pytest.raises(ValueError, match="planner_samples"):
        OnlineTrainer._validate_resume_payload(
            payload,
            incompatible,
            incompatible.rollout_batch_size,
        )

    payload["step"] = cfg.num_envs
    with pytest.raises(ValueError, match="completed fresh-rollout boundary"):
        OnlineTrainer._validate_resume_payload(
            payload,
            cfg,
            cfg.rollout_batch_size,
        )

    payload["step"] = 0
    payload["extra"]["resumable"] = False
    with pytest.raises(ValueError, match="diagnostic-only"):
        OnlineTrainer._validate_resume_payload(
            payload,
            cfg,
            cfg.rollout_batch_size,
        )


def test_boundary_resume_reproduces_uninterrupted_parameters(tmp_path: Path) -> None:
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.total_steps = 2 * cfg.rollout_batch_size
    cfg.eval_every = 0
    cfg.save_every = cfg.rollout_batch_size
    cfg.log_every = 0
    cfg.torch_deterministic = True
    cfg.log_dir = str(tmp_path / "uninterrupted")
    uninterrupted_checkpoint = OnlineTrainer(cfg).train()
    boundary_checkpoint = (
        Path(cfg.log_dir) / "checkpoints" / f"step_{cfg.rollout_batch_size}.pt"
    )
    boundary = torch.load(boundary_checkpoint, map_location="cpu", weights_only=False)
    assert boundary["extra"]["rng_state"] is not None
    assert boundary["extra"]["environment_state"] is not None

    resumed = M4POConfig(**cfg.to_dict())
    resumed.log_dir = str(tmp_path / "resumed")
    resumed.resume_checkpoint = str(boundary_checkpoint)
    resumed_checkpoint = OnlineTrainer(resumed).train()

    uninterrupted_payload = torch.load(
        uninterrupted_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    resumed_payload = torch.load(
        resumed_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    _assert_learned_components_equal(uninterrupted_payload, resumed_payload)


def test_periodic_evaluation_does_not_change_training_rng(tmp_path: Path) -> None:
    baseline = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    baseline.total_steps = 2 * baseline.rollout_batch_size
    baseline.eval_every = 0
    baseline.save_every = 0
    baseline.log_every = 0
    baseline.torch_deterministic = True
    baseline.log_dir = str(tmp_path / "without_eval")
    baseline_checkpoint = OnlineTrainer(baseline).train()

    with_eval = M4POConfig(**baseline.to_dict())
    with_eval.eval_every = with_eval.rollout_batch_size
    with_eval.log_dir = str(tmp_path / "with_eval")
    with_eval_checkpoint = OnlineTrainer(with_eval).train()

    baseline_payload = torch.load(
        baseline_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    with_eval_payload = torch.load(
        with_eval_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    _assert_learned_components_equal(baseline_payload, with_eval_payload)
