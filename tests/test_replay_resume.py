from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import m4po.trainer.off_policy_trainer as trainer_module
from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import M4POConfig
from m4po.trainer import OnlineTrainer
from test_offpolicy_trainer import _FakeAgent, _TimedEnvironment, training as training


def _episode(length=3, marker=0.0):
    return Episode(
        observations={"state": torch.arange(length + 1).float().view(-1, 1) + marker},
        actions=torch.full((length, 1), 0.25),
        rewards=torch.arange(length).float().view(-1, 1),
        terminated=torch.zeros(length, 1),
        task_ids=torch.zeros(length + 1, dtype=torch.long),
        embodiment_ids=torch.zeros(length + 1, dtype=torch.long),
        action_masks=torch.ones(length, 1),
    )


def test_full_replay_roundtrip_preserves_fifo_data_and_next_samples(tmp_path):
    replay = EpisodeReplayBuffer(7, 4, 32, seed=17)
    for marker in (10.0, 20.0, 30.0):
        replay.add(_episode(marker=marker))
    replay.sample()
    path = tmp_path / "replay.pt"
    torch.save(replay.state_dict(), path)
    snapshot = torch.load(path, weights_only=False, map_location="cpu")
    restored = EpisodeReplayBuffer(7, 4, 32, seed=999)
    restored.load_state_dict(snapshot)
    assert len(restored) == 6 and restored.num_episodes == 2
    assert [float(ep.observations["state"][0, 0]) for ep in restored._episodes] == [
        20,
        30,
    ]
    expected, actual = replay.sample(), restored.sample()
    assert torch.equal(expected.observations["state"], actual.observations["state"])
    assert torch.equal(expected.valid, actual.valid)
    assert all(ep.actions.device.type == "cpu" for ep in restored._episodes)
    snapshot["episodes"][0]["actions"].fill_(0.9)
    assert restored._episodes[0].actions.eq(0.25).all()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 9, "schema"),
        ("schema_version", True, "schema"),
        ("capacity", 99, "capacity"),
        ("horizon", 99, "horizon"),
        ("batch_size", 99, "batch_size"),
        ("observation_spec", {"state_dim": 99}, "observation_spec"),
        ("size", 1, "size disagrees"),
        ("rng_state", {"generator": torch.ones(2)}, "RNG state"),
        ("episodes", "not-episodes", "episodes must be"),
    ],
)
def test_invalid_replay_snapshot_does_not_replace_live_buffer(field, value, message):
    replay = EpisodeReplayBuffer(8, 3, 2)
    replay.add(_episode(marker=20))
    snapshot = deepcopy(replay.state_dict())
    snapshot[field] = value
    before = replay.rng_state_dict()["generator"]
    with pytest.raises(ValueError, match=message):
        replay.load_state_dict(snapshot)
    assert len(replay) == 3
    assert replay._episodes[0].observations["state"][0, 0] == 20
    assert torch.equal(before, replay.rng_state_dict()["generator"])


def test_replay_restore_rejects_corrupt_episode_and_capacity_overflow():
    replay = EpisodeReplayBuffer(5, 3, 2)
    replay.add(_episode())
    snapshot = deepcopy(replay.state_dict())
    snapshot["episodes"][0]["rewards"][0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        replay.load_state_dict(snapshot)
    snapshot = replay.state_dict()
    snapshot["episodes"] *= 2
    with pytest.raises(ValueError, match="exceed capacity"):
        replay.load_state_dict(snapshot)


def test_empty_replay_roundtrip_and_observation_contract():
    spec = ObservationSpec((0, 0, 0), 0, 1)
    replay = EpisodeReplayBuffer(8, 3, 2, observation_spec=spec)
    restored = EpisodeReplayBuffer(8, 3, 2, observation_spec=spec)
    restored.load_state_dict(replay.state_dict())
    assert not restored.can_sample
    mismatched = EpisodeReplayBuffer(
        8, 3, 2, observation_spec=replace(spec, state_dim=2)
    )
    with pytest.raises(ValueError, match="observation_spec"):
        mismatched.load_state_dict(replay.state_dict())


def _clock_that_expires_on_update(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        trainer_module,
        "time",
        SimpleNamespace(time=trainer_module.time.time, monotonic=lambda: clock["now"]),
    )
    update = _FakeAgent.update

    def timed_update(self, replay, **kwargs):
        result = update(self, replay, **kwargs)
        clock["now"] = 2.0
        return result

    monkeypatch.setattr(_FakeAgent, "update", timed_update)
    return clock


def test_wall_time_resume_restores_replay_and_finishes_pretraining_without_rewarm(
    training, tmp_path, monkeypatch
):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.total_steps = 16
    cfg.rollout_steps = 1
    monkeypatch.setattr(
        _TimedEnvironment, "preserve_episodes_between_rollouts", True, raising=False
    )
    _clock_that_expires_on_update(monkeypatch)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 4
    assert initial["cfg"]["total_steps"] == 16
    assert initial["extra"]["stop_reason"] == "wall_time_limit"
    assert initial["extra"]["training_complete"] is False
    assert initial["extra"]["resumable"] is True
    assert initial["extra"]["pretrain_updates_completed"] == 1
    assert initial["extra"]["pretrain_done"] is False
    assert initial["extra"]["replay_state"]["size"] == 4

    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final = torch.load(OnlineTrainer(resumed).train(), weights_only=False)
    assert final["step"] == final["cfg"]["total_steps"] == 16
    assert final["extra"]["training_complete"] is True
    assert final["extra"]["replay_saved"] is True
    assert final["extra"]["resume_rebuilds_replay"] is False
    assert final["extra"]["replay_state"]["size"] == 16
    assert final["extra"]["pretrain_updates_completed"] == 2
    assert final["extra"]["update_count"] == 5
    assert _FakeAgent.instances[-1].action_calls == 6
    # The first resumed update is the second pretraining update, using OLD replay.
    assert len(_FakeAgent.instances[-1].batches) == 4


def test_wall_time_resume_keeps_pending_update_credit(training, tmp_path, monkeypatch):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.total_steps = 16
    cfg.pretrain_updates = 0
    cfg.updates_per_step = 2.0
    monkeypatch.setattr(
        _TimedEnvironment, "preserve_episodes_between_rollouts", True, raising=False
    )
    _clock_that_expires_on_update(monkeypatch)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 6
    assert initial["extra"]["update_credit"] == 3.0
    assert initial["extra"]["discarded_partial_transitions"] == 2
    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final = torch.load(OnlineTrainer(resumed).train(), weights_only=False)
    assert final["extra"]["update_count"] == 24
    assert final["extra"]["update_credit"] == 0
    assert _FakeAgent.instances[-1].action_calls == 5


def test_only_latest_checkpoint_embeds_large_replay(training):
    cfg, _ = training
    cfg.save_replay = True
    cfg.total_steps = 8
    cfg.save_every = 4
    path = OnlineTrainer(cfg).train()
    latest = torch.load(path, weights_only=False)
    archived = torch.load(path.parent / "step_8.pt", weights_only=False)
    assert latest["extra"]["replay_saved"] is True
    assert "replay_state" in latest["extra"]
    assert archived["extra"]["replay_saved"] is False
    assert "replay_state" not in archived["extra"]
    status = json.loads((path.parent / "latest_status.json").read_text())
    assert status["step"] == latest["step"]
    for key in (
        "update_count",
        "training_complete",
        "replay_saved",
        "stop_reason",
        "pretrain_updates_completed",
        "update_credit",
        "checkpoint_phase",
        "resumable",
    ):
        assert status[key] == latest["extra"][key]
    assert status["checkpoint_size"] == path.stat().st_size
    assert status["checkpoint_mtime_ns"] == path.stat().st_mtime_ns
    assert not list(path.parent.glob(".*.tmp"))


def test_reaching_step_budget_with_pending_updates_is_not_training_complete(
    training, tmp_path, monkeypatch
):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.total_steps = 6
    cfg.pretrain_updates = 0
    cfg.updates_per_step = 2.0
    _clock_that_expires_on_update(monkeypatch)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == cfg.total_steps
    assert initial["extra"]["update_credit"] == 3.0
    assert initial["extra"]["training_complete"] is False
    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final = torch.load(OnlineTrainer(resumed).train(), weights_only=False)
    assert final["step"] == cfg.total_steps
    assert final["extra"]["training_complete"] is True
    assert final["extra"]["update_count"] == 4
    assert _FakeAgent.instances[-1].action_calls == 0


@pytest.mark.parametrize("total_steps", [6, 12])
def test_resume_evaluates_overdue_boundary_after_updates_before_new_actions(
    training, tmp_path, monkeypatch, total_steps
):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.total_steps = total_steps
    cfg.pretrain_updates = 0
    cfg.updates_per_step = 2.0
    cfg.eval_every = 6
    cfg.eval_episodes = 2
    _clock_that_expires_on_update(monkeypatch)
    evaluations = []

    def evaluate(agent, config, *, episodes, **kwargs):
        evaluations.append((agent.action_calls, len(agent.batches), episodes))
        return {"score_mean": 0.25}

    monkeypatch.setattr(trainer_module, "evaluate_policy", evaluate)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 6
    assert initial["extra"]["last_eval_step"] == -1
    assert initial["extra"]["pending_evaluation"] is True
    assert initial["extra"]["training_complete"] is False
    assert initial["extra"]["update_credit"] == 3
    assert evaluations == []

    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final_path = OnlineTrainer(resumed).train()
    final = torch.load(final_path, weights_only=False)
    assert evaluations[0] == (0, 3, 2)  # Drain three updates, evaluate, then act.
    assert final["extra"]["last_eval_step"] == total_steps
    assert final["extra"]["pending_evaluation"] is False
    assert final["extra"]["training_complete"] is True
    rows = [
        json.loads(line)
        for line in (Path(resumed.log_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows if "eval" in row] == (
        [6] if total_steps == 6 else [6, 12]
    )
    status = json.loads((final_path.parent / "latest_status.json").read_text())
    assert status["last_eval_step"] == total_steps
    assert status["pending_evaluation"] is False


def test_resume_can_finish_only_pending_final_evaluation(
    training, tmp_path, monkeypatch
):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.total_steps = 6
    cfg.pretrain_updates = 0
    cfg.updates_per_step = 0.5  # Exactly one update at the final vector step.
    cfg.eval_every = 6
    _clock_that_expires_on_update(monkeypatch)
    evaluations = []

    def evaluate(agent, config, **kwargs):
        evaluations.append((agent.action_calls, len(agent.batches)))
        return {"score_mean": 0.25}

    monkeypatch.setattr(trainer_module, "evaluate_policy", evaluate)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 6
    assert initial["extra"]["update_credit"] == 0
    assert initial["extra"]["pending_evaluation"] is True
    assert initial["extra"]["training_complete"] is False
    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final = torch.load(OnlineTrainer(resumed).train(), weights_only=False)
    assert evaluations == [(0, 0)]
    assert final["step"] == initial["step"]
    assert final["extra"]["update_count"] == initial["extra"]["update_count"]
    assert final["extra"]["last_eval_step"] == 6
    assert final["extra"]["training_complete"] is True


def test_atomic_checkpoint_failure_preserves_previous_latest(tmp_path):
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous-complete-checkpoint")

    class FailingWriter:
        def save(self, temporary, **kwargs):
            Path(temporary).write_bytes(b"partial")
            raise OSError("injected write failure")

    with pytest.raises(OSError, match="write failure"):
        trainer_module.OffPolicyTrainer._save_checkpoint(
            FailingWriter(), path, step=1, extra={}
        )
    assert path.read_bytes() == b"previous-complete-checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_initial_evaluation_is_not_repeated_on_zero_step_chunk_resume(
    training, tmp_path, monkeypatch
):
    cfg, _ = training
    cfg.save_replay = True
    cfg.max_wall_time_seconds = 1
    cfg.eval_at_start = True
    cfg.eval_episodes = 7
    clock = {"now": 0.0}
    calls = []
    monkeypatch.setattr(
        trainer_module,
        "time",
        SimpleNamespace(time=trainer_module.time.time, monotonic=lambda: clock["now"]),
    )

    def evaluate(*args, episodes, **kwargs):
        calls.append(episodes)
        clock["now"] = 2.0
        return {"score_mean": 0.0}

    monkeypatch.setattr(trainer_module, "evaluate_policy", evaluate)
    first = OnlineTrainer(cfg).train()
    initial = torch.load(first, weights_only=False)
    assert initial["step"] == 0
    assert initial["extra"]["initial_evaluation_completed"] is True
    resumed = M4POConfig(**cfg.to_dict())
    resumed.max_wall_time_seconds = 0
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    OnlineTrainer(resumed).train()
    assert calls == [7]


def test_real_learner_resume_preserves_completed_episode_replay(tmp_path):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.save_replay = True
    cfg.total_steps = 32
    cfg.log_dir = str(tmp_path / "first")
    first_path = OnlineTrainer(cfg).train()
    first = torch.load(first_path, weights_only=False)
    resumed = M4POConfig(**cfg.to_dict())
    resumed.total_steps = 64
    resumed.resume_checkpoint = str(first_path)
    resumed.log_dir = str(tmp_path / "resumed")
    final = torch.load(OnlineTrainer(resumed).train(), weights_only=False)
    assert (
        final["extra"]["replay_state"]["size"] >= first["extra"]["replay_state"]["size"]
    )
    assert (
        final["extra"]["pretrain_updates_completed"]
        == first["extra"]["pretrain_updates_completed"]
    )
    assert final["extra"]["update_count"] > first["extra"]["update_count"]
    assert final["extra"]["training_complete"] is True
