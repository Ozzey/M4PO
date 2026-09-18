from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import m4po.trainer.off_policy_trainer as trainer_module
from m4po.common.config import CHECKPOINT_SCHEMA_VERSION, IMPLEMENTATION_ID, M4POConfig
from m4po.trainer import OnlineTrainer


class _FakeAgent:
    instances = []
    fail_update = False

    def __init__(self, observation_spec, action_dim, cfg, _device, **_kwargs):
        self.observation_spec = observation_spec
        self.action_dim = action_dim
        self.cfg = cfg
        self.batches = []
        self.episodes = []
        self.action_calls = 0
        self.instances.append(self)

    def act_np(self, observation, **_kwargs):
        self.action_calls += 1
        return SimpleNamespace(
            action=np.zeros((self.cfg.num_envs, self.action_dim), dtype=np.float32)
        )

    def update(self, replay, **_kwargs):
        if self.fail_update:
            raise RuntimeError("injected optimizer failure")
        self.batches.append(replay.sample())
        self.episodes.extend(replay._episodes)
        return {"loss": 1.0}

    def save(self, path, *, step, extra):
        from dataclasses import asdict

        torch.save(
            {
                "implementation_id": IMPLEMENTATION_ID,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "cfg": self.cfg.to_dict(),
                "observation_spec": asdict(self.observation_spec),
                "action_dim": self.action_dim,
                "step": step,
                "extra": extra,
            },
            path,
        )

    @classmethod
    def load(cls, path, device, *, override_cfg, restore_optimizers, _payload=None):
        from m4po.common.buffer import ObservationSpec

        assert restore_optimizers
        payload = _payload if _payload is not None else torch.load(path, weights_only=False)
        return cls(
            ObservationSpec(**payload["observation_spec"]),
            payload["action_dim"],
            override_cfg,
            device,
        )


class _TimedEnvironment:
    """Use real adapter metadata, but force predictable two-step time limits."""

    def __init__(self, env):
        self.env = env
        self.actions = []
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_rollout(self):
        self.observation = self.env.reset_rollout()
        self.steps = 0
        return self.observation

    def step(self, actions):
        self.actions.append(actions.copy())
        self.steps += 1
        done = np.full(self.num_envs, self.steps % 2 == 0, dtype=np.bool_)
        terminal = {
            name: np.full_like(value, 100 + self.steps)
            for name, value in self.observation.items()
        }
        next_observation = {
            name: np.full_like(value, -100 if done.all() else self.steps)
            for name, value in self.observation.items()
        }
        infos = [
            {
                "terminated": False,
                "truncated": bool(done[index]),
                "success": True,
                "terminal_observation": {
                    name: value[index].copy() for name, value in terminal.items()
                },
            }
            for index in range(self.num_envs)
        ]
        self.observation = next_observation
        return next_observation, np.ones(self.num_envs, dtype=np.float32), done, infos

    def close(self):
        self.closed = True
        self.env.close()


@pytest.fixture
def training(tmp_path, monkeypatch):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.learning_mode = "off_policy"
    cfg.num_envs = 2
    cfg.control_repeats = "1,1"
    cfg.max_episode_steps = 2
    cfg.rollout_steps = 3
    cfg.total_steps = 14
    cfg.seed_steps = 4
    cfg.pretrain_updates = 2
    cfg.updates_per_step = 0.25
    cfg.batch_size = 4
    cfg.model_horizon = 3
    cfg.log_every = 2
    cfg.save_every = 0
    cfg.eval_every = 0
    cfg.log_dir = str(tmp_path / "run")
    factory = trainer_module.make_vector_env
    environments = []

    def make_env(config):
        env = _TimedEnvironment(factory(config))
        environments.append(env)
        return env

    _FakeAgent.instances = []
    _FakeAgent.fail_update = False
    monkeypatch.setattr(trainer_module, "M4POAgent", _FakeAgent)
    monkeypatch.setattr(trainer_module, "make_vector_env", make_env)
    return cfg, environments


def test_continuous_episodes_cross_collection_segments_and_keep_undefined_success(
    training, monkeypatch
):
    import json

    cfg, environments = training
    cfg.rollout_steps = 1
    monkeypatch.setattr(
        _TimedEnvironment, "preserve_episodes_between_rollouts", True, raising=False
    )
    original_step = _TimedEnvironment.step

    def step(self, actions):
        observation, reward, done, infos = original_step(self, actions)
        for info in infos:
            info["success"] = float("nan")
            info["score"] = 0.25
        return observation, reward, done, infos

    monkeypatch.setattr(_TimedEnvironment, "step", step)
    path = OnlineTrainer(cfg).train()
    payload = torch.load(path, weights_only=False)
    assert payload["extra"]["completed_episodes"] == 6
    assert payload["extra"]["discarded_partial_transitions"] == 2
    assert environments[0].steps == 7
    rows = [
        json.loads(line)
        for line in (Path(cfg.log_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["train_success_rate_20"] is None
    assert rows[-1]["train_score_mean_20"] == 0.25
    assert rows[-1]["updates"] > 0


def test_offpolicy_collection_masks_seed_actions_and_preserves_terminal_observations(
    training,
):
    import json

    cfg, environments = training
    path = OnlineTrainer(cfg).train()
    payload = torch.load(path, weights_only=False)
    agent = _FakeAgent.instances[-1]
    assert payload["step"] == 14
    assert payload["extra"]["update_count"] == 4
    assert payload["extra"]["pretrain_updates_completed"] == 2
    assert payload["extra"]["update_credit"] == 0.5
    assert payload["extra"]["completed_episodes"] == 4
    assert payload["extra"]["discarded_partial_transitions"] == 6
    assert agent.action_calls == 5
    assert environments[0].closed
    assert any(np.any(action != 0) for action in environments[0].actions[:2])
    for episode in agent.episodes:
        assert episode.length == 2
        assert episode.terminated.eq(0).all()
        assert episode.observations["proprio"][-1].eq(102).all()
        assert (episode.actions * (1 - episode.action_masks)).eq(0).all()
    rows = [
        json.loads(line)
        for line in (Path(cfg.log_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["phase"] == "off_policy_training"
    assert rows[-1]["train_success_rate_20"] == 1.0
    assert any(
        record["success_rate_20"] == 1.0
        for record in rows[-1]["train_per_task"].values()
    )


def test_resume_rebuilds_replay_without_repeating_pretraining(training, tmp_path):
    cfg, _ = training
    cfg.total_steps = 8
    first = OnlineTrainer(cfg).train()
    first_payload = torch.load(first, weights_only=False)
    resumed = M4POConfig(**cfg.to_dict())
    resumed.total_steps = 16
    resumed.resume_checkpoint = str(first)
    resumed.log_dir = str(tmp_path / "resumed")
    final = OnlineTrainer(resumed).train()
    final_payload = torch.load(final, weights_only=False)
    assert first_payload["extra"]["pretrain_updates_completed"] == 2
    assert final_payload["extra"]["pretrain_updates_completed"] == 2
    assert (
        final_payload["extra"]["update_count"]
        == first_payload["extra"]["update_count"] + 1
    )
    assert len(_FakeAgent.instances[-1].batches) == 1
    assert _FakeAgent.instances[-1].action_calls == 2
    assert final_payload["extra"]["resume_rebuilds_replay"] is True


def test_partial_optimizer_checkpoint_is_diagnostic_only(training):
    cfg, environments = training
    _FakeAgent.fail_update = True
    with pytest.raises(RuntimeError, match="optimizer failure"):
        OnlineTrainer(cfg).train()
    path = Path(cfg.log_dir) / "checkpoints" / "emergency.pt"
    payload = torch.load(path, weights_only=False)
    assert payload["extra"]["resumable"] is False
    assert environments[0].closed
    with pytest.raises(ValueError, match="diagnostic-only"):
        trainer_module.OffPolicyTrainer._validate_off_policy_resume(payload, cfg)


def test_resume_rejects_learning_mode_mismatch(training):
    cfg, _ = training
    path = OnlineTrainer(cfg).train()
    payload = torch.load(path, weights_only=False)
    payload["cfg"]["learning_mode"] = "on_policy"
    with pytest.raises(ValueError, match="on-policy"):
        trainer_module.OffPolicyTrainer._validate_off_policy_resume(payload, cfg)


def test_forced_task_reset_discards_partial_episodes(training):
    cfg, _ = training
    cfg.rollout_steps = 1
    cfg.total_steps = 6
    cfg.seed_steps = 0
    path = OnlineTrainer(cfg).train()
    payload = torch.load(path, weights_only=False)
    assert payload["extra"]["completed_episodes"] == 0
    assert payload["extra"]["discarded_partial_transitions"] == 6
    assert payload["extra"]["update_count"] == 0
    assert not _FakeAgent.instances[-1].batches
    assert _FakeAgent.instances[-1].action_calls == 0


def test_partial_collection_failure_keeps_committed_counters(training, monkeypatch):
    cfg, _ = training
    original_add = trainer_module.EpisodeReplayBuffer.add
    calls = 0

    def fail_second_episode(self, episode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected collection failure")
        original_add(self, episode)

    monkeypatch.setattr(trainer_module.EpisodeReplayBuffer, "add", fail_second_episode)
    with pytest.raises(RuntimeError, match="collection failure"):
        OnlineTrainer(cfg).train()
    payload = torch.load(
        Path(cfg.log_dir) / "checkpoints" / "emergency.pt", weights_only=False
    )
    assert payload["step"] == 2
    assert payload["extra"]["completed_episodes"] == 0
    assert payload["extra"]["resumable"] is True
