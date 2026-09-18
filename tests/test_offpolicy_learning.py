from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import CHECKPOINT_SCHEMA_VERSION, M4POConfig
from m4po.common.evaluation import evaluate_policy
from m4po.m4po import M4POAgent
from m4po.trainer import OnlineTrainer


def learner(tmp_path, *, images=False):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.use_images = images
    cfg.task_context_mode = "learned"
    cfg.log_dir = str(tmp_path / "run")
    spec = ObservationSpec.from_config(cfg)
    agent = M4POAgent(spec, 3, cfg, torch.device("cpu"))
    replay = EpisodeReplayBuffer(
        32, cfg.model_horizon, cfg.batch_size, seed=4, observation_spec=spec
    )
    for task, length in ((0, 3), (1, 1)):
        observations = {
            name: (
                torch.ones(length + 1, *shape)
                if name.endswith("mask")
                else torch.rand(length + 1, *shape)
            )
            for name, shape in spec.shapes.items()
        }
        mask = torch.tensor([1.0, float(task), 1.0]).expand(length, 3).clone()
        replay.add(
            Episode(
                observations=observations,
                actions=(torch.rand(length, 3) * 2 - 1) * mask,
                rewards=torch.full((length, 1), 0.5 + task),
                terminated=torch.cat((torch.zeros(length - 1, 1), torch.ones(1, 1))),
                task_ids=torch.full((length + 1,), task, dtype=torch.long),
                embodiment_ids=torch.full((length + 1,), task, dtype=torch.long),
                action_masks=mask,
            )
        )
    return agent, replay


@pytest.mark.parametrize("images", [False, True])
def test_replay_reuse_updates_q_and_actor_without_ppo(tmp_path, monkeypatch, images):
    torch.manual_seed(2)
    agent, replay = learner(tmp_path, images=images)
    before = deepcopy(agent.actor.state_dict())
    target_before = deepcopy(agent.model.target_q_functions.state_dict())

    def no_ppo(*_args, **_kwargs):
        pytest.fail("Replay learning must not call PPO or augmented-reward targets")

    monkeypatch.setattr(agent, "_prepare_targets", no_ppo)
    monkeypatch.setattr(agent.planner, "recompute", no_ppo)
    actor_step = agent.actor_optimizer.step

    def isolated_actor_step(*args, **kwargs):
        assert all(parameter.grad is None for parameter in agent.model.parameters())
        assert not any(
            parameter.requires_grad for parameter in agent.model.parameters()
        )
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in agent.actor.parameters()
        )
        return actor_step(*args, **kwargs)

    monkeypatch.setattr(agent.actor_optimizer, "step", isolated_actor_step)
    for _ in range(2):
        metrics = agent.update(replay)
        assert all(math.isfinite(value) for value in metrics.values())
        assert metrics["exploration_bonus_abs_mean"] == 0
        assert metrics["policy_optimization_effective_weight"] == 0
        assert metrics["actor_learning_rate"] == agent.cfg.actor_lr
        assert "approximate_kl" not in metrics
    assert len(replay) == 4
    assert replay.num_episodes == 2
    assert not hasattr(agent, "external_critic")
    assert any(
        not torch.equal(before[name], value)
        for name, value in agent.actor.state_dict().items()
    )
    assert any(
        not torch.equal(target_before[name], value)
        for name, value in agent.model.target_q_functions.state_dict().items()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.model.target_q_functions.parameters()
    )


def test_replay_td_target_uses_extrinsic_reward_and_terminal_mask(tmp_path):
    agent, replay = learner(tmp_path)
    batch = replay.sample()
    # Zero terminations represent time limits: preserve the final-state bootstrap.
    batch = replace(batch, terminated=torch.zeros_like(batch.terminated))
    with torch.no_grad():
        for head in agent.model.target_q_functions:
            head[-1].weight.zero_()
            head[-1].bias.fill_(2.0)
    weights = batch.valid * (
        agent.cfg.rho ** torch.arange(agent.cfg.model_horizon)
    ).view(-1, 1, 1)
    expected = (
        (batch.rewards + agent.cfg.discount * 2.0) * weights
    ).sum() / weights.sum()
    terminal_agent = deepcopy(agent)
    metrics = agent._update_replay(batch)
    assert metrics["td_target_mean"] == pytest.approx(float(expected))
    terminal_batch = replace(batch, terminated=torch.ones_like(batch.terminated))
    terminal_metrics = terminal_agent._update_replay(terminal_batch)
    expected_terminal = (batch.rewards * weights).sum() / weights.sum()
    assert terminal_metrics["td_target_mean"] == pytest.approx(float(expected_terminal))


def test_padding_rewards_do_not_affect_learning(tmp_path):
    agent, replay = learner(tmp_path)
    other = deepcopy(agent)
    batch = replay.sample()
    assert (batch.valid == 0).any()
    changed = replace(
        batch,
        rewards=torch.where(
            batch.valid.bool(), batch.rewards, torch.full_like(batch.rewards, 1e6)
        ),
    )
    state = torch.get_rng_state()
    first = agent._update_replay(batch)
    torch.set_rng_state(state)
    second = other._update_replay(changed)
    for key in ("model_loss", "actor_loss", "q_loss", "td_target_mean"):
        assert first[key] == pytest.approx(second[key], abs=1e-7)
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name])


def test_actor_follows_action_value_gradient_with_zero_entropy(tmp_path, monkeypatch):
    agent, replay = learner(tmp_path)
    agent.cfg.entropy_coef = 0.0
    with torch.no_grad():
        agent.actor.network[-1].weight.zero_()
        agent.actor.network[-1].bias.zero_()

    def linear_q(
        latent,
        action,
        action_mask=None,
        *_args,
        target=False,
        return_type="all",
        **_kwargs,
    ):
        value = (action * action_mask).sum(dim=-1, keepdim=True)
        if target:
            return torch.zeros_like(value)
        if return_type == "all":
            return value.unsqueeze(0).expand(agent.cfg.num_q, *value.shape)
        return value

    monkeypatch.setattr(agent.model, "q", linear_q)
    fixed_latent = torch.zeros(4, agent.model.latent_dim)
    mask = torch.tensor([1.0, 0.0, 1.0]).expand(4, 3)
    before = agent.actor.mean_action(fixed_latent, mask).detach()
    agent.update(replay)
    after = agent.actor.mean_action(fixed_latent, mask).detach()
    assert (after[:, [0, 2]] > before[:, [0, 2]]).all()
    assert after[:, 1].eq(0).all()


def test_offpolicy_checkpoint_roundtrip_and_mode_rejection(tmp_path):
    agent, replay = learner(tmp_path)
    agent.update(replay)
    checkpoint = tmp_path / "offpolicy.pt"
    agent.save(checkpoint, step=16)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["learning_mode"] == "off_policy"
    assert "external_critic" not in payload
    restored = M4POAgent.load(checkpoint, torch.device("cpu"), restore_optimizers=True)
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    for state in restored.actor_optimizer.state.values():
        assert int(state["step"]) == 1
    with pytest.raises(ValueError, match="learning_mode"):
        M4POAgent.load(
            checkpoint,
            torch.device("cpu"),
            override_cfg=replace(agent.cfg, learning_mode="on_policy"),
        )


def test_schema_two_checkpoint_stays_on_policy(tmp_path):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.learning_mode = "on_policy"
    agent = M4POAgent(ObservationSpec.from_config(cfg), 3, cfg, torch.device("cpu"))
    path = tmp_path / "legacy.pt"
    agent.save(path, step=0)
    payload = torch.load(path, weights_only=False)
    payload["checkpoint_schema_version"] = 2
    payload.pop("learning_mode")
    for name in (
        "learning_mode",
        "seed_steps",
        "pretrain_updates",
        "updates_per_step",
        "replay_capacity",
        "batch_size",
        "num_q",
        "rho",
        "policy_optimization_enabled",
        "policy_optimization_weight",
        "exploration_bonus_enabled",
        "exploration_bonus_weight",
    ):
        payload["cfg"].pop(name)
    torch.save(payload, path)
    loaded = M4POAgent.load(path, torch.device("cpu"), restore_optimizers=True)
    assert loaded.cfg.learning_mode == "on_policy"
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])


@pytest.mark.parametrize("images", [False, True])
def test_offpolicy_real_train_evaluate_and_resume(tmp_path, images):
    cfg = M4POConfig.from_yaml("m4po/configs/mock_smoke.yaml")
    cfg.use_images = images
    cfg.log_dir = str(tmp_path / "first")
    cfg.total_steps = 64
    checkpoint = OnlineTrainer(cfg).train()
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["extra"]["update_count"] > cfg.pretrain_updates
    assert payload["extra"]["replay_saved"] is False
    assert payload["extra"]["pretrain_updates_completed"] == cfg.pretrain_updates
    records = [
        json.loads(line)
        for line in (checkpoint.parent.parent / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[-1]["phase"] == "off_policy_training"
    assert records[-1]["replay_episodes"] > 0
    agent = M4POAgent.load(checkpoint, torch.device("cpu"))
    before = deepcopy(agent.state_dict())
    result = evaluate_policy(agent, agent.cfg, episodes=1, deterministic=True)
    assert result["episodes"] == cfg.num_tasks * cfg.num_embodiments
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before[name])
    resumed = replace(
        cfg,
        total_steps=128,
        log_dir=str(tmp_path / "resumed"),
        resume_checkpoint=str(checkpoint),
    )
    next_checkpoint = OnlineTrainer(resumed).train()
    next_payload = torch.load(next_checkpoint, weights_only=False)
    assert next_payload["step"] == 128
    assert next_payload["extra"]["update_count"] > payload["extra"]["update_count"]
    assert next_payload["extra"]["pretrain_updates_completed"] == cfg.pretrain_updates


@pytest.mark.parametrize(
    "overrides",
    [
        {"discrepancy_beta": 0.1},
        {"policy_optimization_enabled": True},
        {"exploration_bonus_weight": 0.1},
        {"num_q": 1},
        {"updates_per_step": float("nan")},
    ],
)
def test_offpolicy_rejects_ambiguous_or_invalid_settings(overrides):
    with pytest.raises(ValueError):
        replace(M4POConfig(), **overrides).validate()
