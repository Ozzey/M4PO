from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from m4po.common.buffer import ObservationSpec
from m4po.m4po import M4POAgent
from test_offpolicy_learning import learner


def assert_identical(first, second):
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


def test_default_actor_mode_matches_explicit_q_max_bit_for_bit(tmp_path):
    torch.manual_seed(51)
    default, replay = learner(tmp_path)
    explicit = deepcopy(default)
    explicit_replay = deepcopy(replay)
    rng = torch.get_rng_state()
    default_metrics = default.update(replay)
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    explicit_metrics = explicit.update(explicit_replay, actor_mode="q_max")
    assert default_metrics == explicit_metrics
    assert default_metrics["bc_loss"] == 0
    assert default_metrics["actor_bc_enabled"] == 0
    assert "policy_q" in default_metrics
    assert_identical(default, explicit)
    assert torch.equal(final_rng, torch.get_rng_state())


def test_behavior_cloning_updates_actor_and_world_without_q_actor_or_scale(
    tmp_path, monkeypatch
):
    torch.manual_seed(52)
    agent, replay = learner(tmp_path)
    actor_before = deepcopy(agent.actor.state_dict())
    model_before = deepcopy(agent.model.state_dict())
    agent.scale.value.fill_(4.0)
    scale_before = deepcopy(agent.scale.state_dict())
    original_q = agent.model.q

    def q_without_actor_max(*args, **kwargs):
        assert kwargs.get("return_type") != "avg"
        return original_q(*args, **kwargs)

    def forbidden_scale_update(*args, **kwargs):
        pytest.fail("Demonstration cloning must not update RunningScale")

    optimizer_step = agent.actor_optimizer.step

    def isolated_actor_step(*args, **kwargs):
        assert all(parameter.grad is None for parameter in agent.model.parameters())
        assert not any(
            parameter.requires_grad for parameter in agent.model.parameters()
        )
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in agent.actor.parameters()
        )
        return optimizer_step(*args, **kwargs)

    monkeypatch.setattr(agent.model, "q", q_without_actor_max)
    monkeypatch.setattr(agent.scale, "update", forbidden_scale_update)
    monkeypatch.setattr(agent.actor_optimizer, "step", isolated_actor_step)
    metrics = agent.update(replay, actor_mode="behavior_cloning")
    assert all(
        isinstance(value, float) and math.isfinite(value) for value in metrics.values()
    )
    assert metrics["actor_bc_enabled"] == 1
    assert metrics["bc_loss"] > 0
    assert "policy_q" not in metrics
    assert any(
        not torch.equal(actor_before[key], value)
        for key, value in agent.actor.state_dict().items()
    )
    assert any(
        not torch.equal(model_before[key], value)
        for key, value in agent.model.state_dict().items()
    )
    for key, value in scale_before.items():
        torch.testing.assert_close(agent.scale.state_dict()[key], value, rtol=0, atol=0)


def test_bc_loss_is_valid_coordinate_average_with_rho_padding_and_entropy(
    tmp_path, monkeypatch
):
    agent, replay = learner(tmp_path)
    agent.cfg.entropy_coef = 0.2
    batch = replay.sample()
    captured = []
    original_sample = agent.actor.sample_squashed

    def sample(*args, **kwargs):
        output = original_sample(*args, **kwargs)
        captured.append(output)
        return output

    monkeypatch.setattr(agent.actor, "sample_squashed", sample)
    metrics = agent._update_replay(batch, actor_mode="behavior_cloning")
    actions, info = captured[-1]
    weights = batch.valid * (
        agent.cfg.rho ** torch.arange(agent.cfg.model_horizon)
    ).view(-1, 1, 1)
    errors = (actions.detach() - batch.actions).square() * batch.action_masks
    coordinate_mean = errors.sum(dim=-1, keepdim=True) / batch.action_masks.sum(
        dim=-1, keepdim=True
    )
    expected_bc = float((coordinate_mean * weights).sum() / weights.sum())
    expected_entropy = float(
        (info["scaled_entropy"].detach() * weights).sum() / weights.sum()
    )
    assert metrics["bc_loss"] == pytest.approx(expected_bc)
    assert metrics["actor_loss"] == pytest.approx(expected_bc - 0.2 * expected_entropy)
    assert len(captured) == 2  # TD bootstrap sample and current actor sample.


@pytest.mark.parametrize("corruption", ["invalid_coordinates", "padded_transitions"])
def test_bc_ignores_invalid_action_coordinates_and_padded_transitions(
    tmp_path, corruption
):
    torch.manual_seed(53)
    agent, replay = learner(tmp_path)
    other = deepcopy(agent)
    batch = replay.sample()
    if corruption == "invalid_coordinates":
        assert (~batch.action_masks.bool()).any()
        actions = torch.where(
            batch.action_masks.bool(),
            batch.actions,
            torch.full_like(batch.actions, float("nan")),
        )
    else:
        assert (~batch.valid.bool()).any()
        actions = torch.where(
            batch.valid.bool(), batch.actions, torch.full_like(batch.actions, 1000.0)
        )
    changed = replace(batch, actions=actions)
    rng = torch.get_rng_state()
    original_metrics = agent._update_replay(batch, actor_mode="behavior_cloning")
    torch.set_rng_state(rng)
    changed_metrics = other._update_replay(changed, actor_mode="behavior_cloning")
    for key in ("actor_loss", "bc_loss", "model_loss", "q_loss"):
        assert original_metrics[key] == pytest.approx(changed_metrics[key], abs=1e-7)
    assert_identical(agent, other)


def test_bc_keeps_world_and_q_learning_identical_to_q_max(tmp_path):
    torch.manual_seed(54)
    agent, replay = learner(tmp_path)
    cloned = deepcopy(agent)
    batch = replay.sample()
    rng = torch.get_rng_state()
    q_metrics = agent._update_replay(batch)
    torch.set_rng_state(rng)
    bc_metrics = cloned._update_replay(batch, actor_mode="behavior_cloning")
    for key in (
        "model_loss",
        "consistency_loss",
        "reward_loss",
        "q_loss",
        "termination_loss",
        "td_target_mean",
    ):
        assert q_metrics[key] == bc_metrics[key]
    assert_identical(agent.model, cloned.model)


@pytest.mark.parametrize("public", [True, False])
def test_invalid_actor_mode_is_rejected_before_any_mutation(tmp_path, public):
    agent, replay = learner(tmp_path)
    before = deepcopy(agent)
    batch = replay.sample()
    rng = torch.get_rng_state()
    replay_rng = replay.rng_state_dict()["generator"].clone()
    with pytest.raises(ValueError, match="actor_mode"):
        if public:
            agent.update(replay, actor_mode="unknown")
        else:
            agent._update_replay(batch, actor_mode="unknown")
    assert_identical(agent, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(replay.rng_state_dict()["generator"], replay_rng)
    assert not agent.actor_optimizer.state
    assert not agent.world_model_optimizer.state


def test_legacy_on_policy_rejects_behavior_cloning_before_updates(tmp_path):
    off_policy, replay = learner(tmp_path)
    cfg = replace(off_policy.cfg, learning_mode="on_policy")
    agent = M4POAgent(ObservationSpec.from_config(cfg), 3, cfg, torch.device("cpu"))
    before = deepcopy(agent)
    with pytest.raises(ValueError, match="requires off-policy"):
        agent.update(replay, actor_mode="behavior_cloning")
    assert_identical(agent, before)
