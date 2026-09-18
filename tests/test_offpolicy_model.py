from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from m4po.common.buffer import ObservationSpec
from m4po.common.config import M4POConfig
from m4po.common.planner import StochasticMPPIPlanner
from m4po.common.policy import GaussianActor
from m4po.common.world_model import HierarchicalWorldModel


def small_config(**overrides) -> M4POConfig:
    return replace(
        M4POConfig(
            learning_mode="off_policy",
            use_images=False,
            proprio_dim=4,
            state_dim=0,
            num_tasks=2,
            num_embodiments=2,
            task_latent_dim=8,
            body_latent_dim=8,
            task_context_dim=4,
            embodiment_context_dim=4,
            encoder_dim=16,
            mlp_dim=16,
            simnorm_dim=4,
            dropout=0.0,
            planning_horizon=2,
            planner_samples=4,
            num_q=5,
            device="cpu",
        ),
        **overrides,
    )


def make_model(cfg: M4POConfig) -> HierarchicalWorldModel:
    model = HierarchicalWorldModel(ObservationSpec.from_config(cfg), 3, cfg)
    if cfg.learning_mode == "off_policy":
        # Nonzero heads make masking and action-gradient assertions substantive.
        with torch.no_grad():
            for head in model.q_functions:
                head[-1].weight.normal_(std=0.1)
        model.hard_update_targets()
    return model


def test_q_masks_nonfinite_padding_and_retains_action_gradients():
    torch.manual_seed(3)
    cfg = small_config()
    model = make_model(cfg)
    latent = torch.randn(2, 4, model.latent_dim)
    actions = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 1.0]).expand_as(actions)
    altered = actions.detach().clone()
    altered[0, :, 1] = float("nan")
    altered[1, :, 1] = float("inf")
    tasks = torch.tensor([0, 1, 0, 1])
    bodies = 1 - tasks
    expected = model.q(latent, actions, mask, tasks, bodies)
    actual = model.q(latent, altered, mask, tasks, bodies)
    assert expected.shape == (cfg.num_q, 2, 4, 1)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    expected.sum().backward()
    assert torch.equal(actions.grad[..., 1], torch.zeros(2, 4))
    assert actions.grad[..., [0, 2]].abs().sum() > 0
    assert model.task_embedding.weight.grad.abs().sum() > 0
    assert model.embodiment_embedding.weight.grad.abs().sum() > 0


def test_q_aggregation_uses_all_heads_or_random_pair():
    cfg = small_config()
    model = make_model(cfg)
    with torch.no_grad():
        for index, head in enumerate(model.q_functions):
            head[-1].weight.zero_()
            head[-1].bias.fill_(float(index))
    latent = torch.randn(4, model.latent_dim)
    action = torch.zeros(4, 3)
    torch.testing.assert_close(
        model.q(latent, action, return_type="avg"), torch.full((4, 1), 2.0)
    )
    torch.manual_seed(12)
    expected_min = torch.randperm(cfg.num_q)[:2].min().float()
    torch.manual_seed(12)
    torch.testing.assert_close(
        model.q(latent, action, return_type="min"), expected_min.expand(4, 1)
    )
    with pytest.raises(ValueError, match="return_type"):
        model.q(latent, action, return_type="invalid")


def test_target_q_is_frozen_uses_online_context_and_polyak_update():
    torch.manual_seed(4)
    cfg = small_config()
    model = make_model(cfg)
    model.train()
    assert model.q_functions.training
    assert not model.target_q_functions.training
    assert not any(
        parameter.requires_grad for parameter in model.target_q_functions.parameters()
    )
    latent = torch.randn(4, model.latent_dim)
    action = torch.randn(4, 3)
    with torch.no_grad():
        model.target_encoder.task_embedding.weight.add_(10.0)
        model.target_encoder.embodiment_embedding.weight.sub_(10.0)
    online = model.q(latent, action)
    target = model.q(latent, action, target=True)
    torch.testing.assert_close(online, target)
    assert not target.requires_grad
    before = [
        parameter.detach().clone()
        for parameter in model.target_q_functions.parameters()
    ]
    with torch.no_grad():
        for parameter in model.q_functions.parameters():
            parameter.add_(0.4)
    model.soft_update_targets(decay=0.75)
    for previous, target_parameter in zip(
        before, model.target_q_functions.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, previous + 0.1)


def test_frozen_q_retains_actor_gradient_and_restores_training_state():
    torch.manual_seed(5)
    cfg = small_config(dropout=0.1)
    model = make_model(cfg)
    actor = GaussianActor(model.latent_dim, model.action_dim, cfg)
    latent = torch.randn(4, model.latent_dim)
    mask = torch.tensor([1.0, 0.0, 1.0]).expand(4, 3)
    model.train()
    with model.frozen_parameters():
        action, _ = actor.sample_squashed(latent, mask)
        loss = -model.q(latent, action, mask, return_type="avg").mean()
        loss.backward()
        assert not model.training
    assert model.training
    assert not model.target_q_functions.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in actor.parameters()) > 0
    assert all(parameter.requires_grad for parameter in model.q_functions.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.target_q_functions.parameters()
    )


def test_squashed_actor_density_and_m3po_entropy_scaling():
    torch.manual_seed(6)
    cfg = small_config()
    actor = GaussianActor(cfg.model_latent_dim, 3, cfg)
    latent = torch.randn(4, cfg.model_latent_dim)
    mask = torch.tensor([1.0, 0.0, 1.0]).expand(4, 3)
    noise = torch.randn(4, 3)
    noise[:, 1] = float("nan")
    action, info = actor.sample_squashed(latent, mask, noise=noise)
    assert torch.isfinite(action).all()
    assert torch.equal(action[:, 1], torch.zeros(4))
    mean, std = actor.distribution_parameters(latent)
    expected_per_dimension = torch.distributions.Normal(mean, std).log_prob(
        info["pre_tanh"]
    )
    expected_per_dimension -= torch.log1p(-action.square())
    expected = (expected_per_dimension * mask).sum(-1, keepdim=True)
    torch.testing.assert_close(info["log_prob"], expected)
    torch.testing.assert_close(actor.squashed_log_prob(latent, action, mask), expected)
    torch.testing.assert_close(info["entropy"], -expected)
    torch.testing.assert_close(info["scaled_entropy"], -2.0 * expected)
    raw_density = actor.log_prob(latent, info["pre_tanh"], mask)
    assert not torch.allclose(raw_density, expected)


def test_squashed_density_is_finite_for_saturated_and_padded_actions():
    cfg = small_config()
    actor = GaussianActor(cfg.model_latent_dim, 3, cfg)
    latent = torch.zeros(2, cfg.model_latent_dim)
    pre_tanh = torch.tensor([[100.0, float("nan"), -100.0]]).expand(2, 3)
    mask = torch.tensor([1.0, 0.0, 1.0]).expand(2, 3)
    density = actor.squashed_log_prob(latent, pre_tanh.tanh(), mask, pre_tanh=pre_tanh)
    assert torch.isfinite(density).all()
    density.sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_offpolicy_planner_bootstraps_actor_action_q(monkeypatch):
    cfg = small_config()
    model = make_model(cfg)
    actor = GaussianActor(model.latent_dim, model.action_dim, cfg)
    planner = StochasticMPPIPlanner(cfg)
    calls = []

    def q(latent, action, mask, task_ids, embodiment_ids, *, return_type):
        calls.append((latent, action, mask, task_ids, embodiment_ids, return_type))
        return action.sum(-1, keepdim=True)

    def unexpected_value(*args, **kwargs):
        pytest.fail("Off-policy planner must bootstrap Q, not the legacy value head")

    monkeypatch.setattr(model, "q", q)
    monkeypatch.setattr(model, "value", unexpected_value)
    latent = torch.randn(2, model.latent_dim)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    result = planner.plan(
        model, actor, latent, mask, torch.tensor([0, 1]), torch.tensor([1, 0])
    )
    assert result.action.shape == (2, 3)
    assert torch.isfinite(result.action).all()
    assert len(calls) == 1
    terminal_latent, terminal_action, terminal_mask, tasks, bodies, reduction = calls[0]
    assert terminal_latent.shape == (2 * cfg.planner_samples, model.latent_dim)
    assert reduction == "avg"
    assert (terminal_action[terminal_mask == 0] == 0).all()
    torch.testing.assert_close(
        tasks, torch.tensor([0, 1]).repeat_interleave(cfg.planner_samples)
    )
    torch.testing.assert_close(
        bodies, torch.tensor([1, 0]).repeat_interleave(cfg.planner_samples)
    )


def test_legacy_model_has_no_q_parameters_and_recomputes_planner_density():
    cfg = small_config(learning_mode="on_policy")
    model = make_model(cfg)
    actor = GaussianActor(model.latent_dim, model.action_dim, cfg)
    planner = StochasticMPPIPlanner(cfg)
    assert not any("q_functions" in key for key in model.state_dict())
    latent = torch.randn(2, model.latent_dim)
    mask = torch.ones(2, 3)
    output = planner.plan(model, actor, latent, mask)
    recomputed = planner.recompute(
        model, actor, latent, mask, output.candidate_noise, output.pre_tanh
    )
    torch.testing.assert_close(recomputed.log_prob, output.log_prob)
    with pytest.raises(RuntimeError, match="off_policy"):
        model.q(latent, output.action)
