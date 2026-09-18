from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import m4po.m4po as m4po_module
from m4po.common.action_tokenizer import ActionTokenizer
from m4po.common.buffer import RolloutBuffer
from m4po.common.config import IMPLEMENTATION_ID, M4POConfig
from m4po.common.losses import generalized_advantage_estimate, lambda_return
from m4po.envs import make_vector_env
from m4po.m4po import M4POAgent, _policy_ratio_statistics


def small_cfg(tmp_path: Path, *, images: bool = False) -> M4POConfig:
    return M4POConfig(
        learning_mode="on_policy",
        env="mock",
        task="reach",
        tasks="reach,push",
        embodiment="arm",
        embodiments="arm,hand",
        multitask=True,
        multiembodiment=True,
        num_tasks=2,
        num_embodiments=2,
        use_images=images,
        image_size=16,
        image_channels=4,
        proprio_dim=6,
        state_dim=7,
        seed=3,
        num_envs=4,
        max_episode_steps=4,
        control_repeats="1,2",
        total_steps=16,
        rollout_steps=2,
        updates_per_rollout=1,
        ppo_epochs=1,
        minibatch_size=4,
        eval_every=0,
        save_every=0,
        log_every=0,
        log_dir=str(tmp_path / "run"),
        model_horizon=2,
        planning_horizon=2,
        task_latent_dim=8,
        body_latent_dim=8,
        task_context_dim=8,
        embodiment_context_dim=4,
        encoder_dim=16,
        mlp_dim=16,
        num_encoder_layers=2,
        simnorm_dim=4,
        dropout=0.0,
        planner_samples=4,
        device="cpu",
        quiet=True,
    )


def collect_rollout(env, agent: M4POAgent, steps: int):
    observation = env.reset()
    buffer = RolloutBuffer(env.observation_spec, env.num_envs)
    for _ in range(steps):
        output = agent.act_np(
            observation,
            task_ids=env.task_ids,
            embodiment_ids=env.embodiment_ids,
            action_mask=env.action_masks,
        )
        next_observation, reward, done, infos = env.step(output.action)
        transition_next = {name: value.copy() for name, value in next_observation.items()}
        for index, finished in enumerate(done):
            if finished:
                terminal = infos[index]["terminal_observation"]
                for name in transition_next:
                    transition_next[name][index] = terminal[name]
        buffer.append(
            observation,
            transition_next,
            output.action,
            output.pre_tanh_action,
            output.candidate_noise,
            output.log_prob,
            reward,
            done,
            env.task_ids,
            env.embodiment_ids,
            env.action_masks,
        )
        observation = next_observation
    return buffer.finalize(), observation


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
        return
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for name in first:
            assert_nested_equal(first[name], second[name])
        return
    if isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second, strict=True):
            assert_nested_equal(first_item, second_item)
        return
    assert first == second


def test_action_tokenizer_roundtrip_and_nonprefix_masks():
    tokenizer = ActionTokenizer(
        [np.array([-2.0, -1.0]), np.array([-3.0, -2.0, -1.0])],
        [np.array([2.0, 3.0]), np.array([1.0, 2.0, 5.0])],
        [np.array([2, 0]), np.array([1, 3, 0])],
        shared_action_dim=4,
    )
    physical = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 2.0, 0.0]], np.float32)
    embodiment_ids = np.array([0, 1], np.int64)
    shared = tokenizer.encode(physical, embodiment_ids)
    decoded = tokenizer.decode(shared, embodiment_ids)
    assert np.allclose(decoded[0, :2], physical[0, :2])
    assert np.allclose(decoded[1, :3], physical[1, :3])
    assert np.all(shared[tokenizer.mask_for(embodiment_ids) == 0.0] == 0.0)
    with pytest.raises(ValueError):
        ActionTokenizer([[0.0]], [[0.0]])


def test_multimodal_masks_and_hierarchical_action_mask(tmp_path):
    cfg = small_cfg(tmp_path, images=True)
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    observation = env.reset()
    altered = {name: value.copy() for name, value in observation.items()}
    altered["proprio"][altered["proprio_mask"] == 0.0] = 1000.0
    altered["state"][altered["state_mask"] == 0.0] = -1000.0
    with torch.no_grad():
        first = agent.model.encode(observation, env.task_ids, env.embodiment_ids)
        second = agent.model.encode(altered, env.task_ids, env.embodiment_ids)
    assert torch.allclose(first, second, atol=1e-6)
    assert first.shape == (cfg.num_envs, cfg.task_latent_dim + cfg.body_latent_dim)

    action = torch.randn(cfg.num_envs, env.action_dim)
    mask = torch.as_tensor(env.action_masks)
    invalid_changed = action + 123.0 * (1.0 - mask)
    with torch.no_grad():
        next_first = agent.model.next(first, action, mask, env.task_ids, env.embodiment_ids)
        next_second = agent.model.next(
            first, invalid_changed, mask, env.task_ids, env.embodiment_ids
        )
    assert torch.allclose(next_first, next_second, atol=1e-6)
    env.close()


def test_planner_likelihood_recompute_and_actor_only_gradient(tmp_path):
    cfg = small_cfg(tmp_path)
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    observation = env.reset()
    observation_t = agent._observation_tensor(observation)
    latent = agent.model.encode(observation_t, env.task_ids, env.embodiment_ids)
    output = agent.planner.plan(
        agent.model,
        agent.actor,
        latent,
        torch.as_tensor(env.action_masks),
        torch.as_tensor(env.task_ids),
        torch.as_tensor(env.embodiment_ids),
    )
    recomputed = agent.planner.recompute(
        agent.model,
        agent.actor,
        latent,
        torch.as_tensor(env.action_masks),
        output.candidate_noise,
        output.pre_tanh,
        torch.as_tensor(env.task_ids),
        torch.as_tensor(env.embodiment_ids),
    )
    assert torch.allclose(output.log_prob, recomputed.log_prob, atol=1e-6)
    (-recomputed.log_prob.mean()).backward()
    assert any(parameter.grad is not None for parameter in agent.actor.parameters())
    assert all(parameter.grad is None for parameter in agent.model.parameters())
    env.close()


def test_ppo_metrics_use_post_step_planner_likelihood(tmp_path):
    torch.manual_seed(7)
    cfg = replace(
        small_cfg(tmp_path),
        actor_lr=3e-3,
        ppo_epochs=1,
        minibatch_size=8,
        ppo_target_kl=None,
    )
    cfg.validate()
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    rollout, _ = collect_rollout(env, agent, cfg.rollout_steps)
    batch = rollout.to(agent.device)
    targets = agent._prepare_targets(batch, progress_fraction=0.25)

    metrics = agent._update_actor_critics(batch, targets)
    flat_batch = batch.flatten()
    flat_latent = targets.latent.flatten(0, 1)
    with torch.no_grad():
        post_output = agent.planner.recompute(
            agent.model,
            agent.actor,
            flat_latent,
            flat_batch.action_masks,
            flat_batch.candidate_noise,
            flat_batch.pre_tanh_actions,
            flat_batch.task_ids,
            flat_batch.embodiment_ids,
        )
        expected = _policy_ratio_statistics(
            post_output.log_prob - flat_batch.old_log_prob,
            cfg.ppo_clip_ratio,
        )

    assert metrics["actor_optimizer_steps"].item() == 1
    assert expected.approximate_kl.item() > 1e-8
    torch.testing.assert_close(
        metrics["approximate_kl"],
        expected.approximate_kl,
    )
    torch.testing.assert_close(
        metrics["max_observed_approximate_kl"],
        expected.approximate_kl,
    )
    torch.testing.assert_close(
        metrics["clip_fraction"],
        expected.clip_fraction,
    )
    torch.testing.assert_close(
        metrics["max_abs_log_ratio"],
        expected.max_abs_log_ratio,
    )
    torch.testing.assert_close(
        metrics["log_ratio_saturation_fraction"],
        expected.saturation_fraction,
    )
    env.close()


def test_target_kl_backtracks_and_persists_accepted_learning_rate(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(11)
    cfg = replace(
        small_cfg(tmp_path),
        actor_lr=1e-3,
        ppo_epochs=1,
        minibatch_size=8,
        ppo_target_kl=0.02,
    )
    cfg.validate()
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    rollout, _ = collect_rollout(env, agent, cfg.rollout_steps)
    batch = rollout.to(agent.device)
    targets = agent._prepare_targets(batch, progress_fraction=0.25)
    real_statistics = m4po_module._policy_ratio_statistics
    calls = 0

    def controlled_statistics(raw_log_ratio, clip_ratio):
        nonlocal calls
        calls += 1
        statistics = real_statistics(raw_log_ratio, clip_ratio)
        if calls == 2:
            return replace(
                statistics,
                approximate_kl=torch.full_like(statistics.approximate_kl, 0.04),
            )
        if calls == 3:
            return replace(
                statistics,
                approximate_kl=torch.full_like(statistics.approximate_kl, 0.01),
            )
        return statistics

    monkeypatch.setattr(
        m4po_module,
        "_policy_ratio_statistics",
        controlled_statistics,
    )
    metrics = agent._update_actor_critics(batch, targets)

    assert metrics["actor_optimizer_steps"].item() == 1
    assert metrics["actor_step_attempts"].item() == 2
    assert metrics["actor_backtrack_steps"].item() == 1
    assert metrics["actor_early_stopped"].item() == 0
    assert metrics["approximate_kl"].item() == pytest.approx(0.01)
    assert metrics["max_observed_approximate_kl"].item() == pytest.approx(0.04)
    assert metrics["actor_learning_rate"].item() == pytest.approx(cfg.actor_lr / 2)
    actor_steps = {
        int(state["step"])
        for state in agent.actor_optimizer.state.values()
        if "step" in state
    }
    assert actor_steps == {1}
    env.close()


def test_target_kl_rejection_restores_actor_and_keeps_critic_updates(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(13)
    cfg = replace(
        small_cfg(tmp_path),
        actor_lr=1e-3,
        ppo_epochs=3,
        minibatch_size=8,
        ppo_target_kl=0.02,
    )
    cfg.validate()
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )

    # Prime Adam so rollback is checked for moment tensors and the step counter,
    # not only for an initially empty optimizer state.
    agent.actor_optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().sum() for parameter in agent.actor.parameters()).backward()
    agent.actor_optimizer.step()
    agent.actor_optimizer.zero_grad(set_to_none=True)

    rollout, _ = collect_rollout(env, agent, cfg.rollout_steps)
    batch = rollout.to(agent.device)
    targets = agent._prepare_targets(batch, progress_fraction=0.25)
    actor_before = deepcopy(agent.actor.state_dict())
    optimizer_before = deepcopy(agent.actor_optimizer.state_dict())
    real_statistics = m4po_module._policy_ratio_statistics
    calls = 0

    def reject_post_step_statistics(raw_log_ratio, clip_ratio):
        nonlocal calls
        calls += 1
        statistics = real_statistics(raw_log_ratio, clip_ratio)
        if calls > 1:
            return replace(
                statistics,
                approximate_kl=torch.full_like(
                    statistics.approximate_kl,
                    torch.inf,
                ),
            )
        return statistics

    monkeypatch.setattr(
        m4po_module,
        "_policy_ratio_statistics",
        reject_post_step_statistics,
    )
    metrics = agent._update_actor_critics(batch, targets)

    assert metrics["actor_optimizer_steps"].item() == 0
    assert metrics["actor_step_attempts"].item() == (
        m4po_module._MAX_KL_BACKTRACK_STEPS + 1
    )
    assert metrics["actor_backtrack_steps"].item() == (
        m4po_module._MAX_KL_BACKTRACK_STEPS + 1
    )
    assert metrics["actor_early_stopped"].item() == 1
    assert metrics["max_observed_approximate_kl"].item() == torch.finfo(
        torch.float32
    ).max
    assert_nested_equal(actor_before, agent.actor.state_dict())
    assert_nested_equal(optimizer_before, agent.actor_optimizer.state_dict())
    critic_steps = {
        int(state["step"])
        for state in agent.critic_optimizer.state.values()
        if "step" in state
    }
    assert critic_steps == {cfg.ppo_epochs}
    env.close()


def test_target_kl_reports_cross_minibatch_precheck_that_stops_actor(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(17)
    cfg = replace(
        small_cfg(tmp_path),
        ppo_epochs=2,
        minibatch_size=8,
        ppo_target_kl=0.02,
    )
    cfg.validate()
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    rollout, _ = collect_rollout(env, agent, cfg.rollout_steps)
    batch = rollout.to(agent.device)
    targets = agent._prepare_targets(batch, progress_fraction=0.25)
    real_statistics = m4po_module._policy_ratio_statistics
    calls = 0

    def stop_on_second_precheck(raw_log_ratio, clip_ratio):
        nonlocal calls
        calls += 1
        statistics = real_statistics(raw_log_ratio, clip_ratio)
        if calls == 2:
            return replace(
                statistics,
                approximate_kl=torch.full_like(statistics.approximate_kl, 0.01),
            )
        if calls == 3:
            return replace(
                statistics,
                approximate_kl=torch.full_like(statistics.approximate_kl, 0.03),
            )
        return statistics

    monkeypatch.setattr(
        m4po_module,
        "_policy_ratio_statistics",
        stop_on_second_precheck,
    )
    metrics = agent._update_actor_critics(batch, targets)

    assert metrics["actor_optimizer_steps"].item() == 1
    assert metrics["actor_early_stopped"].item() == 1
    assert metrics["approximate_kl"].item() == pytest.approx(0.01)
    assert metrics["max_observed_approximate_kl"].item() == pytest.approx(0.03)
    critic_steps = {
        int(state["step"])
        for state in agent.critic_optimizer.state.values()
        if "step" in state
    }
    assert critic_steps == {cfg.ppo_epochs}
    env.close()


def test_terminal_masked_returns_are_numeric():
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    terminated = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    values = torch.tensor([[[0.5]], [[0.6]], [[0.7]], [[0.8]]])
    advantage, returns = generalized_advantage_estimate(
        rewards, values, terminated, discount=0.9, gae_lambda=0.8
    )
    expected_lambda = lambda_return(rewards, values, terminated, 0.9, 0.8)
    assert torch.allclose(returns, advantage + values[:-1])
    assert torch.allclose(returns, expected_lambda)
    assert torch.allclose(returns[1], rewards[1])


def test_fresh_rollout_update_checkpoint_and_load(tmp_path):
    cfg = small_cfg(tmp_path)
    env = make_vector_env(cfg)
    agent = M4POAgent(
        env.observation_spec,
        env.action_dim,
        cfg,
        torch.device("cpu"),
        task_contexts=env.task_contexts,
    )
    rollout, observation = collect_rollout(env, agent, cfg.rollout_steps)
    metrics = agent.update(rollout, progress_fraction=0.25)
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert rollout.candidate_noise.shape == (
        cfg.rollout_steps,
        cfg.num_envs,
        cfg.planner_samples,
        cfg.planning_horizon,
        env.action_dim,
    )

    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint, step=8)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    loaded = M4POAgent.load(checkpoint, torch.device("cpu"))
    torch.manual_seed(99)
    first = agent.act_np(
        observation,
        task_ids=env.task_ids,
        embodiment_ids=env.embodiment_ids,
        action_mask=env.action_masks,
        deterministic=True,
    ).action
    torch.manual_seed(99)
    second = loaded.act_np(
        observation,
        task_ids=env.task_ids,
        embodiment_ids=env.embodiment_ids,
        action_mask=env.action_masks,
        deterministic=True,
    ).action
    assert np.allclose(first, second)
    assert np.allclose(second * (1.0 - env.action_masks), 0.0)
    env.close()


def test_config_rejects_nonfresh_step_budget(tmp_path):
    cfg = replace(small_cfg(tmp_path), total_steps=17)
    with pytest.raises(ValueError, match="complete fresh rollout"):
        cfg.validate()


def test_config_rejects_nonpositive_ppo_target_kl(tmp_path):
    cfg = replace(small_cfg(tmp_path), ppo_target_kl=0.0)
    with pytest.raises(ValueError, match="ppo_target_kl must be positive"):
        cfg.validate()


def test_rollout_resampling_covers_all_task_embodiment_pairs(tmp_path):
    cfg = replace(
        small_cfg(tmp_path),
        num_envs=1,
        total_steps=2,
        minibatch_size=1,
    )
    env = make_vector_env(cfg)
    observed = set()
    for _ in range(cfg.num_tasks * cfg.num_embodiments):
        env.reset_rollout()
        observed.add((int(env.task_ids[0]), int(env.embodiment_ids[0])))
    assert observed == {
        (task_id, embodiment_id)
        for task_id in range(cfg.num_tasks)
        for embodiment_id in range(cfg.num_embodiments)
    }
    env.close()
