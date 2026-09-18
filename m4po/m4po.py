from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("M4PO_DISABLE_TRITON", "1")
if os.environ.get("M4PO_DISABLE_TRITON") == "1" and "triton" not in sys.modules:
    sys.modules["triton"] = None

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from m4po.common.buffer import (
    EpisodeReplayBuffer,
    ObservationSpec,
    ReplayBatch,
    RolloutBatch,
)
from m4po.common.config import CHECKPOINT_SCHEMA_VERSION, IMPLEMENTATION_ID, M4POConfig
from m4po.common.layers import mlp, weight_init
from m4po.common.losses import (
    diagonal_gaussian_entropy,
    generalized_advantage_estimate,
    masked_normalize,
    normalize_value_discrepancy,
)
from m4po.common.planner import PlannerOutput, StochasticMPPIPlanner
from m4po.common.policy import GaussianActor
from m4po.common.utils import to_numpy
from m4po.common.world_model import HierarchicalWorldModel


TensorObservation = Dict[str, torch.Tensor]

_POLICY_LOG_RATIO_CLIP = 20.0
_DIAGNOSTIC_LOG_RATIO_CLIP = 50.0
_MAX_KL_BACKTRACK_STEPS = 6
_KL_BACKTRACK_FACTOR = 0.5


@dataclass(frozen=True)
class _PolicyRatioStatistics:
    ratio: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    max_abs_log_ratio: torch.Tensor
    saturation_fraction: torch.Tensor


def _policy_ratio_statistics(
    raw_log_ratio: torch.Tensor,
    clip_ratio: float,
) -> _PolicyRatioStatistics:
    """Compute PPO terms while retaining raw log-ratio diagnostics."""

    safe_log_ratio = raw_log_ratio.clamp(
        -_POLICY_LOG_RATIO_CLIP,
        _POLICY_LOG_RATIO_CLIP,
    )
    ratio = safe_log_ratio.exp()

    # The non-negative ``expm1(x) - x`` estimator is more informative than
    # applying the policy clamp to the entire expression.  Only its positive
    # exponential input is capped for finite diagnostics; large negative raw
    # log-ratios therefore remain visible instead of all reporting KL=19.
    diagnostic_log_ratio = raw_log_ratio.detach().double()
    approximate_kl = (
        torch.expm1(diagnostic_log_ratio.clamp(max=_DIAGNOSTIC_LOG_RATIO_CLIP))
        - diagnostic_log_ratio
    ).mean().to(dtype=raw_log_ratio.dtype)
    lower_log_ratio = math.log1p(-clip_ratio)
    upper_log_ratio = math.log1p(clip_ratio)
    clip_fraction = (
        (raw_log_ratio < lower_log_ratio) | (raw_log_ratio > upper_log_ratio)
    ).float().mean()
    max_abs_log_ratio = raw_log_ratio.detach().abs().max()
    saturation_fraction = (
        raw_log_ratio.detach().abs() >= _POLICY_LOG_RATIO_CLIP
    ).float().mean()
    return _PolicyRatioStatistics(
        ratio=ratio,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        max_abs_log_ratio=max_abs_log_ratio,
        saturation_fraction=saturation_fraction,
    )


@dataclass(frozen=True)
class ActionBatch:
    """NumPy planner output stored at one vector-environment step."""

    action: np.ndarray
    pre_tanh_action: np.ndarray
    candidate_noise: np.ndarray
    log_prob: np.ndarray


@dataclass(frozen=True)
class OnPolicyTargets:
    """Detached rollout targets held fixed across all PPO epochs."""

    latent: torch.Tensor
    next_latent: torch.Tensor
    external_advantage: torch.Tensor
    augmented_advantage: torch.Tensor
    normalized_augmented_advantage: torch.Tensor
    external_return: torch.Tensor
    augmented_return: torch.Tensor
    old_external_value: torch.Tensor
    old_augmented_value: torch.Tensor
    discrepancy: torch.Tensor
    bonus: torch.Tensor
    augmented_reward: torch.Tensor
    beta: float


class ValueCritic(nn.Module):
    """Model-free state-value critic used for GAE."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = mlp(
            latent_dim,
            [hidden_dim, hidden_dim],
            1,
            dropout=dropout,
        )
        self.apply(weight_init)
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class RunningScale(nn.Module):
    """EMA of the replay Q-value spread, used only to scale the actor loss."""

    def __init__(self, decay: float, device: torch.device) -> None:
        super().__init__()
        self.decay = float(decay)
        self.register_buffer("value", torch.ones((), device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel():
            quantiles = torch.quantile(values, values.new_tensor([0.05, 0.95]))
            estimate = (quantiles[1] - quantiles[0]).clamp_min(1.0)
            self.value.lerp_(estimate, 1.0 - self.decay)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.value.detach().clone()


class M4POAgent(nn.Module):
    """Hierarchical MPPI agent with replay learning and an explicit legacy PPO mode."""

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_dim: int,
        cfg: M4POConfig,
        device: torch.device,
        *,
        task_contexts: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        cfg.device = str(device)
        cfg.validate()
        self.observation_spec = observation_spec
        self.action_dim = int(action_dim)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.cfg = cfg
        self.device = device
        self.model = HierarchicalWorldModel(
            observation_spec,
            self.action_dim,
            cfg,
            task_contexts=task_contexts,
        ).to(device)
        self.actor = GaussianActor(self.model.latent_dim, self.action_dim, cfg).to(
            device
        )
        self.planner = StochasticMPPIPlanner(cfg)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5
        )
        if cfg.learning_mode == "off_policy":
            self.scale = RunningScale(cfg.ema_decay, device)
            # Off-policy planning bootstraps Q(z, pi(z)); the old V head is unused.
            self.model.value_head.requires_grad_(False)
            q_parameters = list(self.model.q_functions.parameters())
            q_parameter_ids = {id(parameter) for parameter in q_parameters}
            world_parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad and id(parameter) not in q_parameter_ids
            ]
            self.world_model_optimizer = torch.optim.Adam(
                [
                    {"params": world_parameters, "lr": cfg.world_model_lr},
                    {"params": q_parameters, "lr": cfg.critic_lr},
                ]
            )
        else:
            self.external_critic = ValueCritic(
                self.model.latent_dim, cfg.mlp_dim, cfg.dropout
            ).to(device)
            self.augmented_critic = ValueCritic(
                self.model.latent_dim, cfg.mlp_dim, cfg.dropout
            ).to(device)
            world_parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            self.world_model_optimizer = torch.optim.Adam(
                world_parameters, lr=cfg.world_model_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                [
                    *self.external_critic.parameters(),
                    *self.augmented_critic.parameters(),
                ],
                lr=cfg.critic_lr,
                eps=1e-5,
            )
            self.external_critic.eval()
            self.augmented_critic.eval()
        self.model.eval()
        self.actor.eval()

    def _observation_tensor(
        self, observation: Mapping[str, np.ndarray | torch.Tensor]
    ) -> TensorObservation:
        self.observation_spec.validate(observation)
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for name, value in observation.items()
        }

    def _ids_tensor(
        self,
        value: np.ndarray | torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return torch.as_tensor(value, dtype=torch.long, device=self.device).reshape(
            batch_size
        )

    def _mask_tensor(
        self,
        value: np.ndarray | torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, self.action_dim, device=self.device)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(
            batch_size, self.action_dim
        )

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, np.ndarray | torch.Tensor],
        *,
        task_ids: np.ndarray | torch.Tensor | None = None,
        embodiment_ids: np.ndarray | torch.Tensor | None = None,
        action_mask: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PlannerOutput:
        observation_t = self._observation_tensor(observation)
        reference = next(iter(observation_t.values()))
        batch_size = int(reference.shape[0])
        task_t = self._ids_tensor(task_ids, batch_size)
        embodiment_t = self._ids_tensor(embodiment_ids, batch_size)
        mask_t = self._mask_tensor(action_mask, batch_size)
        latent = self.model.encode(observation_t, task_t, embodiment_t)
        assert isinstance(latent, torch.Tensor)
        return self.planner.plan(
            self.model,
            self.actor,
            latent,
            mask_t,
            task_t,
            embodiment_t,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def act_np(
        self,
        observation: Mapping[str, np.ndarray | torch.Tensor],
        *,
        task_ids: np.ndarray | torch.Tensor | None = None,
        embodiment_ids: np.ndarray | torch.Tensor | None = None,
        action_mask: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ActionBatch:
        output = self.act(
            observation,
            task_ids=task_ids,
            embodiment_ids=embodiment_ids,
            action_mask=action_mask,
            deterministic=deterministic,
        )
        return ActionBatch(
            action=to_numpy(output.action).astype(np.float32),
            pre_tanh_action=to_numpy(output.pre_tanh).astype(np.float32),
            candidate_noise=to_numpy(output.candidate_noise).astype(np.float32),
            log_prob=to_numpy(output.log_prob).astype(np.float32),
        )

    @torch.no_grad()
    def value_np(
        self,
        observation: Mapping[str, np.ndarray | torch.Tensor],
        *,
        task_ids: np.ndarray | torch.Tensor | None = None,
        embodiment_ids: np.ndarray | torch.Tensor | None = None,
        action_mask: np.ndarray | torch.Tensor | None = None,
        augmented: bool = False,
    ) -> np.ndarray:
        observation_t = self._observation_tensor(observation)
        reference = next(iter(observation_t.values()))
        batch_size = int(reference.shape[0])
        task_t = self._ids_tensor(task_ids, batch_size)
        embodiment_t = self._ids_tensor(embodiment_ids, batch_size)
        latent = self.model.encode(observation_t, task_t, embodiment_t)
        assert isinstance(latent, torch.Tensor)
        if self.cfg.learning_mode == "off_policy":
            mask = self._mask_tensor(action_mask, batch_size)
            action = self.actor.mean_action(latent, mask)
            value = self.model.q(
                latent, action, mask, task_t, embodiment_t, return_type="avg"
            )
            return to_numpy(value.squeeze(-1)).astype(np.float32)
        critic = self.augmented_critic if augmented else self.external_critic
        return to_numpy(critic(latent).squeeze(-1)).astype(np.float32)

    @staticmethod
    def _flatten_time_batch(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(0, 1)

    @torch.no_grad()
    def _prepare_targets(self, batch: RolloutBatch, progress_fraction: float) -> OnPolicyTargets:
        cfg = self.cfg
        latent = self.model.encode(batch.observations, batch.task_ids, batch.embodiment_ids)
        next_latent = self.model.encode(
            batch.next_observations, batch.task_ids, batch.embodiment_ids
        )
        assert isinstance(latent, torch.Tensor) and isinstance(next_latent, torch.Tensor)

        external_value = self.external_critic(latent)
        augmented_value = self.augmented_critic(latent)
        final_external_value = self.external_critic(next_latent[-1])
        final_augmented_value = self.augmented_critic(next_latent[-1])

        predicted_next = self.model.next(
            latent,
            batch.actions,
            batch.action_masks,
            batch.task_ids,
            batch.embodiment_ids,
        )
        predicted_reward = self.model.reward(
            latent,
            batch.actions,
            batch.action_masks,
            batch.task_ids,
            batch.embodiment_ids,
        )
        predicted_termination = self.model.termination(
            latent,
            batch.actions,
            batch.action_masks,
            batch.task_ids,
            batch.embodiment_ids,
        )
        predicted_value = self.model.value(
            predicted_next,
            batch.task_ids,
            batch.embodiment_ids,
        )
        model_based = predicted_reward + cfg.discount * (1.0 - predicted_termination) * predicted_value
        model_free = batch.rewards + cfg.discount * (1.0 - batch.terminated) * self.external_critic(
            next_latent
        )
        discrepancy = (model_based - model_free).abs()
        bonus = normalize_value_discrepancy(
            discrepancy,
            cfg.discrepancy_max,
            eps=cfg.normalization_epsilon,
        )
        assert isinstance(bonus, torch.Tensor)
        annealing = max(
            0.0,
            1.0 - progress_fraction / max(cfg.discrepancy_decay_fraction, 1e-12),
        )
        beta = cfg.discrepancy_beta * annealing
        augmented_reward = batch.rewards + beta * bonus

        external_values = torch.cat((external_value, final_external_value.unsqueeze(0)), dim=0)
        augmented_values = torch.cat((augmented_value, final_augmented_value.unsqueeze(0)), dim=0)
        external_advantage, external_return = generalized_advantage_estimate(
            batch.rewards,
            external_values,
            batch.terminated,
            discount=cfg.discount,
            gae_lambda=cfg.gae_lambda,
        )
        augmented_advantage, augmented_return = generalized_advantage_estimate(
            augmented_reward,
            augmented_values,
            batch.terminated,
            discount=cfg.discount,
            gae_lambda=cfg.gae_lambda,
        )
        normalized_augmented_advantage = masked_normalize(
            augmented_advantage,
            eps=cfg.normalization_epsilon,
        )
        return OnPolicyTargets(
            latent=latent.detach(),
            next_latent=next_latent.detach(),
            external_advantage=external_advantage.detach(),
            augmented_advantage=augmented_advantage.detach(),
            normalized_augmented_advantage=normalized_augmented_advantage.detach(),
            external_return=external_return.detach(),
            augmented_return=augmented_return.detach(),
            old_external_value=external_value.detach(),
            old_augmented_value=augmented_value.detach(),
            discrepancy=discrepancy.detach(),
            bonus=bonus.detach(),
            augmented_reward=augmented_reward.detach(),
            beta=beta,
        )

    @staticmethod
    def _mean_metric(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.zeros((), device=device)
        return torch.stack([value.detach().float().mean() for value in values]).mean()

    def _update_actor_critics(
        self,
        batch: RolloutBatch,
        targets: OnPolicyTargets,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        flat_batch = batch.flatten()
        flat_latent = self._flatten_time_batch(targets.latent)
        flat_advantage = self._flatten_time_batch(targets.normalized_augmented_advantage)
        flat_external_return = self._flatten_time_batch(targets.external_return)
        flat_augmented_return = self._flatten_time_batch(targets.augmented_return)
        flat_old_external = self._flatten_time_batch(targets.old_external_value)
        flat_old_augmented = self._flatten_time_batch(targets.old_augmented_value)

        actor_losses: list[torch.Tensor] = []
        critic_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        approximate_kls: list[torch.Tensor] = []
        observed_approximate_kls: list[torch.Tensor] = []
        clip_fractions: list[torch.Tensor] = []
        max_abs_log_ratios: list[torch.Tensor] = []
        saturation_fractions: list[torch.Tensor] = []
        actor_grad_norms: list[torch.Tensor] = []
        critic_grad_norms: list[torch.Tensor] = []
        actor_optimizer_steps = 0
        actor_step_attempts = 0
        actor_backtrack_steps = 0
        actor_early_stopped = False
        actor_parameters = list(self.actor.parameters())
        self.actor.train()
        self.external_critic.train()
        self.augmented_critic.train()
        with self.model.frozen_parameters():
            for _ in range(cfg.ppo_epochs):
                permutation = torch.randperm(batch.num_transitions, device=self.device)
                for start in range(0, batch.num_transitions, cfg.minibatch_size):
                    index = permutation[start : start + cfg.minibatch_size]
                    minibatch = flat_batch.select(index)
                    latent = flat_latent.index_select(0, index)
                    advantage = flat_advantage.index_select(0, index)
                    if not actor_early_stopped:
                        output = self.planner.recompute(
                            self.model,
                            self.actor,
                            latent,
                            minibatch.action_masks,
                            minibatch.candidate_noise,
                            minibatch.pre_tanh_actions,
                            minibatch.task_ids,
                            minibatch.embodiment_ids,
                        )
                        raw_log_ratio = output.log_prob - minibatch.old_log_prob
                        pre_statistics = _policy_ratio_statistics(
                            raw_log_ratio,
                            cfg.ppo_clip_ratio,
                        )
                        observed_approximate_kls.append(
                            torch.nan_to_num(
                                pre_statistics.approximate_kl.detach().float(),
                                nan=torch.finfo(torch.float32).max,
                                posinf=torch.finfo(torch.float32).max,
                                neginf=0.0,
                            )
                        )
                        if (
                            not bool(
                                torch.isfinite(pre_statistics.approximate_kl).item()
                            )
                            or (
                                cfg.ppo_target_kl is not None
                                and float(pre_statistics.approximate_kl.item())
                                > cfg.ppo_target_kl
                            )
                        ):
                            actor_early_stopped = True
                            self.actor_optimizer.zero_grad(set_to_none=True)
                        else:
                            clipped_ratio = pre_statistics.ratio.clamp(
                                1.0 - cfg.ppo_clip_ratio,
                                1.0 + cfg.ppo_clip_ratio,
                            )
                            surrogate = torch.minimum(
                                pre_statistics.ratio * advantage,
                                clipped_ratio * advantage,
                            )
                            entropy = diagonal_gaussian_entropy(
                                output.std.clamp_min(1e-8).log(),
                                minibatch.action_masks,
                            )
                            actor_loss = (
                                -surrogate.mean()
                                - cfg.entropy_coef * entropy.mean()
                            )
                            self.actor_optimizer.zero_grad(set_to_none=True)
                            actor_loss.backward()
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.actor.parameters(), cfg.grad_clip_norm
                            )
                            parameter_snapshot = [
                                parameter.detach().clone()
                                for parameter in actor_parameters
                            ]
                            optimizer_snapshot = deepcopy(
                                self.actor_optimizer.state_dict()
                            )
                            base_learning_rates = [
                                float(group["lr"])
                                for group in self.actor_optimizer.param_groups
                            ]
                            max_backtracks = (
                                _MAX_KL_BACKTRACK_STEPS
                                if cfg.ppo_target_kl is not None
                                else 0
                            )
                            accepted_statistics: _PolicyRatioStatistics | None = None
                            accepted_entropy: torch.Tensor | None = None
                            for backtrack in range(max_backtracks + 1):
                                if backtrack:
                                    with torch.no_grad():
                                        for parameter, snapshot in zip(
                                            actor_parameters,
                                            parameter_snapshot,
                                            strict=True,
                                        ):
                                            parameter.copy_(snapshot)
                                    self.actor_optimizer.load_state_dict(
                                        deepcopy(optimizer_snapshot)
                                    )
                                learning_rate_scale = _KL_BACKTRACK_FACTOR**backtrack
                                for group, base_learning_rate in zip(
                                    self.actor_optimizer.param_groups,
                                    base_learning_rates,
                                    strict=True,
                                ):
                                    group["lr"] = (
                                        base_learning_rate * learning_rate_scale
                                    )
                                self.actor_optimizer.step()
                                actor_step_attempts += 1

                                # PPO diagnostics describe the proposed policy
                                # after its optimizer step.  Unsafe proposals are
                                # rolled back together with Adam's moment state.
                                with torch.no_grad():
                                    post_output = self.planner.recompute(
                                        self.model,
                                        self.actor,
                                        latent,
                                        minibatch.action_masks,
                                        minibatch.candidate_noise,
                                        minibatch.pre_tanh_actions,
                                        minibatch.task_ids,
                                        minibatch.embodiment_ids,
                                    )
                                    post_statistics = _policy_ratio_statistics(
                                        post_output.log_prob
                                        - minibatch.old_log_prob,
                                        cfg.ppo_clip_ratio,
                                    )
                                    post_entropy = diagonal_gaussian_entropy(
                                        post_output.std.clamp_min(1e-8).log(),
                                        minibatch.action_masks,
                                    )
                                observed_approximate_kls.append(
                                    torch.nan_to_num(
                                        post_statistics.approximate_kl.detach().float(),
                                        nan=torch.finfo(torch.float32).max,
                                        posinf=torch.finfo(torch.float32).max,
                                        neginf=0.0,
                                    )
                                )
                                finite_kl = bool(
                                    torch.isfinite(
                                        post_statistics.approximate_kl
                                    ).item()
                                )
                                within_target = (
                                    cfg.ppo_target_kl is None
                                    or float(
                                        post_statistics.approximate_kl.item()
                                    )
                                    <= cfg.ppo_target_kl
                                )
                                if finite_kl and within_target:
                                    accepted_statistics = post_statistics
                                    accepted_entropy = post_entropy
                                    break
                                actor_backtrack_steps += 1

                            if accepted_statistics is None:
                                with torch.no_grad():
                                    for parameter, snapshot in zip(
                                        actor_parameters,
                                        parameter_snapshot,
                                        strict=True,
                                    ):
                                        parameter.copy_(snapshot)
                                self.actor_optimizer.load_state_dict(
                                    deepcopy(optimizer_snapshot)
                                )
                                self.actor_optimizer.zero_grad(set_to_none=True)
                                actor_early_stopped = True
                            else:
                                assert accepted_entropy is not None
                                actor_optimizer_steps += 1
                                actor_losses.append(actor_loss)
                                entropies.append(accepted_entropy.mean())
                                approximate_kls.append(
                                    accepted_statistics.approximate_kl
                                )
                                clip_fractions.append(
                                    accepted_statistics.clip_fraction
                                )
                                max_abs_log_ratios.append(
                                    accepted_statistics.max_abs_log_ratio
                                )
                                saturation_fractions.append(
                                    accepted_statistics.saturation_fraction
                                )
                                actor_grad_norms.append(
                                    torch.as_tensor(
                                        actor_grad_norm,
                                        device=self.device,
                                    )
                                )

                    external_value = self.external_critic(latent)
                    augmented_value = self.augmented_critic(latent)
                    old_external = flat_old_external.index_select(0, index)
                    old_augmented = flat_old_augmented.index_select(0, index)
                    external_target = flat_external_return.index_select(0, index)
                    augmented_target = flat_augmented_return.index_select(0, index)
                    if cfg.value_clip_ratio is None:
                        external_loss = (external_value - external_target).square().mean()
                        augmented_loss = (augmented_value - augmented_target).square().mean()
                    else:
                        clipped_external = old_external + (external_value - old_external).clamp(
                            -cfg.value_clip_ratio, cfg.value_clip_ratio
                        )
                        clipped_augmented = old_augmented + (augmented_value - old_augmented).clamp(
                            -cfg.value_clip_ratio, cfg.value_clip_ratio
                        )
                        external_loss = torch.maximum(
                            (external_value - external_target).square(),
                            (clipped_external - external_target).square(),
                        ).mean()
                        augmented_loss = torch.maximum(
                            (augmented_value - augmented_target).square(),
                            (clipped_augmented - augmented_target).square(),
                        ).mean()
                    critic_loss = cfg.critic_coef * 0.5 * (external_loss + augmented_loss)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        [*self.external_critic.parameters(), *self.augmented_critic.parameters()],
                        cfg.grad_clip_norm,
                    )
                    self.critic_optimizer.step()

                    critic_losses.append(critic_loss)
                    critic_grad_norms.append(torch.as_tensor(critic_grad_norm, device=self.device))

        self.actor.eval()
        self.external_critic.eval()
        self.augmented_critic.eval()
        return {
            "actor_loss": self._mean_metric(actor_losses, self.device),
            "critic_loss": self._mean_metric(critic_losses, self.device),
            "planner_entropy": self._mean_metric(entropies, self.device),
            "approximate_kl": self._mean_metric(approximate_kls, self.device),
            "max_observed_approximate_kl": (
                torch.stack(observed_approximate_kls).max()
                if observed_approximate_kls
                else torch.zeros((), device=self.device)
            ),
            "clip_fraction": self._mean_metric(clip_fractions, self.device),
            "max_abs_log_ratio": (
                torch.stack(max_abs_log_ratios).max()
                if max_abs_log_ratios
                else torch.zeros((), device=self.device)
            ),
            "log_ratio_saturation_fraction": self._mean_metric(
                saturation_fractions,
                self.device,
            ),
            "actor_optimizer_steps": torch.tensor(
                actor_optimizer_steps,
                device=self.device,
                dtype=torch.float32,
            ),
            "actor_step_attempts": torch.tensor(
                actor_step_attempts,
                device=self.device,
                dtype=torch.float32,
            ),
            "actor_backtrack_steps": torch.tensor(
                actor_backtrack_steps,
                device=self.device,
                dtype=torch.float32,
            ),
            "actor_early_stopped": torch.tensor(
                float(actor_early_stopped),
                device=self.device,
            ),
            "actor_learning_rate": torch.tensor(
                float(self.actor_optimizer.param_groups[0]["lr"]),
                device=self.device,
            ),
            "actor_grad_norm": self._mean_metric(actor_grad_norms, self.device),
            "critic_grad_norm": self._mean_metric(critic_grad_norms, self.device),
        }

    @staticmethod
    def _gather_observation(
        observation: Mapping[str, torch.Tensor],
        time_index: torch.Tensor,
        env_index: torch.Tensor,
    ) -> TensorObservation:
        return {name: value[time_index, env_index] for name, value in observation.items()}

    def _world_model_step(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        sample_count = cfg.minibatch_size
        max_start = batch.length - cfg.model_horizon + 1
        start = torch.randint(max_start, (sample_count,), device=self.device)
        env = torch.randint(batch.num_envs, (sample_count,), device=self.device)
        offsets = torch.arange(cfg.model_horizon, device=self.device)[:, None]
        time = start[None, :] + offsets
        env_time = env[None, :].expand_as(time)

        observations = self._gather_observation(batch.observations, start, env)
        actions = batch.actions[time, env_time]
        rewards = batch.rewards[time, env_time]
        terminated = batch.terminated[time, env_time]
        action_masks = batch.action_masks[time, env_time]
        task_ids = batch.task_ids[time, env_time]
        embodiment_ids = batch.embodiment_ids[time, env_time]

        with torch.no_grad():
            target_latents: list[torch.Tensor] = []
            target_values: list[torch.Tensor] = []
            for index in range(cfg.model_horizon):
                next_observation = self._gather_observation(
                    batch.next_observations, time[index], env_time[index]
                )
                target_latent = self.model.target_encode(
                    next_observation, task_ids[index], embodiment_ids[index]
                )
                assert isinstance(target_latent, torch.Tensor)
                target_latents.append(target_latent)
                target_values.append(
                    self.model.target_value(
                        target_latent, task_ids[index], embodiment_ids[index]
                    )
                )
            target_latent_tensor = torch.stack(target_latents, dim=0)
            target_value_tensor = torch.stack(target_values, dim=0)
            value_targets = torch.empty_like(rewards)
            running = target_value_tensor[-1]
            for index in range(cfg.model_horizon - 1, -1, -1):
                bootstrap = (
                    (1.0 - cfg.wm_lambda) * target_value_tensor[index]
                    + cfg.wm_lambda * running
                )
                running = rewards[index] + cfg.discount * (1.0 - terminated[index]) * bootstrap
                value_targets[index] = running

        latent = self.model.encode(observations, task_ids[0], embodiment_ids[0])
        assert isinstance(latent, torch.Tensor)
        latent_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        termination_losses: list[torch.Tensor] = []
        for index in range(cfg.model_horizon):
            reward_prediction = self.model.reward(
                latent,
                actions[index],
                action_masks[index],
                task_ids[index],
                embodiment_ids[index],
            )
            value_prediction = self.model.value(
                latent, task_ids[index], embodiment_ids[index]
            )
            termination_logits = self.model.termination_logits(
                latent,
                actions[index],
                action_masks[index],
                task_ids[index],
                embodiment_ids[index],
            )
            latent = self.model.next(
                latent,
                actions[index],
                action_masks[index],
                task_ids[index],
                embodiment_ids[index],
            )
            latent_losses.append(
                (latent - target_latent_tensor[index].detach()).square().sum(dim=-1, keepdim=True)
            )
            reward_losses.append(
                F.smooth_l1_loss(reward_prediction, rewards[index], reduction="none")
            )
            value_losses.append(
                F.smooth_l1_loss(value_prediction, value_targets[index].detach(), reduction="none")
            )
            termination_losses.append(
                F.binary_cross_entropy_with_logits(
                    termination_logits, terminated[index], reduction="none"
                )
            )

        latent_loss_tensor = torch.stack(latent_losses, dim=0)
        reward_loss_tensor = torch.stack(reward_losses, dim=0)
        value_loss_tensor = torch.stack(value_losses, dim=0)
        termination_loss_tensor = torch.stack(termination_losses, dim=0)
        continuation = torch.ones_like(terminated)
        for index in range(1, cfg.model_horizon):
            continuation[index] = continuation[index - 1] * (1.0 - terminated[index - 1])
        latent_mask = continuation * (1.0 - terminated)

        def segment_average(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            numerator = (loss * mask).sum(dim=0)
            denominator = mask.sum(dim=0)
            return (numerator / (denominator + cfg.normalization_epsilon)).mean()

        consistency_loss = segment_average(latent_loss_tensor, latent_mask)
        reward_loss = segment_average(reward_loss_tensor, continuation)
        value_loss = segment_average(value_loss_tensor, continuation)
        termination_loss = segment_average(termination_loss_tensor, continuation)
        total_loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * value_loss
            + cfg.termination_coef * termination_loss
        )
        self.world_model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
        self.world_model_optimizer.step()
        self.model.soft_update_targets()
        return {
            "model_loss": total_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "world_value_loss": value_loss.detach(),
            "termination_loss": termination_loss.detach(),
            "model_grad_norm": torch.as_tensor(grad_norm, device=self.device).detach(),
        }

    def _update_replay(
        self, batch: ReplayBatch, *, actor_mode: str = "q_max"
    ) -> Dict[str, float]:
        """Fit the world model and Q targets, then optimize the selected actor loss."""

        if actor_mode not in {"q_max", "behavior_cloning"}:
            raise ValueError("actor_mode must be 'q_max' or 'behavior_cloning'")
        cfg = self.cfg
        batch = batch.to(self.device)
        horizon = batch.actions.shape[0]
        if horizon != cfg.model_horizon:
            raise ValueError(
                f"Replay horizon must equal model_horizon={cfg.model_horizon}"
            )
        weights = batch.valid * (
            cfg.rho ** torch.arange(horizon, device=self.device)
        ).view(-1, 1, 1)
        if not bool((weights.sum() > 0).item()):
            raise ValueError("Replay batch must contain at least one valid transition")

        def weighted_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * weights).sum() / weights.sum()

        self.model.eval()
        self.actor.eval()
        with torch.no_grad():
            next_observations = {
                name: value[1:] for name, value in batch.observations.items()
            }
            target_latents = self.model.target_encode(
                next_observations, batch.task_ids[1:], batch.embodiment_ids[1:]
            )
            # Q and its target share the online latent coordinates, as in M3PO.
            next_latents = self.model.encode(
                next_observations, batch.task_ids[1:], batch.embodiment_ids[1:]
            )
            next_actions, _ = self.actor.sample_squashed(
                next_latents, batch.action_masks
            )
            next_q = self.model.q(
                next_latents,
                next_actions,
                batch.action_masks,
                batch.task_ids[1:],
                batch.embodiment_ids[1:],
                target=True,
                return_type="min",
            )
            td_targets = (
                batch.rewards + cfg.discount * (1.0 - batch.terminated) * next_q
            )

        self.model.train()
        latent = self.model.encode(
            {name: value[0] for name, value in batch.observations.items()},
            batch.task_ids[0],
            batch.embodiment_ids[0],
        )
        actor_latents: list[torch.Tensor] = []
        consistency_losses: list[torch.Tensor] = []
        reward_losses: list[torch.Tensor] = []
        q_losses: list[torch.Tensor] = []
        termination_losses: list[torch.Tensor] = []
        for index in range(horizon):
            actor_latents.append(latent.detach())
            action, mask = batch.actions[index], batch.action_masks[index]
            task, embodiment = batch.task_ids[index], batch.embodiment_ids[index]
            reward = self.model.reward(latent, action, mask, task, embodiment)
            q_values = self.model.q(latent, action, mask, task, embodiment)
            termination = self.model.termination_logits(
                latent, action, mask, task, embodiment
            )
            latent = self.model.next(latent, action, mask, task, embodiment)
            consistency_losses.append(
                (latent - target_latents[index]).square().mean(dim=-1, keepdim=True)
            )
            reward_losses.append(
                F.smooth_l1_loss(reward, batch.rewards[index], reduction="none")
            )
            q_losses.append(
                F.smooth_l1_loss(
                    q_values,
                    td_targets[index].unsqueeze(0).expand_as(q_values),
                    reduction="none",
                ).mean(dim=0)
            )
            termination_losses.append(
                F.binary_cross_entropy_with_logits(
                    termination, batch.terminated[index], reduction="none"
                )
            )
        consistency_loss = weighted_mean(torch.stack(consistency_losses))
        reward_loss = weighted_mean(torch.stack(reward_losses))
        q_loss = weighted_mean(torch.stack(q_losses))
        termination_loss = weighted_mean(torch.stack(termination_losses))
        model_loss = (
            cfg.consistency_coef * consistency_loss
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * q_loss
            + cfg.termination_coef * termination_loss
        )
        self.world_model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            cfg.grad_clip_norm,
        )
        self.world_model_optimizer.step()
        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.model.eval()

        latents = torch.stack(actor_latents).detach()
        self.actor.train()
        # Both objectives isolate the actor from the world model. Q maximization
        # retains dQ/da; demonstration cloning requires no actor-stage Q call.
        with self.model.frozen_parameters():
            actions, info = self.actor.sample_squashed(latents, batch.action_masks)
            policy_q = None
            bc_loss = actions.new_zeros(())
            if actor_mode == "behavior_cloning":
                difference = torch.where(
                    batch.action_masks.bool(),
                    actions - batch.actions,
                    torch.zeros_like(actions),
                )
                valid_dimensions = batch.action_masks.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0)
                bc_loss = weighted_mean(
                    difference.square().sum(dim=-1, keepdim=True) / valid_dimensions
                )
                actor_loss = bc_loss - cfg.entropy_coef * weighted_mean(
                    info["scaled_entropy"]
                )
            else:
                policy_q = self.model.q(
                    latents,
                    actions,
                    batch.action_masks,
                    batch.task_ids[:-1],
                    batch.embodiment_ids[:-1],
                    return_type="avg",
                )
                self.scale.update(policy_q[batch.valid.bool()])
                actor_loss = -weighted_mean(
                    self.scale(policy_q) + cfg.entropy_coef * info["scaled_entropy"]
                )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), cfg.grad_clip_norm
            )
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
        self.actor.eval()
        self.model.soft_update_targets()
        metrics = {
            "model_loss": model_loss,
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "q_loss": q_loss,
            "termination_loss": termination_loss,
            "model_grad_norm": torch.as_tensor(model_grad_norm),
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "actor_grad_norm": torch.as_tensor(actor_grad_norm),
            "policy_entropy": weighted_mean(info["entropy"]),
            **({"policy_q": weighted_mean(policy_q)} if policy_q is not None else {}),
            "policy_scale": self.scale.value,
            "td_target_mean": weighted_mean(td_targets),
            "replay_valid_fraction": batch.valid.mean(),
        }
        return {
            **{
                name: float(value.detach().mean().item())
                for name, value in metrics.items()
            },
            "actor_learning_rate": float(self.actor_optimizer.param_groups[0]["lr"]),
            "actor_optimizer_steps": 1.0,
            "actor_bc_enabled": float(actor_mode == "behavior_cloning"),
            "exploration_effective_weight": 0.0,
            "exploration_bonus_abs_mean": 0.0,
            "policy_optimization_effective_weight": 0.0,
        }

    def update(
        self,
        rollout: RolloutBatch | EpisodeReplayBuffer,
        *,
        progress_fraction: float = 0.0,
        actor_mode: str = "q_max",
    ) -> Dict[str, float]:
        if actor_mode not in {"q_max", "behavior_cloning"}:
            raise ValueError("actor_mode must be 'q_max' or 'behavior_cloning'")
        if actor_mode == "behavior_cloning" and self.cfg.learning_mode != "off_policy":
            raise ValueError("Behavior cloning requires off-policy replay learning")
        if not 0.0 <= progress_fraction <= 1.0:
            raise ValueError("progress_fraction must be in [0, 1]")
        if self.cfg.learning_mode == "off_policy":
            if not isinstance(rollout, EpisodeReplayBuffer):
                raise TypeError("Off-policy updates require an EpisodeReplayBuffer")
            return self._update_replay(rollout.sample(), actor_mode=actor_mode)
        if not isinstance(rollout, RolloutBatch):
            raise TypeError("On-policy updates require a fresh RolloutBatch")
        if not rollout.is_time_major:
            raise ValueError("M4PO updates require a time-major fresh rollout")
        if rollout.length != self.cfg.rollout_steps:
            raise ValueError(
                f"Expected exactly {self.cfg.rollout_steps} fresh rollout steps, "
                f"got {rollout.length}"
            )
        batch = rollout.to(self.device)
        self.model.eval()
        targets = self._prepare_targets(batch, progress_fraction)
        policy_metrics = self._update_actor_critics(batch, targets)

        self.model.train()
        model_metrics: dict[str, list[torch.Tensor]] = {}
        for _ in range(self.cfg.updates_per_rollout):
            step_metrics = self._world_model_step(batch)
            for name, value in step_metrics.items():
                model_metrics.setdefault(name, []).append(value)
        self.model.eval()
        combined = {
            **policy_metrics,
            **{
                name: self._mean_metric(values, self.device)
                for name, values in model_metrics.items()
            },
            "external_advantage_mean": targets.external_advantage.mean(),
            "augmented_advantage_mean": targets.augmented_advantage.mean(),
            "external_return_mean": targets.external_return.mean(),
            "augmented_return_mean": targets.augmented_return.mean(),
            "discrepancy_mean": targets.discrepancy.mean(),
            "discrepancy_bonus_mean": targets.bonus.mean(),
            "discrepancy_beta": torch.tensor(targets.beta, device=self.device),
        }
        return {name: float(value.detach().mean().item()) for name, value in combined.items()}

    def save(
        self, path: str | Path, step: int, extra: dict[str, Any] | None = None
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        task_contexts = self.model.encoder.task_context_features
        learner_state = (
            {"scale": self.scale.state_dict()}
            if self.cfg.learning_mode == "off_policy"
            else {
                "external_critic": self.external_critic.state_dict(),
                "augmented_critic": self.augmented_critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            }
        )
        torch.save(
            {
                "algorithm": "m4po",
                "implementation_id": IMPLEMENTATION_ID,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "learning_mode": self.cfg.learning_mode,
                "step": int(step),
                "observation_spec": {
                    "image_shape": self.observation_spec.image_shape,
                    "proprio_dim": self.observation_spec.proprio_dim,
                    "state_dim": self.observation_spec.state_dim,
                },
                "action_dim": self.action_dim,
                "cfg": self.cfg.to_dict(),
                "task_contexts": None
                if task_contexts is None
                else task_contexts.detach().cpu(),
                "model": self.model.state_dict(),
                "actor": self.actor.state_dict(),
                "world_model_optimizer": self.world_model_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                **learner_state,
                "extra": extra or {},
            },
            path,
        )

    def load_checkpoint(
        self, path: str | Path, *, restore_optimizers: bool = False
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return self._load_payload(payload, restore_optimizers=restore_optimizers)

    def _load_payload(
        self, payload: dict[str, Any], *, restore_optimizers: bool = False
    ) -> dict[str, Any]:
        if payload.get("algorithm") != "m4po":
            raise ValueError(
                f"Expected algorithm='m4po', got {payload.get('algorithm')!r}"
            )
        if payload.get("implementation_id") != IMPLEMENTATION_ID:
            raise ValueError(
                f"Expected implementation_id={IMPLEMENTATION_ID!r}, "
                f"got {payload.get('implementation_id')!r}"
            )
        if payload.get("checkpoint_schema_version") not in {
            2,
            CHECKPOINT_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"Expected checkpoint_schema_version={CHECKPOINT_SCHEMA_VERSION}, "
                f"got {payload.get('checkpoint_schema_version')!r}"
            )
        saved_mode = M4POConfig.from_checkpoint_dict(payload["cfg"]).learning_mode
        if saved_mode != self.cfg.learning_mode:
            raise ValueError(
                f"Checkpoint learning_mode={saved_mode!r} cannot load into "
                f"learning_mode={self.cfg.learning_mode!r}; start a fresh run"
            )
        if payload.get("learning_mode", saved_mode) != saved_mode:
            raise ValueError(
                "Checkpoint learning_mode metadata disagrees with its configuration"
            )
        if payload.get("checkpoint_schema_version") == 2 and saved_mode != "on_policy":
            raise ValueError("Schema-2 checkpoints only support on_policy learning")
        self.model.load_state_dict(payload["model"])
        self.actor.load_state_dict(payload["actor"])
        if saved_mode == "off_policy":
            self.scale.load_state_dict(payload["scale"])
        else:
            self.external_critic.load_state_dict(payload["external_critic"])
            self.augmented_critic.load_state_dict(payload["augmented_critic"])
            self.external_critic.eval()
            self.augmented_critic.eval()
        if restore_optimizers:
            self.world_model_optimizer.load_state_dict(payload["world_model_optimizer"])
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            if saved_mode == "on_policy":
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.model.eval()
        self.actor.eval()
        return payload

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device,
        override_cfg: M4POConfig | None = None,
        *,
        restore_optimizers: bool = False,
        _payload: dict[str, Any] | None = None,
    ) -> M4POAgent:
        # Replay snapshots can occupy many GB. Keep them on the CPU and allow
        # the trainer/evaluator to reuse its already-validated deserialization.
        payload = _payload if _payload is not None else torch.load(
            Path(path), map_location="cpu", weights_only=False
        )
        if payload.get("implementation_id") != IMPLEMENTATION_ID or payload.get(
            "checkpoint_schema_version"
        ) not in {2, CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError("Checkpoint is incompatible with this M4PO implementation")
        cfg = M4POConfig.from_checkpoint_dict(payload["cfg"])
        if override_cfg is not None:
            if override_cfg.learning_mode != cfg.learning_mode:
                raise ValueError(
                    "Checkpoint learning_mode cannot be changed when loading; start a fresh run"
                )
            values = cfg.to_dict()
            overrides = override_cfg.to_dict()
            runtime_keys = {
                "env",
                "seed",
                "num_envs",
                "max_episode_steps",
                "total_steps",
                "eval_every",
                "save_every",
                "log_every",
                "log_dir",
                "device",
                "torch_deterministic",
                "quiet",
                "resume_checkpoint",
                "save_replay",
                "max_wall_time_seconds",
                "mmbench_root",
                "mmbench_eval_num_envs",
            }
            for key in runtime_keys:
                values[key] = overrides[key]
            cfg = M4POConfig(**values)
        cfg.device = str(device)
        cfg.validate()
        spec = ObservationSpec(**payload["observation_spec"])
        agent = cls(
            spec,
            int(payload["action_dim"]),
            cfg,
            device,
            task_contexts=payload.get("task_contexts"),
        )
        agent._load_payload(payload, restore_optimizers=restore_optimizers)
        return agent
