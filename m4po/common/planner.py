from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import M4POConfig
from .losses import diagonal_gaussian_log_prob
from .policy import GaussianActor
from .world_model import HierarchicalWorldModel


@dataclass(frozen=True)
class PlannerOutput:
    """One draw from the conditional stochastic planner policy."""

    action: torch.Tensor
    pre_tanh: torch.Tensor
    candidate_noise: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class StochasticMPPIPlanner:
    """Differentiable actor-guided MPPI used as M4PO's executed policy.

    Candidate noise has canonical shape ``[batch, samples, horizon, action]``.
    Holding that noise, the initial latent, and the world model fixed makes the
    fitted first-step Gaussian a differentiable function of actor parameters.
    """

    def __init__(self, cfg: M4POConfig) -> None:
        self.cfg = cfg
        self.horizon = int(cfg.planning_horizon)
        self.num_samples = int(cfg.planner_samples)
        self.inverse_temperature = float(cfg.inverse_temperature)
        self.min_std = float(cfg.min_std)
        self.discount = float(cfg.discount)

    @staticmethod
    def _action_mask(
        action_mask: torch.Tensor | None,
        batch_size: int,
        action_dim: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if action_mask is None:
            return torch.ones(
                batch_size,
                action_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        action_mask = torch.as_tensor(
            action_mask,
            device=reference.device,
            dtype=reference.dtype,
        )
        try:
            return torch.broadcast_to(action_mask, (batch_size, action_dim))
        except RuntimeError as exc:
            raise ValueError(
                f"Action mask must broadcast to {(batch_size, action_dim)}, got "
                f"{tuple(action_mask.shape)}"
            ) from exc

    @staticmethod
    def _batch_ids(
        value: torch.Tensor | Any | None,
        batch_size: int,
        num_samples: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        ids = torch.as_tensor(value, device=device, dtype=torch.long)
        if ids.shape == (batch_size, 1):
            ids = ids.squeeze(-1)
        try:
            ids = torch.broadcast_to(ids, (batch_size,))
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must broadcast to {(batch_size,)}, got {tuple(ids.shape)}"
            ) from exc
        return ids.repeat_interleave(num_samples)

    def _candidate_noise(
        self,
        latent: torch.Tensor,
        action_dim: int,
        candidate_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = (latent.shape[0], self.num_samples, self.horizon, action_dim)
        if candidate_noise is None:
            return torch.randn(*shape, device=latent.device, dtype=latent.dtype)
        candidate_noise = torch.as_tensor(
            candidate_noise,
            device=latent.device,
            dtype=latent.dtype,
        )
        if candidate_noise.shape != shape:
            raise ValueError(
                f"Candidate noise must have shape {shape}, got {tuple(candidate_noise.shape)}"
            )
        return candidate_noise

    def _candidate_moments(
        self,
        model: HierarchicalWorldModel,
        actor: GaussianActor,
        latent: torch.Tensor,
        action_mask: torch.Tensor,
        candidate_noise: torch.Tensor,
        task_ids: torch.Tensor | Any | None,
        embodiment_ids: torch.Tensor | Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2:
            raise ValueError(
                f"Planner latent must have shape [batch, latent], got {latent.shape}"
            )
        batch_size = int(latent.shape[0])
        samples = self.num_samples
        action_dim = int(actor.action_dim)
        rollout_latent = (
            latent[:, None, :]
            .expand(-1, samples, -1)
            .reshape(
                batch_size * samples,
                latent.shape[-1],
            )
        )
        rollout_mask = (
            action_mask[:, None, :]
            .expand(-1, samples, -1)
            .reshape(
                batch_size * samples,
                action_dim,
            )
        )
        rollout_task_ids = self._batch_ids(
            task_ids,
            batch_size,
            samples,
            latent.device,
            "task_ids",
        )
        rollout_embodiment_ids = self._batch_ids(
            embodiment_ids,
            batch_size,
            samples,
            latent.device,
            "embodiment_ids",
        )

        model_return = torch.zeros(
            batch_size,
            samples,
            1,
            device=latent.device,
            dtype=latent.dtype,
        )
        continuation = torch.ones_like(model_return)
        first_pre_tanh: torch.Tensor | None = None
        for step in range(self.horizon):
            noise = candidate_noise[:, :, step, :].reshape(
                batch_size * samples, action_dim
            )
            pre_tanh, _ = actor.sample_pre_tanh(
                rollout_latent,
                rollout_mask,
                noise=noise,
            )
            if first_pre_tanh is None:
                first_pre_tanh = pre_tanh.reshape(batch_size, samples, action_dim)
            action = torch.tanh(pre_tanh) * rollout_mask
            reward = model.reward(
                rollout_latent,
                action,
                rollout_mask,
                rollout_task_ids,
                rollout_embodiment_ids,
            ).reshape(batch_size, samples, 1)
            termination = model.termination(
                rollout_latent,
                action,
                rollout_mask,
                rollout_task_ids,
                rollout_embodiment_ids,
            ).reshape(batch_size, samples, 1)
            model_return = model_return + (self.discount**step) * continuation * reward
            continuation = continuation * (1.0 - termination)
            rollout_latent = model.next(
                rollout_latent,
                action,
                rollout_mask,
                rollout_task_ids,
                rollout_embodiment_ids,
            )

        terminal_value = model.value(
            rollout_latent,
            rollout_task_ids,
            rollout_embodiment_ids,
        ).reshape(batch_size, samples, 1)
        model_return = (
            model_return + (self.discount**self.horizon) * continuation * terminal_value
        )
        score = torch.nan_to_num(
            model_return.squeeze(-1),
            nan=0.0,
            posinf=torch.finfo(model_return.dtype).max / 100.0,
            neginf=torch.finfo(model_return.dtype).min / 100.0,
        )
        # Center before applying the inverse temperature.  ``softmax`` itself is
        # stable, but multiplying a large finite return by a large kappa first
        # can overflow to infinity and turn an otherwise valid row into NaNs.
        centered_score = score - score.amax(dim=1, keepdim=True)
        weights = torch.softmax(self.inverse_temperature * centered_score, dim=1)
        assert first_pre_tanh is not None
        mean = (weights.unsqueeze(-1) * first_pre_tanh).sum(dim=1)
        variance = (
            weights.unsqueeze(-1) * (first_pre_tanh - mean.unsqueeze(1)).square()
        ).sum(dim=1)
        variance = variance + self.min_std**2
        mean = mean * action_mask
        std = torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny))
        return mean, std

    @staticmethod
    def log_prob(
        pre_tanh: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_mask is not None:
            mask = torch.as_tensor(
                action_mask,
                device=pre_tanh.device,
                dtype=pre_tanh.dtype,
            )
            # Exclude invalid coordinates before arithmetic as well as during
            # reduction, so arbitrary padded values cannot create inf * 0.
            pre_tanh = torch.where(mask.bool(), pre_tanh, mean)
        return diagonal_gaussian_log_prob(
            pre_tanh,
            mean,
            std.clamp_min(torch.finfo(std.dtype).tiny).log(),
            action_mask,
        )

    def plan(
        self,
        model: HierarchicalWorldModel,
        actor: GaussianActor,
        latent: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        candidate_noise: torch.Tensor | None = None,
        execution_noise: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PlannerOutput:
        """Collect one action and all fixed variables needed by PPO."""

        if latent.ndim != 2:
            raise ValueError("Planner expects a batched [batch, latent_dim] tensor")
        if actor.action_dim != model.action_dim:
            raise ValueError("Actor and world-model action dimensions must match")
        batch_size = int(latent.shape[0])
        mask = self._action_mask(
            action_mask,
            batch_size,
            actor.action_dim,
            latent,
        )
        noise = self._candidate_noise(latent, actor.action_dim, candidate_noise)
        # Collection never retains a graph.  The same conditional distribution
        # is rebuilt by ``recompute`` for PPO.
        with torch.no_grad(), model.frozen_parameters():
            mean, std = self._candidate_moments(
                model,
                actor,
                latent,
                mask,
                noise,
                task_ids,
                embodiment_ids,
            )
            if deterministic:
                pre_tanh = mean
            else:
                if execution_noise is None:
                    execution_noise = torch.randn_like(mean)
                else:
                    execution_noise = torch.as_tensor(
                        execution_noise,
                        device=mean.device,
                        dtype=mean.dtype,
                    )
                    if execution_noise.shape != mean.shape:
                        raise ValueError(
                            "Execution noise must have the same shape as the planner mean"
                        )
                pre_tanh = mean + std * execution_noise
            pre_tanh = pre_tanh * mask
            action = torch.tanh(pre_tanh) * mask
            log_prob = self.log_prob(pre_tanh, mean, std, mask)
        return PlannerOutput(
            action=action,
            pre_tanh=pre_tanh,
            candidate_noise=noise.detach(),
            log_prob=log_prob,
            mean=mean,
            std=std,
        )

    def recompute(
        self,
        model: HierarchicalWorldModel,
        actor: GaussianActor,
        latent: torch.Tensor,
        action_mask: torch.Tensor | None,
        candidate_noise: torch.Tensor,
        pre_tanh: torch.Tensor,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
    ) -> PlannerOutput:
        """Recompute planner likelihood with fixed model, latent, and noise."""

        if latent.ndim != 2:
            raise ValueError("Planner expects a batched [batch, latent_dim] tensor")
        if actor.action_dim != model.action_dim:
            raise ValueError("Actor and world-model action dimensions must match")
        latent = latent.detach()
        batch_size = int(latent.shape[0])
        mask = self._action_mask(
            action_mask,
            batch_size,
            actor.action_dim,
            latent,
        ).detach()
        noise = self._candidate_noise(
            latent,
            actor.action_dim,
            candidate_noise,
        ).detach()
        pre_tanh = torch.as_tensor(
            pre_tanh,
            device=latent.device,
            dtype=latent.dtype,
        ).detach()
        if pre_tanh.shape != (batch_size, actor.action_dim):
            raise ValueError(
                "Stored pre-tanh action must have shape "
                f"{(batch_size, actor.action_dim)}, got {tuple(pre_tanh.shape)}"
            )
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.detach()
        if isinstance(embodiment_ids, torch.Tensor):
            embodiment_ids = embodiment_ids.detach()
        with model.frozen_parameters():
            mean, std = self._candidate_moments(
                model,
                actor,
                latent,
                mask,
                noise,
                task_ids,
                embodiment_ids,
            )
            action = torch.tanh(pre_tanh) * mask
            log_prob = self.log_prob(pre_tanh, mean, std, mask)
        return PlannerOutput(
            action=action,
            pre_tanh=pre_tanh,
            candidate_noise=noise,
            log_prob=log_prob,
            mean=mean,
            std=std,
        )

    recompute_likelihood = recompute


# A short compatibility alias mirrors the M3PO module's planner naming style.
MPPIPlanner = StochasticMPPIPlanner
