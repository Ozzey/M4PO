from __future__ import annotations

import math

import torch
from torch import nn

from .config import M4POConfig
from .layers import mlp, weight_init


class GaussianActor(nn.Module):
    """Task- and embodiment-conditioned Gaussian prior in pre-tanh space.

    M4PO plans with samples from this prior rather than using the prior as the
    executed policy directly.  Supplying ``noise`` makes sampling a pure,
    repeatable reparameterization, which is what allows the planner likelihood
    to be recomputed during PPO while holding candidate noise fixed.
    """

    def __init__(self, latent_dim: int, action_dim: int, cfg: M4POConfig) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        hidden_dim = int(getattr(cfg, "actor_hidden_dim", cfg.mlp_dim))
        self.network = mlp(
            self.latent_dim, [hidden_dim, hidden_dim], 2 * self.action_dim
        )
        self.register_buffer("log_std_min", torch.tensor(float(cfg.log_std_min)))
        self.register_buffer(
            "log_std_difference",
            torch.tensor(float(cfg.log_std_max - cfg.log_std_min)),
        )
        self.apply(weight_init)

    @staticmethod
    def _bounded_log_std(
        value: torch.Tensor,
        minimum: torch.Tensor,
        difference: torch.Tensor,
    ) -> torch.Tensor:
        return minimum + 0.5 * difference * (torch.tanh(value) + 1.0)

    def _distribution_parameters(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, got {latent.shape[-1]}"
            )
        mean, raw_log_std = self.network(latent).chunk(2, dim=-1)
        log_std = self._bounded_log_std(
            raw_log_std,
            self.log_std_min,
            self.log_std_difference,
        )
        return mean, log_std

    def distribution_parameters(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the prior mean and standard deviation in pre-tanh space."""

        mean, log_std = self._distribution_parameters(latent)
        return mean, log_std.exp()

    def _action_mask(
        self,
        action_mask: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if action_mask is None:
            return torch.ones_like(reference)
        if action_mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action-mask dimension {self.action_dim}, got {action_mask.shape[-1]}"
            )
        return action_mask.to(device=reference.device, dtype=reference.dtype)

    def sample_pre_tanh(
        self,
        latent: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reparameterize a prior sample, optionally with caller-supplied noise."""

        mean, log_std = self._distribution_parameters(latent)
        std = log_std.exp()
        if noise is None:
            noise = torch.randn_like(mean)
        else:
            if noise.shape != mean.shape:
                raise ValueError(
                    f"Actor noise must have shape {tuple(mean.shape)}, got {tuple(noise.shape)}"
                )
            noise = noise.to(device=mean.device, dtype=mean.dtype)
        mask = self._action_mask(action_mask, mean)
        pre_tanh = (mean + std * noise) * mask
        return pre_tanh, {
            "mean": mean * mask,
            "std": std,
            "log_std": log_std,
            "noise": noise,
        }

    def mean_action(
        self,
        latent: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, _ = self._distribution_parameters(latent)
        mask = self._action_mask(action_mask, mean)
        return torch.tanh(mean) * mask

    def log_prob(
        self,
        latent: torch.Tensor,
        pre_tanh: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log density of a pre-tanh prior sample over valid coordinates."""

        mean, log_std = self._distribution_parameters(latent)
        if pre_tanh.shape != mean.shape:
            raise ValueError(
                f"Pre-tanh action must have shape {tuple(mean.shape)}, got {tuple(pre_tanh.shape)}"
            )
        mask = self._action_mask(action_mask, mean)
        safe_pre_tanh = torch.where(mask.bool(), pre_tanh, mean)
        normalized = (safe_pre_tanh - mean) * torch.exp(-log_std)
        per_dimension = (
            -0.5 * normalized.square() - log_std - 0.5 * math.log(2.0 * math.pi)
        )
        return (per_dimension * mask).sum(dim=-1, keepdim=True)

    def forward(
        self,
        latent: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pre_tanh, info = self.sample_pre_tanh(latent, action_mask, noise=noise)
        mask = self._action_mask(action_mask, pre_tanh)
        action = torch.tanh(pre_tanh) * mask
        log_prob = self.log_prob(latent, pre_tanh, mask)
        valid_dimensions = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return action, {
            **info,
            "pre_tanh": pre_tanh,
            "log_prob": log_prob,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob / valid_dimensions,
        }
