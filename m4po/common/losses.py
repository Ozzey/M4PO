from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def _broadcast_mask(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(mask, device=value.device)
    try:
        return torch.broadcast_tensors(value, mask)[1].to(dtype=value.dtype)
    except RuntimeError:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.broadcast_tensors(value, mask)[1].to(dtype=value.dtype)


def masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: int | Sequence[int] | None = None,
    *,
    keepdim: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean over valid elements, returning zero when no element is valid."""

    if mask is None:
        return value.mean(dim=dim, keepdim=keepdim)
    valid = _broadcast_mask(value, mask)
    weighted = torch.where(valid != 0, value * valid, torch.zeros_like(value))
    numerator = weighted.sum(dim=dim, keepdim=keepdim)
    denominator = valid.sum(dim=dim, keepdim=keepdim)
    return numerator / denominator.clamp_min(eps)


def masked_normalize(
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: int | Sequence[int] | None = None,
    *,
    eps: float = 1e-8,
    clip: float | None = None,
) -> torch.Tensor:
    """Normalize valid values with population variance and zero invalid entries."""

    if dim is None:
        reduce_dim: int | tuple[int, ...] = tuple(range(value.ndim))
    elif isinstance(dim, Sequence):
        reduce_dim = tuple(dim)
    else:
        reduce_dim = dim
    mean = masked_mean(value, mask, dim=reduce_dim, keepdim=True, eps=eps)
    variance = masked_mean(
        (value - mean).square(), mask, dim=reduce_dim, keepdim=True, eps=eps
    )
    normalized = (value - mean) * torch.rsqrt(variance + eps)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        normalized = normalized.clamp(-clip, clip)
    if mask is not None:
        valid = _broadcast_mask(normalized, mask)
        normalized = torch.where(
            valid != 0, normalized * valid, torch.zeros_like(normalized)
        )
    return normalized


def diagonal_gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    keepdim: bool = True,
) -> torch.Tensor:
    """Log density of a diagonal Gaussian, summed over valid action coordinates."""

    value, mean, log_std = torch.broadcast_tensors(value, mean, log_std)
    inverse_std = torch.exp(-log_std)
    component = -0.5 * ((value - mean) * inverse_std).square()
    component = component - log_std - 0.5 * math.log(2.0 * math.pi)
    if mask is not None:
        valid = _broadcast_mask(component, mask)
        component = torch.where(
            valid != 0, component * valid, torch.zeros_like(component)
        )
    return component.sum(dim=-1, keepdim=keepdim)


masked_diagonal_gaussian_log_prob = diagonal_gaussian_log_prob


def diagonal_gaussian_entropy(
    log_std: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    keepdim: bool = True,
) -> torch.Tensor:
    entropy = log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
    if mask is not None:
        valid = _broadcast_mask(entropy, mask)
        entropy = torch.where(valid != 0, entropy * valid, torch.zeros_like(entropy))
    return entropy.sum(dim=-1, keepdim=keepdim)


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    bootstrap_value: torch.Tensor | None = None,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute time-major terminal-masked GAE and value-return targets.

    ``values`` may contain either T values plus ``bootstrap_value`` or T+1
    values, in which case the last value is used for rollout-end bootstrapping.
    A rollout truncation therefore bootstraps while a true terminal does not.
    """

    if rewards.shape != terminated.shape:
        raise ValueError("rewards and terminated must have identical shapes")
    if rewards.ndim < 1:
        raise ValueError("rewards must have a time dimension")
    steps = rewards.shape[0]
    if steps == 0:
        raise ValueError("rewards must contain at least one rollout step")
    if values.shape[0] == steps + 1:
        current_values = values[:-1]
        next_values = values[1:]
    elif values.shape[0] == steps:
        if bootstrap_value is None:
            raise ValueError(
                "bootstrap_value is required when values contains only T entries"
            )
        current_values = values
        bootstrap_value = torch.as_tensor(
            bootstrap_value, device=values.device, dtype=values.dtype
        )
        target_shape = current_values.shape[1:]
        if bootstrap_value.numel() == math.prod(target_shape):
            bootstrap_value = bootstrap_value.reshape(target_shape)
        else:
            try:
                bootstrap_value = torch.broadcast_to(bootstrap_value, target_shape)
            except RuntimeError as exc:
                raise ValueError(
                    "bootstrap_value must broadcast to one value entry"
                ) from exc
        next_values = torch.cat((values[1:], bootstrap_value.unsqueeze(0)), dim=0)
    else:
        raise ValueError("values must contain T or T+1 entries")
    if current_values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError(
            "values, rewards, and terminated must agree after the time dimension"
        )
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    rewards = rewards.to(dtype=current_values.dtype)
    not_terminal = 1.0 - terminated.to(dtype=current_values.dtype)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for index in range(steps - 1, -1, -1):
        delta = (
            rewards[index]
            + discount * not_terminal[index] * next_values[index]
            - current_values[index]
        )
        running = delta + discount * gae_lambda * not_terminal[index] * running
        advantages[index] = running
    return advantages, advantages + current_values


compute_gae = generalized_advantage_estimate


@torch.no_grad()
def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    discount: float,
    trace_lambda: float,
) -> torch.Tensor:
    """Finite-horizon lambda returns with a T+1 target-value sequence."""

    if rewards.shape != terminated.shape:
        raise ValueError("rewards and terminated must have identical shapes")
    if values.shape[0] != rewards.shape[0] + 1 or values.shape[1:] != rewards.shape[1:]:
        raise ValueError("values must have shape [T+1, ...] matching rewards [T, ...]")
    if not 0.0 <= discount <= 1.0 or not 0.0 <= trace_lambda <= 1.0:
        raise ValueError("discount and trace_lambda must be in [0, 1]")
    result = torch.empty_like(rewards)
    running = values[-1]
    not_terminal = 1.0 - terminated.to(dtype=values.dtype)
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = (1.0 - trace_lambda) * values[index + 1] + trace_lambda * running
        running = rewards[index] + discount * not_terminal[index] * bootstrap
        result[index] = running
    return result


def continuation_masks(terminated: torch.Tensor) -> torch.Tensor:
    """Return M_0..M_T where M_(t+1) = M_t * (1 - terminated_t)."""

    if terminated.ndim < 1:
        raise ValueError("terminated must have a time dimension")
    terminated = terminated.float()
    first = torch.ones(
        (1, *terminated.shape[1:]), device=terminated.device, dtype=terminated.dtype
    )
    if terminated.shape[0] == 0:
        return first
    subsequent = torch.cumprod(1.0 - terminated, dim=0)
    return torch.cat((first, subsequent), dim=0)


def normalize_value_discrepancy(
    discrepancy: torch.Tensor,
    maximum: float,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize absolute value discrepancy and clip it to a non-negative bonus."""

    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    error = discrepancy.abs()
    mean = masked_mean(error, mask, keepdim=True, eps=eps)
    variance = masked_mean((error - mean).square(), mask, keepdim=True, eps=eps)
    std = torch.sqrt(variance.clamp_min(0.0))
    bonus = ((error - mean) / (std + eps)).clamp(0.0, maximum)
    if mask is not None:
        valid = _broadcast_mask(bonus, mask)
        bonus = torch.where(valid != 0, bonus * valid, torch.zeros_like(bonus))
    if return_stats:
        return bonus, mean.squeeze(), std.squeeze()
    return bonus


def value_discrepancy_bonus(
    model_based_value: torch.Tensor,
    model_free_value: torch.Tensor,
    maximum: float,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    return normalize_value_discrepancy(
        model_based_value - model_free_value,
        maximum,
        mask,
        eps=eps,
    )


def bounded_log_std(
    value: torch.Tensor, minimum: float, maximum: float
) -> torch.Tensor:
    if maximum <= minimum:
        raise ValueError("maximum must exceed minimum")
    return minimum + 0.5 * (maximum - minimum) * (torch.tanh(value) + 1.0)


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def soft_binary_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")
