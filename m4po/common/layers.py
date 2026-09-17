from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SimNorm(nn.Module):
    """Simplicial normalization over fixed-width latent groups."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % self.dim:
            raise ValueError(
                f"Last dimension {value.shape[-1]} must be divisible by "
                f"SimNorm dim {self.dim}"
            )
        shape = value.shape
        value = value.reshape(*shape[:-1], -1, self.dim)
        return F.softmax(value, dim=-1).reshape(*shape)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class NormedLinear(nn.Linear):
    """Linear layer followed by optional dropout, LayerNorm, and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = (
            activation if activation is not None else nn.Mish(inplace=False)
        )
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.linear(value, self.weight, self.bias)
        if self.dropout is not None:
            value = self.dropout(value)
        return self.activation(self.norm(value))


def mlp(
    input_dim: int,
    hidden_dims: int | Sequence[int],
    output_dim: int,
    *,
    output_activation: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    dims = [input_dim, *hidden_dims, output_dim]
    modules: list[nn.Module] = []
    for index in range(len(dims) - 2):
        modules.append(
            NormedLinear(
                dims[index], dims[index + 1], dropout=dropout if index == 0 else 0.0
            )
        )
    if output_activation is None:
        modules.append(nn.Linear(dims[-2], dims[-1]))
    else:
        modules.append(NormedLinear(dims[-2], dims[-1], activation=output_activation))
    return nn.Sequential(*modules)


def weight_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
