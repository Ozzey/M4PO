from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .config import M4POConfig
from .layers import SimNorm, mlp, weight_init

if TYPE_CHECKING:
    from .buffer import ObservationSpec


def _observation_value(observation: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)


class _ImageEncoder(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int], output_dim: int) -> None:
        super().__init__()
        channels, _, _ = image_shape
        first_width = max(16, min(64, output_dim // 4))
        second_width = max(32, min(128, output_dim // 2))
        self.image_shape = tuple(int(value) for value in image_shape)
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, first_width, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=False),
            nn.Conv2d(first_width, second_width, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=False),
            nn.Conv2d(second_width, second_width, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = mlp(
            second_width,
            [],
            output_dim,
            output_activation=nn.Mish(inplace=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-3:]) != self.image_shape:
            raise ValueError(
                f"Expected image shape {self.image_shape}, got {tuple(image.shape[-3:])}"
            )
        leading_shape = image.shape[:-3]
        flattened = image.reshape(-1, *self.image_shape)
        features = self.convolution(flattened).flatten(1)
        features = self.projection(features)
        return features.reshape(*leading_shape, features.shape[-1])


class _MaskedVectorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        hidden_dims = [output_dim] * max(num_layers - 1, 1)
        # Concatenating the mask lets the shared encoder distinguish a real
        # normalized zero from padding while still zeroing invalid values.
        self.network = mlp(
            2 * self.input_dim,
            hidden_dims,
            output_dim,
            output_activation=nn.Mish(inplace=False),
        )

    def forward(self, value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected vector dimension {self.input_dim}, got {value.shape[-1]}"
            )
        if mask is None:
            mask = torch.ones_like(value)
        else:
            if mask.shape != value.shape:
                raise ValueError(
                    f"Observation mask must have shape {tuple(value.shape)}, got {tuple(mask.shape)}"
                )
            mask = mask.to(device=value.device, dtype=value.dtype)
        return self.network(torch.cat((value * mask, mask), dim=-1))


class _HierarchicalEncoder(nn.Module):
    """Multimodal/context encoder producing ``z = [g, b]``."""

    def __init__(
        self,
        observation_spec: ObservationSpec,
        cfg: M4POConfig,
        task_contexts: torch.Tensor | Any | None,
    ) -> None:
        super().__init__()
        raw_image_shape = observation_spec.image_shape
        self.image_shape = (
            (0, 0, 0)
            if raw_image_shape is None
            else tuple(int(value) for value in raw_image_shape)
        )
        self.proprio_dim = int(observation_spec.proprio_dim)
        self.state_dim = int(observation_spec.state_dim)
        self.task_context_dim = int(cfg.task_context_dim)
        self.embodiment_context_dim = int(cfg.embodiment_context_dim)
        self.task_latent_dim = int(cfg.task_latent_dim)
        self.body_latent_dim = int(cfg.body_latent_dim)
        self.num_tasks = int(cfg.num_tasks)
        self.num_embodiments = int(cfg.num_embodiments)

        feature_dim = int(cfg.encoder_dim)
        self.image_encoder = (
            _ImageEncoder(self.image_shape, feature_dim)
            if all(dimension > 0 for dimension in self.image_shape)
            else None
        )
        self.proprio_encoder = (
            _MaskedVectorEncoder(self.proprio_dim, feature_dim, cfg.num_encoder_layers)
            if self.proprio_dim
            else None
        )
        self.state_encoder = (
            _MaskedVectorEncoder(self.state_dim, feature_dim, cfg.num_encoder_layers)
            if self.state_dim
            else None
        )
        modality_count = sum(
            module is not None
            for module in (self.image_encoder, self.proprio_encoder, self.state_encoder)
        )
        if modality_count == 0:
            raise ValueError("At least one observation modality is required")

        if task_contexts is None:
            self.task_embedding: nn.Embedding | None = nn.Embedding(
                self.num_tasks, self.task_context_dim
            )
            self.task_projection: nn.Module | None = None
            self.register_buffer("task_context_features", None)
        else:
            features = torch.as_tensor(task_contexts, dtype=torch.float32)
            if features.ndim != 2 or features.shape[0] != self.num_tasks:
                raise ValueError(
                    "Frozen task contexts must have shape [num_tasks, feature_dim], got "
                    f"{tuple(features.shape)} for {self.num_tasks} tasks"
                )
            self.task_embedding = None
            self.task_projection = mlp(
                int(features.shape[-1]),
                [],
                self.task_context_dim,
                output_activation=nn.Mish(inplace=False),
            )
            self.register_buffer("task_context_features", features.contiguous())
        self.embodiment_embedding = nn.Embedding(
            self.num_embodiments,
            self.embodiment_context_dim,
        )

        fusion_input_dim = (
            modality_count * feature_dim
            + self.task_context_dim
            + self.embodiment_context_dim
        )
        self.fusion = mlp(
            fusion_input_dim,
            [cfg.mlp_dim],
            cfg.encoder_dim,
            output_activation=nn.Mish(inplace=False),
            dropout=cfg.dropout,
        )
        self.task_latent = mlp(
            cfg.encoder_dim + self.task_context_dim,
            [cfg.mlp_dim],
            self.task_latent_dim,
            output_activation=SimNorm(cfg.simnorm_dim),
            dropout=cfg.dropout,
        )
        self.body_latent = mlp(
            cfg.encoder_dim + self.embodiment_context_dim,
            [cfg.mlp_dim],
            self.body_latent_dim,
            output_activation=SimNorm(cfg.simnorm_dim),
            dropout=cfg.dropout,
        )

    @property
    def latent_dim(self) -> int:
        return self.task_latent_dim + self.body_latent_dim

    def _reference_parameter(self) -> torch.Tensor:
        return next(self.fusion.parameters())

    def _tensor(
        self, value: Any, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        reference = self._reference_parameter()
        return torch.as_tensor(value, device=reference.device, dtype=dtype)

    @staticmethod
    def _leading_shape(observation: Mapping[str, Any] | Any) -> torch.Size:
        image = _observation_value(observation, "image")
        if image is not None:
            return torch.as_tensor(image).shape[:-3]
        proprio = _observation_value(observation, "proprio")
        if proprio is not None:
            return torch.as_tensor(proprio).shape[:-1]
        state = _observation_value(observation, "state")
        if state is not None:
            return torch.as_tensor(state).shape[:-1]
        raise ValueError("Observation contains no image, proprioception, or state")

    def _ids(
        self,
        value: torch.Tensor | Any | None,
        leading_shape: torch.Size | tuple[int, ...],
        *,
        upper_bound: int,
        name: str,
    ) -> torch.Tensor:
        reference = self._reference_parameter()
        if value is None:
            result = torch.zeros(
                leading_shape, device=reference.device, dtype=torch.long
            )
        else:
            result = torch.as_tensor(value, device=reference.device, dtype=torch.long)
            if result.shape == (*leading_shape, 1):
                result = result.squeeze(-1)
            try:
                result = torch.broadcast_to(result, leading_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"{name} must broadcast to {tuple(leading_shape)}, got {tuple(result.shape)}"
                ) from exc
        if result.numel() and (
            bool((result < 0).any()) or bool((result >= upper_bound).any())
        ):
            raise ValueError(f"{name} values must be in [0, {upper_bound})")
        return result

    def contexts(
        self,
        task_ids: torch.Tensor | Any | None,
        embodiment_ids: torch.Tensor | Any | None,
        leading_shape: torch.Size | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids_t = self._ids(
            task_ids,
            leading_shape,
            upper_bound=self.num_tasks,
            name="task_ids",
        )
        embodiment_ids_t = self._ids(
            embodiment_ids,
            leading_shape,
            upper_bound=self.num_embodiments,
            name="embodiment_ids",
        )
        if self.task_embedding is not None:
            task_context = self.task_embedding(task_ids_t)
        else:
            assert self.task_context_features is not None
            assert self.task_projection is not None
            task_context = self.task_projection(self.task_context_features[task_ids_t])
        embodiment_context = self.embodiment_embedding(embodiment_ids_t)
        return task_context, embodiment_context

    def forward(
        self,
        observation: Mapping[str, Any] | Any,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        leading_shape = self._leading_shape(observation)
        features: list[torch.Tensor] = []
        if self.image_encoder is not None:
            image = _observation_value(observation, "image")
            if image is None:
                raise ValueError(
                    "Observation is missing the configured `image` modality"
                )
            image_t = self._tensor(image)
            image_features = self.image_encoder(image_t)
            image_mask = _observation_value(observation, "image_mask")
            if image_mask is not None:
                image_mask_t = self._tensor(image_mask)
                while image_mask_t.ndim < image_features.ndim:
                    image_mask_t = image_mask_t.unsqueeze(-1)
                image_features = image_features * image_mask_t
            features.append(image_features)
        if self.proprio_encoder is not None:
            proprio = _observation_value(observation, "proprio")
            if proprio is None:
                raise ValueError(
                    "Observation is missing the configured `proprio` modality"
                )
            proprio_t = self._tensor(proprio)
            proprio_mask = _observation_value(observation, "proprio_mask")
            proprio_mask_t = (
                None if proprio_mask is None else self._tensor(proprio_mask)
            )
            features.append(self.proprio_encoder(proprio_t, proprio_mask_t))
        if self.state_encoder is not None:
            state = _observation_value(observation, "state")
            if state is None:
                raise ValueError(
                    "Observation is missing the configured `state` modality"
                )
            state_t = self._tensor(state)
            state_mask = _observation_value(observation, "state_mask")
            state_mask_t = None if state_mask is None else self._tensor(state_mask)
            features.append(self.state_encoder(state_t, state_mask_t))

        task_context, embodiment_context = self.contexts(
            task_ids,
            embodiment_ids,
            leading_shape,
        )
        hidden = self.fusion(
            torch.cat((*features, task_context, embodiment_context), dim=-1)
        )
        task_latent = self.task_latent(torch.cat((hidden, task_context), dim=-1))
        body_latent = self.body_latent(torch.cat((hidden, embodiment_context), dim=-1))
        latent = torch.cat((task_latent, body_latent), dim=-1)
        if return_context:
            return latent, task_context, embodiment_context
        return latent


class HierarchicalWorldModel(nn.Module):
    """M4PO multimodal encoder, ordered dynamics, and prediction heads."""

    def __init__(
        self,
        observation_spec: ObservationSpec,
        action_dim: int,
        cfg: M4POConfig,
        task_contexts: torch.Tensor | Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_dim = int(action_dim)
        self.task_latent_dim = int(cfg.task_latent_dim)
        self.body_latent_dim = int(cfg.body_latent_dim)
        self.model_latent_dim = self.task_latent_dim + self.body_latent_dim
        self.task_context_dim = int(cfg.task_context_dim)
        self.embodiment_context_dim = int(cfg.embodiment_context_dim)

        self.encoder = _HierarchicalEncoder(observation_spec, cfg, task_contexts)
        body_dynamics_input = (
            self.model_latent_dim
            + self.action_dim
            + self.action_dim
            + self.embodiment_context_dim
        )
        self.body_dynamics = mlp(
            body_dynamics_input,
            [cfg.mlp_dim, cfg.mlp_dim],
            self.body_latent_dim,
            output_activation=SimNorm(cfg.simnorm_dim),
            dropout=cfg.dropout,
        )
        task_dynamics_input = (
            self.model_latent_dim
            + self.body_latent_dim
            + self.action_dim
            + self.task_context_dim
        )
        self.task_dynamics = mlp(
            task_dynamics_input,
            [cfg.mlp_dim, cfg.mlp_dim],
            self.task_latent_dim,
            output_activation=SimNorm(cfg.simnorm_dim),
            dropout=cfg.dropout,
        )

        head_context_dim = self.task_context_dim + self.embodiment_context_dim
        state_action_head_input = (
            self.model_latent_dim + self.action_dim + head_context_dim
        )
        state_head_input = self.model_latent_dim + head_context_dim
        self.reward_head = mlp(
            state_action_head_input,
            [cfg.mlp_dim, cfg.mlp_dim],
            1,
            dropout=cfg.dropout,
        )
        self.value_head = mlp(
            state_head_input,
            [cfg.mlp_dim, cfg.mlp_dim],
            1,
            dropout=cfg.dropout,
        )
        self.termination_head = mlp(
            state_action_head_input,
            [cfg.mlp_dim, cfg.mlp_dim],
            1,
            dropout=cfg.dropout,
        )

        self.apply(weight_init)
        nn.init.zeros_(self.reward_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.termination_head[-1].weight)

        self.target_encoder = deepcopy(self.encoder)
        self.target_value_head = deepcopy(self.value_head)
        self.target_encoder.requires_grad_(False)
        self.target_value_head.requires_grad_(False)
        self.target_encoder.eval()
        self.target_value_head.eval()

    @property
    def latent_dim(self) -> int:
        return self.model_latent_dim

    @property
    def total_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def image_encoder(self) -> nn.Module | None:
        return self.encoder.image_encoder

    @property
    def proprio_encoder(self) -> nn.Module | None:
        return self.encoder.proprio_encoder

    @property
    def state_encoder(self) -> nn.Module | None:
        return self.encoder.state_encoder

    @property
    def task_embedding(self) -> nn.Embedding | None:
        return self.encoder.task_embedding

    @property
    def embodiment_embedding(self) -> nn.Embedding:
        return self.encoder.embodiment_embedding

    def train(self, mode: bool = True) -> HierarchicalWorldModel:
        super().train(mode)
        self.target_encoder.eval()
        self.target_value_head.eval()
        return self

    def encode(
        self,
        observation: Mapping[str, Any] | Any,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        target: bool = False,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder = self.target_encoder if target else self.encoder
        return encoder(
            observation,
            task_ids,
            embodiment_ids,
            return_context=return_context,
        )

    def encode_target(
        self,
        observation: Mapping[str, Any] | Any,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(
            observation,
            task_ids,
            embodiment_ids,
            target=True,
            return_context=return_context,
        )

    target_encode = encode_target

    def _contexts(
        self,
        latent: torch.Tensor,
        task_ids: torch.Tensor | Any | None,
        embodiment_ids: torch.Tensor | Any | None,
        task_context: torch.Tensor | None,
        embodiment_context: torch.Tensor | None,
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.target_encoder if target else self.encoder
        if task_context is None or embodiment_context is None:
            inferred_task, inferred_embodiment = encoder.contexts(
                task_ids,
                embodiment_ids,
                latent.shape[:-1],
            )
            task_context = inferred_task if task_context is None else task_context
            embodiment_context = (
                inferred_embodiment
                if embodiment_context is None
                else embodiment_context
            )
        task_context = task_context.to(device=latent.device, dtype=latent.dtype)
        embodiment_context = embodiment_context.to(
            device=latent.device, dtype=latent.dtype
        )
        if task_context.shape != (*latent.shape[:-1], self.task_context_dim):
            raise ValueError("Task context shape does not match the latent batch shape")
        if embodiment_context.shape != (
            *latent.shape[:-1],
            self.embodiment_context_dim,
        ):
            raise ValueError(
                "Embodiment context shape does not match the latent batch shape"
            )
        return task_context, embodiment_context

    def _masked_action(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {action.shape[-1]}"
            )
        if action_mask is None:
            action_mask = torch.ones_like(action)
        else:
            if action_mask.shape != action.shape:
                try:
                    action_mask = torch.broadcast_to(action_mask, action.shape)
                except RuntimeError as exc:
                    raise ValueError(
                        f"Action mask must broadcast to {tuple(action.shape)}, got "
                        f"{tuple(action_mask.shape)}"
                    ) from exc
            action_mask = action_mask.to(device=action.device, dtype=action.dtype)
        return action * action_mask, action_mask

    def next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict embodiment dynamics first, then task-progress dynamics."""

        if latent.shape[-1] != self.model_latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.model_latent_dim}, got {latent.shape[-1]}"
            )
        action, action_mask = self._masked_action(action, action_mask)
        task_context, embodiment_context = self._contexts(
            latent,
            task_ids,
            embodiment_ids,
            task_context,
            embodiment_context,
        )
        next_body = self.body_dynamics(
            torch.cat((latent, action, action_mask, embodiment_context), dim=-1)
        )
        next_task = self.task_dynamics(
            torch.cat((latent, next_body, action, task_context), dim=-1)
        )
        return torch.cat((next_task, next_body), dim=-1)

    transition = next

    def _head_inputs(
        self,
        latent: torch.Tensor,
        task_ids: torch.Tensor | Any | None,
        embodiment_ids: torch.Tensor | Any | None,
        task_context: torch.Tensor | None,
        embodiment_context: torch.Tensor | None,
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = self._contexts(
            latent,
            task_ids,
            embodiment_ids,
            task_context,
            embodiment_context,
            target=target,
        )
        return contexts

    def reward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action, _ = self._masked_action(action, action_mask)
        task_context, embodiment_context = self._head_inputs(
            latent,
            task_ids,
            embodiment_ids,
            task_context,
            embodiment_context,
        )
        return self.reward_head(
            torch.cat((latent, action, task_context, embodiment_context), dim=-1)
        )

    def value(
        self,
        latent: torch.Tensor,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        task_context, embodiment_context = self._head_inputs(
            latent,
            task_ids,
            embodiment_ids,
            task_context,
            embodiment_context,
            target=target,
        )
        value_head = self.target_value_head if target else self.value_head
        return value_head(torch.cat((latent, task_context, embodiment_context), dim=-1))

    def target_value(
        self,
        latent: torch.Tensor,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.value(
            latent,
            task_ids,
            embodiment_ids,
            task_context=task_context,
            embodiment_context=embodiment_context,
            target=True,
        )

    def termination_logits(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action, _ = self._masked_action(action, action_mask)
        task_context, embodiment_context = self._head_inputs(
            latent,
            task_ids,
            embodiment_ids,
            task_context,
            embodiment_context,
        )
        return self.termination_head(
            torch.cat((latent, action, task_context, embodiment_context), dim=-1)
        )

    def termination(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.termination_logits(
                latent,
                action,
                action_mask,
                task_ids,
                embodiment_ids,
                task_context=task_context,
                embodiment_context=embodiment_context,
            )
        )

    done = termination

    def predict(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | Any | None = None,
        embodiment_ids: torch.Tensor | Any | None = None,
        *,
        task_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reward = self.reward(
            latent,
            action,
            action_mask,
            task_ids,
            embodiment_ids,
            task_context=task_context,
            embodiment_context=embodiment_context,
        )
        value = self.value(
            latent,
            task_ids,
            embodiment_ids,
            task_context=task_context,
            embodiment_context=embodiment_context,
        )
        termination = self.termination(
            latent,
            action,
            action_mask,
            task_ids,
            embodiment_ids,
            task_context=task_context,
            embodiment_context=embodiment_context,
        )
        return reward, value, termination

    @torch.no_grad()
    def hard_update_targets(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())

    @torch.no_grad()
    def soft_update_targets(self, decay: float | None = None) -> None:
        decay = self.cfg.ema_decay if decay is None else float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        for online, target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)
        for online, target in zip(
            self.value_head.parameters(),
            self.target_value_head.parameters(),
            strict=True,
        ):
            target.mul_(decay).add_(online, alpha=1.0 - decay)
        for online, target in zip(
            self.encoder.buffers(),
            self.target_encoder.buffers(),
            strict=True,
        ):
            if online.is_floating_point():
                target.mul_(decay).add_(online, alpha=1.0 - decay)
            else:
                target.copy_(online)

    update_ema = soft_update_targets

    @contextmanager
    def frozen_parameters(self, *, eval_mode: bool = True) -> Iterator[None]:
        """Freeze weights without disabling gradients through actions/latents."""

        parameters = list(self.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        was_training = self.training
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            if eval_mode:
                self.eval()
            yield
        finally:
            for parameter, state in zip(parameters, requires_grad, strict=True):
                parameter.requires_grad_(state)
            self.train(was_training)

    freeze_parameters = frozen_parameters
