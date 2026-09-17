from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping

import numpy as np
import torch

TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ObservationSpec:
    """Shapes for the padded multimodal observation dictionary."""

    image_shape: tuple[int, ...]
    proprio_dim: int
    state_dim: int

    def __post_init__(self) -> None:
        image_shape = tuple(int(size) for size in self.image_shape)
        if len(image_shape) != 3 or any(size < 0 for size in image_shape):
            raise ValueError(
                "image_shape must be a non-negative [channels, height, width] shape"
            )
        if any(size == 0 for size in image_shape) and any(
            size != 0 for size in image_shape
        ):
            raise ValueError("A disabled image_shape must contain only zeros")
        object.__setattr__(self, "image_shape", image_shape)
        if self.proprio_dim < 0 or self.state_dim < 0:
            raise ValueError("proprio_dim and state_dim must be non-negative")
        image_enabled = bool(self.image_shape) and all(self.image_shape)
        if not image_enabled and self.proprio_dim == 0 and self.state_dim == 0:
            raise ValueError("At least one observation modality must be present")

    @classmethod
    def from_config(cls, cfg) -> "ObservationSpec":
        image_shape = tuple(cfg.image_shape) if cfg.use_images else (0, 0, 0)
        return cls(
            image_shape=image_shape,
            proprio_dim=int(cfg.proprio_dim),
            state_dim=int(cfg.state_dim),
        )

    @property
    def shapes(self) -> Dict[str, tuple[int, ...]]:
        shapes: Dict[str, tuple[int, ...]] = {}
        shapes["image"] = self.image_shape
        shapes["proprio"] = (self.proprio_dim,)
        shapes["proprio_mask"] = (self.proprio_dim,)
        shapes["state"] = (self.state_dim,)
        shapes["state_mask"] = (self.state_dim,)
        return shapes

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.shapes)

    def validate(self, observation: Mapping[str, np.ndarray | torch.Tensor]) -> None:
        expected = self.shapes
        if set(observation) != set(expected):
            missing = sorted(set(expected) - set(observation))
            unexpected = sorted(set(observation) - set(expected))
            raise ValueError(
                "Observation keys do not match the spec; "
                f"missing={missing}, unexpected={unexpected}"
            )
        leading_shape: tuple[int, ...] | None = None
        for name, trailing_shape in expected.items():
            shape = tuple(observation[name].shape)
            if (
                len(shape) < len(trailing_shape)
                or shape[-len(trailing_shape) :] != trailing_shape
            ):
                raise ValueError(
                    f"Observation {name!r} must end in shape {trailing_shape}, "
                    f"got {shape}"
                )
            current_leading = shape[: len(shape) - len(trailing_shape)]
            if leading_shape is None:
                leading_shape = current_leading
            elif current_leading != leading_shape:
                raise ValueError(
                    "All observation modalities must share their leading dimensions"
                )


@dataclass(frozen=True)
class RolloutBatch:
    """One fresh rollout batch, time-major unless produced by :meth:`flatten`."""

    observations: TensorDict
    next_observations: TensorDict
    actions: torch.Tensor
    pre_tanh_actions: torch.Tensor
    candidate_noise: torch.Tensor
    old_log_prob: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    task_ids: torch.Tensor
    embodiment_ids: torch.Tensor
    action_masks: torch.Tensor

    def __post_init__(self) -> None:
        if self.actions.ndim not in (2, 3):
            raise ValueError("actions must have shape [N, A] or [T, B, A]")
        leading = self.actions.shape[:-1]
        if self.pre_tanh_actions.shape != self.actions.shape:
            raise ValueError("pre_tanh_actions and actions must have identical shapes")
        if self.action_masks.shape != self.actions.shape:
            raise ValueError("action_masks and actions must have identical shapes")
        for name, tensor in (
            ("old_log_prob", self.old_log_prob),
            ("rewards", self.rewards),
            ("terminated", self.terminated),
        ):
            if tensor.shape != (*leading, 1):
                raise ValueError(
                    f"{name} must have shape {(*leading, 1)}, got {tuple(tensor.shape)}"
                )
        for name, tensor in (
            ("task_ids", self.task_ids),
            ("embodiment_ids", self.embodiment_ids),
        ):
            if tensor.shape != leading:
                raise ValueError(
                    f"{name} must have shape {tuple(leading)}, "
                    f"got {tuple(tensor.shape)}"
                )
        expected_noise_ndim = self.actions.ndim + 2
        if self.candidate_noise.ndim != expected_noise_ndim:
            raise ValueError(
                "candidate_noise must append [samples, planning_horizon, action_dim] "
                "to the rollout leading dimensions"
            )
        if self.candidate_noise.shape[: len(leading)] != leading:
            raise ValueError("candidate_noise leading dimensions must match actions")
        if self.candidate_noise.shape[-1] != self.actions.shape[-1]:
            raise ValueError(
                "candidate_noise and actions must share the action dimension"
            )
        if set(self.observations) != set(self.next_observations):
            raise ValueError(
                "observations and next_observations must have identical keys"
            )
        for name in self.observations:
            observation = self.observations[name]
            next_observation = self.next_observations[name]
            if observation.shape != next_observation.shape:
                raise ValueError(
                    f"Current and next observation shapes differ for {name!r}"
                )
            if observation.shape[: len(leading)] != leading:
                raise ValueError(
                    f"Observation {name!r} leading dimensions must match actions"
                )

    @property
    def is_time_major(self) -> bool:
        return self.actions.ndim == 3

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.actions.shape[1]) if self.is_time_major else 1

    @property
    def num_transitions(self) -> int:
        return self.length * self.num_envs

    @property
    def device(self) -> torch.device:
        return self.actions.device

    def __len__(self) -> int:
        return self.num_transitions

    def to(
        self, device: str | torch.device, *, non_blocking: bool = False
    ) -> "RolloutBatch":
        def move_dict(values: TensorDict) -> TensorDict:
            return {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in values.items()
            }

        return RolloutBatch(
            observations=move_dict(self.observations),
            next_observations=move_dict(self.next_observations),
            actions=self.actions.to(device, non_blocking=non_blocking),
            pre_tanh_actions=self.pre_tanh_actions.to(
                device, non_blocking=non_blocking
            ),
            candidate_noise=self.candidate_noise.to(device, non_blocking=non_blocking),
            old_log_prob=self.old_log_prob.to(device, non_blocking=non_blocking),
            rewards=self.rewards.to(device, non_blocking=non_blocking),
            terminated=self.terminated.to(device, non_blocking=non_blocking),
            task_ids=self.task_ids.to(device, non_blocking=non_blocking),
            embodiment_ids=self.embodiment_ids.to(device, non_blocking=non_blocking),
            action_masks=self.action_masks.to(device, non_blocking=non_blocking),
        )

    def flatten(self) -> "RolloutBatch":
        if not self.is_time_major:
            return self

        def flatten_dict(values: TensorDict) -> TensorDict:
            return {name: value.flatten(0, 1) for name, value in values.items()}

        return RolloutBatch(
            observations=flatten_dict(self.observations),
            next_observations=flatten_dict(self.next_observations),
            actions=self.actions.flatten(0, 1),
            pre_tanh_actions=self.pre_tanh_actions.flatten(0, 1),
            candidate_noise=self.candidate_noise.flatten(0, 1),
            old_log_prob=self.old_log_prob.flatten(0, 1),
            rewards=self.rewards.flatten(0, 1),
            terminated=self.terminated.flatten(0, 1),
            task_ids=self.task_ids.flatten(0, 1),
            embodiment_ids=self.embodiment_ids.flatten(0, 1),
            action_masks=self.action_masks.flatten(0, 1),
        )

    def select(self, indices: torch.Tensor | np.ndarray | list[int]) -> "RolloutBatch":
        flat = self.flatten()
        index = torch.as_tensor(indices, dtype=torch.long, device=flat.device).reshape(
            -1
        )

        def select_dict(values: TensorDict) -> TensorDict:
            return {
                name: value.index_select(0, index) for name, value in values.items()
            }

        return RolloutBatch(
            observations=select_dict(flat.observations),
            next_observations=select_dict(flat.next_observations),
            actions=flat.actions.index_select(0, index),
            pre_tanh_actions=flat.pre_tanh_actions.index_select(0, index),
            candidate_noise=flat.candidate_noise.index_select(0, index),
            old_log_prob=flat.old_log_prob.index_select(0, index),
            rewards=flat.rewards.index_select(0, index),
            terminated=flat.terminated.index_select(0, index),
            task_ids=flat.task_ids.index_select(0, index),
            embodiment_ids=flat.embodiment_ids.index_select(0, index),
            action_masks=flat.action_masks.index_select(0, index),
        )

    minibatch = select
    __getitem__ = select
    flattened = flatten

    def minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Iterator["RolloutBatch"]:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.num_transitions % minibatch_size:
            raise ValueError("minibatch_size must divide the rollout transition count")
        if shuffle:
            indices = torch.randperm(
                self.num_transitions, generator=generator, device="cpu"
            )
        else:
            indices = torch.arange(self.num_transitions, device="cpu")
        for start in range(0, self.num_transitions, minibatch_size):
            yield self.select(indices[start : start + minibatch_size])


def _numpy_copy(value, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


class RolloutBuffer:
    """Append vector-environment steps and finalize once into a CPU rollout."""

    def __init__(
        self,
        observation_spec: ObservationSpec | None = None,
        num_envs: int | None = None,
    ) -> None:
        if num_envs is not None and num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.observation_spec = observation_spec
        self._num_envs = num_envs
        self._steps: list[dict[str, object]] = []
        self._finalized = False
        self._shapes: dict[str, object] | None = None

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def num_envs(self) -> int | None:
        return self._num_envs

    def __len__(self) -> int:
        return len(self._steps)

    def append(
        self,
        observations: Mapping[str, np.ndarray | torch.Tensor],
        next_observations: Mapping[str, np.ndarray | torch.Tensor],
        actions: np.ndarray | torch.Tensor,
        pre_tanh_actions: np.ndarray | torch.Tensor,
        candidate_noise: np.ndarray | torch.Tensor,
        old_log_prob: np.ndarray | torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        terminated: np.ndarray | torch.Tensor,
        task_ids: np.ndarray | torch.Tensor,
        embodiment_ids: np.ndarray | torch.Tensor,
        action_masks: np.ndarray | torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot append after RolloutBuffer.finalize()")
        if set(observations) != set(next_observations):
            raise ValueError(
                "observations and next_observations must have identical keys"
            )
        if self.observation_spec is not None:
            self.observation_spec.validate(observations)
            self.observation_spec.validate(next_observations)

        obs = {
            name: _numpy_copy(value, np.float32) for name, value in observations.items()
        }
        next_obs = {
            name: _numpy_copy(value, np.float32)
            for name, value in next_observations.items()
        }
        action = _numpy_copy(actions, np.float32)
        pre_tanh = _numpy_copy(pre_tanh_actions, np.float32)
        noise = _numpy_copy(candidate_noise, np.float32)
        mask = _numpy_copy(action_masks, np.float32)
        if action.ndim != 2:
            raise ValueError("Each action step must have shape [num_envs, action_dim]")
        num_envs, action_dim = action.shape
        if self._num_envs is None:
            self._num_envs = num_envs
        elif num_envs != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} environments, got {num_envs}")
        if pre_tanh.shape != action.shape or mask.shape != action.shape:
            raise ValueError(
                "actions, pre_tanh_actions, and action_masks must have identical shapes"
            )
        if (
            noise.ndim != 4
            or noise.shape[0] != num_envs
            or noise.shape[-1] != action_dim
        ):
            raise ValueError(
                "Each candidate_noise step must have shape "
                "[num_envs, samples, planning_horizon, action_dim]"
            )
        for name, value in (*obs.items(), *next_obs.items()):
            if value.ndim == 0 or value.shape[0] != num_envs:
                raise ValueError(
                    f"Observation {name!r} must begin with num_envs={num_envs}"
                )

        def column(value, name: str) -> np.ndarray:
            result = _numpy_copy(value, np.float32)
            if result.shape == (num_envs,):
                result = result[:, None]
            if result.shape != (num_envs, 1):
                raise ValueError(f"{name} must have shape [num_envs] or [num_envs, 1]")
            return result

        def ids(value, name: str) -> np.ndarray:
            result = _numpy_copy(value, np.int64)
            if result.shape == (num_envs, 1):
                result = result[:, 0]
            if result.shape != (num_envs,):
                raise ValueError(f"{name} must have shape [num_envs]")
            return result

        old_log_prob_array = column(old_log_prob, "old_log_prob")
        rewards_array = column(rewards, "rewards")
        terminated_array = column(terminated, "terminated")
        task_id_array = ids(task_ids, "task_ids")
        embodiment_id_array = ids(embodiment_ids, "embodiment_ids")
        floating_values = {
            **{f"observation.{name}": value for name, value in obs.items()},
            **{f"next_observation.{name}": value for name, value in next_obs.items()},
            "actions": action,
            "pre_tanh_actions": pre_tanh,
            "candidate_noise": noise,
            "old_log_prob": old_log_prob_array,
            "rewards": rewards_array,
            "terminated": terminated_array,
            "action_masks": mask,
        }
        for name, value in floating_values.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("action_masks must contain only zeros and ones")
        if np.any(mask.sum(axis=-1) < 1.0):
            raise ValueError("Every rollout worker must have at least one valid action")
        if np.any(np.abs(action) > 1.0 + 1e-6):
            raise ValueError("Normalized rollout actions must lie in [-1, 1]")
        if not np.allclose(action * (1.0 - mask), 0.0, atol=1e-7):
            raise ValueError("Invalid action coordinates must be zero")
        if not np.allclose(pre_tanh * (1.0 - mask), 0.0, atol=1e-7):
            raise ValueError("Invalid pre-tanh action coordinates must be zero")
        if np.any((terminated_array != 0.0) & (terminated_array != 1.0)):
            raise ValueError("terminated must contain only zeros and ones")
        if np.any(task_id_array < 0) or np.any(embodiment_id_array < 0):
            raise ValueError("Task and embodiment IDs must be non-negative")

        step: dict[str, object] = {
            "observations": obs,
            "next_observations": next_obs,
            "actions": action,
            "pre_tanh_actions": pre_tanh,
            "candidate_noise": noise,
            "old_log_prob": old_log_prob_array,
            "rewards": rewards_array,
            "terminated": terminated_array,
            "task_ids": task_id_array,
            "embodiment_ids": embodiment_id_array,
            "action_masks": mask,
        }
        signature = {
            "observations": {name: value.shape for name, value in obs.items()},
            "next_observations": {
                name: value.shape for name, value in next_obs.items()
            },
            **{
                name: value.shape
                for name, value in step.items()
                if isinstance(value, np.ndarray)
            },
        }
        if self._shapes is None:
            self._shapes = signature
        elif signature != self._shapes:
            raise ValueError("All rollout steps must have identical field shapes")
        self._steps.append(step)

    add = append

    def finalize(self) -> RolloutBatch:
        if self._finalized:
            raise RuntimeError("RolloutBuffer.finalize() may only be called once")
        if not self._steps:
            raise RuntimeError("Cannot finalize an empty rollout")
        self._finalized = True

        def stack_dict(field: str) -> TensorDict:
            first = self._steps[0][field]
            assert isinstance(first, dict)
            return {
                name: torch.from_numpy(
                    np.stack(
                        [step[field][name] for step in self._steps],  # type: ignore[index]
                        axis=0,
                    )
                ).to(dtype=torch.float32)
                for name in first
            }

        def stack(field: str, dtype: torch.dtype) -> torch.Tensor:
            values = [step[field] for step in self._steps]
            return torch.from_numpy(np.stack(values, axis=0)).to(dtype=dtype)

        return RolloutBatch(
            observations=stack_dict("observations"),
            next_observations=stack_dict("next_observations"),
            actions=stack("actions", torch.float32),
            pre_tanh_actions=stack("pre_tanh_actions", torch.float32),
            candidate_noise=stack("candidate_noise", torch.float32),
            old_log_prob=stack("old_log_prob", torch.float32),
            rewards=stack("rewards", torch.float32),
            terminated=stack("terminated", torch.float32),
            task_ids=stack("task_ids", torch.long),
            embodiment_ids=stack("embodiment_ids", torch.long),
            action_masks=stack("action_masks", torch.float32),
        )
