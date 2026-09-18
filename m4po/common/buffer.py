from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Episode:
    """A completed worker episode with T transitions and T+1 observations."""

    observations: TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    task_ids: torch.Tensor
    embodiment_ids: torch.Tensor
    action_masks: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class ReplayBatch:
    """Time-major replay sequences; ``valid`` excludes end-of-episode padding."""

    observations: TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    task_ids: torch.Tensor
    embodiment_ids: torch.Tensor
    action_masks: torch.Tensor
    valid: torch.Tensor

    def to(self, device: str | torch.device) -> "ReplayBatch":
        return ReplayBatch(
            observations={
                name: value.to(device) for name, value in self.observations.items()
            },
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "actions",
                    "rewards",
                    "terminated",
                    "task_ids",
                    "embodiment_ids",
                    "action_masks",
                    "valid",
                )
            },
        )


class EpisodeReplayBuffer:
    """FIFO episode replay, uniformly sampled over retained transition starts.

    Sequences never cross a reset. Short tails repeat the final observation and
    context, with zero actions/rewards and a zero validity mask. Storage owns a
    detached CPU copy, so environment buffers and autograd graphs are not retained.
    """

    def __init__(
        self,
        capacity: int,
        horizon: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        observation_spec: ObservationSpec | None = None,
    ) -> None:
        for name, value in (
            ("capacity", capacity),
            ("horizon", horizon),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.observation_spec = observation_spec
        self._episodes: list[Episode] = []
        self._size = 0
        self._signature: dict | None = None
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def can_sample(self) -> bool:
        return bool(self._episodes)

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {"generator": self._generator.get_state().clone()}

    def load_rng_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"generator"}:
            raise ValueError("Replay RNG state must contain a generator state")
        try:
            self._generator.set_state(state["generator"].cpu())
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise ValueError("Replay RNG state is invalid") from exc

    def state_dict(self) -> dict[str, object]:
        """Serialize completed episodes and sampling RNG, without GPU copies.

        As with module state dictionaries, tensors share the immutable stored
        episode data. Save synchronously before modifying the replay buffer.
        Unfinished simulator episodes are deliberately not part of replay.
        """

        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "observation_spec": (
                asdict(self.observation_spec)
                if self.observation_spec is not None
                else None
            ),
            "size": self._size,
            "episodes": [
                {
                    "observations": dict(episode.observations),
                    **{
                        name: getattr(episode, name)
                        for name in (
                            "actions",
                            "rewards",
                            "terminated",
                            "task_ids",
                            "embodiment_ids",
                            "action_masks",
                        )
                    },
                }
                for episode in self._episodes
            ],
            "rng_state": self.rng_state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate a full snapshot before atomically replacing live replay."""

        if (
            not isinstance(state, Mapping)
            or isinstance(state.get("schema_version"), bool)
            or state.get("schema_version") != 1
        ):
            raise ValueError("Unsupported replay checkpoint schema_version")
        for name in ("capacity", "horizon", "batch_size"):
            value = state.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != getattr(self, name)
            ):
                raise ValueError(
                    f"Replay checkpoint {name} does not match configured replay"
                )
        expected_spec = (
            asdict(self.observation_spec) if self.observation_spec is not None else None
        )
        if state.get("observation_spec") != expected_spec:
            raise ValueError("Replay checkpoint observation_spec does not match")
        size = state.get("size")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= self.capacity
        ):
            raise ValueError("Replay checkpoint size is invalid")
        episodes = state.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("Replay checkpoint episodes must be a list")
        staged = EpisodeReplayBuffer(
            self.capacity,
            self.horizon,
            self.batch_size,
            device=self.device,
            observation_spec=self.observation_spec,
        )
        staged.load_rng_state_dict(state.get("rng_state"))
        fields = {
            "observations",
            "actions",
            "rewards",
            "terminated",
            "task_ids",
            "embodiment_ids",
            "action_masks",
        }
        for record in episodes:
            if not isinstance(record, Mapping) or set(record) != fields:
                raise ValueError("Replay checkpoint episode fields are invalid")
            observations = record["observations"]
            if (
                not isinstance(observations, Mapping)
                or any(
                    not isinstance(value, torch.Tensor)
                    for value in observations.values()
                )
                or any(
                    not isinstance(record[name], torch.Tensor)
                    for name in fields - {"observations"}
                )
            ):
                raise ValueError("Replay checkpoint episodes must contain tensors")
            episode = Episode(**record)
            staged._validate_episode(episode)
            if len(staged) + episode.length > self.capacity:
                raise ValueError("Replay checkpoint episodes exceed capacity")
            staged.add(episode)
        if len(staged) != size:
            raise ValueError(
                "Replay checkpoint size disagrees with episode transitions"
            )
        self._episodes = staged._episodes
        self._size = staged._size
        self._signature = staged._signature
        self._generator = staged._generator

    def add(self, episode: Episode) -> None:
        self._validate_episode(episode)

        def copy(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            return value.detach().to(device="cpu", dtype=dtype).clone().contiguous()

        stored = Episode(
            observations={
                name: copy(value, torch.float32)
                for name, value in episode.observations.items()
            },
            actions=copy(episode.actions, torch.float32),
            rewards=copy(episode.rewards, torch.float32),
            terminated=copy(episode.terminated, torch.float32),
            task_ids=copy(episode.task_ids, torch.long),
            embodiment_ids=copy(episode.embodiment_ids, torch.long),
            action_masks=copy(episode.action_masks, torch.float32),
        )
        self._episodes.append(stored)
        self._size += stored.length
        while self._size > self.capacity:
            self._size -= self._episodes.pop(0).length

    def _validate_episode(self, episode: Episode) -> None:
        if (
            episode.actions.ndim != 2
            or episode.actions.shape[0] <= 0
            or episode.actions.shape[1] <= 0
        ):
            raise ValueError("Episode actions must have nonempty shape [T, A]")
        length = episode.length
        if length > self.capacity:
            raise ValueError("An episode cannot exceed replay capacity")
        if not episode.observations:
            raise ValueError("Episode observations cannot be empty")
        if self.observation_spec is not None:
            self.observation_spec.validate(episode.observations)
        for name, value in episode.observations.items():
            if value.ndim < 2 or value.shape[0] != length + 1:
                raise ValueError(f"Episode observation {name!r} must have T+1 entries")
        for name in ("rewards", "terminated"):
            if getattr(episode, name).shape != (length, 1):
                raise ValueError(f"Episode {name} must have shape [T, 1]")
        for name in ("task_ids", "embodiment_ids"):
            value = getattr(episode, name)
            if value.shape != (length + 1,):
                raise ValueError(f"Episode {name} must have shape [T+1]")
            if value.dtype not in (torch.int32, torch.int64) or torch.any(value < 0):
                raise ValueError(f"Episode {name} must be non-negative integers")
            if not torch.all(value == value[0]):
                raise ValueError(
                    f"Episode {name} cannot change across a context/reset boundary"
                )
        if episode.action_masks.shape != episode.actions.shape:
            raise ValueError("Episode action masks must match actions")
        values = {
            **episode.observations,
            "actions": episode.actions,
            "rewards": episode.rewards,
            "terminated": episode.terminated,
            "action_masks": episode.action_masks,
        }
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Episode tensors must contain finite values")
        masks = episode.action_masks
        if torch.any((masks != 0) & (masks != 1)) or torch.any(masks.sum(-1) < 1):
            raise ValueError(
                "Episode action masks must be binary with at least one valid action"
            )
        if not torch.all(masks == masks[0]):
            raise ValueError(
                "Episode action masks cannot change across a context/reset boundary"
            )
        if torch.any(episode.actions.abs() > 1 + 1e-6):
            raise ValueError("Normalized episode actions must lie in [-1, 1]")
        if torch.any((episode.actions * (1 - masks)).abs() > 1e-7):
            raise ValueError("Invalid action coordinates must be zero")
        if torch.any((episode.terminated != 0) & (episode.terminated != 1)):
            raise ValueError("Episode terminated must contain only zeros and ones")
        if torch.any(episode.terminated[:-1] != 0):
            raise ValueError(
                "An episode cannot contain an internal terminal/reset boundary"
            )
        signature = {
            name: tuple(value.shape[1:]) for name, value in episode.observations.items()
        }
        signature["action_dim"] = int(episode.actions.shape[-1])
        if self._signature is not None and signature != self._signature:
            raise ValueError("Replay episodes must share observation and action shapes")
        self._signature = signature

    def sample(self) -> ReplayBatch:
        if not self.can_sample:
            raise RuntimeError("Replay buffer has no completed episode to sample")
        selections = torch.randint(
            self._size, (self.batch_size,), generator=self._generator
        ).numpy()
        ends = np.cumsum([episode.length for episode in self._episodes])
        indices = np.searchsorted(ends, selections, side="right")
        starts = selections - np.where(indices > 0, ends[np.maximum(indices - 1, 0)], 0)
        sequences: list[ReplayBatch] = []
        for episode_index, start in zip(indices.tolist(), starts.tolist(), strict=True):
            episode = self._episodes[episode_index]
            length = min(self.horizon, episode.length - start)
            end = start + length

            def padded(
                value: torch.Tensor,
                *,
                observations: bool = False,
                fill: float | None = None,
            ) -> torch.Tensor:
                result = value[start : end + int(observations)]
                padding = self.horizon - length
                if not padding:
                    return result
                tail = result[-1:].expand(padding, *result.shape[1:])
                if fill is not None:
                    tail = torch.full_like(tail, fill)
                return torch.cat((result, tail), dim=0)

            sequences.append(
                ReplayBatch(
                    observations={
                        name: padded(value, observations=True)
                        for name, value in episode.observations.items()
                    },
                    actions=padded(episode.actions, fill=0),
                    rewards=padded(episode.rewards, fill=0),
                    terminated=padded(episode.terminated, fill=1),
                    task_ids=padded(episode.task_ids, observations=True),
                    embodiment_ids=padded(episode.embodiment_ids, observations=True),
                    action_masks=padded(episode.action_masks),
                    valid=torch.cat(
                        (torch.ones(length, 1), torch.zeros(self.horizon - length, 1))
                    ),
                )
            )
        return ReplayBatch(
            observations={
                name: torch.stack(
                    [batch.observations[name] for batch in sequences], dim=1
                )
                for name in sequences[0].observations
            },
            **{
                name: torch.stack([getattr(batch, name) for batch in sequences], dim=1)
                for name in (
                    "actions",
                    "rewards",
                    "terminated",
                    "task_ids",
                    "embodiment_ids",
                    "action_masks",
                    "valid",
                )
            },
        ).to(self.device)


ReplayBuffer = EpisodeReplayBuffer
