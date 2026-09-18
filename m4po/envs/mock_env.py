from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from m4po.common.action_tokenizer import ActionTokenizer
from m4po.common.buffer import ObservationSpec

Observation = Dict[str, np.ndarray]


def _names(value: str | Sequence[str] | None, count: int, prefix: str) -> list[str]:
    if count <= 0:
        raise ValueError(f"num_{prefix}s must be positive")
    if value is None:
        return [f"{prefix}_{index}" for index in range(count)]
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in value]
    if len(names) != count:
        raise ValueError(f"Expected {count} {prefix} names, got {len(names)}")
    if any(not name for name in names):
        raise ValueError(f"{prefix.capitalize()} names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{prefix.capitalize()} names must be unique")
    return names


def _default_action_tokenizer(embodiment_names: Sequence[str]) -> ActionTokenizer:
    num_embodiments = len(embodiment_names)
    action_dims = [2 + embodiment_id % 3 for embodiment_id in range(num_embodiments)]
    shared_action_dim = max(action_dims)
    action_lows: list[np.ndarray] = []
    action_highs: list[np.ndarray] = []
    action_indices: list[np.ndarray] = []
    shared_indices = np.arange(shared_action_dim, dtype=np.int64)
    for embodiment_id, action_dim in enumerate(action_dims):
        coordinate = np.arange(action_dim, dtype=np.float32)
        action_lows.append(-(1.0 + 0.25 * embodiment_id + 0.10 * coordinate))
        action_highs.append(0.8 + 0.20 * embodiment_id + 0.05 * coordinate)
        permutation = np.roll(shared_indices, embodiment_id + 1)
        action_indices.append(permutation[:action_dim].copy())
    return ActionTokenizer(
        action_lows,
        action_highs,
        action_indices,
        embodiment_names=embodiment_names,
        shared_action_dim=shared_action_dim,
    )


class MockContinuousEnv:
    """One deterministic-shape mock task/embodiment environment.

    The dynamics are intentionally small and stable. They exist to exercise the
    multimodal observation boundary, physical action decoding, control repeats,
    termination, and success bookkeeping without simulator dependencies.
    """

    def __init__(
        self,
        *,
        seed: int,
        task_id: int,
        embodiment_id: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        proprio_dim: int,
        state_dim: int,
        max_low_level_steps: int,
        image_size: int,
        image_channels: int,
        use_images: bool,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.task_id = int(task_id)
        self.embodiment_id = int(embodiment_id)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1).copy()
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1).copy()
        self.action_dim = int(self.action_low.size)
        self.proprio_dim = int(proprio_dim)
        self.state_dim = int(state_dim)
        self.max_low_level_steps = int(max_low_level_steps)
        self.image_size = int(image_size)
        self.image_channels = int(image_channels) if use_images else 0
        self.use_images = bool(use_images)
        if self.action_dim <= 0 or self.proprio_dim < 0 or self.state_dim < 0:
            raise ValueError(
                "Mock action dimensions must be positive and observation dimensions non-negative"
            )
        if not self.use_images and self.proprio_dim == 0 and self.state_dim == 0:
            raise ValueError("At least one mock observation modality must be enabled")
        if self.max_low_level_steps <= 0:
            raise ValueError("max_low_level_steps must be positive")
        if self.use_images and (self.image_size <= 0 or self.image_channels <= 0):
            raise ValueError(
                "image_size and image_channels must be positive when use_images=True"
            )

        self._action_to_proprio = self.rng.normal(
            0.0, 0.12, size=(self.proprio_dim, self.action_dim)
        ).astype(np.float32)
        self._action_to_state = self.rng.normal(
            0.0, 0.08, size=(self.state_dim, self.action_dim)
        ).astype(np.float32)
        phase = 0.7 * (self.task_id + 1) + 0.2 * self.embodiment_id
        coordinate = np.arange(self.state_dim, dtype=np.float32)
        self._target = (0.35 * np.sin(phase + coordinate * 0.5)).astype(np.float32)
        if self.use_images:
            axis = np.linspace(0.0, 1.0, self.image_size, dtype=np.float32)
            self._grid_x, self._grid_y = np.meshgrid(axis, axis, indexing="xy")
        else:
            self._grid_x = np.empty((0, 0), dtype=np.float32)
            self._grid_y = np.empty((0, 0), dtype=np.float32)
        self.proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.low_level_step = 0

    def _image(self) -> np.ndarray:
        if not self.use_images:
            return np.empty((0, 0, 0), dtype=np.float32)
        state_0 = float(self.state[0]) if self.state_dim else 0.0
        state_1 = float(self.state[1]) if self.state_dim > 1 else state_0
        proprio_0 = float(self.proprio[0]) if self.proprio_dim else 0.0
        patterns = (
            0.45 + 0.25 * state_0 + 0.20 * self._grid_x,
            0.45 + 0.25 * state_1 + 0.20 * self._grid_y,
            0.35 + 0.20 * proprio_0 + 0.15 * (self._grid_x + self._grid_y),
            0.55 + 0.15 * state_0 - 0.10 * self._grid_x + 0.10 * self._grid_y,
        )
        image = np.empty(
            (self.image_channels, self.image_size, self.image_size), dtype=np.float32
        )
        for channel in range(self.image_channels):
            pattern = patterns[channel % len(patterns)] + 0.025 * (channel // len(patterns))
            image[channel] = np.clip(pattern, 0.0, 1.0)
        return image

    def _observation(self) -> Observation:
        return {
            "image": self._image(),
            "proprio": np.clip(self.proprio, -1.0, 1.0).astype(np.float32, copy=True),
            "state": np.clip(self.state, -1.0, 1.0).astype(np.float32, copy=True),
        }

    def reset(self) -> Observation:
        self.low_level_step = 0
        self.proprio = self.rng.normal(0.0, 0.30, size=self.proprio_dim).astype(np.float32)
        self.state = self.rng.normal(0.0, 0.35, size=self.state_dim).astype(np.float32)
        return self._observation()

    def step(self, physical_action: np.ndarray) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        action = np.asarray(physical_action, dtype=np.float32).reshape(-1)
        if action.shape != self.action_low.shape:
            raise ValueError(
                f"Expected physical action shape {self.action_low.shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Physical actions must be finite")
        action = np.clip(action, self.action_low, self.action_high)
        normalized_action = 2.0 * (action - self.action_low) / (
            self.action_high - self.action_low
        ) - 1.0
        proprio_noise = self.rng.normal(0.0, 0.005, size=self.proprio_dim).astype(np.float32)
        state_noise = self.rng.normal(0.0, 0.004, size=self.state_dim).astype(np.float32)
        self.proprio = (
            0.90 * self.proprio
            + self._action_to_proprio @ normalized_action
            + proprio_noise
        ).astype(np.float32)
        coupled = np.zeros(self.state_dim, dtype=np.float32)
        coupled[: min(self.state_dim, self.proprio_dim)] = self.proprio[
            : min(self.state_dim, self.proprio_dim)
        ]
        self.state = (
            0.91 * self.state
            + 0.035 * self._target
            + 0.025 * coupled
            + self._action_to_state @ normalized_action
            + state_noise
        ).astype(np.float32)
        self.low_level_step += 1

        if self.state_dim:
            state_error = float(np.square(self.state - self._target).mean())
        elif self.proprio_dim:
            state_error = float(np.square(self.proprio).mean())
        else:
            state_error = 0.0
        action_cost = float(np.square(normalized_action).mean())
        reward = float(np.exp(-2.0 * state_error) - 0.02 * action_cost)
        success = bool(state_error < 0.0125)
        timeout = self.low_level_step >= self.max_low_level_steps
        done = bool(success or timeout)
        return self._observation(), reward, done, {
            "state_error": state_error,
            "action_cost": action_cost,
            "success": success,
            "timeout": bool(timeout),
        }

    def close(self) -> None:
        pass


class MockVectorEnv:
    """Dependency-free multi-task, multi-embodiment vector environment.

    Actions use the shared normalized token space. Every worker decodes those
    tokens to embodiment-specific physical controls and repeats them at its
    configured low-level control rate. Observations are padded dictionaries;
    vector steps auto-reset completed workers and preserve their true final
    observation in ``info["terminal_observation"]``.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        seed: int = 0,
        max_episode_steps: int = 32,
        image_size: int = 32,
        image_channels: int = 4,
        use_images: bool = True,
        proprio_dim: int | None = None,
        state_dim: int | None = None,
        num_tasks: int = 2,
        num_embodiments: int = 2,
        task_context_dim: int = 16,
        discount: float = 0.99,
        low_level_discount: float | Sequence[float] | None = 1.0,
        tasks: str | Sequence[str] | None = None,
        embodiments: str | Sequence[str] | None = None,
        control_repeats: Sequence[int] | None = None,
        action_tokenizer: ActionTokenizer | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_tasks = int(num_tasks)
        self.num_embodiments = int(num_embodiments)
        self.max_episode_steps = int(max_episode_steps)
        self.image_size = int(image_size)
        self.image_channels = int(image_channels) if use_images else 0
        self.use_images = bool(use_images)
        self.discount = float(discount)
        self._base_seed = int(seed)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.num_tasks <= 0 or self.num_embodiments <= 0:
            raise ValueError("num_tasks and num_embodiments must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if task_context_dim < 0:
            raise ValueError("task_context_dim must be non-negative")

        self.task_names = _names(tasks, self.num_tasks, "task")
        configured_embodiment_names = _names(
            embodiments, self.num_embodiments, "embodiment"
        )
        if action_tokenizer is None:
            action_tokenizer = _default_action_tokenizer(configured_embodiment_names)
        if action_tokenizer.num_embodiments != self.num_embodiments:
            raise ValueError(
                "Action tokenizer embodiment count does not match num_embodiments: "
                f"{action_tokenizer.num_embodiments} != {self.num_embodiments}"
            )
        if (
            embodiments is not None
            and list(action_tokenizer.embodiment_names) != configured_embodiment_names
        ):
            raise ValueError("Action tokenizer names do not match configured embodiment names")
        self.action_tokenizer = action_tokenizer
        self.embodiment_names = list(action_tokenizer.embodiment_names)
        self.action_dim = int(action_tokenizer.shared_action_dim)
        self.action_low = -np.ones(self.action_dim, dtype=np.float32)
        self.action_high = np.ones(self.action_dim, dtype=np.float32)
        self.embodiment_action_dims = np.asarray(
            action_tokenizer.action_dims, dtype=np.int64
        )
        self.embodiment_action_indices = tuple(
            indices.copy() for indices in action_tokenizer.action_indices
        )
        self.physical_action_lows = tuple(
            bounds.copy() for bounds in action_tokenizer.action_lows
        )
        self.physical_action_highs = tuple(
            bounds.copy() for bounds in action_tokenizer.action_highs
        )

        if control_repeats is None:
            repeats = np.asarray(
                [1 + embodiment_id % 3 for embodiment_id in range(self.num_embodiments)],
                dtype=np.int64,
            )
        elif isinstance(control_repeats, str):
            try:
                repeats = np.asarray(
                    [int(item.strip()) for item in control_repeats.split(",") if item.strip()],
                    dtype=np.int64,
                )
            except ValueError as exc:
                raise ValueError("control_repeats must be comma-separated integers") from exc
        else:
            raw_repeats = np.asarray(control_repeats)
            if not np.issubdtype(raw_repeats.dtype, np.integer):
                raise ValueError("control_repeats must contain integers")
            repeats = raw_repeats.astype(np.int64, copy=True).reshape(-1)
        if repeats.shape != (self.num_embodiments,) or np.any(repeats <= 0):
            raise ValueError("control_repeats must contain one positive value per embodiment")
        self.control_repeats = repeats

        if low_level_discount is None:
            self.low_level_discounts = np.power(
                self.discount, 1.0 / self.control_repeats.astype(np.float64)
            ).astype(np.float32)
        elif np.isscalar(low_level_discount):
            self.low_level_discounts = np.full(
                self.num_embodiments, float(low_level_discount), dtype=np.float32
            )
        else:
            self.low_level_discounts = np.asarray(
                low_level_discount, dtype=np.float32
            ).reshape(-1)
        if self.low_level_discounts.shape != (self.num_embodiments,):
            raise ValueError("low_level_discount must be scalar or have one value per embodiment")
        if np.any(self.low_level_discounts < 0.0) or np.any(self.low_level_discounts > 1.0):
            raise ValueError("low-level discounts must lie in [0, 1]")
        aligned = np.isclose(self.low_level_discounts, 1.0, atol=1e-7) | np.isclose(
            np.power(self.low_level_discounts, self.control_repeats),
            self.discount,
            rtol=1e-5,
            atol=1e-7,
        )
        if not bool(np.all(aligned)):
            raise ValueError(
                "Each low-level discount must be 1 or satisfy "
                "low_level_discount ** control_repeat == discount"
            )

        self.task_contexts = self._make_task_contexts(self.num_tasks, int(task_context_dim))

        generated_proprio_dims = np.asarray(
            [
                self.action_tokenizer.action_dims[embodiment_id]
                + 2
                + embodiment_id % 2
                for embodiment_id in range(self.num_embodiments)
            ],
            dtype=np.int64,
        )
        if proprio_dim is None:
            self.proprio_dim = int(generated_proprio_dims.max())
            self.proprio_dims = generated_proprio_dims
        else:
            self.proprio_dim = int(proprio_dim)
            if self.proprio_dim < 0:
                raise ValueError("Mock proprio_dim must be non-negative")
            self.proprio_dims = np.minimum(generated_proprio_dims, self.proprio_dim)
        generated_state_dim = int(
            max(
                5 + (task + embodiment) % 3
                for task in range(self.num_tasks)
                for embodiment in range(self.num_embodiments)
            )
        )
        if state_dim is None:
            self.state_dim = generated_state_dim
        else:
            self.state_dim = int(state_dim)
            if self.state_dim < 0:
                raise ValueError("Mock state_dim must be non-negative")
        image_shape = (
            (self.image_channels, self.image_size, self.image_size)
            if self.use_images
            else (0, 0, 0)
        )
        self.image_shape = image_shape
        self.observation_spec = ObservationSpec(
            image_shape=image_shape,
            proprio_dim=self.proprio_dim,
            state_dim=self.state_dim,
        )
        self.observation_shapes = self.observation_spec.shapes

        self._task_embodiment_pairs = np.asarray(
            [
                (task_id, embodiment_id)
                for task_id in range(self.num_tasks)
                for embodiment_id in range(self.num_embodiments)
            ],
            dtype=np.int64,
        )
        self._assignment_rng = np.random.default_rng(self._base_seed + 104_729)
        self._assignment_deck = np.empty((0, 2), dtype=np.int64)
        self._assignment_cursor = 0
        self._rollout_generation = 0
        self._rollout_started = False
        self.task_ids = np.empty(self.num_envs, dtype=np.int64)
        self.embodiment_ids = np.empty(self.num_envs, dtype=np.int64)
        self.action_masks = np.empty((self.num_envs, self.action_dim), dtype=np.float32)
        self.worker_control_repeats = np.empty(self.num_envs, dtype=np.int64)
        self.worker_state_dims = np.empty(self.num_envs, dtype=np.int64)
        self.envs: List[MockContinuousEnv] = []
        task_ids, embodiment_ids = self._worker_assignments()
        self._rebuild_workers(task_ids, embodiment_ids)

    def _refill_assignment_deck(self) -> None:
        order = self._assignment_rng.permutation(len(self._task_embodiment_pairs))
        self._assignment_deck = self._task_embodiment_pairs[order].copy()
        self._assignment_cursor = 0

    def _worker_assignments(self) -> tuple[np.ndarray, np.ndarray]:
        assignments: list[np.ndarray] = []
        while len(assignments) < self.num_envs:
            if self._assignment_cursor >= len(self._assignment_deck):
                self._refill_assignment_deck()
            remaining = self.num_envs - len(assignments)
            available = len(self._assignment_deck) - self._assignment_cursor
            count = min(remaining, available)
            assignments.extend(
                self._assignment_deck[
                    self._assignment_cursor : self._assignment_cursor + count
                ]
            )
            self._assignment_cursor += count
        array = np.asarray(assignments, dtype=np.int64)
        return array[:, 0].copy(), array[:, 1].copy()

    def _worker_seed(self, worker_id: int, task_id: int, embodiment_id: int) -> int:
        seed_sequence = np.random.SeedSequence(
            [
                self._base_seed,
                self._rollout_generation,
                int(worker_id),
                int(task_id),
                int(embodiment_id),
            ]
        )
        return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

    def _rebuild_workers(
        self,
        task_ids: np.ndarray,
        embodiment_ids: np.ndarray,
    ) -> None:
        task_ids = np.asarray(task_ids, dtype=np.int64).reshape(-1)
        embodiment_ids = np.asarray(embodiment_ids, dtype=np.int64).reshape(-1)
        if task_ids.shape != (self.num_envs,) or embodiment_ids.shape != (self.num_envs,):
            raise ValueError("Worker assignments must contain one task and embodiment per worker")
        if np.any(task_ids < 0) or np.any(task_ids >= self.num_tasks):
            raise ValueError("Worker task IDs are out of range")
        if np.any(embodiment_ids < 0) or np.any(embodiment_ids >= self.num_embodiments):
            raise ValueError("Worker embodiment IDs are out of range")

        for env in self.envs:
            env.close()
        self.task_ids = task_ids.copy()
        self.embodiment_ids = embodiment_ids.copy()
        self.action_masks = self.action_tokenizer.mask_for(self.embodiment_ids)
        self.worker_control_repeats = self.control_repeats[self.embodiment_ids].copy()
        generated_worker_state_dims = np.asarray(
            [
                5 + (int(task_id) + int(embodiment_id)) % 3
                for task_id, embodiment_id in zip(
                    self.task_ids, self.embodiment_ids, strict=True
                )
            ],
            dtype=np.int64,
        )
        self.worker_state_dims = np.minimum(generated_worker_state_dims, self.state_dim)

        self.envs = []
        for worker_id, (task_id, embodiment_id) in enumerate(
            zip(self.task_ids, self.embodiment_ids, strict=True)
        ):
            embodiment_id = int(embodiment_id)
            control_repeat = int(self.control_repeats[embodiment_id])
            self.envs.append(
                MockContinuousEnv(
                    seed=self._worker_seed(
                        worker_id,
                        int(task_id),
                        embodiment_id,
                    ),
                    task_id=int(task_id),
                    embodiment_id=embodiment_id,
                    action_low=self.action_tokenizer.action_lows[embodiment_id],
                    action_high=self.action_tokenizer.action_highs[embodiment_id],
                    proprio_dim=int(self.proprio_dims[embodiment_id]),
                    state_dim=int(self.worker_state_dims[worker_id]),
                    max_low_level_steps=self.max_episode_steps * control_repeat,
                    image_size=self.image_size,
                    image_channels=self.image_channels,
                    use_images=self.use_images,
                )
            )
        self._has_reset = False

    @staticmethod
    def _make_task_contexts(num_tasks: int, context_dim: int) -> np.ndarray:
        if context_dim == 0:
            return np.empty((num_tasks, 0), dtype=np.float32)
        task = np.arange(1, num_tasks + 1, dtype=np.float32)[:, None]
        coordinate = np.arange(1, context_dim + 1, dtype=np.float32)[None, :]
        contexts = np.sin(0.37 * task * coordinate) + np.cos(0.19 * task * coordinate)
        norm = np.linalg.norm(contexts, axis=1, keepdims=True).clip(min=1e-6)
        return (contexts / norm).astype(np.float32)

    def _pad_observation(self, observation: Observation, worker_id: int) -> Observation:
        embodiment_id = int(self.embodiment_ids[worker_id])
        expected_proprio_dim = int(self.proprio_dims[embodiment_id])
        expected_state_dim = int(self.worker_state_dims[worker_id])
        if observation["proprio"].shape != (expected_proprio_dim,):
            raise ValueError(
                f"Worker {worker_id} produced proprio shape {observation['proprio'].shape}, "
                f"expected {(expected_proprio_dim,)}"
            )
        if observation["state"].shape != (expected_state_dim,):
            raise ValueError(
                f"Worker {worker_id} produced state shape {observation['state'].shape}, "
                f"expected {(expected_state_dim,)}"
            )
        if observation["image"].shape != self.image_shape:
            raise ValueError(
                f"Worker {worker_id} produced image shape {observation['image'].shape}, "
                f"expected {self.image_shape}"
            )
        proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        proprio_mask = np.zeros(self.proprio_dim, dtype=np.float32)
        proprio_dim = int(observation["proprio"].shape[0])
        proprio[:proprio_dim] = observation["proprio"]
        proprio_mask[:proprio_dim] = 1.0

        state = np.zeros(self.state_dim, dtype=np.float32)
        state_mask = np.zeros(self.state_dim, dtype=np.float32)
        state_dim = int(observation["state"].shape[0])
        state[:state_dim] = observation["state"]
        state_mask[:state_dim] = 1.0
        return {
            "image": observation["image"].astype(np.float32, copy=True),
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "state": state,
            "state_mask": state_mask,
        }

    @staticmethod
    def _stack_observations(observations: Sequence[Observation]) -> Observation:
        return {
            key: np.stack([observation[key] for observation in observations], axis=0).astype(
                np.float32, copy=False
            )
            for key in ("image", "proprio", "proprio_mask", "state", "state_mask")
        }

    def reset(self) -> Observation:
        observations = [
            self._pad_observation(env.reset(), worker_id)
            for worker_id, env in enumerate(self.envs)
        ]
        self._has_reset = True
        self._rollout_started = True
        return self._stack_observations(observations)

    def reset_rollout(self) -> Observation:
        """Start a fresh rollout with balanced randomized worker contexts.

        Assignments are drawn from shuffled decks containing every configured
        task--embodiment pair exactly once. Consequently, small vector batches
        continue through the current deck on later rollouts instead of silently
        starving pairs that did not fit in the first batch.
        """

        if self._rollout_started:
            task_ids, embodiment_ids = self._worker_assignments()
            self._rollout_generation += 1
        else:
            task_ids = self.task_ids
            embodiment_ids = self.embodiment_ids
        self._rebuild_workers(task_ids, embodiment_ids)
        return self.reset()

    def rollout_state_dict(self) -> Dict[str, Any]:
        """Return the sampler state needed to continue at a rollout boundary."""

        return {
            "schema_version": 1,
            "assignment_rng_state": deepcopy(self._assignment_rng.bit_generator.state),
            "assignment_deck": self._assignment_deck.copy(),
            "assignment_cursor": int(self._assignment_cursor),
            "rollout_generation": int(self._rollout_generation),
            "rollout_started": bool(self._rollout_started),
            "task_ids": self.task_ids.copy(),
            "embodiment_ids": self.embodiment_ids.copy(),
        }

    def load_rollout_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a state produced by :meth:`rollout_state_dict`."""

        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("Invalid mock rollout state")
        try:
            deck = np.asarray(state["assignment_deck"], dtype=np.int64)
            cursor = int(state["assignment_cursor"])
            generation = int(state["rollout_generation"])
            rollout_started = bool(state["rollout_started"])
            task_ids = np.asarray(state["task_ids"], dtype=np.int64)
            embodiment_ids = np.asarray(state["embodiment_ids"], dtype=np.int64)
            rng_state = deepcopy(state["assignment_rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid mock rollout state fields") from exc
        if deck.ndim != 2 or deck.shape[1:] != (2,):
            raise ValueError("Invalid assignment deck in mock rollout state")
        if cursor < 0 or cursor > len(deck) or generation < 0:
            raise ValueError("Invalid cursor or generation in mock rollout state")
        if deck.size and (
            np.any(deck[:, 0] < 0)
            or np.any(deck[:, 0] >= self.num_tasks)
            or np.any(deck[:, 1] < 0)
            or np.any(deck[:, 1] >= self.num_embodiments)
        ):
            raise ValueError("Assignment deck IDs are out of range")
        try:
            self._assignment_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid assignment RNG in mock rollout state") from exc
        self._assignment_deck = deck.copy()
        self._assignment_cursor = cursor
        self._rollout_generation = generation
        self._rollout_started = rollout_started
        self._rebuild_workers(task_ids, embodiment_ids)

    # Standard aliases let external vector adapters expose the same optional
    # checkpoint contract without depending on mock-specific method names.
    state_dict = rollout_state_dict
    load_state_dict = load_rollout_state_dict

    def step(self, actions: np.ndarray):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step()")
        shared_actions = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, self.action_dim)
        if shared_actions.shape != expected_shape:
            raise ValueError(
                f"Expected shared action shape {expected_shape}, got {shared_actions.shape}"
            )
        if not np.all(np.isfinite(shared_actions)):
            raise ValueError("Actions must be finite")
        shared_actions = np.clip(shared_actions, -1.0, 1.0) * self.action_masks

        next_observations: list[Observation] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[Dict[str, Any]] = []
        for worker_id, (env, shared_action) in enumerate(
            zip(self.envs, shared_actions, strict=True)
        ):
            embodiment_id = int(self.embodiment_ids[worker_id])
            physical_action = self.action_tokenizer.decode(shared_action, embodiment_id)
            local_normalized_action = self.action_tokenizer.detokenize(
                shared_action, embodiment_id
            )
            repeat = int(self.control_repeats[embodiment_id])
            low_level_discount = float(self.low_level_discounts[embodiment_id])
            low_level_rewards: list[float] = []
            done = False
            success = False
            low_info: Dict[str, Any] = {}
            observation: Observation | None = None
            for _ in range(repeat):
                observation, low_reward, done, low_info = env.step(physical_action)
                low_level_rewards.append(float(low_reward))
                success = success or bool(low_info.get("success", False))
                if done:
                    break
            assert observation is not None
            aggregated_reward = sum(
                (low_level_discount**index) * reward
                for index, reward in enumerate(low_level_rewards)
            )
            padded_observation = self._pad_observation(observation, worker_id)
            info: Dict[str, Any] = {
                **low_info,
                "task_id": int(self.task_ids[worker_id]),
                "task_name": self.task_names[int(self.task_ids[worker_id])],
                "embodiment_id": embodiment_id,
                "embodiment_name": self.embodiment_names[embodiment_id],
                "success": bool(success),
                "terminated": bool(success),
                "truncated": bool(low_info.get("timeout", False)),
                "control_repeat": repeat,
                "low_level_steps_executed": len(low_level_rewards),
                "low_level_rewards": low_level_rewards,
                "low_level_discount": low_level_discount,
                "shared_action": shared_action.copy(),
                "local_normalized_action": local_normalized_action.copy(),
                "physical_action": physical_action.copy(),
            }
            if done:
                info["terminal_observation"] = {
                    key: value.copy() for key, value in padded_observation.items()
                }
                padded_observation = self._pad_observation(env.reset(), worker_id)
            next_observations.append(padded_observation)
            rewards.append(float(aggregated_reward))
            dones.append(bool(done))
            infos.append(info)

        return (
            self._stack_observations(next_observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


MockMultiTaskVectorEnv = MockVectorEnv
MockMultiEmbodimentVectorEnv = MockVectorEnv


__all__ = [
    "MockContinuousEnv",
    "MockMultiEmbodimentVectorEnv",
    "MockMultiTaskVectorEnv",
    "MockVectorEnv",
]
