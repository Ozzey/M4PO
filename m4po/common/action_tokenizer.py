from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ActionTokenizer:
    """Map embodiment-specific physical controls to shared normalized tokens.

    Each embodiment owns a vector of physical lower/upper bounds and a vector
    of shared token indices. Local coordinates are normalized to ``[-1, 1]``
    before being scattered into the shared action space. Coordinates that are
    invalid for an embodiment are always zero.

    A scalar embodiment ID supports arbitrary leading batch dimensions with a
    local trailing action dimension. Mixed-embodiment batches use a padded
    local trailing dimension equal to :attr:`action_dim`.
    """

    def __init__(
        self,
        action_lows: Sequence[Sequence[float] | np.ndarray],
        action_highs: Sequence[Sequence[float] | np.ndarray],
        action_indices: Sequence[Sequence[int] | np.ndarray] | None = None,
        *,
        embodiment_names: Sequence[str] | None = None,
        shared_action_dim: int | None = None,
    ) -> None:
        if len(action_lows) == 0:
            raise ValueError("ActionTokenizer requires at least one embodiment")
        if len(action_lows) != len(action_highs):
            raise ValueError("action_lows and action_highs must have the same length")

        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for embodiment_id, (low, high) in enumerate(zip(action_lows, action_highs, strict=True)):
            low_array = np.asarray(low, dtype=np.float32).reshape(-1).copy()
            high_array = np.asarray(high, dtype=np.float32).reshape(-1).copy()
            if low_array.size == 0:
                raise ValueError(
                    f"Embodiment {embodiment_id} must have at least one action coordinate"
                )
            if low_array.shape != high_array.shape:
                raise ValueError(
                    f"Action bounds for embodiment {embodiment_id} must have matching shapes, "
                    f"got {low_array.shape} and {high_array.shape}"
                )
            if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
                raise ValueError(f"Action bounds for embodiment {embodiment_id} must be finite")
            if np.any(high_array <= low_array):
                raise ValueError(
                    f"Every action upper bound must exceed its lower bound for embodiment "
                    f"{embodiment_id}"
                )
            lows.append(low_array)
            highs.append(high_array)

        action_dims = tuple(int(low.size) for low in lows)
        default_shared_dim = max(action_dims)
        if shared_action_dim is None:
            shared_action_dim = default_shared_dim
        shared_action_dim = int(shared_action_dim)
        if shared_action_dim < default_shared_dim:
            raise ValueError(
                "shared_action_dim must be at least the largest embodiment action dimension, "
                f"got {shared_action_dim} < {default_shared_dim}"
            )

        if action_indices is None:
            action_indices = [np.arange(action_dim, dtype=np.int64) for action_dim in action_dims]
        if len(action_indices) != len(lows):
            raise ValueError("action_indices must contain one index vector per embodiment")

        indices: list[np.ndarray] = []
        for embodiment_id, (raw_indices, action_dim) in enumerate(
            zip(action_indices, action_dims, strict=True)
        ):
            raw_array = np.asarray(raw_indices)
            if not np.issubdtype(raw_array.dtype, np.integer):
                raise ValueError(f"Action indices for embodiment {embodiment_id} must be integers")
            index_array = raw_array.astype(np.int64, copy=True).reshape(-1)
            if index_array.size != action_dim:
                raise ValueError(
                    f"Embodiment {embodiment_id} has {action_dim} action bounds but "
                    f"{index_array.size} shared indices"
                )
            if np.any(index_array < 0) or np.any(index_array >= shared_action_dim):
                raise ValueError(
                    f"Action indices for embodiment {embodiment_id} must lie in "
                    f"[0, {shared_action_dim})"
                )
            if np.unique(index_array).size != index_array.size:
                raise ValueError(f"Action indices for embodiment {embodiment_id} must be unique")
            indices.append(index_array)

        if embodiment_names is None:
            embodiment_names = [f"embodiment_{index}" for index in range(len(lows))]
        names = tuple(str(name).strip() for name in embodiment_names)
        if len(names) != len(lows):
            raise ValueError("embodiment_names must contain one name per embodiment")
        if any(not name for name in names):
            raise ValueError("Embodiment names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("Embodiment names must be unique")

        self.action_lows = tuple(lows)
        self.action_highs = tuple(highs)
        self.action_indices = tuple(indices)
        self.action_dims = action_dims
        self.embodiment_names = names
        self.num_embodiments = len(lows)
        self.shared_action_dim = shared_action_dim
        self.action_dim = shared_action_dim
        self.action_masks = np.zeros(
            (self.num_embodiments, self.shared_action_dim), dtype=np.float32
        )
        for embodiment_id, index_array in enumerate(self.action_indices):
            self.action_masks[embodiment_id, index_array] = 1.0

    def _resolve_embodiment_ids(
        self,
        embodiment_ids: int | Sequence[int] | np.ndarray | None,
        embodiment_id: int | None,
    ) -> np.ndarray:
        if embodiment_id is not None:
            if embodiment_ids is not None:
                raise ValueError("Pass either embodiment_ids or embodiment_id, not both")
            embodiment_ids = embodiment_id
        if embodiment_ids is None:
            if self.num_embodiments != 1:
                raise ValueError(
                    "An embodiment ID is required when multiple embodiments are configured"
                )
            embodiment_ids = 0
        ids = np.asarray(embodiment_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Embodiment IDs must be integers")
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.num_embodiments):
            raise ValueError(
                f"Embodiment IDs must lie in [0, {self.num_embodiments}), got {ids.tolist()}"
            )
        return ids

    @staticmethod
    def _as_action_array(actions: Sequence[float] | np.ndarray) -> np.ndarray:
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim == 0:
            raise ValueError("Actions must have a trailing coordinate dimension")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("Actions must be finite")
        return action_array

    def _validate_batch_ids(self, action_array: np.ndarray, ids: np.ndarray) -> None:
        if ids.ndim == 0:
            return
        if tuple(ids.shape) != tuple(action_array.shape[:-1]):
            raise ValueError(
                "A mixed-embodiment ID array must match the action leading dimensions, "
                f"got IDs {ids.shape} for actions {action_array.shape}"
            )
        if action_array.shape[-1] != self.shared_action_dim:
            raise ValueError(
                "Mixed-embodiment local actions must be padded to shared_action_dim, "
                f"expected {self.shared_action_dim}, got {action_array.shape[-1]}"
            )

    def mask_for(
        self,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
    ) -> np.ndarray:
        """Return shared action masks for one or more embodiments."""

        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        return self.action_masks[ids].copy()

    def normalize(
        self,
        physical_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Normalize local physical actions while preserving local coordinate order."""

        actions = self._as_action_array(physical_actions)
        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        self._validate_batch_ids(actions, ids)
        if ids.ndim == 0:
            index = int(ids)
            action_dim = self.action_dims[index]
            if actions.shape[-1] not in {action_dim, self.shared_action_dim}:
                raise ValueError(
                    f"Embodiment {index} expects {action_dim} local action coordinates, "
                    f"got {actions.shape[-1]}"
                )
            local = actions[..., :action_dim]
            normalized = 2.0 * (local - self.action_lows[index]) / (
                self.action_highs[index] - self.action_lows[index]
            ) - 1.0
            if clip:
                normalized = np.clip(normalized, -1.0, 1.0)
            if actions.shape[-1] == action_dim:
                return normalized.astype(np.float32, copy=False)
            output = np.zeros_like(actions, dtype=np.float32)
            output[..., :action_dim] = normalized
            return output

        output = np.zeros_like(actions, dtype=np.float32)
        flat_actions = actions.reshape(-1, self.shared_action_dim)
        flat_output = output.reshape(-1, self.shared_action_dim)
        for row, index in enumerate(ids.reshape(-1)):
            action_dim = self.action_dims[int(index)]
            normalized = 2.0 * (
                flat_actions[row, :action_dim] - self.action_lows[int(index)]
            ) / (self.action_highs[int(index)] - self.action_lows[int(index)]) - 1.0
            flat_output[row, :action_dim] = (
                np.clip(normalized, -1.0, 1.0) if clip else normalized
            )
        return output

    def denormalize(
        self,
        normalized_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Convert local normalized actions to embodiment-specific physical controls."""

        actions = self._as_action_array(normalized_actions)
        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        self._validate_batch_ids(actions, ids)
        if ids.ndim == 0:
            index = int(ids)
            action_dim = self.action_dims[index]
            if actions.shape[-1] not in {action_dim, self.shared_action_dim}:
                raise ValueError(
                    f"Embodiment {index} expects {action_dim} local action coordinates, "
                    f"got {actions.shape[-1]}"
                )
            local = actions[..., :action_dim]
            if clip:
                local = np.clip(local, -1.0, 1.0)
            physical = self.action_lows[index] + 0.5 * (local + 1.0) * (
                self.action_highs[index] - self.action_lows[index]
            )
            if actions.shape[-1] == action_dim:
                return physical.astype(np.float32, copy=False)
            output = np.zeros_like(actions, dtype=np.float32)
            output[..., :action_dim] = physical
            return output

        output = np.zeros_like(actions, dtype=np.float32)
        flat_actions = actions.reshape(-1, self.shared_action_dim)
        flat_output = output.reshape(-1, self.shared_action_dim)
        for row, index in enumerate(ids.reshape(-1)):
            action_dim = self.action_dims[int(index)]
            local = flat_actions[row, :action_dim]
            if clip:
                local = np.clip(local, -1.0, 1.0)
            flat_output[row, :action_dim] = self.action_lows[int(index)] + 0.5 * (
                local + 1.0
            ) * (self.action_highs[int(index)] - self.action_lows[int(index)])
        return output

    def tokenize(
        self,
        normalized_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Scatter local normalized coordinates into the shared token space."""

        actions = self._as_action_array(normalized_actions)
        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        self._validate_batch_ids(actions, ids)
        output = np.zeros((*actions.shape[:-1], self.shared_action_dim), dtype=np.float32)
        if ids.ndim == 0:
            index = int(ids)
            action_dim = self.action_dims[index]
            if actions.shape[-1] not in {action_dim, self.shared_action_dim}:
                raise ValueError(
                    f"Embodiment {index} expects {action_dim} local action coordinates, "
                    f"got {actions.shape[-1]}"
                )
            local = actions[..., :action_dim]
            if clip:
                local = np.clip(local, -1.0, 1.0)
            output[..., self.action_indices[index]] = local
            return output

        flat_actions = actions.reshape(-1, self.shared_action_dim)
        flat_output = output.reshape(-1, self.shared_action_dim)
        for row, index in enumerate(ids.reshape(-1)):
            index = int(index)
            local = flat_actions[row, : self.action_dims[index]]
            if clip:
                local = np.clip(local, -1.0, 1.0)
            flat_output[row, self.action_indices[index]] = local
        return output

    def detokenize(
        self,
        shared_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Extract embodiment-local normalized coordinates from shared tokens."""

        actions = self._as_action_array(shared_actions)
        if actions.shape[-1] != self.shared_action_dim:
            raise ValueError(
                f"Shared actions must have trailing dimension {self.shared_action_dim}, "
                f"got {actions.shape[-1]}"
            )
        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        if ids.ndim == 0:
            local = actions[..., self.action_indices[int(ids)]]
            return np.clip(local, -1.0, 1.0) if clip else local.copy()
        if tuple(ids.shape) != tuple(actions.shape[:-1]):
            raise ValueError(
                "A mixed-embodiment ID array must match the action leading dimensions, "
                f"got IDs {ids.shape} for actions {actions.shape}"
            )
        output = np.zeros_like(actions, dtype=np.float32)
        flat_actions = actions.reshape(-1, self.shared_action_dim)
        flat_output = output.reshape(-1, self.shared_action_dim)
        for row, index in enumerate(ids.reshape(-1)):
            index = int(index)
            local = flat_actions[row, self.action_indices[index]]
            flat_output[row, : self.action_dims[index]] = (
                np.clip(local, -1.0, 1.0) if clip else local
            )
        return output

    def encode(
        self,
        physical_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Normalize and tokenize physical actions in one operation."""

        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        normalized = self.normalize(physical_actions, ids, clip=clip)
        return self.tokenize(normalized, ids, clip=clip)

    def decode(
        self,
        shared_actions: Sequence[float] | np.ndarray,
        embodiment_ids: int | Sequence[int] | np.ndarray | None = None,
        *,
        embodiment_id: int | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        """Extract and denormalize shared action tokens in one operation."""

        ids = self._resolve_embodiment_ids(embodiment_ids, embodiment_id)
        normalized = self.detokenize(shared_actions, ids, clip=clip)
        return self.denormalize(normalized, ids, clip=clip)

    physical_to_shared = encode
    shared_to_physical = decode
    to_shared = encode
    from_shared = decode


__all__ = ["ActionTokenizer"]
