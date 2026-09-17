from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def environment_signature(
    env: Any,
    task_contexts: np.ndarray | None,
    task_context_mode: str,
) -> dict[str, Any]:
    """Serialize environment semantics that must match a checkpoint."""

    external_signature = getattr(env, "environment_signature", None)
    if callable(external_signature):
        external_signature = external_signature()
    tokenizer = getattr(env, "action_tokenizer", None)
    if tokenizer is None:
        # A simulator adapter without the shared tokenizer must describe any
        # embodiment-specific mapping in its external signature. Worker masks
        # cannot be stored here because their shape depends on ``num_envs``.
        action_interface: dict[str, Any] = {
            "action_dim": int(env.action_dim),
            "embodiment_masks": None,
        }
    else:
        action_interface = {
            "action_dim": int(tokenizer.shared_action_dim),
            "action_lows": [value.tolist() for value in tokenizer.action_lows],
            "action_highs": [value.tolist() for value in tokenizer.action_highs],
            "action_indices": [value.tolist() for value in tokenizer.action_indices],
        }

    context_hash = None
    context_shape = None
    if task_contexts is not None:
        contiguous = np.ascontiguousarray(task_contexts, dtype=np.float32)
        context_hash = hashlib.sha256(contiguous.tobytes()).hexdigest()
        context_shape = list(contiguous.shape)
    spec = env.observation_spec
    return {
        "adapter": f"{type(env).__module__}.{type(env).__qualname__}",
        "task_names": list(env.task_names),
        "embodiment_names": list(env.embodiment_names),
        "observation_spec": {
            "image_shape": None if spec.image_shape is None else list(spec.image_shape),
            "proprio_dim": int(spec.proprio_dim),
            "state_dim": int(spec.state_dim),
        },
        "action_interface": action_interface,
        "max_episode_steps": (
            None
            if getattr(env, "max_episode_steps", None) is None
            else int(env.max_episode_steps)
        ),
        "control_repeats": np.asarray(
            getattr(env, "control_repeats", []), dtype=np.int64
        ).tolist(),
        "low_level_discounts": np.asarray(
            getattr(env, "low_level_discounts", []), dtype=np.float32
        ).tolist(),
        "task_context_mode": task_context_mode,
        "task_context_shape": context_shape,
        "task_context_sha256": context_hash,
        "external": external_signature,
    }


__all__ = ["environment_signature"]
