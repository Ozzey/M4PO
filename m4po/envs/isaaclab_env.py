from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np

from m4po.common.buffer import ObservationSpec


def _load_factory(path: str) -> Callable[[Any], Any]:
    """Resolve ``package.module:function`` without importing IsaacLab eagerly."""

    module_name, separator, attribute = path.partition(":")
    if not separator:
        module_name, separator, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            "isaaclab_factory must look like 'package.module:function' or "
            "'package.module.function'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import IsaacLab environment module {module_name!r}. "
            "Install the simulator/task package in the active environment."
        ) from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"IsaacLab factory {path!r} does not resolve to a callable")
    return factory


def _validate_environment(env: Any) -> None:
    required_attributes = (
        "num_envs",
        "num_tasks",
        "num_embodiments",
        "task_ids",
        "embodiment_ids",
        "task_names",
        "embodiment_names",
        "action_dim",
        "action_masks",
        "observation_spec",
        "task_contexts",
    )
    missing = [name for name in required_attributes if not hasattr(env, name)]
    if missing:
        raise TypeError(f"IsaacLab adapter is missing required attributes: {missing}")
    for method in ("reset", "step", "close"):
        if not callable(getattr(env, method, None)):
            raise TypeError(f"IsaacLab adapter must provide a callable {method}()")
    if not isinstance(env.observation_spec, ObservationSpec):
        raise TypeError("IsaacLab adapter observation_spec must be an ObservationSpec")
    num_envs = int(env.num_envs)
    if num_envs <= 0:
        raise ValueError("IsaacLab adapter num_envs must be positive")
    if np.asarray(env.task_ids).shape != (num_envs,):
        raise ValueError("IsaacLab adapter task_ids must have shape [num_envs]")
    if np.asarray(env.embodiment_ids).shape != (num_envs,):
        raise ValueError("IsaacLab adapter embodiment_ids must have shape [num_envs]")
    if np.asarray(env.action_masks).shape != (num_envs, int(env.action_dim)):
        raise ValueError(
            "IsaacLab adapter action_masks must have shape [num_envs, action_dim]"
        )
    contexts = np.asarray(env.task_contexts)
    if contexts.ndim != 2 or contexts.shape[0] != int(env.num_tasks):
        raise ValueError(
            "IsaacLab adapter task_contexts must have shape [num_tasks, feature_dim]"
        )


def make_isaaclab_vector_env(cfg):
    """Build a simulator adapter supplied by an IsaacLab task package.

    The external factory receives the resolved :class:`M4POConfig` and must
    return the vector-environment contract documented in ``docs/DEVELOPMENT``.
    Keeping task registrations outside this repository avoids hard-coding
    private assets or a particular IsaacLab release while leaving collection,
    learning, resume validation, and evaluation fully shared.
    """

    path = getattr(cfg, "isaaclab_factory", None)
    if not path:
        raise ValueError(
            "env='isaaclab' requires --isaaclab-factory package.module:function"
        )
    env = _load_factory(str(path))(cfg)
    _validate_environment(env)
    return env


__all__ = ["make_isaaclab_vector_env"]
