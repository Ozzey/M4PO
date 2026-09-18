from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _configured_names(value: Any) -> str | Sequence[str] | None:
    if value is None or isinstance(value, (str, list, tuple)):
        return value
    raise ValueError("Task and embodiment names must be comma-separated strings or sequences")


def make_vector_env(cfg):
    """Construct the configured vector environment through a lazy adapter."""

    env_name = str(getattr(cfg, "env", "mock")).lower().strip()
    if env_name == "mock":
        from .mock_env import MockVectorEnv

        control_repeats = getattr(
            cfg, "control_repeat_values", getattr(cfg, "control_repeats", None)
        )
        control_decimation = int(getattr(cfg, "control_decimation", 1))
        has_explicit_repeats = control_repeats is not None
        if isinstance(control_repeats, str):
            has_explicit_repeats = bool(control_repeats.strip())
        elif isinstance(control_repeats, (list, tuple)):
            has_explicit_repeats = len(control_repeats) > 0
        if not has_explicit_repeats and control_decimation > 1:
            control_repeats = [control_decimation] * int(cfg.num_embodiments)
        return MockVectorEnv(
            num_envs=int(cfg.num_envs),
            seed=int(cfg.seed),
            max_episode_steps=int(cfg.max_episode_steps or 32),
            image_size=int(cfg.image_size),
            image_channels=int(getattr(cfg, "image_channels", 4)),
            use_images=bool(cfg.use_images),
            proprio_dim=(
                None
                if getattr(cfg, "proprio_dim", None) is None
                else int(cfg.proprio_dim)
            ),
            state_dim=(
                None if getattr(cfg, "state_dim", None) is None else int(cfg.state_dim)
            ),
            num_tasks=int(cfg.num_tasks),
            num_embodiments=int(cfg.num_embodiments),
            task_context_dim=int(cfg.task_context_dim),
            discount=float(cfg.discount),
            low_level_discount=getattr(cfg, "low_level_discount", 1.0),
            tasks=_configured_names(
                getattr(cfg, "task_names", getattr(cfg, "tasks", None))
            ),
            embodiments=_configured_names(
                getattr(cfg, "embodiment_names", getattr(cfg, "embodiments", None))
            ),
            control_repeats=control_repeats,
        )
    if env_name in {"isaaclab", "isaac_lab"}:
        from .isaaclab_env import make_isaaclab_vector_env

        return make_isaaclab_vector_env(cfg)
    if env_name == "mmbench":
        from .mmbench_env import make_mmbench_vector_env

        return make_mmbench_vector_env(cfg)
    raise ValueError(
        f"Unknown environment {env_name!r}. The built-in environment is 'mock'; "
        "additional simulators must be registered as lazy adapters in m4po.envs.factory."
    )


__all__ = ["make_vector_env"]
