from __future__ import annotations

import gymnasium as gym


gym.register(
    id="M4PO-Inference-G1-InspireFTP-GR00T-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_inspire_pickplace_groot_env_cfg:PickPlaceG1InspireFTPGR00TEnvCfg"
        ),
    },
)
