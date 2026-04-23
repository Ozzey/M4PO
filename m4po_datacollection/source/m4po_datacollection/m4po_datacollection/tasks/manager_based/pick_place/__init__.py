# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="Template-M4po-DataCollection-G1-InspireFTP-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_unitree_g1_inspire_hand_env_cfg:PickPlaceG1InspireFTPEnvCfg",
    },
)
