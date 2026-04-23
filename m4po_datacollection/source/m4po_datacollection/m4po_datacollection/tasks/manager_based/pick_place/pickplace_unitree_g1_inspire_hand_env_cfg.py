# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local mirror of the official Isaac Lab G1 Inspire pick-place environment config."""

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg as IsaacLabPickPlaceG1InspireFTPEnvCfg,
)

from m4po_datacollection.paths import ASSETS_DIR, ROBOT_ASSET_PATH

LOCAL_TABLE_ASSET_PATH = ASSETS_DIR / "objects" / "PackingTable" / "PackingTable.usd"
LEFT_WRIST_FRAME = "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link"
RIGHT_WRIST_FRAME = "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link"


class PickPlaceG1InspireFTPEnvCfg(IsaacLabPickPlaceG1InspireFTPEnvCfg):
    """Local wrapper around the upstream Isaac Lab config with offline asset paths."""

    def __post_init__(self):
        """Swap remote assets for local workspace copies before the upstream setup runs."""
        self.scene.robot.spawn.usd_path = str(ROBOT_ASSET_PATH)
        super().__post_init__()

        if not ROBOT_ASSET_PATH.is_file():
            raise FileNotFoundError(f"Robot USD not found: {ROBOT_ASSET_PATH}")
        if not LOCAL_TABLE_ASSET_PATH.is_file():
            raise FileNotFoundError(f"Packing table USD not found: {LOCAL_TABLE_ASSET_PATH}")

        frame_tasks = self.actions.pink_ik_cfg.controller.variable_input_tasks
        frame_tasks[0].frame = LEFT_WRIST_FRAME
        frame_tasks[1].frame = RIGHT_WRIST_FRAME
        frame_tasks[2].controlled_frames = [LEFT_WRIST_FRAME, RIGHT_WRIST_FRAME]

        self.scene.packing_table.spawn = UsdFileCfg(
            usd_path=str(LOCAL_TABLE_ASSET_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        self.scene.ground = None
        self.scene.object.spawn = sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=0.05),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.2, 0.8), roughness=0.4),
        )
