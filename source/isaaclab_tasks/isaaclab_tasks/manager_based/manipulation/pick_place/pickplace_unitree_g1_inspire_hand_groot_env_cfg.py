# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T-oriented inference environment for the G1 Inspire FTP pick-place task.

This environment keeps the underlying task and action space from
``PickPlaceG1InspireFTPEnvCfg`` but exposes observations in a format that is
closer to the public Arena G1 GR00T checkpoint:

- one RGB camera stream named ``ego_view``
- structured state terms grouped by body part
- wrist poses exposed explicitly as 7D position + quaternion terms

The public checkpoint is still not embodiment-matched to the Inspire 5-finger
hand, so a policy adapter is still required at inference time.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp
from .pickplace_unitree_g1_inspire_hand_env_cfg import (
    ObjectTableSceneCfg,
    PickPlaceG1InspireFTPEnvCfg,
)


def eef_pose(env, link_name: str) -> torch.Tensor:
    """Return 7D end-effector pose as position + quaternion."""

    return torch.cat((mdp.get_eef_pos(env, link_name), mdp.get_eef_quat(env, link_name)), dim=-1)


@configclass
class ObjectTableGR00TSceneCfg(ObjectTableSceneCfg):
    """Scene with a single ego-view camera for GR00T-style inference."""

    ego_view = CameraCfg(
        prim_path="{ENV_REGEX_NS}/EgoViewCam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12, 1.68), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"),
    )


@configclass
class GR00TObservationsCfg:
    """Observation layout aligned with the public Arena G1 GR00T checkpoint."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for GR00T inference."""

        left_leg = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ]
            },
        )
        right_leg = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ]
            },
        )
        waist = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]},
        )
        left_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_yaw_joint",
                ]
            },
        )
        left_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "L_index_proximal_joint",
                    "L_middle_proximal_joint",
                    "L_thumb_proximal_yaw_joint",
                    "L_index_intermediate_joint",
                    "L_middle_intermediate_joint",
                    "L_thumb_proximal_pitch_joint",
                    "L_thumb_intermediate_joint",
                ]
            },
        )
        right_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_yaw_joint",
                ]
            },
        )
        right_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={
                "joint_names": [
                    "R_index_proximal_joint",
                    "R_middle_proximal_joint",
                    "R_thumb_proximal_yaw_joint",
                    "R_index_intermediate_joint",
                    "R_middle_intermediate_joint",
                    "R_thumb_proximal_pitch_joint",
                    "R_thumb_intermediate_joint",
                ]
            },
        )
        left_wrist_pose = ObsTerm(func=eef_pose, params={"link_name": "left_wrist_yaw_link"})
        right_wrist_pose = ObsTerm(func=eef_pose, params={"link_name": "right_wrist_yaw_link"})
        ego_view = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ego_view"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceG1InspireFTPGR00TEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    """Inspire FTP pick-place environment with GR00T-style observations."""

    scene: ObjectTableGR00TSceneCfg = ObjectTableGR00TSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: GR00TObservationsCfg = GR00TObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["ego_view"]
