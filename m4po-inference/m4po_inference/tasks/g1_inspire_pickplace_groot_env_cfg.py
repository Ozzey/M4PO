from __future__ import annotations

import torch

from m4po_inference.bootstrap import configure_python_path


configure_python_path()

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    ObjectTableSceneCfg,
)
from m4po_datacollection.tasks.manager_based.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg,
)


def eef_pose(env, link_name: str) -> torch.Tensor:
    """Return the wrist pose as position plus quaternion."""

    return torch.cat((mdp.get_eef_pos(env, link_name), mdp.get_eef_quat(env, link_name)), dim=-1)


@configclass
class ObjectTableGR00TSceneCfg(ObjectTableSceneCfg):
    """Local-assets scene with a single ego-view camera for GR00T inference."""

    ego_view = CameraCfg(
        prim_path="{ENV_REGEX_NS}/EgoViewCam",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.12, 1.68),
            rot=(-0.19848, 0.9801, 0.0, 0.0),
            convention="ros",
        ),
    )


@configclass
class GR00TObservationsCfg:
    """Observation layout aligned with the public Arena G1 GR00T checkpoint."""

    @configclass
    class PolicyCfg(ObsGroup):
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
    """Local-assets G1 Inspire pick-place env with GR00T-style observations."""

    scene: ObjectTableGR00TSceneCfg = ObjectTableGR00TSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: GR00TObservationsCfg = GR00TObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["ego_view"]
