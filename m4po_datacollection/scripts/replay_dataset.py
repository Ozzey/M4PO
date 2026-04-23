#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay HDF5 demonstrations in the local G1 Inspire pick-place scene."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

try:
    import pinocchio  # noqa: F401
except ImportError:
    pinocchio = None  # noqa: F841

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Template-M4po-DataCollection-G1-InspireFTP-Abs-v0"
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "isaaclab" / "datasets" / "dataset_annotated_g1_locomanip.hdf5"
)

parser = argparse.ArgumentParser(description="Replay demonstrations with the local G1 Inspire env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task to instantiate for replay.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Leave empty to replay every episode.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=str(DEFAULT_DATASET),
    help="Dataset file to replay.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=("auto", "actions", "states"),
    default="auto",
    help="Replay actions when possible, otherwise fall back to direct state playback.",
)
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Report success using the environment's success termination condition.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m4po_datacollection.tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _patch_missing_env_metadata(dataset_file_handler: HDF5DatasetFileHandler) -> None:
    """Make datasets without ``env_args`` replayable."""
    if "env_args" not in dataset_file_handler._hdf5_data_group.attrs:
        dataset_file_handler.get_env_name = lambda: None  # type: ignore[method-assign]


def _resolve_replay_mode(mode: str, episode_data: EpisodeData, env) -> str:
    """Choose between action replay and direct state playback."""
    actions = episode_data.data.get("actions")
    states = episode_data.data.get("states")
    env_action_dim = env.action_space.shape[-1]
    action_dim = None if actions is None else actions.shape[-1]

    if mode == "actions":
        if actions is None:
            raise ValueError("The dataset does not contain actions.")
        if action_dim != env_action_dim:
            raise ValueError(
                f"Action replay requested, but dataset actions are {action_dim}-D and env actions are {env_action_dim}-D."
            )
        return "actions"

    if mode == "states":
        if states is None:
            raise ValueError("The dataset does not contain states for trajectory replay.")
        return "states"

    if actions is not None and action_dim == env_action_dim:
        return "actions"
    if states is not None:
        return "states"
    raise ValueError("No compatible replay source found in the dataset.")


def _finish_episode(success_term, env, env_id: int, episode_index: int | None, failed_demo_ids: list[int]) -> bool:
    """Return whether the current episode counts as successful."""
    if success_term is None or episode_index is None:
        return False
    if bool(success_term.func(env, **success_term.params)[env_id]):
        return True
    if episode_index not in failed_demo_ids:
        failed_demo_ids.append(episode_index)
    return False


def _adapt_state_to_env(state: dict | None, env, env_id: int) -> dict | None:
    """Pad or trim dataset joint states so they match the instantiated robot."""
    if state is None:
        return None

    robot_state = state.get("articulation", {}).get("robot")
    if robot_state is None:
        return state

    target_joint_count = env.scene["robot"].data.default_joint_pos.shape[-1]
    current_joint_count = robot_state["joint_position"].shape[-1]
    if current_joint_count == target_joint_count:
        return state

    if current_joint_count > target_joint_count:
        robot_state["joint_position"] = robot_state["joint_position"][..., :target_joint_count]
        robot_state["joint_velocity"] = robot_state["joint_velocity"][..., :target_joint_count]
        return state

    padded_joint_pos = env.scene["robot"].data.default_joint_pos[env_id : env_id + 1].clone()
    padded_joint_vel = torch.zeros_like(padded_joint_pos)
    padded_joint_pos[..., :current_joint_count] = robot_state["joint_position"]
    padded_joint_vel[..., :current_joint_count] = robot_state["joint_velocity"]
    robot_state["joint_position"] = padded_joint_pos
    robot_state["joint_velocity"] = padded_joint_vel
    return state


def main():
    """Replay episodes loaded from the dataset file."""
    dataset_path = Path(args_cli.dataset_file).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"The dataset file does not exist: {dataset_path}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(str(dataset_path))
    _patch_missing_env_metadata(dataset_file_handler)

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        raise RuntimeError("No episodes found in the dataset.")

    episode_indices_to_replay = args_cli.select_episodes or list(range(episode_count))

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    success_term = None
    if args_cli.validate_success_rate and hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success

    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    env.reset()

    probe_episode = dataset_file_handler.load_episode(episode_names[episode_indices_to_replay[0]], env.device)
    replay_mode = _resolve_replay_mode(args_cli.mode, probe_episode, env)
    print(f"[INFO] Replaying {len(episode_indices_to_replay)} episode(s) from {dataset_path}")
    print(f"[INFO] Replay mode: {replay_mode}")

    current_episode_indices = [None] * args_cli.num_envs
    env_episode_data_map = {index: EpisodeData() for index in range(args_cli.num_envs)}
    replayed_episode_count = 0
    recorded_episode_count = 0
    failed_demo_ids: list[int] = []
    device_env_ids = [torch.tensor([index], device=env.device, dtype=torch.long) for index in range(args_cli.num_envs)]

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            active_env_ids: list[int] = []
            actions = None
            if replay_mode == "actions":
                idle_action = getattr(env_cfg, "idle_action", torch.zeros(env.action_space.shape[-1], device=env.device))
                actions = idle_action.repeat(args_cli.num_envs, 1)

            for env_id in range(args_cli.num_envs):
                if replay_mode == "actions":
                    next_item = env_episode_data_map[env_id].get_next_action()
                else:
                    next_item = env_episode_data_map[env_id].get_next_state()

                if next_item is None:
                    if _finish_episode(
                        success_term, env, env_id, current_episode_indices[env_id], failed_demo_ids
                    ):
                        recorded_episode_count += 1

                    next_episode_index = None
                    while episode_indices_to_replay:
                        candidate = episode_indices_to_replay.pop(0)
                        if candidate < episode_count:
                            next_episode_index = candidate
                            break

                    if next_episode_index is None:
                        current_episode_indices[env_id] = None
                        continue

                    replayed_episode_count += 1
                    current_episode_indices[env_id] = next_episode_index
                    episode_data = dataset_file_handler.load_episode(episode_names[next_episode_index], env.device)
                    env_episode_data_map[env_id] = episode_data
                    print(f"{replayed_episode_count:4}: Loading #{next_episode_index} into env_{env_id}")

                    initial_state = episode_data.get_initial_state()
                    if initial_state is not None:
                        initial_state = _adapt_state_to_env(initial_state, env, env_id)
                        env.reset_to(initial_state, device_env_ids[env_id], is_relative=True)

                    if replay_mode == "actions":
                        next_item = env_episode_data_map[env_id].get_next_action()
                    else:
                        next_item = env_episode_data_map[env_id].get_next_state()

                if next_item is None:
                    continue

                active_env_ids.append(env_id)
                if replay_mode == "actions":
                    actions[env_id] = next_item
                else:
                    next_item = _adapt_state_to_env(next_item, env, env_id)
                    env.scene.reset_to(next_item, env_ids=device_env_ids[env_id], is_relative=True)

            if not active_env_ids:
                break

            if replay_mode == "actions":
                env.step(actions)
            else:
                env.sim.forward()
                env.sim.step(render=not args_cli.headless)
                env.scene.update(dt=env.physics_dt)

    for env_id in range(args_cli.num_envs):
        if _finish_episode(success_term, env, env_id, current_episode_indices[env_id], failed_demo_ids):
            recorded_episode_count += 1

    print(f"Finished replaying {replayed_episode_count} episode(s).")
    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")
        if failed_demo_ids:
            print(f"Failed demo IDs: {sorted(failed_demo_ids)}")

    env.close()
    dataset_file_handler.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
