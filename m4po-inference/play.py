# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a GR00T policy on the local G1 Inspire 5-finger pick-place task."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

from m4po_inference.bootstrap import configure_python_path


configure_python_path()

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "assets" / "model" / "GN1x-Tuned-Arena-G1-Loco-Manipulation"


parser = argparse.ArgumentParser(description="Play a GR00T policy on the local G1 Inspire pick-place task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--task",
    type=str,
    default="M4PO-Inference-G1-InspireFTP-GR00T-Abs-v0",
    help="Task name. Defaults to the local GR00T-ready G1 Inspire env.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=str(DEFAULT_MODEL_DIR),
    help="Local checkpoint path or Hugging Face model id.",
)
parser.add_argument(
    "--embodiment_tag",
    type=str,
    default="new_embodiment",
    help="GR00T embodiment tag. The default matches the public Arena G1 checkpoint metadata.",
)
parser.add_argument(
    "--instruction",
    type=str,
    default="pick up the block and place it on the packing table",
    help="Language instruction passed to the policy.",
)
parser.add_argument("--horizon", type=int, default=400, help="Maximum env steps per rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument(
    "--chunk_size",
    type=int,
    default=1,
    help="How many actions from the predicted action chunk to execute before re-running inference.",
)
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--policy_device",
    type=str,
    default=None,
    help="Device for the GR00T policy. Defaults to --device when omitted.",
)
parser.add_argument(
    "--server_host",
    type=str,
    default=None,
    help="Optional GR00T policy-server host. When set, use a lightweight PolicyClient instead of local model inference.",
)
parser.add_argument("--server_port", type=int, default=5555, help="GR00T policy-server port for --server_host mode.")
parser.add_argument(
    "--server_timeout_ms",
    type=int,
    default=15000,
    help="ZMQ timeout in milliseconds for --server_host mode.",
)
parser.add_argument(
    "--server_api_token",
    type=str,
    default=None,
    help="Optional API token for the GR00T policy server.",
)
parser.add_argument(
    "--no_strict",
    action="store_true",
    default=False,
    help="Disable GR00T input/output validation. Keep strict mode on while debugging format mismatches.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import m4po_inference.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


POLICY_STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "left_wrist_pose",
    "right_wrist_pose",
]
POLICY_VIDEO_KEYS = ["ego_view"]
EXPECTED_ACTION_KEYS = [
    "left_wrist_pose",
    "right_wrist_pose",
    "left_hand",
    "right_hand",
    "base_height_command",
    "navigate_command",
]


def _require_gr00t_policy():
    try:
        from gr00t.policy import Gr00tPolicy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GR00T is not installed in the active environment. Install NVIDIA Isaac-GR00T "
            "or use --server_host to talk to a separate GR00T policy server."
        ) from exc
    return Gr00tPolicy


def _require_gr00t_policy_client():
    try:
        from gr00t.policy.server_client import PolicyClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GR00T client modules are not installed in the active environment. "
            "Install the lightweight client pieces or run local inference."
        ) from exc
    return PolicyClient


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env(value: torch.Tensor | np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _first_action_step(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array[0]
    if array.ndim == 2:
        return array
    if array.ndim == 1:
        return array[None, :]
    raise ValueError(f"Unsupported action tensor shape: {array.shape}")


def _stack_history(buffer: deque[np.ndarray], horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError(f"History horizon must be positive, got {horizon}.")
    if len(buffer) == 0:
        raise ValueError("Cannot build policy observation from an empty history buffer.")
    frames = list(buffer)
    while len(frames) < horizon:
        frames.insert(0, frames[0])
    return np.stack(frames[-horizon:], axis=0)


def _extract_horizon(modality_cfg) -> int:
    if isinstance(modality_cfg, dict):
        delta_indices = modality_cfg.get("delta_indices")
    else:
        delta_indices = getattr(modality_cfg, "delta_indices", None)
    if delta_indices is None:
        return 1
    return len(delta_indices)


class GR00TObservationAdapter:
    """Build temporally stacked GR00T observations from Isaac Lab policy observations."""

    def __init__(self, state_horizon: int, video_horizon: int, instruction: str):
        self.state_horizon = state_horizon
        self.video_horizon = video_horizon
        self.instruction = instruction
        self.state_buffers = {key: deque(maxlen=state_horizon) for key in POLICY_STATE_KEYS}
        self.video_buffers = {key: deque(maxlen=video_horizon) for key in POLICY_VIDEO_KEYS}

    def reset(self) -> None:
        for buffer in self.state_buffers.values():
            buffer.clear()
        for buffer in self.video_buffers.values():
            buffer.clear()

    def append(self, policy_obs: dict[str, torch.Tensor]) -> None:
        for key in POLICY_STATE_KEYS:
            self.state_buffers[key].append(_first_env(policy_obs[key], np.float32))
        for key in POLICY_VIDEO_KEYS:
            self.video_buffers[key].append(_first_env(policy_obs[key], np.uint8))

    def build(self) -> dict[str, dict[str, np.ndarray | list[list[str]]]]:
        obs = {
            "state": {},
            "video": {},
            "language": {"task": [[self.instruction]]},
        }
        for key, buffer in self.state_buffers.items():
            obs["state"][key] = _stack_history(buffer, self.state_horizon)[None, ...].astype(np.float32, copy=False)
        for key, buffer in self.video_buffers.items():
            obs["video"][key] = _stack_history(buffer, self.video_horizon)[None, ...].astype(np.uint8, copy=False)
        return obs


def _pack_action_dict(action_dict: dict[str, np.ndarray]) -> np.ndarray:
    action_dict = {
        key[7:] if key.startswith("action.") else key: value for key, value in action_dict.items()
    }
    if all(key in action_dict for key in EXPECTED_ACTION_KEYS):
        parts = []
        for key in EXPECTED_ACTION_KEYS:
            action = _first_action_step(np.asarray(action_dict[key], dtype=np.float32))
            parts.append(action)
        return np.concatenate(parts, axis=-1)

    if len(action_dict) == 1:
        only_value = next(iter(action_dict.values()))
        return _first_action_step(np.asarray(only_value, dtype=np.float32))

    raise ValueError(
        "Unsupported GR00T action dictionary layout. "
        f"Available keys: {sorted(action_dict.keys())}. Expected either a single concatenated key "
        f"or the structured keys {EXPECTED_ACTION_KEYS}."
    )


def _map_trihand_to_inspire(hand: np.ndarray) -> np.ndarray:
    """Map the 7-DoF tri-hand command to one 12-joint Inspire hand."""

    index_prox, middle_prox, thumb_yaw, index_inter, middle_inter, thumb_pitch, thumb_tip = hand
    return np.asarray(
        [
            index_prox,
            middle_prox,
            middle_prox,
            middle_prox,
            thumb_yaw,
            index_inter,
            middle_inter,
            middle_inter,
            middle_inter,
            thumb_pitch,
            thumb_tip,
            thumb_tip,
        ],
        dtype=np.float32,
    )


def adapt_action_to_inspire_ftp(action_chunk: np.ndarray) -> np.ndarray:
    """Convert a public Arena G1 GR00T action chunk to the Inspire FTP action space."""

    if action_chunk.shape[-1] == 38:
        return action_chunk.astype(np.float32, copy=False)
    if action_chunk.shape[-1] != 32:
        raise ValueError(
            "Expected a 32-dim Arena G1 action or a 38-dim Inspire action, "
            f"but received shape {action_chunk.shape}."
        )

    adapted = []
    for action in action_chunk:
        left_wrist_pose = action[0:7]
        right_wrist_pose = action[7:14]
        left_hand = _map_trihand_to_inspire(action[14:21])
        right_hand = _map_trihand_to_inspire(action[21:28])

        inspire_hand = np.asarray(
            [
                left_hand[0],
                left_hand[1],
                left_hand[2],
                left_hand[3],
                left_hand[4],
                right_hand[0],
                right_hand[1],
                right_hand[2],
                right_hand[3],
                right_hand[4],
                left_hand[5],
                left_hand[6],
                left_hand[7],
                left_hand[8],
                left_hand[9],
                right_hand[5],
                right_hand[6],
                right_hand[7],
                right_hand[8],
                right_hand[9],
                left_hand[10],
                right_hand[10],
                left_hand[11],
                right_hand[11],
            ],
            dtype=np.float32,
        )
        adapted.append(np.concatenate((left_wrist_pose, right_wrist_pose, inspire_hand), axis=-1))

    return np.stack(adapted, axis=0)


def load_policy(
    model_path: str,
    embodiment_tag: str,
    device: str,
    strict: bool,
    server_host: str | None = None,
    server_port: int = 5555,
    server_timeout_ms: int = 15000,
    server_api_token: str | None = None,
):
    if server_host:
        PolicyClient = _require_gr00t_policy_client()
        policy = PolicyClient(
            host=server_host,
            port=server_port,
            timeout_ms=server_timeout_ms,
            api_token=server_api_token,
            strict=False,
        )
        if not policy.ping():
            raise RuntimeError(
                f"Unable to connect to the GR00T policy server at {server_host}:{server_port}. "
                "Start the server first, then re-run this script."
            )
        return policy

    local_model_path = Path(model_path)
    if local_model_path.is_dir():
        weights = list(local_model_path.glob("model-*.safetensors")) or list(local_model_path.glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(
                f"Local model directory exists but no weight files were found: {local_model_path}\n"
                "Run `python m4po-inference/download_model.py` first, or use `--server_host`."
            )

    Gr00tPolicy = _require_gr00t_policy()
    try:
        policy = Gr00tPolicy(model_path=model_path, embodiment_tag=embodiment_tag, device=device, strict=strict)
    except Exception as exc:
        if "gr00t_n1_5" in str(exc):
            raise RuntimeError(
                "Failed to load the public Arena G1 checkpoint because it reports `model_type=gr00t_n1_5` "
                "while the active local GR00T code expects a newer model family. "
                "Use `m4po-inference/run_gr00t_server.sh`, which now auto-selects the N1.5 compatibility "
                "server for this checkpoint, or switch to a checkpoint that matches the installed GR00T release."
            ) from exc
        raise RuntimeError(
            "Failed to load the GR00T policy. If this is an embodiment-tag mismatch, try overriding "
            "`--embodiment_tag` and inspect the checkpoint metadata. "
            "For the public Arena G1 checkpoint the expected tag is typically `new_embodiment`."
        ) from exc
    return policy


def rollout(policy, env, success_term, horizon: int, chunk_size: int, instruction: str) -> bool:
    policy_obs, _ = env.reset()
    env_obs = policy_obs["policy"]

    modality_cfg = policy.get_modality_config()
    state_horizon = _extract_horizon(modality_cfg["state"])
    video_horizon = _extract_horizon(modality_cfg["video"])
    adapter = GR00TObservationAdapter(state_horizon=state_horizon, video_horizon=video_horizon, instruction=instruction)
    adapter.reset()
    adapter.append(env_obs)

    policy.reset()

    steps = 0
    while steps < horizon:
        gr00t_obs = adapter.build()
        action_dict, _ = policy.get_action(gr00t_obs)
        raw_action_chunk = _pack_action_dict(action_dict)
        env_action_chunk = adapt_action_to_inspire_ftp(raw_action_chunk)

        for action in env_action_chunk[: max(1, chunk_size)]:
            action_tensor = torch.from_numpy(action).to(device=env.unwrapped.device).view(1, env.action_space.shape[1])
            policy_obs, _, terminated, truncated, _ = env.step(action_tensor)
            env_obs = policy_obs["policy"]
            adapter.append(env_obs)
            steps += 1

            if bool(success_term.func(env, **success_term.params)[0]):
                return True
            if terminated or truncated or steps >= horizon:
                return False

    return False


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    policy_device = args_cli.policy_device or args_cli.device
    policy = load_policy(
        model_path=args_cli.model_path,
        embodiment_tag=args_cli.embodiment_tag,
        device=policy_device,
        strict=not args_cli.no_strict,
        server_host=args_cli.server_host,
        server_port=args_cli.server_port,
        server_timeout_ms=args_cli.server_timeout_ms,
        server_api_token=args_cli.server_api_token,
    )

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting GR00T trial {trial}")
        succeeded = rollout(
            policy=policy,
            env=env,
            success_term=success_term,
            horizon=args_cli.horizon,
            chunk_size=args_cli.chunk_size,
            instruction=args_cli.instruction,
        )
        results.append(succeeded)
        print(f"[INFO] Trial {trial}: {succeeded}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
