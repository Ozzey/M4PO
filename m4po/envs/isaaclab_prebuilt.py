from __future__ import annotations

import hashlib
import importlib.metadata
import math
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing.connection import Connection
from types import MethodType
from typing import Any

import numpy as np

from m4po.common.buffer import ObservationSpec

DEFAULT_FORGE_TASKS = (
    "Isaac-Forge-PegInsert-Direct-v0",
    "Isaac-Forge-GearMesh-Direct-v0",
    "Isaac-Forge-NutThread-Direct-v0",
)
DEFAULT_CARTPOLE_TASKS = (
    "Isaac-Cartpole-Direct-v0",
    "Isaac-Cartpole-v0",
)

_FORGE_SUCCESS_DEFINITION = "forge_task_completion"
_SURVIVAL_SUCCESS_DEFINITION = "survived_horizon_without_cart_or_pole_termination"


def _to_numpy(value: Any) -> np.ndarray:
    """Copy an array or tensor to a NumPy array without importing torch."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value)


def _flatten_group(value: Any, batch_size: int) -> np.ndarray:
    """Flatten a batched tensor or nested observation group deterministically."""

    if value is None:
        return np.empty((batch_size, 0), dtype=np.float32)
    if isinstance(value, Mapping):
        parts = [_flatten_group(item, batch_size) for item in value.values()]
        return (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((batch_size, 0), dtype=np.float32)
        )
    if isinstance(value, (tuple, list)):
        parts = [_flatten_group(item, batch_size) for item in value]
        return (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((batch_size, 0), dtype=np.float32)
        )

    array = _to_numpy(value)
    if array.ndim == 0 or array.shape[0] != batch_size:
        raise ValueError(
            "IsaacLab observations must have one leading row per environment; "
            f"expected {batch_size}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("IsaacLab observations must be numeric")
    array = np.asarray(array, dtype=np.float32).reshape(batch_size, -1)
    if not np.all(np.isfinite(array)):
        raise ValueError("IsaacLab observations must be finite")
    return array


def _pad_rows(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad a two-dimensional feature matrix and return its binary validity mask."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature matrix, got {array.shape}"
        )
    if width < array.shape[1]:
        raise ValueError(
            f"Feature width {array.shape[1]} exceeds configured padded width {width}"
        )
    output = np.zeros((array.shape[0], width), dtype=np.float32)
    mask = np.zeros_like(output)
    output[:, : array.shape[1]] = array
    mask[:, : array.shape[1]] = 1.0
    return output, mask


def _space_flat_dim(specification: Any) -> int:
    """Return a flat continuous-space size using only duck-typed metadata."""

    if specification is None:
        return 0
    if isinstance(specification, bool):
        raise TypeError("Boolean values are not valid space specifications")
    if isinstance(specification, (int, np.integer)):
        if int(specification) < 0:
            raise ValueError("Space dimensions must be non-negative")
        return int(specification)
    if isinstance(specification, Mapping):
        return sum(_space_flat_dim(value) for value in specification.values())
    if isinstance(specification, tuple):
        return sum(_space_flat_dim(value) for value in specification)
    if isinstance(specification, list) and all(
        isinstance(value, (int, np.integer)) for value in specification
    ):
        return int(np.prod(specification, dtype=np.int64))

    spaces = getattr(specification, "spaces", None)
    if isinstance(spaces, Mapping):
        return sum(_space_flat_dim(value) for value in spaces.values())
    if isinstance(spaces, (tuple, list)):
        return sum(_space_flat_dim(value) for value in spaces)
    shape = getattr(specification, "shape", None)
    if shape is not None:
        return int(np.prod(tuple(int(value) for value in shape), dtype=np.int64))
    raise ValueError(
        "M4PO's IsaacLab adapter requires continuous Box-like observation and action spaces"
    )


def _deterministic_contexts(names: Sequence[str], width: int) -> np.ndarray:
    """Create stable unit task features without a simulator or text-model dependency."""

    if width <= 0:
        raise ValueError("Task context width must be positive")
    contexts = np.empty((len(names), width), dtype=np.float32)
    for index, name in enumerate(names):
        digest = hashlib.sha256(str(name).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        values = np.random.default_rng(seed).standard_normal(width).astype(np.float32)
        contexts[index] = values / max(float(np.linalg.norm(values)), 1e-6)
    return contexts


@dataclass(frozen=True)
class _TaskSpec:
    registry_id: str
    embodiment: str
    action_dim: int
    policy_dim_hint: int
    state_dim_hint: int
    decimation: int
    simulation_dt: float
    episode_length_offset: int
    add_cartpole_pole_termination: bool
    success_definition: str
    policy_observation_permutation: tuple[int, ...]
    reward_scale: float

    @property
    def step_dt(self) -> float:
        return self.decimation * self.simulation_dt


_SUPPORTED_TASK_METADATA = {
    name: _TaskSpec(
        registry_id=name,
        embodiment="franka-forge",
        action_dim=7,
        policy_dim_hint=24,
        state_dim_hint=0,
        decimation=8,
        simulation_dt=1.0 / 120.0,
        episode_length_offset=1,
        add_cartpole_pole_termination=False,
        success_definition=_FORGE_SUCCESS_DEFINITION,
        policy_observation_permutation=tuple(range(24)),
        reward_scale=1.0,
    )
    for name in DEFAULT_FORGE_TASKS
}
_SUPPORTED_TASK_METADATA.update(
    {
        DEFAULT_CARTPOLE_TASKS[0]: _TaskSpec(
            registry_id=DEFAULT_CARTPOLE_TASKS[0],
            embodiment="cartpole",
            action_dim=1,
            policy_dim_hint=4,
            state_dim_hint=0,
            decimation=2,
            simulation_dt=1.0 / 120.0,
            episode_length_offset=1,
            add_cartpole_pole_termination=False,
            success_definition=_SURVIVAL_SUCCESS_DEFINITION,
            # DirectRLEnv already emits the canonical Cartpole order:
            # [pole_pos, pole_vel, cart_pos, cart_vel].
            policy_observation_permutation=(0, 1, 2, 3),
            # Its reward terms are rates, so integrate them over the outer step.
            reward_scale=1.0 / 60.0,
        ),
        DEFAULT_CARTPOLE_TASKS[1]: _TaskSpec(
            registry_id=DEFAULT_CARTPOLE_TASKS[1],
            embodiment="cartpole",
            action_dim=1,
            policy_dim_hint=4,
            state_dim_hint=0,
            decimation=2,
            simulation_dt=1.0 / 120.0,
            episode_length_offset=0,
            add_cartpole_pole_termination=True,
            success_definition=_SURVIVAL_SUCCESS_DEFINITION,
            # ManagerBasedRLEnv emits
            # [cart_pos, pole_pos, cart_vel, pole_vel].
            policy_observation_permutation=(1, 3, 0, 2),
            reward_scale=1.0,
        ),
    }
)
_WORKER_START_TIMEOUT_SECONDS = 600.0
_WORKER_START_ATTEMPTS = 3
_WORKER_START_RETRY_DELAY_SECONDS = 5.0
_WORKER_COMMAND_TIMEOUT_SECONDS = 300.0
_WORKER_CLOSE_TIMEOUT_SECONDS = 60.0
_WORKER_PROTOCOL_VERSION = 1


def _policy_observation(
    raw_observation: Any,
    num_envs: int,
    permutation: Sequence[int],
) -> np.ndarray:
    """Return a task observation in its verified canonical feature order."""

    if isinstance(raw_observation, Mapping) and "policy" in raw_observation:
        raw_observation = raw_observation["policy"]
    policy = _flatten_group(raw_observation, num_envs)
    indices = tuple(int(index) for index in permutation)
    if len(indices) != policy.shape[1] or sorted(indices) != list(
        range(policy.shape[1])
    ):
        raise ValueError(
            "IsaacLab policy observation permutation must contain every feature "
            f"index exactly once; shape={policy.shape}, permutation={indices}"
        )
    return policy[:, indices].copy()


def _scaled_rewards(
    raw_reward: Any, num_envs: int, reward_scale: float
) -> np.ndarray:
    """Scale a worker reward only after validating the raw and scaled values."""

    scale = float(reward_scale)
    if not math.isfinite(scale):
        raise ValueError("IsaacLab reward scale must be finite")
    raw_values = _to_numpy(raw_reward)
    if not np.issubdtype(raw_values.dtype, np.number):
        raise ValueError("IsaacLab rewards must be numeric")
    try:
        raw_values = raw_values.reshape(num_envs)
    except ValueError as exc:
        raise ValueError(
            "IsaacLab rewards must have one value per environment; "
            f"expected {num_envs}, got {raw_values.shape}"
        ) from exc
    if not np.all(np.isfinite(raw_values)):
        raise RuntimeError("IsaacLab returned non-finite raw rewards")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_values = np.asarray(raw_values, dtype=np.float32) * np.float32(scale)
    if not np.all(np.isfinite(scaled_values)):
        raise RuntimeError("IsaacLab reward scaling produced non-finite rewards")
    return scaled_values


def _current_raw_observation(unwrapped: Any) -> Any:
    """Read the pre-reset observation from either IsaacLab environment API."""

    get_observations = getattr(unwrapped, "_get_observations", None)
    if callable(get_observations):
        return get_observations()
    observation_manager = getattr(unwrapped, "observation_manager", None)
    compute_observations = getattr(observation_manager, "compute", None)
    if callable(compute_observations):
        return compute_observations(update_history=False)
    raise RuntimeError(
        "Supported IsaacLab tasks must expose either _get_observations() or an "
        "observation_manager"
    )


def _reset_success_buffers(unwrapped: Any, num_envs: int, torch: Any) -> None:
    reset_buffers = getattr(unwrapped, "_reset_buffers", None)
    if callable(reset_buffers):
        env_ids = torch.arange(num_envs, device=unwrapped.device, dtype=torch.long)
        reset_buffers(env_ids)
        return
    for name in ("ep_succeeded", "ep_success_times"):
        value = getattr(unwrapped, name, None)
        zero = getattr(value, "zero_", None)
        if callable(zero):
            zero()


def _forge_successes(unwrapped: Any, num_envs: int) -> np.ndarray:
    succeeded = getattr(unwrapped, "ep_succeeded", None)
    if succeeded is None:
        success_values = np.zeros(num_envs, dtype=np.bool_)
    else:
        success_values = _to_numpy(succeeded).astype(np.bool_, copy=True)

    get_successes = getattr(unwrapped, "_get_curr_successes", None)
    task_cfg = getattr(unwrapped, "cfg_task", None)
    if callable(get_successes) and task_cfg is not None:
        current = get_successes(
            success_threshold=task_cfg.success_threshold,
            check_rot=task_cfg.name == "nut_thread",
        )
        success_values |= _to_numpy(current).astype(np.bool_, copy=False)
    return success_values.reshape(num_envs)


def _survival_successes(unwrapped: Any, num_envs: int) -> np.ndarray:
    """Treat reaching the time limit without task termination as success."""

    time_outs = getattr(unwrapped, "reset_time_outs", None)
    terminated = getattr(unwrapped, "reset_terminated", None)
    if time_outs is None or terminated is None:
        raise RuntimeError(
            "The configured survival success metric requires IsaacLab termination "
            "and time-out buffers"
        )
    time_out_values = _to_numpy(time_outs).astype(np.bool_, copy=False)
    terminated_values = _to_numpy(terminated).astype(np.bool_, copy=False)
    if time_out_values.size != num_envs or terminated_values.size != num_envs:
        raise RuntimeError("IsaacLab termination buffers have an unexpected shape")
    return (
        time_out_values.reshape(num_envs) & ~terminated_values.reshape(num_envs)
    ).copy()


def _task_successes(
    unwrapped: Any, num_envs: int, success_definition: str
) -> np.ndarray:
    if success_definition == _FORGE_SUCCESS_DEFINITION:
        return _forge_successes(unwrapped, num_envs)
    if success_definition == _SURVIVAL_SUCCESS_DEFINITION:
        return _survival_successes(unwrapped, num_envs)
    raise ValueError(f"Unsupported IsaacLab success definition {success_definition!r}")


def _apply_task_overrides(task_cfg: Any, request: Mapping[str, Any]) -> None:
    """Apply explicit compatibility overrides from the verified task contract."""

    if not bool(request["add_cartpole_pole_termination"]):
        return

    from isaaclab.envs import mdp
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    terminations = getattr(task_cfg, "terminations", None)
    if terminations is None:
        raise RuntimeError(
            "The Cartpole pole-termination override requires a termination config"
        )
    if getattr(terminations, "pole_out_of_bounds", None) is not None:
        raise RuntimeError(
            "IsaacLab already defines pole_out_of_bounds; update the verified task "
            "contract instead of applying a duplicate override"
        )
    terminations.pole_out_of_bounds = TerminationTermCfg(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "bounds": (-math.pi / 2.0, math.pi / 2.0),
        },
    )


def _worker_message(sequence: int, message_type: str, **values: Any) -> dict[str, Any]:
    return {
        "protocol_version": _WORKER_PROTOCOL_VERSION,
        "sequence": int(sequence),
        "type": message_type,
        **values,
    }


def _validate_worker_command(
    command: Any, expected_sequence: int, expected_operation: str | None = None
) -> Mapping[str, Any]:
    if not isinstance(command, Mapping):
        raise TypeError("IsaacLab worker commands must be mappings")
    if command.get("protocol_version") != _WORKER_PROTOCOL_VERSION:
        raise ValueError("IsaacLab worker protocol version mismatch")
    if command.get("sequence") != expected_sequence:
        raise ValueError(
            "IsaacLab worker command sequence mismatch: "
            f"expected {expected_sequence}, got {command.get('sequence')!r}"
        )
    if (
        expected_operation is not None
        and command.get("operation") != expected_operation
    ):
        raise ValueError(
            f"Expected IsaacLab worker operation {expected_operation!r}, "
            f"got {command.get('operation')!r}"
        )
    return command


def _run_isaaclab_worker(
    connection: Any, request: Mapping[str, Any], app_launcher: Any
) -> None:
    """Own one SimulationApp and one Gym environment in an isolated process."""

    backend: Any | None = None
    sequence = 0
    try:
        # The entry module launched SimulationApp before importing this module.
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        registry_id = str(request["registry_id"])
        num_envs = int(request["num_envs"])
        policy_observation_permutation = tuple(
            int(index) for index in request["policy_observation_permutation"]
        )
        reward_scale = float(request["reward_scale"])
        task_cfg = parse_env_cfg(
            registry_id,
            device=str(request["device"]),
            num_envs=num_envs,
            use_fabric=bool(request["use_fabric"]),
        )
        _apply_task_overrides(task_cfg, request)
        # DirectRLEnv tasks time out one count before ``max_episode_length``;
        # ManagerBasedRLEnv's standard time-out term uses the full count. The
        # verified task contract records that API-specific offset so M4PO's
        # requested N is exactly N outer actions, including the terminal one.
        expected_backend_horizon = int(request["max_episode_steps"]) + int(
            request["episode_length_offset"]
        )
        task_cfg.episode_length_s = (
            expected_backend_horizon * float(task_cfg.sim.dt) * int(task_cfg.decimation)
        )
        task_cfg.sim.render_interval = int(task_cfg.decimation)
        backend = gym.make(registry_id, cfg=task_cfg, render_mode=None)
        unwrapped = backend.unwrapped
        if int(unwrapped.max_episode_length) != expected_backend_horizon:
            raise RuntimeError(
                "IsaacLab rounded the configured episode horizon unexpectedly: "
                f"expected {expected_backend_horizon}, got "
                f"{int(unwrapped.max_episode_length)}"
            )

        original_reset_idx = unwrapped._reset_idx
        capture_terminal = False
        terminal_policy: dict[int, np.ndarray] = {}
        terminal_success: dict[int, bool] = {}

        def reset_with_terminal_capture(instance: Any, env_ids: Any) -> Any:
            if capture_terminal:
                ids = _to_numpy(env_ids).astype(np.int64, copy=False).reshape(-1)
                policy = _policy_observation(
                    _current_raw_observation(instance),
                    num_envs,
                    policy_observation_permutation,
                )
                successes = _task_successes(
                    instance, num_envs, str(request["success_definition"])
                )
                for env_id in ids.tolist():
                    terminal_policy[int(env_id)] = policy[int(env_id)].copy()
                    terminal_success[int(env_id)] = bool(successes[int(env_id)])
            return original_reset_idx(env_ids)

        unwrapped._reset_idx = MethodType(reset_with_terminal_capture, unwrapped)

        def reset(seed: int) -> np.ndarray:
            nonlocal capture_terminal
            capture_terminal = False
            output = backend.reset(seed=int(seed))
            _reset_success_buffers(unwrapped, num_envs, torch)
            raw_observation = output[0] if isinstance(output, tuple) else output
            return _policy_observation(
                raw_observation, num_envs, policy_observation_permutation
            )

        initial_observation = reset(int(request["seed"]))
        connection.send(
            _worker_message(
                sequence,
                "ready",
                registry_id=registry_id,
                observation=initial_observation,
                policy_dim=int(initial_observation.shape[1]),
                action_dim=_space_flat_dim(unwrapped.single_action_space),
                decimation=int(task_cfg.decimation),
                simulation_dt=float(task_cfg.sim.dt),
                episode_length_offset=int(request["episode_length_offset"]),
                backend_horizon=int(unwrapped.max_episode_length),
                add_cartpole_pole_termination=bool(
                    request["add_cartpole_pole_termination"]
                ),
                success_definition=str(request["success_definition"]),
                policy_observation_permutation=policy_observation_permutation,
                reward_scale=reward_scale,
            )
        )

        while True:
            try:
                command = connection.recv()
            except EOFError:
                break
            sequence += 1
            command = _validate_worker_command(command, sequence)
            operation = command.get("operation")
            if operation == "close":
                connection.send(_worker_message(sequence, "closed"))
                break
            if operation == "reset":
                observation = reset(int(command["seed"]))
                connection.send(
                    _worker_message(sequence, "reset", observation=observation)
                )
                continue
            if operation != "step":
                raise ValueError(f"Unknown IsaacLab worker operation {operation!r}")

            action_array = np.asarray(command["actions"], dtype=np.float32)
            local_actions = torch.as_tensor(
                action_array, dtype=torch.float32, device=unwrapped.device
            )
            terminal_policy.clear()
            terminal_success.clear()
            capture_terminal = True
            try:
                raw_observation, reward, terminated, truncated, _ = backend.step(
                    local_actions
                )
            finally:
                capture_terminal = False

            connection.send(
                _worker_message(
                    sequence,
                    "step",
                    observation=_policy_observation(
                        raw_observation, num_envs, policy_observation_permutation
                    ),
                    reward=_scaled_rewards(reward, num_envs, reward_scale),
                    terminated=_to_numpy(terminated).astype(np.bool_, copy=True),
                    truncated=_to_numpy(truncated).astype(np.bool_, copy=True),
                    terminal_policy=[
                        terminal_policy.get(env_id) for env_id in range(num_envs)
                    ],
                    terminal_success=np.asarray(
                        [
                            terminal_success.get(env_id, False)
                            for env_id in range(num_envs)
                        ],
                        dtype=np.bool_,
                    ),
                )
            )
    except Exception as exc:  # noqa: BLE001 - propagate remote simulator failures.
        try:
            connection.send(
                _worker_message(
                    sequence,
                    "error",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"IsaacLab worker environment close failed: {exc}", file=sys.stderr
                )
        try:
            app_launcher.app.close()
        except Exception as exc:  # noqa: BLE001
            print(f"IsaacLab worker application close failed: {exc}", file=sys.stderr)
        try:
            connection.close()
        except OSError:
            pass


class IsaacLabPrebuiltEnv:
    """Sequential adapter for compatible prebuilt IsaacLab vector tasks.

    IsaacLab owns one global simulation context per process and cannot reliably
    recreate another task after closing that context. The parent therefore
    keeps one simulator subprocess active for an entire fresh M4PO rollout
    and fully replaces the process when the balanced sampler changes tasks.
    """

    sequential_evaluation = True

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.max_episode_steps = int(cfg.max_episode_steps)
        if self.num_envs <= 0 or self.max_episode_steps <= 0:
            raise ValueError("IsaacLab num_envs and max_episode_steps must be positive")

        configured_tasks = list(cfg.task_names)
        if not configured_tasks:
            configured_tasks = list(DEFAULT_FORGE_TASKS)
        self.task_names = [str(name).strip() for name in configured_tasks]
        if any(not name for name in self.task_names):
            raise ValueError("IsaacLab task registration names must be non-empty")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("IsaacLab task registration names must be unique")
        self.num_tasks = len(self.task_names)

        configured_embodiments = list(cfg.embodiment_names)
        if len(configured_embodiments) != 1:
            raise ValueError(
                "The sequential prebuilt adapter currently requires one common embodiment"
            )
        self.embodiment_names = [str(configured_embodiments[0]).strip()]
        if not self.embodiment_names[0]:
            raise ValueError("IsaacLab embodiment name must be non-empty")
        self.num_embodiments = 1
        self.task_contexts = _deterministic_contexts(
            self.task_names, int(cfg.task_context_dim)
        )

        self._base_seed = int(cfg.seed)
        self._assignment_rng = np.random.default_rng(self._base_seed + 524_287)
        self._assignment_deck = np.empty(0, dtype=np.int64)
        self._assignment_cursor = 0
        self._rollout_generation = 0
        self._rollout_started = False
        self._reset_count = 0
        self._requested_task_id: int | None = None
        self._active_task_id = 0
        self._worker_task_id: int | None = None
        self._worker_process: Any | None = None
        self._worker_connection: Any | None = None
        self._worker_sequence = 0
        self._pending_observation: np.ndarray | None = None
        self._closed = False

        self._device = str(getattr(cfg, "isaaclab_device", "cuda:0"))
        if self._device in {"", "auto", "None"}:
            self._device = "cuda:0"
        self._headless = bool(getattr(cfg, "isaaclab_headless", True))
        self._enable_cameras = bool(getattr(cfg, "isaaclab_enable_cameras", False))
        self._use_fabric = bool(getattr(cfg, "isaaclab_use_fabric", True))
        if bool(getattr(cfg, "use_images", False)) or self._enable_cameras:
            raise ValueError(
                "The built-in prebuilt adapter currently supports policy-vector "
                "observations only; disable use_images and isaaclab_enable_cameras"
            )

        unsupported = sorted(set(self.task_names) - set(_SUPPORTED_TASK_METADATA))
        if unsupported:
            raise ValueError(
                "The built-in prebuilt adapter has no verified contract for "
                f"these IsaacLab tasks: {unsupported}"
            )
        self._task_specs = [_SUPPORTED_TASK_METADATA[name] for name in self.task_names]
        task_embodiments = {spec.embodiment for spec in self._task_specs}
        if len(task_embodiments) != 1:
            raise ValueError(
                "Configured IsaacLab tasks do not share one supported embodiment"
            )
        expected_embodiment = next(iter(task_embodiments))
        if self.embodiment_names != [expected_embodiment]:
            raise ValueError(
                "Configured IsaacLab embodiment does not match the task contract: "
                f"expected {expected_embodiment!r}, got {self.embodiment_names[0]!r}"
            )
        task_control_rates = {
            (spec.decimation, spec.simulation_dt) for spec in self._task_specs
        }
        if len(task_control_rates) != 1:
            raise ValueError(
                "Configured IsaacLab tasks must share one control rate for a common "
                "embodiment"
            )
        self._proprio_dim = max(spec.policy_dim_hint for spec in self._task_specs)
        self._state_dim = max(spec.state_dim_hint for spec in self._task_specs)
        self.action_dim = max(spec.action_dim for spec in self._task_specs)
        self.observation_spec = ObservationSpec(
            image_shape=(0, 0, 0),
            proprio_dim=self._proprio_dim,
            state_dim=self._state_dim,
        )
        self.observation_shapes = self.observation_spec.shapes
        self.control_repeats = np.asarray(
            [self._task_specs[0].decimation], dtype=np.int64
        )
        # This reports physics substeps already aggregated inside DirectRLEnv;
        # the parent never repeats an outer M4PO action.
        self.low_level_discounts = np.asarray(
            [float(cfg.discount) ** (1.0 / self.control_repeats[0])],
            dtype=np.float32,
        )

        try:
            self._activate_task(self._next_task_id())
            self._set_metadata(self._active_task_id)
        except BaseException:
            self.close()
            raise

    def _next_reset_seed(self) -> int:
        seed = self._base_seed + 1_000_003 * self._active_task_id + self._reset_count
        self._reset_count += 1
        return seed

    def _receive_worker(
        self, timeout: float, operation: str, expected_sequence: int
    ) -> Mapping[str, Any]:
        connection = self._worker_connection
        process = self._worker_process
        if connection is None or process is None:
            raise RuntimeError("IsaacLab simulator worker is not running")
        if not connection.poll(timeout):
            exit_code = process.poll()
            raise TimeoutError(
                f"IsaacLab worker timed out during {operation} after {timeout:g}s; "
                f"exit_code={exit_code}"
            )
        try:
            response = connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"IsaacLab worker exited during {operation}; exit_code={process.poll()}"
            ) from exc
        if not isinstance(response, Mapping):
            raise TypeError("IsaacLab worker returned a malformed response")
        if response.get("protocol_version") != _WORKER_PROTOCOL_VERSION:
            raise RuntimeError("IsaacLab worker response protocol version mismatch")
        if response.get("sequence") != expected_sequence:
            raise RuntimeError(
                "IsaacLab worker response sequence mismatch: "
                f"expected {expected_sequence}, got {response.get('sequence')!r}"
            )
        if response.get("type") == "error":
            details = str(
                response.get("traceback", response.get("error", "unknown error"))
            )
            raise RuntimeError(f"IsaacLab worker failed during {operation}:\n{details}")
        return response

    def _request_worker(
        self,
        operation: str,
        *,
        timeout: float = _WORKER_COMMAND_TIMEOUT_SECONDS,
        **values: Any,
    ) -> Mapping[str, Any]:
        if self._worker_connection is None:
            raise RuntimeError("IsaacLab simulator worker is not running")
        self._worker_sequence += 1
        sequence = self._worker_sequence
        try:
            self._worker_connection.send(
                {
                    "protocol_version": _WORKER_PROTOCOL_VERSION,
                    "sequence": sequence,
                    "operation": operation,
                    **values,
                }
            )
            return self._receive_worker(timeout, operation, sequence)
        except (BrokenPipeError, EOFError, OSError) as exc:
            # A timed-out or malformed RPC leaves message ordering unknown.
            # Poison this worker instead of allowing a stale reply to satisfy
            # a later command.
            self._stop_worker(graceful=False)
            raise RuntimeError(
                f"IsaacLab worker connection failed during {operation}"
            ) from exc
        except Exception:
            self._stop_worker(graceful=False)
            raise

    def _stop_worker(self, *, graceful: bool = True) -> None:
        connection = self._worker_connection
        process = self._worker_process
        self._worker_connection = None
        self._worker_process = None
        self._worker_task_id = None
        sequence = self._worker_sequence + 1
        self._worker_sequence = 0
        self._pending_observation = None
        close_acknowledged = False
        graceful_deadline = time.monotonic() + _WORKER_CLOSE_TIMEOUT_SECONDS

        def worker_is_running() -> bool:
            if process is None:
                return False
            try:
                return process.poll() is None
            except Exception:  # noqa: BLE001 - still attempt terminate and kill.
                return True

        try:
            if process is None:
                return

            if graceful and connection is not None and worker_is_running():
                try:
                    connection.send(
                        {
                            "protocol_version": _WORKER_PROTOCOL_VERSION,
                            "sequence": sequence,
                            "operation": "close",
                        }
                    )
                    remaining = max(0.0, graceful_deadline - time.monotonic())
                    if connection.poll(remaining):
                        response = connection.recv()
                        close_acknowledged = bool(
                            isinstance(response, Mapping)
                            and response.get("protocol_version")
                            == _WORKER_PROTOCOL_VERSION
                            and response.get("sequence") == sequence
                            and response.get("type") == "closed"
                        )
                except Exception:  # noqa: BLE001 - teardown must remain best-effort.
                    close_acknowledged = False

            if close_acknowledged and worker_is_running():
                try:
                    process.wait(timeout=max(0.0, graceful_deadline - time.monotonic()))
                except Exception:  # noqa: BLE001,S110 - fall through to termination.
                    pass

            if worker_is_running():
                try:
                    process.terminate()
                except Exception:  # noqa: BLE001,S110 - continue to kill fallback.
                    pass
                try:
                    process.wait(timeout=10.0)
                except Exception:  # noqa: BLE001,S110 - continue to kill fallback.
                    pass
            if worker_is_running():
                try:
                    process.kill()
                except Exception:  # noqa: BLE001,S110 - cleanup remains best-effort.
                    pass
                try:
                    process.wait(timeout=10.0)
                except Exception:  # noqa: BLE001,S110 - cleanup remains best-effort.
                    pass
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def _start_worker(self, task_id: int) -> None:
        request = {
            "registry_id": self.task_names[task_id],
            "num_envs": self.num_envs,
            "max_episode_steps": self.max_episode_steps,
            "seed": self._next_reset_seed(),
            "device": self._device,
            "headless": self._headless,
            "enable_cameras": self._enable_cameras,
            "use_fabric": self._use_fabric,
            "episode_length_offset": self._task_specs[task_id].episode_length_offset,
            "add_cartpole_pole_termination": self._task_specs[
                task_id
            ].add_cartpole_pole_termination,
            "success_definition": self._task_specs[task_id].success_definition,
            "policy_observation_permutation": self._task_specs[
                task_id
            ].policy_observation_permutation,
            "reward_scale": self._task_specs[task_id].reward_scale,
        }
        for attempt in range(1, _WORKER_START_ATTEMPTS + 1):
            try:
                self._start_worker_attempt(task_id, request)
                return
            except Exception as exc:
                if attempt == _WORKER_START_ATTEMPTS:
                    raise RuntimeError(
                        "IsaacLab worker failed to start after "
                        f"{_WORKER_START_ATTEMPTS} attempts"
                    ) from exc
                print(
                    "IsaacLab worker startup attempt "
                    f"{attempt}/{_WORKER_START_ATTEMPTS} failed; retrying in "
                    f"{_WORKER_START_RETRY_DELAY_SECONDS:g}s:\n{exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(_WORKER_START_RETRY_DELAY_SECONDS)

    def _start_worker_attempt(self, task_id: int, request: Mapping[str, Any]) -> None:
        parent_socket, child_socket = socket.socketpair()
        try:
            child_fd = child_socket.fileno()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "m4po._isaaclab_worker",
                    "--connection-fd",
                    str(child_fd),
                ],
                close_fds=True,
                pass_fds=(child_fd,),
            )
            self._worker_process = process
            self._worker_task_id = task_id
            self._worker_sequence = 0
            child_socket.close()
            parent_connection = Connection(parent_socket.detach())
            self._worker_connection = parent_connection
            parent_connection.send(
                {
                    "protocol_version": _WORKER_PROTOCOL_VERSION,
                    "sequence": 0,
                    "operation": "bootstrap",
                    "request": request,
                }
            )
            response = self._receive_worker(_WORKER_START_TIMEOUT_SECONDS, "startup", 0)
            if response.get("type") != "ready":
                raise RuntimeError("IsaacLab worker did not send its ready handshake")
            spec = self._task_specs[task_id]
            actual = {
                "registry_id": str(response["registry_id"]),
                "action_dim": int(response["action_dim"]),
                "policy_dim": int(response["policy_dim"]),
                "decimation": int(response["decimation"]),
                "simulation_dt": float(response["simulation_dt"]),
                "episode_length_offset": int(response["episode_length_offset"]),
                "backend_horizon": int(response["backend_horizon"]),
                "add_cartpole_pole_termination": bool(
                    response["add_cartpole_pole_termination"]
                ),
                "success_definition": str(response["success_definition"]),
                "policy_observation_permutation": tuple(
                    int(index)
                    for index in response["policy_observation_permutation"]
                ),
                "reward_scale": float(response["reward_scale"]),
            }
            expected = {
                "registry_id": spec.registry_id,
                "action_dim": spec.action_dim,
                "policy_dim": spec.policy_dim_hint,
                "decimation": spec.decimation,
                "simulation_dt": spec.simulation_dt,
                "episode_length_offset": spec.episode_length_offset,
                "backend_horizon": self.max_episode_steps + spec.episode_length_offset,
                "add_cartpole_pole_termination": (spec.add_cartpole_pole_termination),
                "success_definition": spec.success_definition,
                "policy_observation_permutation": (
                    spec.policy_observation_permutation
                ),
                "reward_scale": spec.reward_scale,
            }
            if actual != expected:
                raise RuntimeError(
                    "IsaacLab task interface differs from the supported contract: "
                    f"expected={expected}, actual={actual}"
                )
            observation = np.asarray(response["observation"], dtype=np.float32)
            expected_observation_shape = (
                self.num_envs,
                spec.policy_dim_hint,
            )
            if observation.shape != expected_observation_shape:
                raise RuntimeError(
                    "IsaacLab worker returned an unexpected initial observation shape "
                    f"{observation.shape}; expected {expected_observation_shape}"
                )
            self._pending_observation = observation.copy()
        except BaseException:
            self._stop_worker(graceful=False)
            raise
        finally:
            parent_socket.close()
            child_socket.close()

    def _activate_task(self, task_id: int) -> None:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} is out of range")
        self._active_task_id = task_id
        if (
            self._worker_process is not None
            and self._worker_process.poll() is None
            and self._worker_task_id == task_id
        ):
            self._set_metadata(task_id)
            return

        self._stop_worker()
        self._start_worker(task_id)
        self._set_metadata(task_id)

    def _set_metadata(self, task_id: int) -> None:
        self.task_ids = np.full(self.num_envs, int(task_id), dtype=np.int64)
        self.embodiment_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.action_masks = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        local_action_dim = self._task_specs[int(task_id)].action_dim
        self.action_masks[:, :local_action_dim] = 1.0

    def _canonical_observation(self, raw_observation: Any) -> dict[str, np.ndarray]:
        policy = _flatten_group(raw_observation, self.num_envs)
        proprio, proprio_mask = _pad_rows(policy, self._proprio_dim)
        return {
            "image": np.empty((self.num_envs, 0, 0, 0), dtype=np.float32),
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "state": np.empty((self.num_envs, 0), dtype=np.float32),
            "state_mask": np.empty((self.num_envs, 0), dtype=np.float32),
        }

    def _terminal_observation(self, policy: Any) -> dict[str, np.ndarray]:
        policy_array = np.asarray(policy, dtype=np.float32).reshape(-1)
        if policy_array.size > self._proprio_dim:
            raise ValueError("Terminal policy observation exceeds the padded width")
        proprio = np.zeros(self._proprio_dim, dtype=np.float32)
        proprio_mask = np.zeros(self._proprio_dim, dtype=np.float32)
        proprio[: policy_array.size] = policy_array
        proprio_mask[: policy_array.size] = 1.0
        return {
            "image": np.empty((0, 0, 0), dtype=np.float32),
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "state": np.empty(0, dtype=np.float32),
            "state_mask": np.empty(0, dtype=np.float32),
        }

    def _reset_backend(self) -> Any:
        if self._pending_observation is not None:
            observation = self._pending_observation
            self._pending_observation = None
            return observation
        response = self._request_worker("reset", seed=self._next_reset_seed())
        if response.get("type") != "reset":
            raise RuntimeError("IsaacLab worker returned a malformed reset response")
        return response["observation"]

    def _refill_assignment_deck(self) -> None:
        self._assignment_deck = self._assignment_rng.permutation(self.num_tasks).astype(
            np.int64, copy=False
        )
        self._assignment_cursor = 0

    def _next_task_id(self) -> int:
        if self._assignment_cursor >= len(self._assignment_deck):
            self._refill_assignment_deck()
        task_id = int(self._assignment_deck[self._assignment_cursor])
        self._assignment_cursor += 1
        return task_id

    def reset(self) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed IsaacLab environment")
        task_id = (
            self._active_task_id
            if self._requested_task_id is None
            else self._requested_task_id
        )
        self._activate_task(task_id)
        return self._canonical_observation(self._reset_backend())

    def reset_rollout(self) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed IsaacLab environment")
        task_id = (
            (
                self._active_task_id
                if not self._rollout_started
                else self._next_task_id()
            )
            if self._requested_task_id is None
            else self._requested_task_id
        )
        self._rollout_started = True
        self._rollout_generation += 1
        self._activate_task(task_id)
        return self._canonical_observation(self._reset_backend())

    def set_evaluation_pair(self, task_id: int, embodiment_id: int) -> None:
        """Select the sole task/embodiment pair used by the next reset."""

        task_id = int(task_id)
        embodiment_id = int(embodiment_id)
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError(f"Evaluation task ID {task_id} is out of range")
        if embodiment_id != 0:
            raise ValueError("The prebuilt adapter exposes only embodiment ID 0")
        self._requested_task_id = task_id
        self._activate_task(task_id)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._closed or self._worker_process is None:
            raise RuntimeError("Cannot step a closed IsaacLab environment")
        action_array = np.asarray(actions, dtype=np.float32)
        expected_shape = (self.num_envs, self.action_dim)
        if action_array.shape != expected_shape:
            raise ValueError(
                f"Expected shared action shape {expected_shape}, got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("IsaacLab actions must be finite")
        action_array = np.clip(action_array, -1.0, 1.0) * self.action_masks
        local_dim = self._task_specs[self._active_task_id].action_dim

        response = self._request_worker(
            "step", actions=action_array[:, :local_dim].copy()
        )
        if response.get("type") != "step":
            raise RuntimeError("IsaacLab worker returned a malformed step response")
        reward_array = np.asarray(response["reward"], dtype=np.float32).reshape(
            self.num_envs
        )
        if not np.all(np.isfinite(reward_array)):
            raise RuntimeError("IsaacLab worker returned non-finite rewards")
        terminated_array = np.asarray(response["terminated"], dtype=np.bool_).reshape(
            self.num_envs
        )
        truncated_array = np.asarray(response["truncated"], dtype=np.bool_).reshape(
            self.num_envs
        )
        done = terminated_array | truncated_array
        observation = self._canonical_observation(response["observation"])
        terminal_policies = list(response["terminal_policy"])
        terminal_success = np.asarray(
            response["terminal_success"], dtype=np.bool_
        ).reshape(self.num_envs)
        if len(terminal_policies) != self.num_envs:
            raise RuntimeError(
                "IsaacLab worker returned malformed terminal observations"
            )

        infos: list[dict[str, Any]] = []
        for env_id in range(self.num_envs):
            info: dict[str, Any] = {
                "task_id": self._active_task_id,
                "task_name": self.task_names[self._active_task_id],
                "embodiment_id": 0,
                "embodiment_name": self.embodiment_names[0],
                "success": bool(terminal_success[env_id]),
                "terminated": bool(terminated_array[env_id]),
                "truncated": bool(truncated_array[env_id]),
            }
            if done[env_id]:
                terminal_policy = terminal_policies[env_id]
                if terminal_policy is None:
                    raise RuntimeError(
                        "IsaacLab reset a completed environment before its terminal "
                        "observation could be captured"
                    )
                info["terminal_observation"] = self._terminal_observation(
                    terminal_policy
                )
            infos.append(info)
        return observation, reward_array.copy(), done.copy(), infos

    def rollout_state_dict(self) -> dict[str, Any]:
        """Save only boundary sampler state; partial simulator state is discarded."""

        return {
            "schema_version": 2,
            "task_names": list(self.task_names),
            "assignment_rng_state": deepcopy(self._assignment_rng.bit_generator.state),
            "assignment_deck": self._assignment_deck.copy(),
            "assignment_cursor": int(self._assignment_cursor),
            "rollout_generation": int(self._rollout_generation),
            "rollout_started": bool(self._rollout_started),
            "reset_count": int(self._reset_count),
        }

    def load_rollout_state_dict(self, state: Any) -> None:
        if not isinstance(state, Mapping) or state.get("schema_version") != 2:
            raise ValueError("Invalid IsaacLab rollout sampler state")
        if list(state.get("task_names", [])) != self.task_names:
            raise ValueError("IsaacLab rollout state has different ordered tasks")
        try:
            deck = np.asarray(state["assignment_deck"], dtype=np.int64).reshape(-1)
            cursor = int(state["assignment_cursor"])
            generation = int(state["rollout_generation"])
            rollout_started = bool(state["rollout_started"])
            reset_count = int(state["reset_count"])
            rng_state = deepcopy(state["assignment_rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid IsaacLab rollout sampler state fields") from exc
        if cursor < 0 or cursor > len(deck) or generation < 0 or reset_count < 0:
            raise ValueError("Invalid IsaacLab rollout sampler counters")
        if deck.size:
            if deck.shape != (self.num_tasks,) or not np.array_equal(
                np.sort(deck), np.arange(self.num_tasks, dtype=np.int64)
            ):
                raise ValueError("Invalid IsaacLab rollout task deck")
        elif cursor != 0 or generation != 0 or rollout_started:
            raise ValueError("An empty IsaacLab rollout task deck must be pristine")
        try:
            self._assignment_rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid IsaacLab rollout sampler RNG state") from exc
        self._assignment_deck = deck.copy()
        self._assignment_cursor = cursor
        self._rollout_generation = generation
        self._rollout_started = rollout_started
        self._reset_count = reset_count
        # The constructor's initial worker used a pre-restore seed. Force the
        # next boundary reset even if the restored sampler selects that task.
        self._pending_observation = None

    state_dict = rollout_state_dict
    load_state_dict = load_rollout_state_dict

    def environment_signature(self) -> dict[str, Any]:
        try:
            isaaclab_version = importlib.metadata.version("isaaclab")
        except importlib.metadata.PackageNotFoundError:
            isaaclab_version = None
        return {
            "kind": "isaaclab_sequential_prebuilt_subprocess",
            "isaaclab_version": isaaclab_version,
            "tasks": [
                {
                    "registry_id": spec.registry_id,
                    "embodiment": spec.embodiment,
                    "action_dim": spec.action_dim,
                    "policy_dim_hint": spec.policy_dim_hint,
                    "state_dim_hint": spec.state_dim_hint,
                    "decimation": spec.decimation,
                    "simulation_dt": spec.simulation_dt,
                    "step_dt": spec.step_dt,
                    "episode_length_offset": spec.episode_length_offset,
                    "add_cartpole_pole_termination": (
                        spec.add_cartpole_pole_termination
                    ),
                    "success_definition": spec.success_definition,
                    "policy_observation_permutation": list(
                        spec.policy_observation_permutation
                    ),
                    "reward_scale": spec.reward_scale,
                }
                for spec in self._task_specs
            ],
            "shared_action_dim": self.action_dim,
            "proprio_dim": self._proprio_dim,
            "state_dim": self._state_dim,
            "headless": self._headless,
            "enable_cameras": self._enable_cameras,
            "use_fabric": self._use_fabric,
            "worker_start_method": "subprocess_socketpair",
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_worker()


def make_isaaclab_prebuilt_env(cfg: Any) -> IsaacLabPrebuiltEnv:
    """Factory entry point for M4PO's lazy IsaacLab loader."""

    return IsaacLabPrebuiltEnv(cfg)


__all__ = [
    "DEFAULT_CARTPOLE_TASKS",
    "DEFAULT_FORGE_TASKS",
    "IsaacLabPrebuiltEnv",
    "make_isaaclab_prebuilt_env",
]
