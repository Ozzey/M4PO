from __future__ import annotations

import numpy as np
import pytest

import m4po.envs.isaaclab_prebuilt as prebuilt_module
from m4po.common.config import M4POConfig
from m4po.envs.isaaclab_prebuilt import (
    DEFAULT_CARTPOLE_TASKS,
    IsaacLabPrebuiltEnv,
    _current_raw_observation,
    _deterministic_contexts,
    _flatten_group,
    _pad_rows,
    _policy_observation,
    _scaled_rewards,
    _space_flat_dim,
    _survival_successes,
    _validate_worker_command,
)


def test_flatten_nested_observation_group_preserves_mapping_order() -> None:
    group = {
        "position": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "nested": (
            np.asarray([[5.0], [6.0]], dtype=np.float32),
            {"force": np.asarray([[7.0, 8.0], [9.0, 10.0]])},
        ),
    }
    flattened = _flatten_group(group, batch_size=2)
    np.testing.assert_array_equal(
        flattened,
        np.asarray(
            [[1.0, 2.0, 5.0, 7.0, 8.0], [3.0, 4.0, 6.0, 9.0, 10.0]],
            dtype=np.float32,
        ),
    )


def test_pad_rows_returns_zero_padding_and_binary_mask() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    padded, mask = _pad_rows(values, 4)
    np.testing.assert_array_equal(
        padded,
        np.asarray([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        mask,
        np.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="exceeds"):
        _pad_rows(values, 1)


def test_cartpole_policy_observations_share_one_canonical_order() -> None:
    canonical_reset = np.asarray(
        [[0.1, 0.2, 0.3, 0.4], [1.1, 1.2, 1.3, 1.4]], dtype=np.float32
    )
    canonical_terminal = canonical_reset + 10.0

    for canonical in (canonical_reset, canonical_terminal):
        direct_raw = {"policy": canonical}
        manager_raw = {
            "policy": canonical[:, [2, 0, 3, 1]],
        }
        np.testing.assert_array_equal(
            _policy_observation(direct_raw, 2, (0, 1, 2, 3)), canonical
        )
        np.testing.assert_array_equal(
            _policy_observation(manager_raw, 2, (1, 3, 0, 2)), canonical
        )


def test_policy_observation_rejects_invalid_permutation() -> None:
    raw = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="every feature index exactly once"):
        _policy_observation(raw, 2, (0, 1, 1, 3))
    with pytest.raises(ValueError, match="every feature index exactly once"):
        _policy_observation(raw, 2, (0, 1, 2))


def test_cartpole_reward_contract_aligns_direct_and_manager_scales() -> None:
    direct_raw = np.asarray([60.0, -30.0], dtype=np.float32)
    manager_raw = np.asarray([1.0, -0.5], dtype=np.float32)
    np.testing.assert_allclose(
        _scaled_rewards(direct_raw, 2, 1.0 / 60.0),
        manager_raw,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(_scaled_rewards(manager_raw, 2, 1.0), manager_raw)


@pytest.mark.parametrize(
    ("raw_reward", "reward_scale", "error"),
    [
        (np.asarray([np.inf]), 1.0, "non-finite raw rewards"),
        (np.asarray([np.finfo(np.float64).max]), 1.0, "produced non-finite"),
        (np.asarray([1.0]), np.inf, "scale must be finite"),
    ],
)
def test_reward_scaling_rejects_non_finite_values(
    raw_reward: np.ndarray, reward_scale: float, error: str
) -> None:
    with pytest.raises((RuntimeError, ValueError), match=error):
        _scaled_rewards(raw_reward, 1, reward_scale)


def test_space_flat_dim_handles_continuous_nested_specs() -> None:
    assert _space_flat_dim(7) == 7
    assert _space_flat_dim([2, 3]) == 6
    assert _space_flat_dim({"a": 2, "b": ([3], 4)}) == 9


def test_task_contexts_are_stable_distinct_and_normalized() -> None:
    names = ["peg insert", "gear mesh", "nut thread"]
    first = _deterministic_contexts(names, 16)
    second = _deterministic_contexts(names, 16)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert len({row.tobytes() for row in first}) == len(names)


def test_current_raw_observation_supports_direct_and_manager_apis() -> None:
    direct_observation = {"policy": np.asarray([[1.0, 2.0]], dtype=np.float32)}

    class DirectEnvironment:
        def _get_observations(self):
            return direct_observation

    assert _current_raw_observation(DirectEnvironment()) is direct_observation

    manager_observation = {"policy": np.asarray([[3.0, 4.0]], dtype=np.float32)}

    class ObservationManager:
        def __init__(self) -> None:
            self.update_history: bool | None = None

        def compute(self, *, update_history: bool):
            self.update_history = update_history
            return manager_observation

    class ManagerEnvironment:
        def __init__(self) -> None:
            self.observation_manager = ObservationManager()

    manager_environment = ManagerEnvironment()
    assert _current_raw_observation(manager_environment) is manager_observation
    assert manager_environment.observation_manager.update_history is False


def test_survival_success_requires_timeout_without_termination() -> None:
    class Environment:
        reset_time_outs = np.asarray([True, True, False, False])
        reset_terminated = np.asarray([False, True, True, False])

    np.testing.assert_array_equal(
        _survival_successes(Environment(), 4),
        np.asarray([True, False, False, False]),
    )


def _cartpole_config() -> M4POConfig:
    cfg = M4POConfig.from_yaml("m4po/configs/isaaclab_forge_smoke.yaml")
    cfg.task = DEFAULT_CARTPOLE_TASKS[0]
    cfg.tasks = ",".join(DEFAULT_CARTPOLE_TASKS)
    cfg.embodiment = "cartpole"
    cfg.num_tasks = len(DEFAULT_CARTPOLE_TASKS)
    cfg.proprio_dim = 4
    cfg.control_decimation = 2
    cfg.control_repeats = "2"
    return cfg


def test_cartpole_tasks_expose_one_verified_shared_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(IsaacLabPrebuiltEnv, "_activate_task", lambda *_args: None)
    adapter = IsaacLabPrebuiltEnv(_cartpole_config())
    try:
        assert adapter.task_names == list(DEFAULT_CARTPOLE_TASKS)
        assert adapter.embodiment_names == ["cartpole"]
        assert adapter.action_dim == 1
        assert adapter.observation_spec.proprio_dim == 4
        np.testing.assert_array_equal(adapter.control_repeats, [2])
        signature = adapter.environment_signature()
        assert {task["success_definition"] for task in signature["tasks"]} == {
            "survived_horizon_without_cart_or_pole_termination"
        }
        assert {task["embodiment"] for task in signature["tasks"]} == {"cartpole"}
        assert {
            task["registry_id"]: task["episode_length_offset"]
            for task in signature["tasks"]
        } == {
            "Isaac-Cartpole-Direct-v0": 1,
            "Isaac-Cartpole-v0": 0,
        }
        assert {
            task["registry_id"]: task["add_cartpole_pole_termination"]
            for task in signature["tasks"]
        } == {
            "Isaac-Cartpole-Direct-v0": False,
            "Isaac-Cartpole-v0": True,
        }
        assert {
            task["registry_id"]: task["policy_observation_permutation"]
            for task in signature["tasks"]
        } == {
            "Isaac-Cartpole-Direct-v0": [0, 1, 2, 3],
            "Isaac-Cartpole-v0": [1, 3, 0, 2],
        }
        assert {
            task["registry_id"]: task["reward_scale"]
            for task in signature["tasks"]
        } == pytest.approx(
            {
                "Isaac-Cartpole-Direct-v0": 1.0 / 60.0,
                "Isaac-Cartpole-v0": 1.0,
            }
        )
    finally:
        adapter.close()


def test_cartpole_contract_rejects_mislabeled_embodiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(IsaacLabPrebuiltEnv, "_activate_task", lambda *_args: None)
    cfg = _cartpole_config()
    cfg.embodiment = "franka-forge"
    with pytest.raises(ValueError, match="does not match the task contract"):
        IsaacLabPrebuiltEnv(cfg)


class _FakeConnection:
    def __init__(self, response: object | None = None, *, fail_send: bool = False):
        self.response = response
        self.fail_send = fail_send
        self.sent: list[object] = []
        self.closed = False

    def send(self, message: object) -> None:
        if self.fail_send:
            raise BrokenPipeError("closed")
        self.sent.append(message)

    def poll(self, _timeout: float) -> bool:
        return self.response is not None

    def recv(self) -> object:
        return self.response

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self):
        self.return_code: int | None = None
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        if self.return_code is None:
            raise TimeoutError("still running")
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9


def _adapter_with_fake_worker(
    connection: _FakeConnection, process: _FakeProcess
) -> IsaacLabPrebuiltEnv:
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter._worker_connection = connection
    adapter._worker_process = process
    adapter._worker_task_id = 0
    adapter._worker_sequence = 4
    adapter._pending_observation = np.ones((1, 1), dtype=np.float32)
    return adapter


def test_worker_protocol_rejects_stale_commands() -> None:
    command = {"protocol_version": 1, "sequence": 4, "operation": "step"}
    assert _validate_worker_command(command, 4, "step") is command
    with pytest.raises(ValueError, match="sequence mismatch"):
        _validate_worker_command(command, 5)


def test_stop_worker_terminates_immediately_after_malformed_close_reply() -> None:
    connection = _FakeConnection({"protocol_version": 1, "sequence": 4, "type": "step"})
    process = _FakeProcess()
    adapter = _adapter_with_fake_worker(connection, process)

    adapter._stop_worker()

    assert process.terminated
    assert not process.killed
    assert connection.closed
    assert adapter._worker_process is None
    assert adapter._worker_connection is None
    assert all(timeout <= 10.0 for timeout in process.wait_timeouts)


def test_failed_rpc_poisons_and_reaps_worker() -> None:
    connection = _FakeConnection(fail_send=True)
    process = _FakeProcess()
    adapter = _adapter_with_fake_worker(connection, process)

    with pytest.raises(RuntimeError, match="connection failed"):
        adapter._request_worker("step", actions=np.zeros((1, 1), dtype=np.float32))

    assert process.terminated
    assert connection.closed
    assert adapter._worker_process is None


def test_constructor_closes_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = M4POConfig.from_yaml("m4po/configs/isaaclab_forge_smoke.yaml")
    stop_calls: list[bool] = []

    def interrupt_activation(_self, _task_id: int) -> None:
        raise KeyboardInterrupt

    def track_stop(_self, *, graceful: bool = True) -> None:
        stop_calls.append(graceful)

    monkeypatch.setattr(IsaacLabPrebuiltEnv, "_activate_task", interrupt_activation)
    monkeypatch.setattr(IsaacLabPrebuiltEnv, "_stop_worker", track_stop)

    with pytest.raises(KeyboardInterrupt):
        IsaacLabPrebuiltEnv(cfg)

    assert stop_calls == [True]


def test_startup_keyboard_interrupt_reaps_spawned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def __init__(self, descriptor: int):
            self.descriptor = descriptor
            self.closed = False
            self.detached = False

        def fileno(self) -> int:
            return self.descriptor

        def close(self) -> None:
            self.closed = True

        def detach(self) -> int:
            self.detached = True
            return self.descriptor

    parent_socket = FakeSocket(41)
    child_socket = FakeSocket(42)
    connection = _FakeConnection()
    process = _FakeProcess()
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter.task_names = ["Isaac-Forge-PegInsert-Direct-v0"]
    adapter._task_specs = [
        prebuilt_module._SUPPORTED_TASK_METADATA[adapter.task_names[0]]
    ]
    adapter.num_envs = 1
    adapter.max_episode_steps = 4
    adapter._base_seed = 7
    adapter._active_task_id = 0
    adapter._reset_count = 0
    adapter._device = "cuda:0"
    adapter._headless = True
    adapter._enable_cameras = False
    adapter._use_fabric = True
    adapter._worker_connection = None
    adapter._worker_process = None
    adapter._worker_task_id = None
    adapter._worker_sequence = 0
    adapter._pending_observation = None

    def interrupt_receive(*_args, **_kwargs):
        raise KeyboardInterrupt

    adapter._receive_worker = interrupt_receive
    monkeypatch.setattr(
        prebuilt_module.socket,
        "socketpair",
        lambda: (parent_socket, child_socket),
    )
    monkeypatch.setattr(
        prebuilt_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        prebuilt_module,
        "Connection",
        lambda _descriptor: connection,
    )

    with pytest.raises(KeyboardInterrupt):
        adapter._start_worker(0)

    assert child_socket.closed
    assert parent_socket.detached
    assert process.terminated
    assert connection.closed
    assert adapter._worker_process is None
    assert adapter._worker_connection is None


@pytest.mark.parametrize(
    ("task_id", "permutation", "reward_scale"),
    [
        (0, (0, 1, 2, 3), 1.0 / 60.0),
        (1, (1, 3, 0, 2), 1.0),
    ],
)
def test_cartpole_worker_request_carries_semantic_contract(
    monkeypatch: pytest.MonkeyPatch,
    task_id: int,
    permutation: tuple[int, ...],
    reward_scale: float,
) -> None:
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter.task_names = list(DEFAULT_CARTPOLE_TASKS)
    adapter._task_specs = [
        prebuilt_module._SUPPORTED_TASK_METADATA[name]
        for name in DEFAULT_CARTPOLE_TASKS
    ]
    adapter.num_envs = 2
    adapter.max_episode_steps = 8
    adapter._base_seed = 7
    adapter._active_task_id = task_id
    adapter._reset_count = 0
    adapter._device = "cuda:0"
    adapter._headless = True
    adapter._enable_cameras = False
    adapter._use_fabric = True
    requests: list[dict[str, object]] = []

    monkeypatch.setattr(
        adapter,
        "_start_worker_attempt",
        lambda _task_id, request: requests.append(dict(request)),
    )
    adapter._start_worker(task_id)

    assert len(requests) == 1
    assert requests[0]["policy_observation_permutation"] == permutation
    assert requests[0]["reward_scale"] == pytest.approx(reward_scale)


def test_worker_startup_retry_reuses_bootstrap_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter.task_names = ["Isaac-Forge-PegInsert-Direct-v0"]
    adapter._task_specs = [
        prebuilt_module._SUPPORTED_TASK_METADATA[adapter.task_names[0]]
    ]
    adapter.num_envs = 1
    adapter.max_episode_steps = 4
    adapter._base_seed = 7
    adapter._active_task_id = 0
    adapter._reset_count = 0
    adapter._device = "cuda:0"
    adapter._headless = True
    adapter._enable_cameras = False
    adapter._use_fabric = True
    requests: list[dict[str, object]] = []
    delays: list[float] = []

    def start_attempt(_task_id: int, request: dict[str, object]) -> None:
        requests.append(dict(request))
        if len(requests) == 1:
            raise RuntimeError("transient startup failure")

    monkeypatch.setattr(adapter, "_start_worker_attempt", start_attempt)
    monkeypatch.setattr(prebuilt_module.time, "sleep", delays.append)

    adapter._start_worker(0)

    assert [request["seed"] for request in requests] == [7, 7]
    assert adapter._reset_count == 1
    assert delays == [prebuilt_module._WORKER_START_RETRY_DELAY_SECONDS]


def test_worker_startup_retry_exhaustion_chains_final_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter.task_names = ["Isaac-Forge-PegInsert-Direct-v0"]
    adapter._task_specs = [
        prebuilt_module._SUPPORTED_TASK_METADATA[adapter.task_names[0]]
    ]
    adapter.num_envs = 1
    adapter.max_episode_steps = 4
    adapter._base_seed = 11
    adapter._active_task_id = 0
    adapter._reset_count = 0
    adapter._device = "cuda:0"
    adapter._headless = True
    adapter._enable_cameras = False
    adapter._use_fabric = True
    requests: list[dict[str, object]] = []

    def fail_start_attempt(_task_id: int, request: dict[str, object]) -> None:
        requests.append(dict(request))
        raise RuntimeError(f"failure {len(requests)}")

    monkeypatch.setattr(adapter, "_start_worker_attempt", fail_start_attempt)
    monkeypatch.setattr(prebuilt_module.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="failed to start after 3 attempts") as error:
        adapter._start_worker(0)

    assert len(requests) == prebuilt_module._WORKER_START_ATTEMPTS
    assert {request["seed"] for request in requests} == {11}
    assert adapter._reset_count == 1
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "failure 3"


def test_rollout_state_rejects_partial_task_deck() -> None:
    adapter = object.__new__(IsaacLabPrebuiltEnv)
    adapter.task_names = ["one", "two", "three"]
    adapter.num_tasks = 3
    adapter._assignment_rng = np.random.default_rng(7)
    adapter._pending_observation = None
    state = {
        "schema_version": 2,
        "task_names": list(adapter.task_names),
        "assignment_rng_state": adapter._assignment_rng.bit_generator.state,
        "assignment_deck": np.asarray([0, 1], dtype=np.int64),
        "assignment_cursor": 1,
        "rollout_generation": 1,
        "rollout_started": True,
        "reset_count": 2,
    }

    with pytest.raises(ValueError, match="task deck"):
        adapter.load_rollout_state_dict(state)
