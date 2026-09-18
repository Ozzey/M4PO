from __future__ import annotations

from copy import deepcopy
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from m4po.common.config import M4POConfig
from m4po.envs.mmbench_env import (
    MMBenchVectorEnv,
    _configure_native_asset_paths,
    configure_mmbench,
    load_mmbench_catalog,
)


@pytest.fixture
def catalog(tmp_path):
    tasks = [f"native-{index}" for index in range(200)]
    metadata = {
        task: {
            "embodiment": "Body A, two joints"
            if index % 2 == 0
            else "Body B, three joints",
            "instruction": task,
            "action_dim": 2 if index % 2 == 0 else 3,
            "max_episode_steps": 3 if index % 2 == 0 else 2,
            "text_embedding": [float(index)] * 512,
        }
        for index, task in enumerate(tasks)
    }
    # Upstream metadata contains held-out tasks as well as the exact soup set.
    metadata["held-out"] = metadata[tasks[0]].copy()
    (tmp_path / "tasks.json").write_text(json.dumps(metadata))
    common = tmp_path / "tdmpc2" / "common"
    common.mkdir(parents=True)
    (common / "__init__.py").write_text(
        "raise RuntimeError('Catalog loading must not execute upstream Python')\n"
        + "TASK_SET = "
        + repr({"dmcontrol": tasks, "soup": tasks})
    )
    return tmp_path


def config(catalog, **kwargs):
    return M4POConfig(
        env="mmbench",
        mmbench_root=str(catalog),
        use_images=False,
        tasks="native-0,native-1",
        num_envs=1,
        **kwargs,
    )


class NativeEnv:
    def __init__(self, task, *, terminated=False):
        self.task = task
        self.index = int(task.rsplit("-", 1)[-1])
        self.dim = 2 if self.index % 2 == 0 else 3
        self.action_space = SimpleNamespace(
            shape=(self.dim,), low=-np.ones(self.dim), high=np.ones(self.dim)
        )
        self.terminated = terminated
        self.elapsed = 0
        self.actions = []
        self.closed = False

    def reset(self):
        self.elapsed = 0
        return np.full(self.dim + 1, self.index, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.elapsed += 1
        return (
            np.full(self.dim + 1, self.elapsed + 10, dtype=np.float32),
            2.0,
            self.terminated,
            self.elapsed >= (3 if self.index % 2 == 0 else 2),
            {"success": float("nan"), "score": 0.125 * self.elapsed},
        )

    def close(self):
        self.closed = True


def factory(**kwargs):
    return NativeEnv(kwargs["task"])


def test_catalog_is_metadata_only_and_preserves_exact_soup(catalog):
    metadata, task_sets, digest = load_mmbench_catalog(catalog)
    assert len(metadata) == 201
    assert len(task_sets["soup"]) == 200
    assert len(digest) == 64
    cfg = config(catalog)
    configure_mmbench(cfg)
    assert cfg.num_tasks == cfg.num_embodiments == 2
    assert cfg.embodiment_names == ["body-a-two-joints", "body-b-three-joints"]
    assert cfg.state_dim == 128 and cfg.proprio_dim == 0
    assert cfg.task_context_dim == 512 and cfg.max_episode_steps == 3
    previous = cfg.to_dict()
    configure_mmbench(cfg)
    assert cfg.to_dict() == previous


def test_full_soup_config_does_not_include_held_out_tasks(catalog):
    cfg = config(catalog)
    cfg.tasks = None
    configure_mmbench(cfg)
    assert cfg.num_tasks == 200
    assert "held-out" not in cfg.task_names
    cfg.tasks = "held-out"
    with pytest.raises(ValueError, match="outside the official"):
        configure_mmbench(cfg)


def test_adapter_preserves_native_rewards_scores_timeout_and_terminal_state(catalog):
    env = MMBenchVectorEnv(config(catalog), native_factory=factory)
    try:
        env.set_evaluation_pair(0, 0)
        observation = env.reset()
        assert observation["state"].shape == (1, 128)
        assert observation["image"].shape == (1, 0, 0, 0)
        assert observation["state_mask"].sum() == 3
        native = env.envs[0]
        for _ in range(3):
            observation, reward, done, infos = env.step(
                np.ones((1, 16), dtype=np.float32)
            )
        assert reward[0] == 2
        assert done[0]
        assert np.isnan(infos[0]["success"])
        assert infos[0]["score"] == 0.375
        assert infos[0]["truncated"] and not infos[0]["terminated"]
        np.testing.assert_array_equal(infos[0]["terminal_observation"]["state"][:3], 13)
        np.testing.assert_array_equal(observation["state"][:3], 0)
        assert len(native.actions) == 3  # Native repeat logic is never repeated again.
        np.testing.assert_array_equal(native.actions[0], [1, 1])
        assert not native.closed
        assert env.envs[0] is native
    finally:
        env.close()


def test_worker_context_rotates_only_after_completed_episode(catalog):
    env = MMBenchVectorEnv(config(catalog), native_factory=factory)
    try:
        env.reset()
        first_task = int(env.task_ids[0])
        first_body = int(env.embodiment_ids[0])
        horizon = int(env.episode_lengths[first_task])
        for _ in range(horizon - 1):
            _, _, done, _ = env.step(np.zeros((1, 16)))
            assert not done[0]
            assert env.task_ids[0] == first_task
        observation, _, done, infos = env.step(np.zeros((1, 16)))
        assert done[0] and env.task_ids[0] != first_task
        assert infos[0]["task_id"] == first_task
        assert infos[0]["embodiment_id"] == first_body
        assert observation["state"][0, 0] == env.task_ids[0]
        assert env.evaluation_pairs == [(0, 0), (1, 1)]
        assert env.preserve_episodes_between_rollouts
        assert env.sequential_evaluation
    finally:
        env.close()


def test_task_deck_visits_all_tasks_with_one_worker(catalog):
    cfg = config(catalog)
    cfg.tasks = "native-0,native-1,native-2,native-3"
    env = MMBenchVectorEnv(cfg, native_factory=factory)
    try:
        env.reset()
        visited = []
        while len(visited) < 4:
            _, _, done, infos = env.step(np.zeros((1, 16)))
            if done[0]:
                visited.append(infos[0]["task_id"])
        assert set(visited) == {0, 1, 2, 3}
    finally:
        env.close()


def test_true_termination_is_not_changed_to_timeout(catalog):
    env = MMBenchVectorEnv(
        config(catalog),
        native_factory=lambda **kw: NativeEnv(kw["task"], terminated=True),
    )
    try:
        env.reset()
        _, _, done, infos = env.step(np.zeros((1, 16)))
        assert done[0] and infos[0]["terminated"] and not infos[0]["truncated"]
    finally:
        env.close()


def test_sampler_state_can_resume_without_serializing_simulator(catalog):
    cfg = config(catalog)
    env = MMBenchVectorEnv(cfg, native_factory=factory)
    restored = MMBenchVectorEnv(deepcopy(cfg), native_factory=factory)
    try:
        state = env.rollout_state_dict()
        restored.load_rollout_state_dict(state)
        np.testing.assert_array_equal(env.task_ids, restored.task_ids)
        assert env._next_task() == restored._next_task()
        restored.reset()
        restored.step(np.zeros((1, 16)))
        with pytest.raises(ValueError, match="no native"):
            restored.set_evaluation_pair(0, 1)
    finally:
        env.close()
        restored.close()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("use_images", True, "state observations only"),
        ("control_decimation", 2, "action repeats"),
        ("task_context_mode", "learned", "frozen"),
        ("task_context_path", "custom.pt", "frozen"),
    ],
)
def test_adapter_rejects_silent_benchmark_contract_changes(
    catalog, field, value, match
):
    cfg = config(catalog)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        configure_mmbench(cfg)


def test_oversized_states_are_rejected_not_truncated():
    with pytest.raises(ValueError, match="at most 128"):
        MMBenchVectorEnv._pad_observation(np.zeros(129))


def test_fixed_sampling_preserves_workers_and_balances_task_transitions(catalog):
    cfg = config(catalog)
    cfg.num_envs = 2
    cfg.mmbench_sampling = "fixed"
    env = MMBenchVectorEnv(cfg, native_factory=factory)
    restored = MMBenchVectorEnv(deepcopy(cfg), native_factory=factory)
    try:
        np.testing.assert_array_equal(env.task_ids, [0, 1])
        natives = list(env.envs)
        env.reset()
        completed = [0, 0]
        for _ in range(6):
            _, _, dones, infos = env.step(np.zeros((2, 16)))
            np.testing.assert_array_equal(env.task_ids, [0, 1])
            for worker, done in enumerate(dones):
                completed[worker] += int(done)
                assert infos[worker]["task_id"] == worker
        assert completed == [2, 3]
        assert env.envs == natives
        assert not any(native.closed for native in natives)
        restored.load_rollout_state_dict(env.rollout_state_dict())
        np.testing.assert_array_equal(restored.task_ids, [0, 1])
        assert "equal transitions" in env.environment_signature["sampling"]
    finally:
        env.close()
        restored.close()


def test_fixed_sampling_requires_full_worker_coverage_except_evaluation(catalog):
    cfg = config(catalog)
    cfg.mmbench_sampling = "fixed"
    with pytest.raises(ValueError, match="multiple of num_tasks"):
        configure_mmbench(cfg)
    cfg._mmbench_evaluation = True
    env = MMBenchVectorEnv(cfg, native_factory=factory)
    try:
        env.set_evaluation_pair(1, 1)
        env.reset()
        for _ in range(5):
            env.step(np.zeros((1, 16)))
            assert env.task_ids[0] == 1
        assert "equal transitions" in env.environment_signature["sampling"]
    finally:
        env.close()


def test_maniskill_asset_alias_is_bound_before_import(monkeypatch, tmp_path):
    monkeypatch.delenv("MS_ASSET_DIR", raising=False)
    monkeypatch.setenv("MANISKILL_ASSET_DIR", str(tmp_path / ".maniskill"))
    monkeypatch.delitem(sys.modules, "mani_skill", raising=False)
    _configure_native_asset_paths()
    assert os.environ["MS_ASSET_DIR"] == str(tmp_path / ".maniskill")
    assert not (tmp_path / ".maniskill").exists()  # No mutation of asset trees.


def test_maniskill_explicit_native_asset_setting_takes_precedence(
    monkeypatch, tmp_path
):
    native_root = tmp_path / "native"
    monkeypatch.setenv("MS_ASSET_DIR", str(native_root))
    monkeypatch.setenv("MANISKILL_ASSET_DIR", str(tmp_path / "legacy"))
    monkeypatch.setitem(
        sys.modules, "mani_skill", SimpleNamespace(ASSET_DIR=native_root / "data")
    )
    _configure_native_asset_paths()
    assert os.environ["MS_ASSET_DIR"] == str(native_root)


def test_maniskill_wrong_already_imported_asset_root_requires_fresh_process(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MS_ASSET_DIR", str(tmp_path / "configured"))
    monkeypatch.setitem(
        sys.modules, "mani_skill", SimpleNamespace(ASSET_DIR=tmp_path / "old" / "data")
    )
    with pytest.raises(RuntimeError, match="start a fresh process"):
        _configure_native_asset_paths()
