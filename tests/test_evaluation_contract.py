from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import m4po.common.evaluation as evaluation_module
from m4po.common.buffer import ObservationSpec
from m4po.common.environment import environment_signature


class _DynamicEvaluationEnv:
    def __init__(self) -> None:
        self.num_envs = 4
        self.num_tasks = 2
        self.num_embodiments = 2
        self.action_dim = 2
        self.max_episode_steps = 1
        self.task_names = ["task-0", "task-1"]
        self.embodiment_names = ["body-0", "body-1"]
        self.observation_spec = ObservationSpec((0, 0, 0), 0, 1)
        self.control_repeats = np.ones(2, dtype=np.int64)
        self.low_level_discounts = np.ones(2, dtype=np.float32)
        # Deliberately stale construction metadata. Evaluation must read again
        # after reset, when the adapter publishes the assigned contexts.
        self.task_ids = np.zeros(4, dtype=np.int64)
        self.embodiment_ids = np.zeros(4, dtype=np.int64)
        self.action_masks = self._masks(self.embodiment_ids)
        self.closed = False

    @staticmethod
    def _masks(embodiment_ids: np.ndarray) -> np.ndarray:
        masks = np.ones((len(embodiment_ids), 2), dtype=np.float32)
        masks[embodiment_ids == 0, 1] = 0.0
        return masks

    def _observation(self) -> dict[str, np.ndarray]:
        return {"state": np.zeros((self.num_envs, 1), dtype=np.float32)}

    def reset(self) -> dict[str, np.ndarray]:
        self.task_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.embodiment_ids = np.asarray([0, 1, 0, 1], dtype=np.int64)
        self.action_masks = self._masks(self.embodiment_ids)
        return self._observation()

    def step(self, actions: np.ndarray):
        assert actions.shape == (self.num_envs, self.action_dim)
        task_ids = self.task_ids.copy()
        embodiment_ids = self.embodiment_ids.copy()
        rewards = (1 + 2 * task_ids + embodiment_ids).astype(np.float32)
        done = np.ones(self.num_envs, dtype=np.bool_)
        infos = [
            {
                "task_id": int(task_id),
                "embodiment_id": int(embodiment_id),
                "success": True,
            }
            for task_id, embodiment_id in zip(task_ids, embodiment_ids, strict=True)
        ]
        # Simulate an auto-reset adapter that reassigns workers. The next call
        # must use these new IDs and masks, while the completed episodes remain
        # attributed to the pre-step IDs above.
        order = np.asarray([3, 2, 1, 0])
        self.task_ids = task_ids[order]
        self.embodiment_ids = embodiment_ids[order]
        self.action_masks = self._masks(self.embodiment_ids)
        return self._observation(), rewards, done, infos

    def close(self) -> None:
        self.closed = True


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def act_np(
        self,
        observation,
        *,
        task_ids,
        embodiment_ids,
        action_mask,
        deterministic,
    ):
        del observation, deterministic
        self.calls.append((task_ids.copy(), embodiment_ids.copy(), action_mask.copy()))
        return SimpleNamespace(
            action=np.zeros((len(task_ids), action_mask.shape[-1]), dtype=np.float32)
        )


class _SequentialEvaluationEnv(_DynamicEvaluationEnv):
    def __init__(self) -> None:
        super().__init__()
        self.num_envs = 3
        self.task_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.embodiment_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.action_masks = self._masks(self.embodiment_ids)
        self.sequential_evaluation = True
        self.selected_pair = (0, 0)
        self.selections: list[tuple[int, int]] = []

    def set_evaluation_pair(self, task_id: int, embodiment_id: int) -> None:
        self.selected_pair = (int(task_id), int(embodiment_id))
        self.selections.append(self.selected_pair)

    def reset(self) -> dict[str, np.ndarray]:
        task_id, embodiment_id = self.selected_pair
        self.task_ids.fill(task_id)
        self.embodiment_ids.fill(embodiment_id)
        self.action_masks = self._masks(self.embodiment_ids)
        return self._observation()

    def step(self, actions: np.ndarray):
        assert actions.shape == (self.num_envs, self.action_dim)
        task_ids = self.task_ids.copy()
        embodiment_ids = self.embodiment_ids.copy()
        rewards = (1 + 2 * task_ids + embodiment_ids).astype(np.float32)
        done = np.ones(self.num_envs, dtype=np.bool_)
        infos = [
            {
                "task_id": int(task_id),
                "embodiment_id": int(embodiment_id),
                "success": True,
            }
            for task_id, embodiment_id in zip(task_ids, embodiment_ids, strict=True)
        ]
        return self._observation(), rewards, done, infos


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        num_tasks=2,
        num_embodiments=2,
        num_envs=1,
        task_context_mode="frozen",
    )


def test_evaluation_refreshes_dynamic_metadata_and_attributes_completed_pair(
    monkeypatch,
) -> None:
    env = _DynamicEvaluationEnv()
    agent = _RecordingAgent()
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    result = evaluation_module.evaluate_policy(agent, _config(), episodes=2)

    assert result["episodes"] == 8
    assert all(record["episodes"] == 2 for record in result["per_pair"].values())
    assert len(agent.calls) == 2
    assert agent.calls[0][0].tolist() == [0, 0, 1, 1]
    assert agent.calls[0][1].tolist() == [0, 1, 0, 1]
    assert agent.calls[1][0].tolist() == [1, 1, 0, 0]
    assert agent.calls[1][1].tolist() == [1, 0, 1, 0]
    for _, embodiment_ids, masks in agent.calls:
        assert np.array_equal(masks, env._masks(embodiment_ids))
    assert env.closed


def test_evaluation_rejects_missing_cartesian_pair_and_closes(monkeypatch) -> None:
    env = _DynamicEvaluationEnv()
    original_reset = env.reset

    def reset_with_duplicate():
        observation = original_reset()
        env.task_ids = np.asarray([0, 0, 1, 0], dtype=np.int64)
        env.embodiment_ids = np.asarray([0, 1, 0, 0], dtype=np.int64)
        env.action_masks = env._masks(env.embodiment_ids)
        return observation

    env.reset = reset_with_duplicate
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    with pytest.raises(ValueError, match="every task/embodiment pair exactly once"):
        evaluation_module.evaluate_policy(_RecordingAgent(), _config(), episodes=1)
    assert env.closed


def test_evaluation_rejects_checkpoint_environment_signature_mismatch(
    monkeypatch,
) -> None:
    env = _DynamicEvaluationEnv()
    expected = environment_signature(env, None, "frozen")
    expected = deepcopy(expected)
    expected["task_names"] = ["different", "task-1"]
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    with pytest.raises(ValueError, match="do not match the checkpoint"):
        evaluation_module.evaluate_policy(
            _RecordingAgent(),
            _config(),
            episodes=1,
            expected_environment_signature=expected,
        )
    assert env.closed


def test_evaluation_can_select_and_run_each_pair_sequentially(monkeypatch) -> None:
    env = _SequentialEvaluationEnv()
    agent = _RecordingAgent()
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    result = evaluation_module.evaluate_policy(agent, _config(), episodes=2)

    assert env.selections == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result["episodes"] == 8
    assert all(record["episodes"] == 2 for record in result["per_pair"].values())
    assert [record["return_mean"] for record in result["per_pair"].values()] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert len(agent.calls) == 4
    for call, pair in zip(agent.calls, env.selections, strict=True):
        task_ids, embodiment_ids, masks = call
        assert np.all(task_ids == pair[0])
        assert np.all(embodiment_ids == pair[1])
        assert np.array_equal(masks, env._masks(embodiment_ids))
    assert env.closed


def test_sequential_evaluation_rejects_workers_outside_selected_pair(
    monkeypatch,
) -> None:
    env = _SequentialEvaluationEnv()
    original_reset = env.reset

    def reset_with_wrong_worker():
        observation = original_reset()
        env.task_ids[-1] = 1 - env.task_ids[-1]
        return observation

    env.reset = reset_with_wrong_worker
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    with pytest.raises(ValueError, match="only the selected task/embodiment pair"):
        evaluation_module.evaluate_policy(_RecordingAgent(), _config(), episodes=1)
    assert env.closed
