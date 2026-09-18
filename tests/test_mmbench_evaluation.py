from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import m4po.common.evaluation as evaluation_module
from m4po.common.buffer import ObservationSpec


class _SparseBenchmarkEnv:
    sequential_evaluation = True

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.num_tasks = 3
        self.num_embodiments = 2
        self.task_names = ["reach", "walk", "balance"]
        self.task_domains = ["manipulation", "locomotion", "locomotion"]
        self.benchmark_task_count = 200
        self.benchmark_full_suite = False
        self.embodiment_names = ["arm", "leg"]
        self.evaluation_pairs = [(0, 0), (0, 1), (1, 1), (2, 0)]
        self.action_dim = 1
        self.action_masks = np.ones((num_envs, 1), dtype=np.float32)
        self.task_ids = np.zeros(num_envs, dtype=np.int64)
        self.embodiment_ids = np.zeros(num_envs, dtype=np.int64)
        self.observation_spec = ObservationSpec((0, 0, 0), 0, 1)
        self.selected_pair = (0, 0)
        self.selections: list[tuple[int, int]] = []
        self.native_scores = {(0, 0): 0.2, (0, 1): 0.4, (1, 1): 0.8, (2, 0): 0.1}
        self.native_success = {
            (0, 0): True,
            (0, 1): False,
            (1, 1): True,
            (2, 0): np.nan,
        }
        self.closed = False

    def set_evaluation_pair(self, task_id: int, embodiment_id: int) -> None:
        self.selected_pair = (task_id, embodiment_id)
        self.selections.append(self.selected_pair)

    def reset(self):
        self.task_ids.fill(self.selected_pair[0])
        self.embodiment_ids.fill(self.selected_pair[1])
        return {"state": np.zeros((self.num_envs, 1), dtype=np.float32)}

    def step(self, actions):
        assert actions.shape == (self.num_envs, 1)
        info = {
            "task_id": self.selected_pair[0],
            "embodiment_id": self.selected_pair[1],
            "success": self.native_success[self.selected_pair],
            "score": self.native_scores[self.selected_pair],
        }
        return (
            self.reset(),
            np.full(self.num_envs, 10.0),
            np.ones(self.num_envs, dtype=np.bool_),
            [dict(info) for _ in range(self.num_envs)],
        )

    def close(self):
        self.closed = True


class _Agent:
    def act_np(
        self, observation, *, task_ids, embodiment_ids, action_mask, deterministic
    ):
        del observation, task_ids, embodiment_ids, deterministic
        return SimpleNamespace(action=np.zeros_like(action_mask))


def _config():
    return SimpleNamespace(
        env="mmbench",
        num_envs=8,
        mmbench_eval_num_envs=2,
        num_tasks=3,
        num_embodiments=2,
        task_context_mode="frozen",
    )


def test_sparse_evaluation_uses_native_scores_and_preserves_worker_budget(monkeypatch):
    env = _SparseBenchmarkEnv()

    def make_env(cfg):
        assert cfg.num_envs == 2  # Not six imaginary Cartesian workers.
        return env

    monkeypatch.setattr(evaluation_module, "make_vector_env", make_env)
    result = evaluation_module.evaluate_policy(_Agent(), _config(), episodes=3)

    assert env.selections == env.evaluation_pairs
    assert result["episodes"] == 12
    assert set(result["per_pair"]) == {
        "reach/arm",
        "reach/leg",
        "walk/leg",
        "balance/arm",
    }
    assert all(record["episodes"] == 3 for record in result["per_pair"].values())
    assert result["return_mean"] == 10.0
    assert result["per_task"]["reach"]["score_mean"] == pytest.approx(0.3)
    assert result["per_embodiment"]["leg"]["score_mean"] == pytest.approx(0.6)
    assert result["score_mean"] == pytest.approx(0.4)
    assert result["score_defined_episodes"] == 12
    assert result["score_aggregation"] == "mean_of_task_mean_scores"
    assert result["per_domain"]["locomotion"]["task_count"] == 2
    assert result["per_domain"]["locomotion"]["score_mean"] == pytest.approx(0.45)
    assert result["per_pair"]["walk/leg"]["domain"] == "locomotion"
    assert result["benchmark_coverage"] == {
        "canonical_task_count": 200,
        "selected_task_count": 3,
        "evaluated_task_count": 3,
        "score_task_count": 3,
        "domain_count": 2,
        "evaluated_pair_count": 4,
        "full_suite_selected": False,
        "complete_score_coverage": True,
        "full_suite": False,
    }
    assert env.closed


def test_nan_success_is_undefined_not_true_or_false(monkeypatch):
    env = _SparseBenchmarkEnv()
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)
    result = evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)

    assert result["successes"] == [1.0, 0.0, 1.0, None]
    assert result["success_rate"] is None
    assert result["worst_embodiment_success"] is None
    assert result["success_defined_episodes"] == 3
    assert result["per_pair"]["balance/arm"]["success_rate"] is None
    assert result["per_task"]["balance"]["success_rate"] is None
    assert result["per_embodiment"]["arm"]["success_rate"] is None
    assert result["per_embodiment"]["leg"]["success_rate"] == 0.5
    json.dumps(result, allow_nan=False)


def test_missing_native_score_is_not_replaced_with_reward(monkeypatch):
    env = _SparseBenchmarkEnv()
    env.native_scores[(2, 0)] = None
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)
    result = evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)

    assert result["return_mean"] == 10.0
    assert result["score_mean"] is None
    assert result["score_defined_episodes"] == 3
    assert result["per_pair"]["balance/arm"]["scores"] == [None]
    assert result["benchmark_coverage"]["complete_score_coverage"] is False
    assert result["benchmark_coverage"]["score_task_count"] == 2


def test_pilot_cannot_claim_full_suite_merely_by_setting_a_flag(monkeypatch):
    env = _SparseBenchmarkEnv()
    env.benchmark_full_suite = True
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)
    result = evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)
    assert result["benchmark_coverage"]["full_suite"] is False
    assert result["benchmark_coverage"]["full_suite_selected"] is False


def test_domain_metadata_must_match_selected_tasks(monkeypatch):
    env = _SparseBenchmarkEnv()
    env.task_domains = ["only-one-domain"]
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)
    with pytest.raises(ValueError, match="one domain per task"):
        evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)
    assert env.closed


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        (None, "explicit evaluation_pairs"),
        ([], "must not be empty"),
        ([(0, 0), (0, 0), (1, 1), (2, 0)], "duplicate"),
        ([(0, 0), (1, 1), (3, 0)], "out of range"),
        ([(0, 0), (1, 1), (2, -1)], "out of range"),
        ([(0.0, 0), (1, 1), (2, 0)], "must be integers"),
        ([(False, 0), (1, 1), (2, 0)], "must be integers"),
        ([(0, 0, 0), (1, 1), (2, 0)], "entries must be"),
        ([(0, 0), (1, 1)], "cover every configured"),
        ([(0, 0), (1, 0), (2, 0)], "cover every configured"),
    ],
)
def test_invalid_sparse_pair_contract_is_rejected_and_closed(
    monkeypatch, pairs, message
):
    env = _SparseBenchmarkEnv()
    env.evaluation_pairs = pairs
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)

    with pytest.raises(ValueError, match=message):
        evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)
    assert env.closed


def test_sequential_adapter_cannot_substitute_an_unsupported_pair(monkeypatch):
    env = _SparseBenchmarkEnv()
    original_reset = env.reset

    def wrong_reset():
        observation = original_reset()
        env.task_ids.fill(1)
        env.embodiment_ids.fill(0)  # In bounds, but this combination does not exist.
        return observation

    env.reset = wrong_reset
    monkeypatch.setattr(evaluation_module, "make_vector_env", lambda cfg: env)
    with pytest.raises(ValueError, match="only the selected task/embodiment pair"):
        evaluation_module.evaluate_policy(_Agent(), _config(), episodes=1)
    assert env.closed


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_undefined_episode_metrics_remain_null(value):
    assert evaluation_module._episode_metric({"success": value}, "success") is None
    assert evaluation_module._episode_metric({"score": value}, "score") is None
    assert evaluation_module._episode_metric({}, "success") is None


def test_native_score_is_preserved_and_success_is_binary():
    assert evaluation_module._episode_metric({"score": 87.5}, "score") == 87.5
    assert evaluation_module._episode_metric({"success": False}, "success") == 0.0
    with pytest.raises(ValueError, match="must be boolean"):
        evaluation_module._episode_metric({"success": 0.5}, "success")
