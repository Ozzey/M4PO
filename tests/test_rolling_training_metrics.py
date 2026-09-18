from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from m4po.common.config import M4POConfig
from m4po.trainer import OnlineTrainer
from m4po.trainer.off_policy_trainer import (
    _RollingMetrics,
    _complete_mean,
    _optional_success,
)
from test_offpolicy_trainer import _TimedEnvironment, training as training


def test_task_macro_is_fair_with_unequal_episode_counts_and_undefined_success():
    metrics = _RollingMetrics(["frequent", "rare", "no_success"])
    for _ in range(10):
        metrics.record("frequent", 1, 2, 0, 0.2)
    metrics.record("rare", 7, 3, 1, 0.8)
    metrics.record("no_success", 2, 5, float("nan"), 0.5)
    result = metrics.summary()
    assert _complete_mean(metrics.global_windows["successes"]) is None
    assert result["train_success_rate_supported_tasks_20"] == 0.5
    assert result["train_success_tasks"] == 2
    assert result["train_success_task_coverage"] == pytest.approx(2 / 3)
    assert result["train_success_episodes_in_window"] == 11
    assert result["train_score_macro_20"] == pytest.approx(0.5)
    assert _complete_mean(metrics.global_windows["scores"]) == pytest.approx(3.3 / 12)
    assert result["train_score_task_count"] == 3
    assert result["train_score_full_coverage"] is True


def test_partly_undefined_task_windows_are_excluded_not_partly_averaged():
    metrics = _RollingMetrics(["mixed", "unsupported", "known"])
    metrics.record("mixed", 0, 1, 1, 0.8)
    metrics.record("mixed", 0, 1, None, None)
    metrics.record("unsupported", 0, 1, float("nan"), float("nan"))
    assert metrics.summary()["train_success_rate_supported_tasks_20"] is None
    assert metrics.summary()["train_score_macro_20"] is None
    assert metrics.summary()["train_success_tasks"] == 0
    metrics.record("known", 0, 1, 0, 0.25)
    result = metrics.summary()
    assert result["train_success_rate_supported_tasks_20"] == 0
    assert result["train_success_tasks"] == 1
    assert result["train_success_episodes_in_window"] == 1
    assert result["train_score_macro_20"] == 0.25
    assert result["train_score_full_coverage"] is False
    assert result["rolling_window_episodes"] == 20


def test_windows_evict_at_twenty_and_committed_snapshots_do_not_mutate():
    metrics = _RollingMetrics(["task"])
    metrics.record("task", 0, 1, None, None)
    committed = metrics.state_dict()
    assert committed is metrics.state_dict()
    for _ in range(20):
        metrics.record("task", 1, 2, True, 0.6)
    assert committed["per_task"]["task"]["episodes_completed"] == 1
    assert committed["global"]["successes"] == [None]
    assert metrics.summary()["train_success_rate_supported_tasks_20"] == 1
    assert metrics.summary()["train_success_episodes_in_window"] == 20
    assert metrics.per_task["task"]["episodes_completed"] == 21


@pytest.mark.parametrize("value", [0.2, 2, -1, "True", "0", np.array([1])])
def test_nonbinary_success_is_never_coerced_to_truthy(value):
    with pytest.raises(ValueError, match="binary"):
        _optional_success(value)


@pytest.mark.parametrize(
    "value,expected",
    [(True, 1), (False, 0), (np.float32(1), 1), (None, None), (float("nan"), None)],
)
def test_native_binary_and_undefined_success(value, expected):
    assert _optional_success(value) == expected


def test_rolling_state_roundtrip_preserves_global_and_per_task_windows():
    original = _RollingMetrics(["first", "second"])
    for index in range(30):
        original.record(
            "first" if index % 3 else "second", index, index + 1, index % 2, index / 30
        )
    saved = deepcopy(original.state_dict())
    restored = _RollingMetrics(["first", "second"])
    restored.load_state_dict(saved, completed_episodes=30)
    assert restored.state_dict() == saved
    assert restored.summary() == original.summary()
    original.record("second", 9, 4, None, 0.8)
    restored.record("second", 9, 4, None, 0.8)
    assert restored.state_dict() == original.state_dict()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update(schema_version=True),
        lambda state: state.update(window_size=21),
        lambda state: state.update(task_names=["other"]),
        lambda state: state.update(history_start_completed_episodes=2),
        lambda state: state["global"]["successes"].append(1),
        lambda state: state["global"]["scores"].__setitem__(0, float("nan")),
        lambda state: state["per_task"]["task"]["successes"].__setitem__(0, 0.5),
        lambda state: state["per_task"]["task"]["lengths"].__setitem__(0, 0),
        lambda state: state["per_task"]["task"].update(episodes_completed=2),
        lambda state: state["per_task"].update(other={}),
    ],
)
def test_invalid_rolling_checkpoints_do_not_replace_live_windows(mutation):
    metrics = _RollingMetrics(["task"])
    metrics.record("task", 2, 3, 1, 0.2)
    saved = deepcopy(metrics.state_dict())
    corrupt = deepcopy(saved)
    mutation(corrupt)
    with pytest.raises(ValueError, match="Rolling metrics"):
        metrics.load_state_dict(corrupt, completed_episodes=1)
    assert metrics.state_dict() == saved


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _last_metrics(cfg):
    rows = (Path(cfg.log_dir) / "metrics.jsonl").read_text().splitlines()
    return json.loads(rows[-1])


def test_actual_trainer_resume_restores_windows_without_resetting_curve(
    training, tmp_path
):
    cfg, _ = training
    cfg.save_replay = True
    initial = OnlineTrainer(cfg).train()
    original = _payload(initial)["extra"]["rolling_metrics_state"]
    resumed = M4POConfig(**cfg.to_dict())
    resumed.resume_checkpoint = str(initial)
    resumed.log_dir = str(tmp_path / "resumed")
    final = OnlineTrainer(resumed).train()
    assert _payload(final)["extra"]["rolling_metrics_state"] == original
    metrics = _last_metrics(resumed)
    assert metrics["train_per_task"] == _last_metrics(cfg)["train_per_task"]
    assert metrics["train_success_rate_supported_tasks_20"] == 1
    assert metrics["rolling_metrics_restored"] is True
    assert metrics["rolling_metrics_reset_on_resume"] is False
    assert metrics["rolling_metrics_history_start_episode"] == 0


def test_legacy_checkpoint_explicitly_reports_missing_rolling_history(
    training, tmp_path
):
    cfg, _ = training
    initial = _payload(OnlineTrainer(cfg).train())
    del initial["extra"]["rolling_metrics_state"]
    legacy = tmp_path / "legacy.pt"
    torch.save(initial, legacy)
    resumed = M4POConfig(**cfg.to_dict())
    resumed.resume_checkpoint = str(legacy)
    resumed.log_dir = str(tmp_path / "resumed")
    final = OnlineTrainer(resumed).train()
    metrics = _last_metrics(resumed)
    assert metrics["rolling_metrics_restored"] is False
    assert metrics["rolling_metrics_reset_on_resume"] is True
    assert metrics["train_success_rate_supported_tasks_20"] is None
    assert metrics["train_success_tasks"] == 0
    assert (
        metrics["rolling_metrics_history_start_episode"]
        == initial["extra"]["completed_episodes"]
    )
    assert (
        _payload(final)["extra"]["rolling_metrics_state"][
            "history_start_completed_episodes"
        ]
        == initial["extra"]["completed_episodes"]
    )


def test_partial_worker_failure_rolls_back_metrics_with_committed_counters(
    training, monkeypatch
):
    cfg, _ = training
    original_step = _TimedEnvironment.step

    def invalid_success(self, actions):
        obs, reward, done, infos = original_step(self, actions)
        if done.all():
            infos[1]["success"] = 0.2
        return obs, reward, done, infos

    monkeypatch.setattr(_TimedEnvironment, "step", invalid_success)
    with pytest.raises(ValueError, match="binary"):
        OnlineTrainer(cfg).train()
    saved = _payload(Path(cfg.log_dir) / "checkpoints" / "emergency.pt")
    assert saved["step"] == 2
    assert saved["extra"]["completed_episodes"] == 0
    rolling = saved["extra"]["rolling_metrics_state"]
    assert rolling["global"]["successes"] == []
    assert sum(row["episodes_completed"] for row in rolling["per_task"].values()) == 0
