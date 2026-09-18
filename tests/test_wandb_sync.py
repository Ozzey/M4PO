from __future__ import annotations

import json
from pathlib import Path

import pytest

from m4po.wandb_sync import (
    SyncCursor,
    _log_benchmark,
    _final_environment_step,
    _last_pretrain_update,
    _refresh_run_config,
    _wait_for_startup_inputs,
    _wandb_tags,
    drain_metrics,
    load_cursor,
    reconcile_history_step,
    record_to_wandb,
)


class FakeConfig(dict[str, object]):
    def __init__(self) -> None:
        super().__init__()
        self.updates: list[tuple[dict[str, object], bool]] = []

    def update(  # type: ignore[override]
        self,
        values: dict[str, object],
        *,
        allow_val_change: bool = False,
    ) -> None:
        self.updates.append((values, allow_val_change))
        super().update(values)


def test_demonstration_metrics_have_update_axis_not_online_step():
    output = record_to_wandb({
        "phase": "demonstration_pretraining", "step": 0,
        "pretrain_updates": 200, "bc_loss": 0.12, "q_loss": 0.3,
    })
    assert output["environment_step"] == 0
    assert output["pretrain_update"] == 200
    assert output["pretrain/bc_loss"] == 0.12
    assert "train/bc_loss" not in output


def test_rolling_metrics_use_environment_steps_and_preserve_native_coverage():
    result = record_to_wandb({
        "step": 40000, "phase": "off_policy_training",
        "train_success_rate_supported_tasks_20": 0.7,
        "train_success_tasks": 2, "train_success_task_coverage": 0.01,
        "train_success_episodes_in_window": 32,
        "train_score_macro_20": 0.4, "train_score_task_coverage": 0.5,
        "train_per_task": {
            "mw-reach": {"success_rate_20": 0.8, "score_mean_20": 0.8},
            "walker-stand": {"success_rate_20": None, "score_mean_20": 0.2},
        },
    })
    assert result["environment_step"] == 40000
    assert result["train/rolling_success_rate"] == 0.7
    assert result["train/rolling_success_task_coverage"] == 0.01
    assert result["train/rolling_success_episode_count"] == 32
    assert result["train/rolling_normalized_score"] == 0.4
    assert result["train/task/mw-reach/rolling_success_rate"] == 0.8
    assert "train/task/walker-stand/rolling_success_rate" not in result


def test_undefined_rolling_success_is_not_logged_as_zero():
    result = record_to_wandb({
        "step": 10000, "train_success_rate_supported_tasks_20": None,
        "train_success_task_coverage": 0.0,
    })
    assert "train/rolling_success_rate" not in result
    assert result["train/rolling_success_task_coverage"] == 0.0


def test_pretrain_artifact_does_not_claim_online_budget(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({
        "phase": "demonstration_pretraining", "step": 0,
        "pretrain_updates": 200000,
    }) + "\n")
    assert _final_environment_step({"total_steps": 100000000}, metrics) == 0
    assert _last_pretrain_update(metrics) == 200000


class FakeRun:
    def __init__(self, *, step: int = 0) -> None:
        self.records: list[tuple[dict[str, object], int, bool]] = []
        self.summary: dict[str, object] = {}
        self.config = FakeConfig()
        self.step = step

    def log(self, data: dict[str, object], *, step: int, commit: bool) -> None:
        self.records.append((data, step, commit))


def _append(path: Path, value: str) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(value)


def test_drain_backfills_and_resumes_without_duplicates(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    cursor_path = tmp_path / "cursor.json"
    metrics.write_text(
        json.dumps({"step": 4096, "model_loss": 3.0})
        + "\n"
        + json.dumps({"step": 8192, "actor_loss": 0.2})
        + "\n",
        encoding="utf-8",
    )
    run = FakeRun()
    cursor = load_cursor(metrics, cursor_path, "run-1")
    cursor, count = drain_metrics(metrics, cursor_path, cursor, run)

    assert count == 2
    assert [record[1] for record in run.records] == [0, 1]
    assert run.records[0][0]["environment_step"] == 4096
    assert run.records[0][0]["train/model_loss"] == 3.0
    assert all(record[2] is True for record in run.records)
    assert cursor.next_line_index == 2
    assert cursor.next_history_step == 2

    resumed = load_cursor(metrics, cursor_path, "run-1")
    resumed, count = drain_metrics(metrics, cursor_path, resumed, run)
    assert count == 0
    assert resumed == cursor
    assert len(run.records) == 2


def test_resume_reconciles_legacy_cursor_with_wandb_history(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    cursor_path = tmp_path / "cursor.json"
    metrics.touch()
    stat = metrics.stat()
    cursor_path.write_text(
        json.dumps(
            {
                "run_id": "resumed-run",
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "byte_offset": 0,
                "next_line_index": 64,
            }
        ),
        encoding="utf-8",
    )

    cursor = load_cursor(metrics, cursor_path, "resumed-run")
    assert cursor.next_history_step == 64
    cursor = reconcile_history_step(cursor_path, cursor, FakeRun(step=65))
    assert cursor.next_line_index == 64
    assert cursor.next_history_step == 65
    assert json.loads(cursor_path.read_text())["next_history_step"] == 65

    _append(metrics, json.dumps({"step": 266_240, "loss": 1.0}) + "\n")
    run = FakeRun()
    cursor, count = drain_metrics(metrics, cursor_path, cursor, run)

    assert count == 1
    assert run.records[0][1] == 65
    assert cursor.next_line_index == 65
    assert cursor.next_history_step == 66


def test_drain_waits_for_complete_newline(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    cursor_path = tmp_path / "cursor.json"
    complete = json.dumps({"step": 1, "loss": 2.0}) + "\n"
    partial = json.dumps({"step": 2, "loss": 1.0})
    metrics.write_text(complete + partial, encoding="utf-8")
    run = FakeRun()
    cursor = load_cursor(metrics, cursor_path, "run-2")
    cursor, count = drain_metrics(metrics, cursor_path, cursor, run)
    assert count == 1

    _append(metrics, "\n")
    cursor, count = drain_metrics(metrics, cursor_path, cursor, run)
    assert count == 1
    assert run.records[-1][0]["environment_step"] == 2


def test_drain_rejects_truncation(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    cursor_path = tmp_path / "cursor.json"
    metrics.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    run = FakeRun()
    cursor = load_cursor(metrics, cursor_path, "run-3")
    cursor, _ = drain_metrics(metrics, cursor_path, cursor, run)
    metrics.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="truncated"):
        drain_metrics(metrics, cursor_path, cursor, run)


def test_eval_flattening_keeps_aggregates_and_skips_arrays() -> None:
    payload = record_to_wandb(
        {
            "step": 100,
            "eval": {
                "success_rate": 0.5,
                "returns": [1.0, 2.0],
                "per_task": {
                    "peg": {
                        "task": "peg",
                        "success_rate": 0.75,
                    }
                },
            },
        }
    )

    assert payload["environment_step"] == 100
    assert payload["eval/success_rate"] == 0.5
    assert payload["eval/per_task/peg/success_rate"] == 0.75
    assert "eval/returns" not in payload
    assert "eval/per_task/peg/task" not in payload


def test_training_success_rate_has_clear_fractional_alias() -> None:
    payload = record_to_wandb(
        {
            "step": 4096,
            "train_success_rate_20": 0.35,
        }
    )

    assert payload["train/train_success_rate_20"] == 0.35
    assert payload["train/success_rate_20"] == 0.35


def test_mmbench_score_is_logged_without_undefined_success(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({
        "score_mean": 0.3,
        "success_rate": None,
        "per_task": {"walker-stand": {"score_mean": 0.3, "success_rate": None}},
        "action_timing": {"mean_ms": 12.0},
    }))
    run = FakeRun()
    cursor = SyncCursor(run_id="mmbench-test", device=1, inode=2)
    _log_benchmark(
        run, benchmark, environment_step=40000,
        cursor_path=tmp_path / "cursor.json", cursor=cursor,
    )
    history = run.records[0][0]
    assert history["eval/score_mean"] == 0.3
    assert history["eval/per_task/walker-stand/score_mean"] == 0.3
    assert history["eval/action_timing/mean_ms"] == 12.0
    assert "eval/success_rate" not in history
    training = record_to_wandb({
        "step": 1000, "train_score_mean_20": 0.2,
        "train_success_rate_20": None,
    })
    assert training["train/score_mean_20"] == 0.2
    assert "train/success_rate_20" not in training


def test_frozen_benchmark_success_is_summary_and_final_history(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "success_rate": 0.5,
                "worst_embodiment_success": 0.25,
                "return_mean": 12.0,
                "per_task": {
                    "peg": {"success_rate": 0.75, "return_mean": 15.0},
                    "gear": {"success_rate": 0.25, "return_mean": 9.0},
                },
            }
        ),
        encoding="utf-8",
    )
    run = FakeRun()

    cursor_path = tmp_path / "cursor.json"
    cursor = SyncCursor(
        run_id="run-1",
        device=1,
        inode=2,
        next_line_index=255,
        next_history_step=256,
    )
    cursor = _log_benchmark(
        run,
        benchmark,
        environment_step=1_044_480,
        cursor_path=cursor_path,
        cursor=cursor,
    )

    assert run.summary["eval/success_rate"] == 0.5
    assert run.summary["eval/per_task/peg/success_rate"] == 0.75
    history, step, commit = run.records[0]
    assert history == {
        "environment_step": 1_044_480,
        "eval/success_rate": 0.5,
        "eval/worst_embodiment_success": 0.25,
        "eval/per_task/peg/success_rate": 0.75,
        "eval/per_task/gear/success_rate": 0.25,
    }
    assert step == 256
    assert commit is True
    assert cursor.next_line_index == 255
    assert cursor.next_history_step == 257
    assert json.loads(cursor_path.read_text())["next_history_step"] == 257


def test_refresh_run_config_updates_stale_resolved_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config_resolved.yaml"
    config_path.write_text(
        "total_steps: 262144\nactor_lr: 0.0003\n",
        encoding="utf-8",
    )
    run = FakeRun()

    refreshed = _refresh_run_config(
        run,
        config_path,
        {"total_steps": 131072, "actor_lr": 0.000003},
    )

    assert refreshed == {"total_steps": 262144, "actor_lr": 0.0003}
    assert run.config == refreshed
    assert run.config.updates == [(refreshed, True)]


def test_startup_waits_for_metrics_and_optional_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    config = tmp_path / "config_resolved.yaml"
    metrics.touch()
    states: list[str] = []

    def running_state(_job_id: str) -> str:
        states.append("RUNNING")
        return "RUNNING"

    def finish_config(_seconds: float) -> None:
        config.write_text("env: isaaclab\n", encoding="utf-8")

    monkeypatch.setattr("m4po.wandb_sync.query_slurm_state", running_state)
    monkeypatch.setattr("m4po.wandb_sync.time.sleep", finish_config)

    _wait_for_startup_inputs(metrics, config, "123", 0.01)

    assert states == ["RUNNING"]

    config.unlink()
    _wait_for_startup_inputs(metrics, None, "123", 0.01)
    assert states == ["RUNNING"]


def test_startup_reports_missing_inputs_when_training_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    config = tmp_path / "config_resolved.yaml"
    monkeypatch.setattr(
        "m4po.wandb_sync.query_slurm_state", lambda _job_id: "FAILED"
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"Training entered FAILED.*metrics=.*config=",
    ):
        _wait_for_startup_inputs(metrics, config, "123", 0.01)


def test_wandb_tags_are_derived_from_environment_and_tasks() -> None:
    cartpole_tags = _wandb_tags(
        {
            "env": "isaaclab",
            "tasks": "Isaac-Cartpole-Direct-v0,Isaac-Cartpole-v0",
            "multitask": True,
        }
    )

    assert cartpole_tags == (
        "m4po",
        "isaaclab",
        "task:Isaac-Cartpole-Direct-v0",
        "task:Isaac-Cartpole-v0",
        "multitask",
    )
    assert "forge" not in cartpole_tags

    forge_tags = _wandb_tags(
        {
            "env": "isaaclab",
            "tasks": (
                "Isaac-Forge-PegInsert-Direct-v0,"
                "Isaac-Forge-GearMesh-Direct-v0"
            ),
            "multitask": True,
        }
    )
    assert "isaaclab" in forge_tags
    assert "task:Isaac-Forge-PegInsert-Direct-v0" in forge_tags
    assert "multitask" in forge_tags
