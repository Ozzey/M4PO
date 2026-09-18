from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class SyncCursor:
    run_id: str
    device: int
    inode: int
    byte_offset: int = 0
    next_line_index: int = 0
    next_history_step: int = 0


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        file.write(value)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_cursor(metrics_path: Path, cursor_path: Path, run_id: str) -> SyncCursor:
    stat = metrics_path.stat()
    if not cursor_path.exists():
        return SyncCursor(run_id=run_id, device=stat.st_dev, inode=stat.st_ino)

    try:
        raw = json.loads(cursor_path.read_text(encoding="utf-8"))
        cursor = SyncCursor(
            run_id=str(raw["run_id"]),
            device=int(raw["device"]),
            inode=int(raw["inode"]),
            byte_offset=int(raw["byte_offset"]),
            next_line_index=int(raw["next_line_index"]),
            # Cursors written before history and source positions were split used
            # the source line index for both.  W&B's resumed run step is
            # reconciled after initialization to account for any later history
            # records, such as a frozen-policy benchmark.
            next_history_step=int(
                raw.get("next_history_step", raw["next_line_index"])
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid W&B sync cursor: {cursor_path}") from exc
    if cursor.run_id != run_id:
        raise ValueError(f"Cursor belongs to W&B run {cursor.run_id!r}, not {run_id!r}")
    _validate_source_file(stat, cursor)
    return cursor


def _validate_source_file(stat: os.stat_result, cursor: SyncCursor) -> None:
    if stat.st_dev != cursor.device or stat.st_ino != cursor.inode:
        raise RuntimeError("Metrics file was replaced after W&B synchronization began")
    if stat.st_size < cursor.byte_offset:
        raise RuntimeError("Metrics file was truncated after W&B synchronization began")
    if (
        cursor.byte_offset < 0
        or cursor.next_line_index < 0
        or cursor.next_history_step < 0
    ):
        raise ValueError("W&B sync cursor counters must be non-negative")


def reconcile_history_step(
    cursor_path: Path,
    cursor: SyncCursor,
    run: Any,
) -> SyncCursor:
    """Advance the local history position to W&B's next accepted run step."""

    remote_step = run.step
    if not isinstance(remote_step, int) or isinstance(remote_step, bool):
        raise TypeError(f"W&B run.step must be an integer, got {remote_step!r}")
    if remote_step < 0:
        raise ValueError("W&B run.step must be non-negative")
    cursor = replace(
        cursor,
        next_history_step=max(cursor.next_history_step, remote_step),
    )
    _atomic_json(cursor_path, asdict(cursor))
    return cursor


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_scalars(
    value: Any,
    *,
    prefix: str,
    output: dict[str, Any],
    numeric_only: bool,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_scalars(
                child,
                prefix=child_prefix,
                output=output,
                numeric_only=numeric_only,
            )
        return
    if isinstance(value, (list, tuple)) or value is None:
        return
    if numeric_only:
        if _is_numeric(value) and (
            not isinstance(value, float) or math.isfinite(value)
        ):
            output[prefix] = value
        return
    if isinstance(value, (str, bool, int, float)):
        output[prefix] = value


def record_to_wandb(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one JSONL record to compact W&B scalar metrics."""

    output: dict[str, Any] = {}
    source_step = record.get("step")
    if _is_numeric(source_step):
        output["environment_step"] = int(source_step)
    source_time = record.get("time")
    if _is_numeric(source_time):
        output["source/wall_time"] = source_time

    if record.get("phase") == "demonstration_pretraining":
        updates = record.get("pretrain_updates")
        if _is_numeric(updates):
            output["pretrain_update"] = int(updates)
        _flatten_scalars(
            {key: value for key, value in record.items() if key not in {"step", "time"}},
            prefix="pretrain", output=output, numeric_only=False,
        )
        return output

    if isinstance(record.get("eval"), Mapping):
        _flatten_scalars(
            record["eval"],
            prefix="eval",
            output=output,
            numeric_only=True,
        )
        return output

    training = {
        key: value
        for key, value in record.items()
        if key not in {"step", "time", "eval"}
    }
    _flatten_scalars(
        training,
        prefix="train",
        output=output,
        numeric_only=False,
    )
    success_rate = training.get("train_success_rate_20")
    if _is_numeric(success_rate) and (
        not isinstance(success_rate, float) or math.isfinite(success_rate)
    ):
        # Keep the source-compatible name while exposing a concise W&B series.
        output["train/success_rate_20"] = success_rate
    score_mean = training.get("train_score_mean_20")
    if _is_numeric(score_mean) and math.isfinite(score_mean):
        output["train/score_mean_20"] = score_mean
    rolling_aliases = {
        "train_success_rate_supported_tasks_20": "train/rolling_success_rate",
        "train_success_task_coverage": "train/rolling_success_task_coverage",
        "train_success_tasks": "train/rolling_success_task_count",
        "train_success_episodes_in_window": "train/rolling_success_episode_count",
        "train_score_macro_20": "train/rolling_normalized_score",
        "train_score_task_coverage": "train/rolling_score_task_coverage",
    }
    for source, destination in rolling_aliases.items():
        value = training.get(source)
        if _is_numeric(value) and math.isfinite(value):
            output[destination] = value
    per_task = training.get("train_per_task")
    if isinstance(per_task, Mapping):
        for task, metrics in per_task.items():
            if not isinstance(metrics, Mapping):
                continue
            for source, name in (
                ("success_rate_20", "rolling_success_rate"),
                ("score_mean_20", "rolling_normalized_score"),
            ):
                value = metrics.get(source)
                if _is_numeric(value) and math.isfinite(value):
                    output[f"train/task/{task}/{name}"] = value
    return output


def drain_metrics(
    metrics_path: Path,
    cursor_path: Path,
    cursor: SyncCursor,
    run: Any,
) -> tuple[SyncCursor, int]:
    """Log every complete JSONL record after ``cursor.byte_offset``."""

    _validate_source_file(metrics_path.stat(), cursor)
    count = 0
    with open(metrics_path, "rb") as file:
        file.seek(cursor.byte_offset)
        while True:
            start = file.tell()
            raw_line = file.readline()
            if not raw_line:
                break
            if not raw_line.endswith(b"\n"):
                file.seek(start)
                break
            end = file.tell()
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid JSONL record at line index {cursor.next_line_index}"
                ) from exc
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"JSONL record {cursor.next_line_index} is not an object"
                )
            run.log(
                record_to_wandb(record),
                step=cursor.next_history_step,
                commit=True,
            )
            cursor = replace(
                cursor,
                byte_offset=end,
                next_line_index=cursor.next_line_index + 1,
                next_history_step=cursor.next_history_step + 1,
            )
            _atomic_json(cursor_path, asdict(cursor))
            count += 1
    return cursor, count


def query_slurm_state(job_id: str | None) -> str:
    if not job_id:
        return "NOT_CONFIGURED"
    queued = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if queued.returncode == 0 and queued.stdout.strip():
        return queued.stdout.splitlines()[0].strip().split("+")[0]

    controlled = subprocess.run(
        ["scontrol", "--oneliner", "show", "job", job_id],
        check=False,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(?:^|\s)JobState=([A-Z_]+)", controlled.stdout)
    return match.group(1) if match else "UNKNOWN"


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _refresh_run_config(
    run: Any,
    config_path: Path | None,
    current_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reload trainer-owned config values and propagate changes to W&B."""

    if config_path is None or not config_path.exists():
        return dict(current_config)
    refreshed = _read_yaml(config_path)
    # A trainer rewriting the file can expose a momentarily empty file.  The
    # resolved M4PO config is never intentionally empty, so retain the last
    # complete view until content is available again.
    if not refreshed and current_config:
        return dict(current_config)
    if refreshed != current_config:
        run.config.update(refreshed, allow_val_change=True)
    return refreshed


def _wandb_tags(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Build stable run tags from the resolved environment configuration."""

    tags = ["m4po"]
    environment = str(config.get("env", "")).strip().lower()
    if environment:
        tags.append(environment)

    configured_tasks = config.get("tasks")
    if isinstance(configured_tasks, str):
        task_names = [
            value.strip() for value in configured_tasks.split(",") if value.strip()
        ]
    elif isinstance(configured_tasks, (list, tuple)):
        task_names = [str(value).strip() for value in configured_tasks if str(value).strip()]
    else:
        task = str(config.get("task", "")).strip()
        task_names = [task] if task else []
    tags.extend(f"task:{name}" for name in task_names)

    if bool(config.get("multitask")) or len(task_names) > 1:
        tags.append("multitask")
    if bool(config.get("multiembodiment")):
        tags.append("multiembodiment")
    return tuple(dict.fromkeys(tags))


def _benchmark_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Benchmark output must contain a JSON object")
    flattened: dict[str, Any] = {}
    _flatten_scalars(
        payload,
        prefix="eval",
        output=flattened,
        numeric_only=True,
    )
    return flattened


def _last_environment_step(path: Path) -> int | None:
    latest: int | None = None
    with open(path, encoding="utf-8") as file:
        for raw_line in file:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            step = record.get("step")
            if _is_numeric(step) and (
                not isinstance(step, float) or math.isfinite(step)
            ):
                latest = int(step)
    return latest


def _last_pretrain_update(path: Path) -> int | None:
    latest = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("phase") == "demonstration_pretraining":
                value = row.get("pretrain_updates")
                if type(value) is int and value >= 0:
                    latest = value
    return latest


def _final_environment_step(
    config: Mapping[str, Any], metrics_path: Path
) -> int | None:
    # Actual progress wins over the intended budget (offline updates stay at 0).
    if metrics_path.exists():
        actual = _last_environment_step(metrics_path)
        if actual is not None:
            return actual
    configured = config.get("total_steps")
    if _is_numeric(configured) and (
        not isinstance(configured, float) or math.isfinite(configured)
    ):
        return int(configured)
    return _last_environment_step(metrics_path)


def _log_benchmark(
    run: Any,
    path: Path | None,
    *,
    environment_step: int | None,
    cursor_path: Path,
    cursor: SyncCursor,
) -> SyncCursor:
    summary = _benchmark_summary(path)
    for key, value in summary.items():
        run.summary[key] = value

    if environment_step is None:
        return cursor
    benchmark_metrics = {
        key: value
        for key, value in summary.items()
        if key == "eval/success_rate"
        or key == "eval/worst_embodiment_success"
        or (key.startswith("eval/per_task/") and key.endswith("/success_rate"))
        or key == "eval/score_mean"
        or key == "eval/worst_embodiment_score"
        or (key.startswith("eval/per_task/") and key.endswith("/score_mean"))
        or key.startswith("eval/action_timing/")
    }
    if benchmark_metrics:
        run.log(
            {"environment_step": environment_step, **benchmark_metrics},
            step=cursor.next_history_step,
            commit=True,
        )
        cursor = replace(
            cursor,
            next_history_step=cursor.next_history_step + 1,
        )
        _atomic_json(cursor_path, asdict(cursor))
    return cursor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill and follow M4PO JSONL metrics into a W&B run."
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cursor", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--evaluation-job-id")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def _wait_for_startup_inputs(
    metrics_path: Path,
    config_path: Path | None,
    training_job_id: str,
    poll_seconds: float,
) -> None:
    """Wait for trainer-owned inputs while the source job can still create them."""

    while True:
        missing: list[tuple[str, Path]] = []
        if not metrics_path.exists():
            missing.append(("metrics", metrics_path))
        if config_path is not None and not config_path.exists():
            missing.append(("config", config_path))
        if not missing:
            return

        state = query_slurm_state(training_job_id)
        if state in TERMINAL_SLURM_STATES:
            details = ", ".join(f"{name}={path}" for name, path in missing)
            raise FileNotFoundError(
                f"Training entered {state} before required W&B sync input"
                f"{'s' if len(missing) != 1 else ''} appeared: {details}"
            )
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _wait_for_startup_inputs(
        args.metrics,
        args.config,
        args.training_job_id,
        args.poll_seconds,
    )

    import wandb

    resolved_config = _read_yaml(args.config)
    initial_config = dict(resolved_config)
    initial_config.update(
        {
            "source_training_slurm_job": args.training_job_id,
            "source_evaluation_slurm_job": args.evaluation_job_id,
            "metrics_source": str(args.metrics),
        }
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow",
        name=args.name,
        group=args.group,
        job_type="training",
        tags=_wandb_tags(initial_config),
        config=initial_config,
        force=True,
        settings=wandb.Settings(init_timeout=120, x_disable_stats=True),
    )
    if run is None or not run.url:
        raise RuntimeError("W&B did not return an online run URL")
    run.define_metric("environment_step")
    run.define_metric("pretrain_update")
    run.define_metric("pretrain/*", step_metric="pretrain_update")
    run.define_metric("train/*", step_metric="environment_step")
    run.define_metric("eval/*", step_metric="environment_step")
    run.summary["metrics/rolling_success_definition"] = (
        "Task-macro native success over each contributing task's last 20 completed "
        "episodes; tasks without native success are excluded, never treated as failures. "
        "See rolling_success_task_coverage; this is not all-200 MMBench success."
    )
    run.summary["metrics/rolling_score_definition"] = (
        "Task-macro native normalized score over each task's last 20 completed "
        "episodes; see rolling_score_task_coverage."
    )
    _atomic_text(args.url_file, f"{run.url}\n")
    print(f"WANDB_RUN_URL={run.url}", flush=True)

    cursor = load_cursor(args.metrics, args.cursor, args.run_id)
    cursor = reconcile_history_step(args.cursor, cursor, run)
    training_state = "UNKNOWN"
    evaluation_state = "NOT_CONFIGURED"
    terminal_idle_polls = 0
    try:
        while True:
            resolved_config = _refresh_run_config(
                run,
                args.config,
                resolved_config,
            )
            cursor, logged = drain_metrics(args.metrics, args.cursor, cursor, run)
            training_state = query_slurm_state(args.training_job_id)
            evaluation_state = query_slurm_state(args.evaluation_job_id)
            run.summary["slurm/training_state"] = training_state
            run.summary["slurm/evaluation_state"] = evaluation_state
            run.summary["pipeline/status"] = (
                "job_completed" if training_state == "COMPLETED"
                else "job_failed" if training_state in TERMINAL_SLURM_STATES
                else "running"
            )
            run.summary["sync/source_records"] = cursor.next_line_index

            training_terminal = training_state in TERMINAL_SLURM_STATES
            evaluation_terminal = evaluation_state in TERMINAL_SLURM_STATES
            pipeline_terminal = training_terminal and (
                training_state != "COMPLETED"
                or not args.evaluation_job_id
                or evaluation_terminal
            )
            if pipeline_terminal and logged == 0:
                terminal_idle_polls += 1
                if terminal_idle_polls >= 2:
                    break
            else:
                terminal_idle_polls = 0
            time.sleep(args.poll_seconds)

        cursor, _ = drain_metrics(args.metrics, args.cursor, cursor, run)
        resolved_config = _refresh_run_config(
            run,
            args.config,
            resolved_config,
        )
        cursor = _log_benchmark(
            run,
            args.benchmark,
            environment_step=_final_environment_step(
                resolved_config,
                args.metrics,
            ),
            cursor_path=args.cursor,
            cursor=cursor,
        )
        for path in (args.config, args.metrics, args.benchmark):
            if path is not None and path.exists():
                run.save(str(path), base_path=str(path.parent), policy="now")
        if (
            training_state == "COMPLETED"
            and args.checkpoint is not None
            and args.checkpoint.exists()
        ):
            artifact = wandb.Artifact(
                f"{args.run_id}-checkpoint",
                type="model",
                metadata={
                    "environment_step": _final_environment_step(resolved_config, args.metrics),
                    "pretrain_updates": _last_pretrain_update(args.metrics),
                },
            )
            artifact.add_file(str(args.checkpoint), name="latest.pt")
            run.log_artifact(artifact, aliases=["latest", "final"])
    finally:
        run.finish()


if __name__ == "__main__":
    main()
