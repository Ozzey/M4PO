from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit


PREPARED_STATUS = "prepared_not_started"
DATASET_REPOSITORY = "nicklashansen/mmbench"
DATASET_REVISION = "a59d457df617400d3e45a5158c8deac8a52055b4"
METRIC_DEFINITIONS = {
    "metrics/rolling_success_definition": (
        "Task-macro native success over each contributing task's last 20 completed "
        "episodes; tasks without native success are excluded, never treated as failures. "
        "See train/rolling_success_task_coverage and episode counts; this is not "
        "all-200 MMBench success."
    ),
    "metrics/rolling_score_definition": (
        "Task-macro native normalized score over each contributing task's last 20 "
        "completed episodes, including reward-based tasks; see "
        "train/rolling_score_task_coverage. Scores are not success percentages."
    ),
    "metrics/evaluation_definition": (
        "Frozen-policy native task metrics, separate from exploratory training "
        "episodes. Undefined native success is omitted, with coverage retained."
    ),
}


def verify_slurm_allocation() -> dict[str, Any]:
    """Check the scheduler, not just inherited login-shell environment variables."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id or not job_id.isdigit():
        raise RuntimeError("W&B preparation requires a running Slurm allocation")
    job = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", job))
    if (
        fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or not fields.get("NodeList")
    ):
        raise RuntimeError(f"Slurm job {job_id} is not a running node allocation")
    nodes = subprocess.run(
        ["scontrol", "show", "hostnames", fields["NodeList"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.split()
    hostname = socket.gethostname()
    if hostname.split(".", 1)[0] not in {node.split(".", 1)[0] for node in nodes}:
        raise RuntimeError(
            f"Host {hostname} is outside Slurm job {job_id}; use srun on its compute node"
        )
    return {"job_id": job_id, "hostname": hostname, "verified": True}


def _validate_url(url: Any, run_path: str) -> str:
    if not isinstance(url, str):
        raise RuntimeError("W&B did not return an online run URL")
    parsed = urlsplit(url)
    entity, project, run_id = run_path.split("/")
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.path.rstrip("/") != f"/{entity}/{project}/runs/{run_id}"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("W&B run URL does not match the requested run identity")
    return url.rstrip("/")


def _validate_local_run(directory: Path, run_path: str) -> None:
    if (directory / "metrics.jsonl").exists():
        raise RuntimeError(f"Refusing to prepare a run with existing metrics: {directory}")
    if (directory / "checkpoints").exists() and any(
        (directory / "checkpoints").iterdir()
    ):
        raise RuntimeError(f"Refusing to prepare a run with checkpoints: {directory}")
    url_file = directory / "wandb_url.txt"
    if url_file.exists():
        _validate_url(url_file.read_text(encoding="utf-8").strip(), run_path)


def _existing_run(api, wandb, run_path: str):
    try:
        return api.run(run_path)
    except (ValueError, wandb.errors.CommError) as exc:
        # Authentication and network failures must not masquerade as absent runs.
        message = str(exc).lower()
        if message.startswith("could not find run "):
            return None
        raise


def _validate_remote_run(run, config: dict[str, Any]) -> None:
    summary = dict(run.summary)
    last_history_step = run.lastHistoryStep
    if (
        run.state != "finished"
        or type(last_history_step) is not int
        or last_history_step != -1
        or summary.get("pipeline/status") != PREPARED_STATUS
        or any(
            key in {"_step", "environment_step", "pretrain_update", "sync/source_records"}
            or key.startswith(("train/", "eval/", "pretrain/", "benchmark/"))
            for key in summary
        )
    ):
        raise RuntimeError("Refusing to overwrite an existing W&B run with progress")
    if any(run.config.get(key) != value for key, value in config.items()):
        raise RuntimeError("Existing prepared W&B run has a different planned protocol")


def _atomic_url(path: Path, url: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(url + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_runs(
    *,
    pretrain_run_dir: str | Path,
    online_run_dir: str | Path,
    pretrain_run_id: str,
    online_run_id: str,
    entity: str = "adityanarendra5",
    project: str = "m4po-mmbench",
) -> dict[str, str]:
    allocation = verify_slurm_allocation()
    for value in (entity, project, pretrain_run_id, online_run_id):
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            raise ValueError("W&B identities must contain only letters, digits, '-' and '_'")
    directories = [Path(pretrain_run_dir).resolve(), Path(online_run_dir).resolve()]
    if directories[0] == directories[1] or pretrain_run_id == online_run_id:
        raise ValueError("Pretraining and online training require distinct directories and IDs")
    run_ids = [pretrain_run_id, online_run_id]
    paths = [f"{entity}/{project}/{run_id}" for run_id in run_ids]
    for directory, path in zip(directories, paths):
        _validate_local_run(directory, path)

    # Keep this entry point usable in the small W&B environment without Torch.
    # Allocation and local-progress checks must precede even the SDK import.
    wandb = importlib.import_module("wandb")
    api = wandb.Api(timeout=30)
    common = {
        "algorithm": "M4PO",
        "env": "mmbench",
        "mmbench_task_set": "soup",
        "num_tasks": 200,
        "num_domains": 10,
        "seed": 0,
        "use_images": False,
        "learning_mode": "off_policy",
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "planned_pretrain_updates": 200_000,
        "planned_online_transitions": 100_000_000,
        "online_demo_mixing": False,
        "rolling_episode_window_per_task": 20,
        "pretrain_run_id": pretrain_run_id,
        "online_run_id": online_run_id,
    }
    configs = [
        {
            **common,
            "pipeline_stage": "demonstration_pretraining",
            "planned_environment_steps": 0,
            "actor_objective": "masked_behavior_cloning_plus_entropy",
        },
        {
            **common,
            "pipeline_stage": "online_training",
            "planned_environment_steps": 100_000_000,
            "actor_objective": "off_policy_replay_q_maximization_plus_entropy",
            "initialization_kind": "demonstration_pretraining",
        },
    ]
    existing = [_existing_run(api, wandb, path) for path in paths]
    for run, config in zip(existing, configs):
        if run is not None:
            _validate_remote_run(run, config)

    urls = {}
    for index, (directory, path, config, remote) in enumerate(
        zip(directories, paths, configs, existing)
    ):
        directory.mkdir(parents=True, exist_ok=True)
        stage = "pretrain" if index == 0 else "online"
        if remote is not None:
            url = _validate_url(remote.url, path)
        else:
            run = wandb.init(
                entity=entity,
                project=project,
                id=run_ids[index],
                # A run created after the API check must not be taken over.
                # Training sidecars may later resume this prepared ID with allow.
                resume="never",
                mode="online",
                force=True,
                dir=str(directory),
                name=f"M4PO MMBench {'demo pretraining' if index == 0 else 'demo-initialized online RL'} seed 0",
                group="mmbench-demo-pretrained-seed0",
                job_type=config["pipeline_stage"],
                notes=(
                    "Prepared only; training has not started. All 200 official MMBench "
                    "tasks across 10 domains, seed 0, state observations. Planned: "
                    "200,000 offline M4PO BC/world-model/TD-Q updates on official "
                    "demonstrations, then 100M online transitions from those weights. "
                    "Unlike NEWT's 50:50 online demonstration/replay mixture, M4PO "
                    "does not mix demonstrations during online RL. Native success "
                    "is unavailable for some tasks; use normalized scores and coverage."
                ),
                tags=["m4po", "mmbench", "200-tasks", "demonstrations", stage, "seed0"],
                config=config,
                settings=wandb.Settings(init_timeout=120, x_disable_stats=True),
            )
            if run is None:
                raise RuntimeError("W&B did not initialize an online run")
            try:
                url = _validate_url(run.url, path)
                if run.step != 0 or run.resumed:
                    raise RuntimeError("W&B run appeared during preparation; refusing to reset it")
                run.define_metric("environment_step")
                run.define_metric("pretrain_update")
                run.define_metric("train/*", step_metric="environment_step")
                run.define_metric("eval/*", step_metric="environment_step")
                run.define_metric("pretrain/*", step_metric="pretrain_update")
                run.summary.update(
                    {
                        **METRIC_DEFINITIONS,
                        "pipeline/status": PREPARED_STATUS,
                        "pipeline/stage": config["pipeline_stage"],
                        "pipeline/prepared_allocation": allocation,
                        "planned/task_count": 200,
                        "planned/domain_count": 10,
                        "planned/pretrain_updates": 200_000,
                        "planned/online_transitions": 100_000_000,
                        "planned/environment_steps_this_stage": config["planned_environment_steps"],
                    }
                )
            except Exception:
                run.finish(exit_code=1)
                raise
            else:
                run.finish()
        _atomic_url(directory / "wandb_url.txt", url)
        urls[stage] = url
        print(f"WANDB_{stage.upper()}_RUN_URL={url}", flush=True)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare honest MMBench W&B run links from an allocated compute node."
    )
    parser.add_argument("--pretrain-run-dir", required=True)
    parser.add_argument("--online-run-dir", required=True)
    parser.add_argument("--pretrain-run-id", required=True)
    parser.add_argument("--online-run-id", required=True)
    parser.add_argument("--entity", default="adityanarendra5")
    parser.add_argument("--project", default="m4po-mmbench")
    print(json.dumps(prepare_runs(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
