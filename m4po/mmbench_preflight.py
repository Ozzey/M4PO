from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from m4po.common.config import M4POConfig
from m4po.envs.mmbench_env import MMBenchVectorEnv, load_mmbench_catalog


_RESULT_MARKER = "MMBENCH_PREFLIGHT_RESULT="
_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "mmbench_all.yaml"


def verify_slurm_allocation() -> dict[str, Any]:
    """Require the executing host to belong to a currently running allocation."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id or not job_id.isdigit():
        raise RuntimeError(
            "Preflight requires a running Slurm allocation, not a login shell"
        )
    job = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", job))
    if fields.get("JobState") != "RUNNING" or not fields.get("NodeList"):
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
            f"Host {hostname} is not in Slurm job {job_id}'s allocation; "
            "launch this command with srun on the allocated compute node"
        )
    return {"job_id": job_id, "hostname": hostname, "nodes": nodes, "verified": True}


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.isfinite(float(array)):
        raise ValueError(f"Native {name} must be a finite scalar")
    return float(array)


def inspect_task(cfg: M4POConfig, task: str, *, native_factory=None) -> dict[str, Any]:
    """Run one genuine native episode, including the adapter's terminal/reset path."""

    if task not in cfg.task_names:
        raise ValueError(f"Task {task!r} is not in the configured preflight task set")
    child_cfg = deepcopy(cfg)
    child_cfg.tasks = task
    child_cfg.task = task
    child_cfg.num_envs = 1
    child_cfg.validate()
    env = MMBenchVectorEnv(child_cfg, native_factory=native_factory)
    start = time.perf_counter()
    try:
        observation = env.reset()
        env.observation_spec.validate(observation)
        if not all(np.isfinite(value).all() for value in observation.values()):
            raise ValueError("Native reset observation contains non-finite values")
        rng = np.random.default_rng(child_cfg.seed)
        episode_return = 0.0
        terminal = None
        action_dim = int(env.action_masks[0].sum())
        state_dim = int(observation["state_mask"][0].sum())
        for step in range(1, env.max_episode_steps + 1):
            actions = rng.uniform(-1, 1, size=(1, env.action_dim)).astype(np.float32)
            actions *= env.action_masks
            observation, rewards, dones, infos = env.step(actions)
            env.observation_spec.validate(observation)
            if not all(np.isfinite(value).all() for value in observation.values()):
                raise ValueError("Native step observation contains non-finite values")
            episode_return += _finite_scalar(rewards[0], "reward")
            info = infos[0]
            _finite_scalar(info.get("score"), "score")
            if int(info["task_id"]) != 0 or int(info["embodiment_id"]) != 0:
                raise ValueError("Single-task preflight changed its native context")
            if bool(dones[0]) != (bool(info["terminated"]) or bool(info["truncated"])):
                raise ValueError("Native done flag disagrees with termination metadata")
            if dones[0]:
                terminal = info
                break
        if terminal is None:
            raise ValueError(
                "Native episode did not finish within its official horizon"
            )
        final_observation = terminal.get("terminal_observation")
        if not isinstance(final_observation, dict) or not all(
            np.isfinite(value).all() for value in final_observation.values()
        ):
            raise ValueError("Native terminal observation is missing or non-finite")
        success = terminal.get("success")
        if success is not None:
            success_array = np.asarray(success)
            if success_array.ndim != 0:
                raise ValueError("Native success is not scalar")
            success = float(success_array)
            success = success if np.isfinite(success) else None
            if success not in (None, 0.0, 1.0):
                raise ValueError("Native success is neither binary nor undefined")
        return {
            "task": task,
            "status": "passed",
            "domain": env.task_domains[0],
            "embodiment": env.embodiment_descriptions[0],
            "native_action_dim": action_dim,
            "native_state_dim": state_dim,
            "state_padding": env.observation_spec.state_dim,
            "action_padding": env.action_dim,
            "task_embedding_dim": int(env.task_contexts.shape[1]),
            "metadata_horizon": env.max_episode_steps,
            "episode_length": step,
            "return": episode_return,
            "score": _finite_scalar(terminal["score"], "score"),
            "success": success,
            "terminated": bool(terminal["terminated"]),
            "truncated": bool(terminal["truncated"]),
            "elapsed_seconds": time.perf_counter() - start,
            "catalog_sha256": env.environment_signature["catalog_sha256"],
            "native_sources_sha256": env.environment_signature["native_sources_sha256"],
        }
    finally:
        env.close()


def _sanitized_tail(value: str | bytes | None, limit: int = 4000) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
    for name, secret in os.environ.items():
        if len(secret) >= 8 and re.search(
            r"key|token|secret|password|credential", name, re.I
        ):
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)((?:api[_-]?key|token|password|secret|authorization)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)(Bearer\s+)\S+", r"\1[REDACTED]", text)
    return text[-limit:]


def _run_child(config: Path, task: str, timeout: float) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "m4po.mmbench_preflight",
        "--config",
        str(config),
        "--child-task",
        task,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    result = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                result = json.loads(line[len(_RESULT_MARKER) :])
            except (ValueError, TypeError):
                pass
            break
    if timed_out:
        result = {
            "task": task,
            "status": "timeout",
            "error": f"Native task exceeded {timeout:g} seconds",
        }
    elif not isinstance(result, dict) or result.get("task") != task:
        result = {
            "task": task,
            "status": "failed",
            "error": "Native subprocess did not return a valid task result",
        }
    elif process.returncode != 0 or result.get("status") != "passed":
        result["status"] = "failed"
    result["process_exit_code"] = int(process.returncode)
    result["process_elapsed_seconds"] = time.perf_counter() - started
    if result["status"] != "passed":
        result["stderr_tail"] = _sanitized_tail(stderr)
        result["stdout_tail"] = _sanitized_tail(stdout)
        result["error"] = _sanitized_tail(
            str(result.get("error", "Native task failed"))
        )
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def run_preflight(
    config: str | Path,
    out: str | Path,
    *,
    timeout: float = 120,
    workers: int = 1,
    require_slurm: bool = False,
) -> dict[str, Any]:
    """Audit every configured task; failed/missing native tasks never count as coverage."""

    if workers not in (1, 2) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "Preflight requires workers=1 or 2 and a positive finite timeout"
        )
    allocation = (
        verify_slurm_allocation()
        if require_slurm
        else {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "verified": False,
        }
    )
    config_path, output_path = Path(config).resolve(), Path(out).resolve()
    cfg = M4POConfig.from_yaml(config_path)
    if cfg.env != "mmbench":
        raise ValueError("MMBench preflight requires env: mmbench")
    _, task_sets, catalog_digest = load_mmbench_catalog(cfg.mmbench_root)
    tasks = list(cfg.task_names)
    full_suite = set(tasks) == set(task_sets["soup"])
    report = {
        "benchmark": "NEWT-MMBench-state",
        "purpose": "native environment preflight, not trained-policy evaluation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "allocation": allocation,
        "catalog_sha256": catalog_digest,
        "official_task_count": len(task_sets["soup"]),
        "configured_task_count": len(tasks),
        "full_suite_configured": full_suite,
        "required_tasks": tasks,
        "workers": workers,
        "timeout_seconds": timeout,
        "completed_task_count": 0,
        "passed_task_count": 0,
        "failed_task_count": 0,
        "complete": False,
        "all_configured_tasks_passed": False,
        "full_suite_passed": False,
        "per_task": {},
    }
    _write_report(output_path, report)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_run_child, config_path, task, timeout): task
            for task in tasks
        }
        for future in as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task": task,
                    "status": "failed",
                    "error": _sanitized_tail(str(exc)),
                }
            report["per_task"][task] = result
            report["completed_task_count"] += 1
            key = (
                "passed_task_count"
                if result.get("status") == "passed"
                else "failed_task_count"
            )
            report[key] += 1
            _write_report(output_path, report)
            print(
                f"[{report['completed_task_count']}/{len(tasks)}] {task}: {result['status']}",
                flush=True,
            )
    report["complete"] = report["completed_task_count"] == len(tasks)
    report["all_configured_tasks_passed"] = report["complete"] and report[
        "passed_task_count"
    ] == len(tasks)
    report["full_suite_passed"] = full_suite and report["all_configured_tasks_passed"]
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate genuine NEWT MMBench episodes in isolated subprocesses."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("runs/mmbench_preflight.json"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--child-task", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_task is not None:
        try:
            cfg = M4POConfig.from_yaml(args.config)
            result = inspect_task(cfg, args.child_task)
        except Exception as exc:
            result = {
                "task": args.child_task,
                "status": "failed",
                "error": _sanitized_tail(f"{type(exc).__name__}: {exc}"),
            }
        print(_RESULT_MARKER + json.dumps(result, allow_nan=False), flush=True)
        raise SystemExit(0 if result["status"] == "passed" else 1)
    report = run_preflight(
        args.config,
        args.out,
        timeout=args.timeout,
        workers=args.workers,
        require_slurm=args.require_slurm,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "passed": report["passed_task_count"],
                "failed": report["failed_task_count"],
                "full_suite_passed": report["full_suite_passed"],
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if report["all_configured_tasks_passed"] else 1)


if __name__ == "__main__":
    main()
