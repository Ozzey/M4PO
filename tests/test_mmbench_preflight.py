from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from m4po import mmbench_preflight as preflight
from test_mmbench_env import catalog as catalog
from test_mmbench_env import config, factory


def test_preflight_checks_full_native_episode_and_undefined_success(catalog):
    cfg = config(catalog)
    cfg.validate()
    result = preflight.inspect_task(cfg, "native-0", native_factory=factory)
    assert result["status"] == "passed"
    assert result["episode_length"] == result["metadata_horizon"] == 3
    assert result["return"] == 6
    assert result["score"] == 0.375
    assert result["success"] is None
    assert result["native_action_dim"] == 2
    assert result["native_state_dim"] == 3
    assert result["task_embedding_dim"] == 512
    assert result["state_padding"] == 128 and result["action_padding"] == 16
    assert result["truncated"] and not result["terminated"]


def test_preflight_rejects_nonfinite_native_score(catalog):
    def invalid_factory(**kwargs):
        native = factory(**kwargs)
        original = native.step

        def invalid_step(action):
            observation, reward, terminated, truncated, info = original(action)
            info["score"] = float("nan")
            return observation, reward, terminated, truncated, info

        native.step = invalid_step
        return native

    cfg = config(catalog)
    cfg.validate()
    with pytest.raises(ValueError, match="finite scalar"):
        preflight.inspect_task(cfg, "native-0", native_factory=invalid_factory)


def test_parent_reports_every_failure_and_never_labels_subset_full(
    catalog, tmp_path, monkeypatch
):
    path, out = tmp_path / "pilot.yaml", tmp_path / "results.json"
    cfg = config(catalog)
    cfg.save_yaml(path)

    def run_child(config_path, task, timeout):
        assert config_path == path
        assert timeout == 7
        return {"task": task, "status": "passed" if task == "native-0" else "failed"}

    monkeypatch.setattr(preflight, "_run_child", run_child)
    report = preflight.run_preflight(path, out, timeout=7, workers=2)
    assert report["complete"]
    assert report["completed_task_count"] == 2
    assert report["passed_task_count"] == report["failed_task_count"] == 1
    assert not report["all_configured_tasks_passed"]
    assert not report["full_suite_passed"]
    assert json.loads(out.read_text()) == report


def test_full_suite_requires_all_200_tasks_to_pass(catalog, tmp_path, monkeypatch):
    path, out = tmp_path / "all.yaml", tmp_path / "results.json"
    cfg = config(catalog)
    cfg.tasks = None
    cfg.save_yaml(path)
    visited = []

    def run_child(config_path, task, timeout):
        visited.append(task)
        return {"task": task, "status": "passed"}

    monkeypatch.setattr(preflight, "_run_child", run_child)
    report = preflight.run_preflight(path, out)
    assert len(set(visited)) == 200
    assert report["full_suite_passed"]
    assert report["failed_task_count"] == 0


def test_subprocess_exception_is_recorded_not_skipped(catalog, tmp_path, monkeypatch):
    path, out = tmp_path / "pilot.yaml", tmp_path / "results.json"
    config(catalog).save_yaml(path)

    def fail(*args):
        raise RuntimeError("dependency missing")

    monkeypatch.setattr(preflight, "_run_child", fail)
    report = preflight.run_preflight(path, out)
    assert report["failed_task_count"] == 2
    assert len(report["per_task"]) == 2
    assert not report["all_configured_tasks_passed"]


def test_slurm_check_rejects_login_node_even_with_job_environment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "login-1")

    def scontrol(command, **kwargs):
        return SimpleNamespace(
            stdout="JobId=123 JobState=RUNNING NodeList=gpu-[1-2]\n"
            if "job" in command
            else "gpu-1\ngpu-2\n"
        )

    monkeypatch.setattr(preflight.subprocess, "run", scontrol)
    with pytest.raises(RuntimeError, match="not in Slurm"):
        preflight.verify_slurm_allocation()
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "gpu-1.cluster")
    assert preflight.verify_slurm_allocation()["verified"]


def test_slurm_check_rejects_absent_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="running Slurm allocation"):
        preflight.verify_slurm_allocation()


def test_native_error_output_redacts_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "long-private-wandb-credential")
    result = preflight._sanitized_tail(
        "token=private-other-token api_key=secret-value Bearer abc123\n"
        "long-private-wandb-credential"
    )
    assert "private-other-token" not in result
    assert "secret-value" not in result
    assert "abc123" not in result
    assert "long-private-wandb-credential" not in result
