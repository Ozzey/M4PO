from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mmbench_wandb.py"
spec = importlib.util.spec_from_file_location("prepare_mmbench_wandb", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CommError(Exception):
    pass


class FakeRun:
    def __init__(self, **kwargs):
        self.url = (
            f"https://wandb.ai/{kwargs['entity']}/{kwargs['project']}/runs/{kwargs['id']}"
        )
        self.config = kwargs["config"]
        self.summary = {}
        self.definitions = []
        self.step = 0
        self.lastHistoryStep = -1
        self.resumed = False
        self.finished = False
        self.state = "running"

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def finish(self, exit_code=0):
        self.finished = True
        self.state = "finished" if exit_code == 0 else "failed"


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = {}
    calls = []

    def find_run(path):
        if path not in runs:
            raise CommError(f"Could not find run {path}")
        return runs[path]

    def initialize(**kwargs):
        calls.append(kwargs)
        run = FakeRun(**kwargs)
        runs[f"{kwargs['entity']}/{kwargs['project']}/{kwargs['id']}"] = run
        return run

    sdk = SimpleNamespace(
        Api=lambda **kwargs: SimpleNamespace(run=find_run),
        init=initialize,
        Settings=lambda **kwargs: kwargs,
        errors=SimpleNamespace(CommError=CommError),
    )
    monkeypatch.setattr(prepare.importlib, "import_module", lambda name: sdk)
    monkeypatch.setattr(
        prepare,
        "verify_slurm_allocation",
        lambda: {"job_id": "123", "hostname": "gpu1", "verified": True},
    )
    return SimpleNamespace(sdk=sdk, runs=runs, calls=calls)


def _arguments(tmp_path):
    return {
        "pretrain_run_dir": tmp_path / "pretrain",
        "online_run_dir": tmp_path / "online",
        "pretrain_run_id": "demo-s0",
        "online_run_id": "online-s0",
    }


def test_prepares_two_real_links_and_axes_without_fake_history(tmp_path, fake_wandb):
    urls = prepare.prepare_runs(**_arguments(tmp_path))
    assert len(fake_wandb.calls) == 2
    assert len(urls) == 2
    for stage, run in zip(("pretrain", "online"), fake_wandb.runs.values()):
        assert run.finished
        assert run.summary["pipeline/status"] == "prepared_not_started"
        assert run.config["num_tasks"] == 200
        assert run.config["num_domains"] == 10
        assert run.config["planned_pretrain_updates"] == 200_000
        assert run.config["planned_online_transitions"] == 100_000_000
        assert run.config["online_demo_mixing"] is False
        assert ("train/*", {"step_metric": "environment_step"}) in run.definitions
        assert ("eval/*", {"step_metric": "environment_step"}) in run.definitions
        assert ("pretrain/*", {"step_metric": "pretrain_update"}) in run.definitions
        assert not any(key.startswith(("train/", "eval/", "pretrain/")) for key in run.summary)
        assert (tmp_path / stage / "wandb_url.txt").read_text().strip() == urls[stage]
    assert fake_wandb.calls[0]["config"]["planned_environment_steps"] == 0
    assert fake_wandb.calls[1]["config"]["planned_environment_steps"] == 100_000_000
    assert all(
        call["resume"] == "never" and call["mode"] == "online"
        for call in fake_wandb.calls
    )


def test_prepared_runs_are_idempotent_without_resuming_or_changing_remote(tmp_path, fake_wandb):
    first = prepare.prepare_runs(**_arguments(tmp_path))
    second = prepare.prepare_runs(**_arguments(tmp_path))
    assert first == second
    assert len(fake_wandb.calls) == 2


def test_local_metrics_refused_before_sdk_import(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "verify_slurm_allocation", lambda: {})
    monkeypatch.setattr(
        prepare.importlib,
        "import_module",
        lambda name: pytest.fail("SDK must not be imported for an existing local run"),
    )
    (tmp_path / "online").mkdir()
    (tmp_path / "online" / "metrics.jsonl").touch()
    with pytest.raises(RuntimeError, match="existing metrics"):
        prepare.prepare_runs(**_arguments(tmp_path))


@pytest.mark.parametrize(
    "progress", ["running", "_step", "history_step_zero", "train/loss", "pipeline/status"]
)
def test_remote_progress_is_never_reset(tmp_path, fake_wandb, progress):
    prepare.prepare_runs(**_arguments(tmp_path))
    run = list(fake_wandb.runs.values())[1]
    if progress == "running":
        run.state = "running"
    elif progress == "history_step_zero":
        run.lastHistoryStep = 0
    elif progress == "pipeline/status":
        run.summary[progress] = "job_completed"
    else:
        run.summary[progress] = 0
    with pytest.raises(RuntimeError, match="with progress"):
        prepare.prepare_runs(**_arguments(tmp_path))
    assert len(fake_wandb.calls) == 2


def test_authentication_errors_are_not_treated_as_missing_runs(tmp_path, fake_wandb):
    def denied(_path):
        raise CommError("Authentication denied")

    fake_wandb.sdk.Api = lambda **kwargs: SimpleNamespace(run=denied)
    with pytest.raises(CommError, match="Authentication denied"):
        prepare.prepare_runs(**_arguments(tmp_path))
    assert not fake_wandb.calls


@pytest.mark.parametrize("state,host", [("PENDING", "gpu1"), ("RUNNING", "login1")])
def test_verifies_actual_allocation_not_only_slurm_environment(monkeypatch, state, host):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(prepare.socket, "gethostname", lambda: host)

    def command(args, **kwargs):
        value = (
            f"JobId=123 JobState={state} NodeList=gpu[1-2]"
            if args[2] == "job"
            else "gpu1\ngpu2\n"
        )
        return SimpleNamespace(stdout=value)

    monkeypatch.setattr(prepare.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="running node allocation|outside Slurm"):
        prepare.verify_slurm_allocation()


def test_allocation_failure_precedes_sdk_import(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        prepare.importlib,
        "import_module",
        lambda name: pytest.fail("SDK must not be imported on login node"),
    )
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        prepare.prepare_runs(**_arguments(tmp_path))
