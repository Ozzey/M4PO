from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from m4po import download_mmbench_demos as download


class Response(io.BytesIO):
    def __init__(self, body: bytes, *, status: int = 200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


@pytest.fixture
def dataset(monkeypatch):
    tasks = [f"task-{index}" for index in range(200)]
    payloads = {f"{task}.pt": task.encode() for task in tasks}
    entries = [
        {
            "type": "file",
            "path": path,
            "size": len(body),
            "lfs": {"size": len(body), "oid": hashlib.sha256(body).hexdigest()},
        }
        for path, body in payloads.items()
    ]
    entries.append({"type": "file", "path": "unrelated.pt", "size": 123})
    calls = []

    def open_url(request, timeout):
        url = request.full_url
        calls.append(url)
        if "/api/" in url:
            return Response(json.dumps(entries).encode())
        name = url.rsplit("/", 1)[-1].split("?", 1)[0]
        return Response(payloads[name])

    monkeypatch.setattr(download, "urlopen", open_url)
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(download, "load_task_catalog", lambda root: (tasks, "a" * 64))
    return tasks, payloads, entries, calls


def test_catalog_reads_official_200_tasks_and_matches_adapter_digest():
    root = Path(__file__).resolve().parents[1] / "external/newt"
    tasks, digest = download.load_task_catalog(root)
    assert len(tasks) == len(set(tasks)) == 200
    assert "cartpole-balance" in tasks
    assert "bipedal-walker" not in tasks
    expected = (root / "tasks.json").read_bytes() + (
        root / "tdmpc2/common/__init__.py"
    ).read_bytes()
    assert digest == hashlib.sha256(expected).hexdigest()


def test_catalog_refuses_executable_expressions(tmp_path):
    source = tmp_path / "tdmpc2/common"
    source.mkdir(parents=True)
    (tmp_path / "tasks.json").write_text("{}")
    (source / "__init__.py").write_text("TASK_SET = dict(soup=['task'] * 200)")
    with pytest.raises(ValueError):
        download.load_task_catalog(tmp_path)


def test_manifest_selects_exact_tasks_and_reproducible_relative_identity(dataset):
    tasks, payloads, _, _ = dataset
    manifest = download.build_manifest(tasks, "a" * 64)
    assert manifest["repo_id"] == "nicklashansen/mmbench"
    assert manifest["revision"] == download.DATASET_REVISION
    assert manifest["task_count"] == 200
    assert [row["task"] for row in manifest["files"]] == tasks
    assert all(not Path(row["path"]).is_absolute() for row in manifest["files"])
    assert manifest["total_bytes"] == sum(map(len, payloads.values()))
    assert manifest["dataset_manifest_sha256"] == download.manifest_digest(manifest)
    assert manifest == download.build_manifest(tasks, "a" * 64)


def test_manifest_handles_pagination(dataset, monkeypatch):
    tasks, _, entries, _ = dataset
    next_url = download._API_ROOT + "?cursor=second"
    pages = [
        Response(
            json.dumps(entries[:100]).encode(),
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        Response(json.dumps(entries[100:]).encode()),
    ]
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: pages.pop(0))
    assert len(download.build_manifest(tasks, "a" * 64)["files"]) == 200
    assert not pages


@pytest.mark.parametrize("fault", ["missing", "duplicate", "hash", "size"])
def test_manifest_rejects_incomplete_or_unverifiable_metadata(dataset, fault):
    tasks, _, entries, _ = dataset
    if fault == "missing":
        entries.pop(0)
    elif fault == "duplicate":
        entries.append(entries[0])
    elif fault == "hash":
        entries[0]["lfs"]["oid"] = "not-a-hash"
    elif fault == "size":
        entries[0]["lfs"]["size"] += 1
    with pytest.raises(ValueError, match="missing|Duplicate|LFS"):
        download.build_manifest(tasks, "a" * 64)


def test_metadata_does_not_follow_unpinned_pagination(dataset, monkeypatch):
    tasks, _, _, calls = dataset

    def bad_page(request, timeout):
        calls.append(request.full_url)
        return Response(
            b"[]", headers={"Link": '<https://example.com/secret>; rel="next"'}
        )

    monkeypatch.setattr(download, "urlopen", bad_page)
    with pytest.raises(RuntimeError, match="3 attempts"):
        download.build_manifest(tasks, "a" * 64)
    assert len(calls) == 3
    assert all(url.startswith(download._API_ROOT) for url in calls)


def test_download_all_exactly_once_then_verify_without_overwriting(dataset, tmp_path):
    _, payloads, _, calls = dataset
    first = download.download_demos(tmp_path, workers=4)
    assert first["downloaded"] == first["checked_files"] == 200
    assert first["verified_existing"] == 0
    assert first["downloaded_bytes"] == sum(map(len, payloads.values()))
    assert len([url for url in calls if "/resolve/" in url]) == 200
    assert not (tmp_path / "unrelated.pt").exists()
    manifest_path = tmp_path / download.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    mtimes = {path: (tmp_path / path).stat().st_mtime_ns for path in payloads}
    calls.clear()
    second = download.download_demos(tmp_path, workers=1)
    assert second["downloaded"] == second["downloaded_bytes"] == 0
    assert second["verified_existing"] == 200
    assert not any("/resolve/" in url for url in calls)
    assert manifest == json.loads(manifest_path.read_text())
    assert mtimes == {path: (tmp_path / path).stat().st_mtime_ns for path in payloads}
    assert not list(tmp_path.glob("*.part"))


def test_download_refuses_to_overwrite_existing_corrupt_file(dataset, tmp_path):
    tasks, _, _, calls = dataset
    row = download.build_manifest(tasks, "a" * 64)["files"][0]
    target = tmp_path / row["path"]
    target.write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="move it aside"):
        download._download_file(tmp_path, row)
    assert target.read_bytes() == b"corrupt"
    assert calls == []


def test_download_retries_partial_and_hash_failures_then_publishes(
    dataset, tmp_path, monkeypatch
):
    tasks, payloads, _, _ = dataset
    row = download.build_manifest(tasks, "a" * 64)["files"][0]
    target = tmp_path / row["path"]
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"stale interrupted transfer")
    responses = [b"short", b"wrong!", payloads[row["path"]]]

    def stream(request, timeout):
        assert not target.exists()
        return Response(responses.pop(0))

    monkeypatch.setattr(download, "urlopen", stream)
    assert download._download_file(tmp_path, row) == "downloaded"
    assert not responses
    assert target.read_bytes() == payloads[row["path"]]
    assert not partial.exists()


def test_failed_download_is_bounded_and_does_not_publish(
    dataset, tmp_path, monkeypatch
):
    tasks, _, _, _ = dataset
    row = download.build_manifest(tasks, "a" * 64)["files"][0]
    attempts = []

    def fail(request, timeout):
        attempts.append(request.full_url)
        raise URLError("do not expose secret=test-token")

    monkeypatch.setattr(download, "urlopen", fail)
    with pytest.raises(RuntimeError, match="3 attempts") as caught:
        download._download_file(tmp_path, row)
    assert "test-token" not in str(caught.value)
    assert len(attempts) == 3
    assert not (tmp_path / row["path"]).exists()
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("target_name", ["task-0.pt", "task-0.pt.part"])
def test_download_refuses_symlinks(dataset, tmp_path, target_name):
    tasks, _, _, _ = dataset
    row = download.build_manifest(tasks, "a" * 64)["files"][0]
    destination = tmp_path / "preserved"
    destination.write_bytes(b"preserve")
    (tmp_path / target_name).symlink_to(destination)
    with pytest.raises(ValueError, match="symlink"):
        download._download_file(tmp_path, row)
    assert destination.read_bytes() == b"preserve"


@pytest.mark.parametrize("workers", [0, 9, -1, True, 1.5])
def test_invalid_concurrency_rejected_before_network(dataset, tmp_path, workers):
    _, _, _, calls = dataset
    with pytest.raises(ValueError, match="workers"):
        download.download_demos(tmp_path, workers=workers)
    assert calls == []


def test_existing_different_manifest_blocks_download(dataset, tmp_path):
    _, _, _, calls = dataset
    (tmp_path / download.MANIFEST_FILENAME).write_text('{"revision":"other"}')
    with pytest.raises(ValueError, match="different dataset manifest"):
        download.download_demos(tmp_path)
    assert not any("/resolve/" in url for url in calls)


def test_slurm_guard_runs_before_metadata_or_output_creation(
    dataset, tmp_path, monkeypatch
):
    _, _, _, calls = dataset

    def reject_login_node():
        raise RuntimeError("Not an allocated node")

    monkeypatch.setitem(
        sys.modules,
        "m4po.mmbench_preflight",
        SimpleNamespace(verify_slurm_allocation=reject_login_node),
    )
    out = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="allocated node"):
        download.download_demos(out, require_slurm=True)
    assert not out.exists()
    assert calls == []


def test_failed_batch_never_publishes_a_complete_manifest(
    dataset, tmp_path, monkeypatch
):
    def fail(root, row):
        raise RuntimeError("failed shard")

    monkeypatch.setattr(download, "_download_file", fail)
    with pytest.raises(RuntimeError, match="failed shard"):
        download.download_demos(tmp_path)
    assert not (tmp_path / download.MANIFEST_FILENAME).exists()
