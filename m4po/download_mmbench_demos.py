from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DATASET_REPO_ID = "nicklashansen/mmbench"
DATASET_REVISION = "a59d457df617400d3e45a5158c8deac8a52055b4"
MANIFEST_FILENAME = "manifest.json"
_API_ROOT = (
    f"https://huggingface.co/api/datasets/{DATASET_REPO_ID}/tree/{DATASET_REVISION}"
)
_ATTEMPTS = 3
_CHUNK_BYTES = 8 * 1024 * 1024


def load_task_catalog(newt_root: str | Path | None = None) -> tuple[list[str], str]:
    """Read the canonical task list without importing Torch or native simulators."""

    configured = newt_root or os.environ.get("M4PO_NEWT_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1] / "external" / "newt"
    )
    metadata_bytes = (root / "tasks.json").read_bytes()
    task_bytes = (root / "tdmpc2/common/__init__.py").read_bytes()
    metadata = json.loads(metadata_bytes)
    task_sets = None
    for node in ast.parse(task_bytes).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_SET"
            for target in node.targets
        ):
            task_sets = ast.literal_eval(node.value)
            break
    tasks = task_sets.get("soup") if isinstance(task_sets, dict) else None
    if (
        not isinstance(tasks, list)
        or len(tasks) != 200
        or any(
            not isinstance(task, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task) is None
            for task in tasks
        )
        or len(set(tasks)) != 200
        or any(task not in metadata for task in tasks)
    ):
        raise ValueError("The official NEWT catalog must define 200 unique soup tasks")
    return tasks, hashlib.sha256(metadata_bytes + task_bytes).hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    identity = {
        key: value
        for key, value in manifest.items()
        if key != "dataset_manifest_sha256"
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _error_label(error: Exception) -> str:
    # Never print redirect URLs, headers, or authentication-bearing exceptions.
    if isinstance(error, HTTPError):
        return f"HTTP {error.code}"
    return type(error).__name__


def _read_tree_page(url: str) -> tuple[list[dict[str, Any]], str | None]:
    for attempt in range(_ATTEMPTS):
        try:
            with urlopen(Request(url), timeout=60) as response:
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ValueError("Dataset metadata page exceeds the safety limit")
                entries = json.loads(body)
                if not isinstance(entries, list):
                    raise ValueError("Dataset tree metadata must be a list")
                next_url = None
                for link in response.headers.get("Link", "").split(","):
                    match = re.match(r'\s*<([^>]+)>;\s*rel="next"', link)
                    if match:
                        next_url = match.group(1)
                if next_url:
                    parsed = urlparse(next_url)
                    if (
                        parsed.scheme != "https"
                        or parsed.netloc != "huggingface.co"
                        or parsed.path != urlparse(_API_ROOT).path
                    ):
                        raise ValueError("Dataset pagination left the pinned source")
                return entries, next_url
        except (OSError, ValueError, URLError) as error:
            if attempt + 1 == _ATTEMPTS:
                raise RuntimeError(
                    f"Cannot read pinned dataset metadata after {_ATTEMPTS} attempts: "
                    f"{_error_label(error)}"
                ) from None
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable metadata retry state")


def build_manifest(tasks: list[str], catalog_sha256: str) -> dict[str, Any]:
    """Resolve exactly the canonical shards and their authoritative LFS hashes."""

    if (
        len(tasks) != 200
        or any(
            not isinstance(task, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task) is None
            for task in tasks
        )
        or len(set(tasks)) != 200
    ):
        raise ValueError("Demonstration download requires all 200 canonical tasks")
    if re.fullmatch(r"[a-f0-9]{64}", catalog_sha256) is None:
        raise ValueError("Invalid catalog SHA-256")
    required = {f"{task}.pt": task for task in tasks}
    found: dict[str, dict[str, Any]] = {}
    url = f"{_API_ROOT}?recursive=false&expand=false&limit=1000"
    visited = set()
    while url:
        if url in visited or len(visited) >= 100:
            raise ValueError("Dataset pagination did not terminate")
        visited.add(url)
        entries, url = _read_tree_page(url)
        for entry in entries:
            path = entry.get("path")
            if path not in required:
                continue
            if path in found:
                raise ValueError(f"Duplicate dataset metadata for {path}")
            lfs = entry.get("lfs", {})
            size = entry.get("size")
            digest = lfs.get("oid")
            if (
                entry.get("type") != "file"
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or lfs.get("size") != size
                or not isinstance(digest, str)
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            ):
                raise ValueError(f"Missing or inconsistent LFS size/SHA-256 for {path}")
            found[path] = {
                "task": required[path],
                "path": path,
                "size": size,
                "sha256": digest,
            }
    missing = sorted(set(required) - set(found))
    if missing:
        raise ValueError(f"Pinned dataset is missing {len(missing)} tasks: {missing}")
    manifest = {
        "schema_version": 1,
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "catalog_sha256": catalog_sha256,
        "task_count": 200,
        "total_bytes": sum(row["size"] for row in found.values()),
        "files": [found[f"{task}.pt"] for task in tasks],
    }
    manifest["dataset_manifest_sha256"] = manifest_digest(manifest)
    return manifest


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(root: Path, row: dict[str, Any]) -> str:
    target = root / row["path"]
    partial = target.with_name(target.name + ".part")
    if target.is_symlink() or partial.is_symlink():
        raise ValueError(f"Refusing a symlink download target: {row['path']}")
    if target.exists():
        if (
            target.is_file()
            and target.stat().st_size == row["size"]
            and _file_digest(target) == row["sha256"]
        ):
            return "verified_existing"
        raise ValueError(
            f"Existing {row['path']} fails verification; move it aside before retrying"
        )
    url = (
        f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/"
        f"{DATASET_REVISION}/{quote(row['path'], safe='')}?download=true"
    )
    for attempt in range(_ATTEMPTS):
        try:
            digest = hashlib.sha256()
            size = 0
            with urlopen(Request(url), timeout=60) as response:
                if response.status != 200:
                    raise ValueError("Expected a complete HTTP response")
                with partial.open("wb") as stream:
                    while chunk := response.read(_CHUNK_BYTES):
                        size += len(chunk)
                        if size > row["size"]:
                            raise ValueError(
                                "Downloaded shard exceeds its declared size"
                            )
                        digest.update(chunk)
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if size != row["size"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("Downloaded shard failed size/SHA-256 verification")
            if target.exists() or target.is_symlink():
                raise ValueError("Download target appeared during transfer")
            os.replace(partial, target)
            return "downloaded"
        except (OSError, ValueError, URLError) as error:
            if partial.exists() and not partial.is_symlink():
                partial.unlink()
            if attempt + 1 == _ATTEMPTS:
                raise RuntimeError(
                    f"Cannot download {row['path']} after {_ATTEMPTS} attempts: "
                    f"{_error_label(error)}"
                ) from None
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable download retry state")


def _publish_manifest(path: Path, manifest: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=".manifest.", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download_demos(
    out_dir: str | Path,
    *,
    workers: int = 4,
    require_slurm: bool = False,
    newt_root: str | Path | None = None,
) -> dict[str, Any]:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 8
    ):
        raise ValueError("workers must be an integer from 1 to 8")
    if require_slurm:
        from m4po.mmbench_preflight import verify_slurm_allocation

        verify_slurm_allocation()
    tasks, catalog_hash = load_task_catalog(newt_root)
    manifest = build_manifest(tasks, catalog_hash)
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".download.lock"
    manifest_path = root / MANIFEST_FILENAME
    if lock_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Refusing symlink manifest/download-lock paths")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another downloader already owns this directory"
            ) from None
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output directory has a different dataset manifest")
        counts = {"downloaded": 0, "verified_existing": 0}
        downloaded_bytes = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_file, root, row): row
                for row in manifest["files"]
            }
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    row = futures[future]
                    counts[result] += 1
                    if result == "downloaded":
                        downloaded_bytes += row["size"]
                    print(
                        f"[{completed}/200] {result}: {row['path']} ({row['size']} bytes)",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        _publish_manifest(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "checked_files": sum(counts.values()),
        "verified_bytes": manifest["total_bytes"],
        "downloaded_bytes": downloaded_bytes,
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify only the 200 official MMBench demonstration shards"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--newt-root", type=Path)
    parser.add_argument("--require-slurm", action="store_true")
    args = parser.parse_args()
    result = download_demos(
        args.out_dir,
        workers=args.workers,
        require_slurm=args.require_slurm,
        newt_root=args.newt_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
