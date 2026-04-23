from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SOURCE_DIRS = (
    REPO_ROOT / "isaaclab" / "source" / "isaaclab",
    REPO_ROOT / "isaaclab" / "source" / "isaaclab_tasks",
    REPO_ROOT / "isaaclab" / "source" / "isaaclab_assets",
    REPO_ROOT / "m4po_datacollection" / "source" / "m4po_datacollection",
    REPO_ROOT / "isaaclab" / "external" / "Isaac-GR00T",
)


def configure_python_path() -> None:
    """Prefer the local workspace packages when running from this folder."""

    for path in reversed(LOCAL_SOURCE_DIRS):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)
