from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
ASSETS_DIR = Path(os.environ.get("M4PO_DATACOLLECTION_ASSETS_DIR", PROJECT_ROOT / "assets")).resolve()
ROBOT_ASSET_PATH = ASSETS_DIR / "robots" / "g1-29dof_wholebody_inspire" / "g1_29dof_with_inspire_rev_1_0.usd"
