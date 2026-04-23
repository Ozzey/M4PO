# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External Isaac Lab package for local data-collection tasks and tooling."""

# Register Gym environments when Isaac Sim / Kit modules are available.
try:
    from .tasks import *  # noqa: F401, F403
except ModuleNotFoundError as exc:
    if exc.name not in {"omni.timeline", "carb", "pxr"}:
        raise

# Register UI extensions when running inside Kit.
try:
    from .ui_extension_example import *  # noqa: F401, F403
except ModuleNotFoundError as exc:
    if exc.name not in {"omni.ext", "omni.ui", "carb", "pxr"}:
        raise
