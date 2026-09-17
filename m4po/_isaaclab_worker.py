from __future__ import annotations

import argparse
import traceback
from collections.abc import Mapping
from multiprocessing.connection import Connection

PROTOCOL_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal isolated IsaacLab worker for M4PO."
    )
    parser.add_argument("--connection-fd", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = Connection(args.connection_fd)
    app_launcher = None
    try:
        bootstrap = connection.recv()
        if not isinstance(bootstrap, Mapping):
            raise TypeError("IsaacLab worker bootstrap must be a mapping")
        if bootstrap.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("IsaacLab worker protocol version mismatch")
        if bootstrap.get("sequence") != 0 or bootstrap.get("operation") != "bootstrap":
            raise ValueError("IsaacLab worker received an invalid bootstrap command")
        request = bootstrap.get("request")
        if not isinstance(request, Mapping):
            raise TypeError("IsaacLab worker bootstrap request must be a mapping")

        # Launch Kit before importing any module that imports torch, Gym, or
        # IsaacLab task extensions.
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(
            headless=bool(request["headless"]),
            enable_cameras=bool(request["enable_cameras"]),
            device=str(request["device"]),
        )
        from m4po.envs.isaaclab_prebuilt import _run_isaaclab_worker

        _run_isaaclab_worker(connection, request, app_launcher)
    except Exception as exc:  # noqa: BLE001 - report bootstrap/Kit failures.
        try:
            connection.send(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "sequence": 0,
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        if app_launcher is not None:
            try:
                app_launcher.app.close()
            except Exception:  # noqa: BLE001,S110
                pass
        connection.close()


if __name__ == "__main__":
    main()
