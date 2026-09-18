from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any

import torch

from m4po.common.config import M4POConfig
from m4po.common.demonstrations import (
    canonical_digest,
    load_demonstrations,
    stream_sha256,
)
from m4po.common.utils import JsonlLogger, ensure_dir, get_device, set_seed
from m4po.envs.mmbench_env import MAX_ACTION_DIM
from m4po.m4po import M4POAgent
from m4po.mmbench_preflight import verify_slurm_allocation
from m4po.trainer.online_trainer import OnlineTrainer


STAGE = "demonstration_pretraining"
ACTOR_OBJECTIVE = "masked_behavior_cloning_plus_entropy"


def _config_digest(cfg: M4POConfig) -> str:
    values = cfg.to_dict()
    for name in (
        "log_dir",
        "device",
        "quiet",
        "resume_checkpoint",
        "init_checkpoint",
        "max_wall_time_seconds",
        "mmbench_root",
        "save_every",
        "log_every",
    ):
        values.pop(name, None)
    return canonical_digest(values)


def _implementation_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_checkpoint(agent, path: Path, extra: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        agent.save(Path(temporary), step=0, extra=extra)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    stat = path.stat()
    _write_json(
        path.with_name("latest_status.json"),
        {
            "stage": STAGE,
            "step": 0,
            **{
                name: extra[name]
                for name in (
                    "updates_completed",
                    "target_updates",
                    "training_complete",
                    "resumable",
                    "stop_reason",
                )
            },
            "dataset_sha256": extra["demonstration_pretraining"]["dataset_sha256"],
            "protocol_sha256": extra["demonstration_pretraining"]["protocol_sha256"],
            "m4po_sources_sha256": extra["demonstration_pretraining"][
                "m4po_sources_sha256"
            ],
            "config_sha256": extra["demonstration_pretraining"]["config_sha256"],
            "allocation": extra["allocation"],
            "device": extra["device"],
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
        },
    )


def _publish_pretrained(source: Path, target: Path) -> None:
    """Publish an immutable completed initialization artifact, never overwrite it."""

    if target.exists():
        with source.open("rb") as file:
            expected = stream_sha256(file)
        with target.open("rb") as file:
            actual = stream_sha256(file)
        if expected != actual:
            raise FileExistsError(
                f"Refusing to replace different pretrained weights: {target}"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=".pretrained_model.", dir=target.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        # link is an atomic, no-overwrite publication on the same filesystem.
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pretrain(
    cfg: M4POConfig,
    *,
    data_dir: str | Path,
    preflight_report: str | Path,
    updates: int = 200_000,
    resume: str | Path | None = None,
    max_wall_time_seconds: float = 0,
    save_every: int = 2_000,
    log_every: int = 100,
    require_slurm: bool = False,
) -> Path:
    """Offline BC plus M4PO world/TD-Q learning; online transition count stays zero."""

    started = time.monotonic()
    allocation = verify_slurm_allocation() if require_slurm else {"verified": False}
    for name, value in (
        ("updates", updates),
        ("save_every", save_every),
        ("log_every", log_every),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(max_wall_time_seconds) or max_wall_time_seconds < 0:
        raise ValueError("max_wall_time_seconds must be finite and nonnegative")
    cfg.validate()
    if cfg.resume_checkpoint or getattr(cfg, "init_checkpoint", None):
        raise ValueError(
            "Offline pretraining uses --resume, not online initialization/resume settings"
        )
    checkpoint_dir = ensure_dir(Path(cfg.log_dir) / "checkpoints")
    latest = checkpoint_dir / "latest.pt"
    final = checkpoint_dir / "pretrained_model.pt"
    if not resume and (latest.exists() or final.exists()):
        raise FileExistsError(
            "Offline output already has weights; use --resume or a new log directory"
        )
    set_seed(cfg.seed, cfg.torch_deterministic)
    device = get_device(cfg.device)
    cfg.device = str(device)
    dataset = load_demonstrations(cfg, data_dir, preflight_report)
    protocol = {
        "schema_version": 1,
        "stage": STAGE,
        "actor_objective": ACTOR_OBJECTIVE,
        "model_objective": "M4PO consistency + reward + terminated-aware TD-Q + termination",
        "actor_representation": "detached replay rollout latents",
        "config_sha256": _config_digest(cfg),
        "m4po_sources_sha256": _implementation_digest(),
        "target_updates": updates,
        **dataset.provenance,
    }
    provenance = {**protocol, "protocol_sha256": canonical_digest(protocol)}
    completed = 0
    payload = None
    if resume:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=False)
        extra = payload.get("extra", {})
        previous = extra.get("demonstration_pretraining", {})
        completed = extra.get("updates_completed")
        if (
            payload.get("step") != 0
            or extra.get("stage") != STAGE
            or extra.get("resumable") is not True
            or extra.get("target_updates") != updates
            or type(completed) is not int
            or not 0 <= completed <= updates
            or extra.get("training_complete") is not (completed == updates)
            or previous.get("completed_updates") != completed
            or previous.get("stage_completed") is not (completed == updates)
            or any(previous.get(key) != value for key, value in provenance.items())
            or not isinstance(extra.get("rng_state"), dict)
            or "replay_rng_state" not in extra
        ):
            raise ValueError(
                "Offline resume checkpoint has incompatible progress or provenance"
            )
        agent = M4POAgent.load(
            Path(resume),
            device,
            override_cfg=cfg,
            restore_optimizers=True,
            _payload=payload,
        )
        dataset.replay.load_rng_state_dict(extra["replay_rng_state"])
        OnlineTrainer._restore_rng_state(extra["rng_state"])
    else:
        agent = M4POAgent(
            dataset.observation_spec,
            MAX_ACTION_DIM,
            cfg,
            device,
            task_contexts=dataset.task_contexts,
        )
    cfg.save_yaml(Path(cfg.log_dir) / "config_resolved.yaml")
    _write_json(Path(cfg.log_dir) / "demonstration_protocol.json", provenance)
    if completed == updates:
        _publish_pretrained(Path(resume), final)
        return final
    del payload
    stop = {"signal": None}

    def request_stop(number, _frame):
        stop["signal"] = int(number)

    previous_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)
    }
    for number in previous_handlers:
        signal.signal(number, request_stop)

    def checkpoint_extra(reason):
        return {
            "stage": STAGE,
            "updates_completed": completed,
            "target_updates": updates,
            "training_complete": completed == updates,
            "resumable": True,
            "stop_reason": reason,
            "rng_state": OnlineTrainer._capture_rng_state(),
            "replay_rng_state": dataset.replay.rng_state_dict(),
            "replay_saved": False,
            "allocation": allocation,
            "device": str(device),
            "demonstration_pretraining": {
                **provenance,
                "completed_updates": completed,
                "stage_completed": completed == updates,
            },
        }

    reason = "completed"
    try:
        with JsonlLogger(Path(cfg.log_dir) / "metrics.jsonl") as logger:
            while completed < updates:
                if stop["signal"] is not None:
                    reason = f"signal_{stop['signal']}"
                    break
                if (
                    max_wall_time_seconds
                    and time.monotonic() - started >= max_wall_time_seconds
                ):
                    reason = "wall_time_limit"
                    break
                metrics = agent.update(dataset.replay, actor_mode="behavior_cloning")
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError(
                        "Non-finite offline pretraining metric; previous checkpoint retained"
                    )
                completed += 1
                if completed % log_every == 0 or completed == updates:
                    logger.write(
                        {
                            "phase": STAGE,
                            "step": 0,
                            "pretrain_updates": completed,
                            "target_updates": updates,
                            "elapsed_seconds": time.monotonic() - started,
                            **metrics,
                        }
                    )
                if completed % save_every == 0 and completed != updates:
                    _save_checkpoint(
                        agent, latest, checkpoint_extra("periodic_checkpoint")
                    )
            _save_checkpoint(agent, latest, checkpoint_extra(reason))
            logger.write(
                {
                    "phase": STAGE,
                    "step": 0,
                    "pretrain_updates": completed,
                    "training_complete": completed == updates,
                    "stop_reason": reason,
                }
            )
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    if completed == updates:
        _publish_pretrained(latest, final)
        return final
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain M4PO on pinned official MMBench demonstrations."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs" / "mmbench_all.yaml")
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--updates", type=int, default=200_000)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-wall-time-seconds", type=float, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--require-slurm", action="store_true")
    args = parser.parse_args()
    cfg = M4POConfig.from_yaml(args.config)
    cfg.log_dir, cfg.device = args.log_dir, args.device
    if args.seed is not None:
        cfg.seed = args.seed
    path = pretrain(
        cfg,
        data_dir=args.data_dir,
        preflight_report=args.preflight_report,
        updates=args.updates,
        resume=args.resume,
        max_wall_time_seconds=args.max_wall_time_seconds,
        save_every=args.save_every,
        log_every=args.log_every,
        require_slurm=args.require_slurm,
    )
    print(json.dumps({"checkpoint": str(path), "log_dir": cfg.log_dir}))


if __name__ == "__main__":
    main()
