from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import numpy as np

from m4po.common.config import IMPLEMENTATION_ID, M4POConfig, update_config_from_args
from m4po.common.evaluation import evaluate_policy
from m4po.common.utils import get_device, set_seed
from m4po.m4po import M4POAgent
from m4po.train import add_config_args


class _TimedAgent:
    """Measure delivered action-batch latency, excluding environment stepping."""

    def __init__(self, agent: M4POAgent, warmup_calls: int = 5):
        self.agent = agent
        self.warmup_calls = warmup_calls
        self.calls = 0
        self.milliseconds: list[float] = []
        self.batch_sizes: list[int] = []

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def act_np(self, *args, **kwargs):
        device = torch.device(self.agent.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = self.agent.act_np(*args, **kwargs)
        # act_np returns CPU numpy arrays, including the delivered action, so
        # GPU work and device-to-host transfers have completed at this point.
        elapsed = (time.perf_counter() - start) * 1000.0
        self.calls += 1
        if self.calls > self.warmup_calls:
            self.milliseconds.append(elapsed)
            self.batch_sizes.append(len(output.action))
        return output

    def summary(self) -> dict[str, object]:
        values = self.milliseconds
        return {
            "measurement": "delivered_action_batch_latency_including_transfers",
            "device": str(self.agent.device),
            "warmup_calls_excluded": min(self.calls, self.warmup_calls),
            "measured_calls": len(values),
            "batch_size_min": min(self.batch_sizes) if self.batch_sizes else None,
            "batch_size_max": max(self.batch_sizes) if self.batch_sizes else None,
            "mean_ms": float(np.mean(values)) if values else None,
            "median_ms": float(np.median(values)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained M4PO checkpoint.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to a .pt checkpoint."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes to evaluate per task/embodiment pair.",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Optional JSON output path."
    )
    parser.add_argument(
        "--csv-out", type=str, default=None, help="Optional CSV summary path."
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    add_config_args(parser)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError(
            f"Checkpoint is not compatible with {IMPLEMENTATION_ID}: "
            f"{payload.get('implementation_id')!r}"
        )
    cfg = M4POConfig.from_checkpoint_dict(payload["cfg"])
    cfg = update_config_from_args(cfg, args)
    set_seed(cfg.seed, cfg.torch_deterministic)
    device = get_device(cfg.device)
    agent = M4POAgent.load(checkpoint, device=device, override_cfg=cfg, _payload=payload)
    payload.get("extra", {}).pop("replay_state", None)
    extra = payload.get("extra", {})
    initialization = extra.get("initialization", {"kind": "random"})
    if extra.get("stage") == "demonstration_pretraining":
        if extra.get("training_complete") is not True:
            raise ValueError("Direct offline evaluation requires completed pretraining")
        from m4po.common.initialization import file_sha256

        initialization = {
            "kind": "demonstration_pretraining",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "demonstration_pretraining": extra["demonstration_pretraining"],
        }
    timed_agent = _TimedAgent(agent)
    evaluation = evaluate_policy(
        timed_agent,
        agent.cfg,
        episodes=args.episodes,
        deterministic=args.deterministic,
        expected_environment_signature=payload.get("extra", {}).get(
            "environment_signature"
        ),
    )
    returns = list(evaluation["returns"])
    results: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload["step"]),
        "learning_mode": agent.cfg.learning_mode,
        "initialization": initialization,
        "env": agent.cfg.env,
        "tasks": agent.cfg.task_names,
        "embodiments": agent.cfg.embodiment_names,
        "seed": agent.cfg.seed,
        "deterministic": bool(args.deterministic),
        **evaluation,
        "action_timing": timed_agent.summary(),
        "return_min": float(min(returns)) if returns else 0.0,
        "return_max": float(max(returns)) if returns else 0.0,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    csv_out = args.csv_out
    if csv_out is None and args.out:
        csv_out = str(Path(args.out).with_suffix(".csv"))
    if csv_out:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        excluded = {
            "returns",
            "lengths",
            "successes",
            "scores",
            "action_timing",
            "per_pair",
            "per_task",
            "per_embodiment",
            "per_domain",
            "benchmark_coverage",
            "initialization",
        }
        row = {key: value for key, value in results.items() if key not in excluded}
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    return results


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    print(json.dumps(results, indent=2))


benchmark = evaluate


if __name__ == "__main__":
    main()
