from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from m4po.common.config import IMPLEMENTATION_ID, M4POConfig, update_config_from_args
from m4po.common.evaluation import evaluate_policy
from m4po.common.utils import get_device, set_seed
from m4po.m4po import M4POAgent
from m4po.train import add_config_args


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
    cfg = M4POConfig(**payload["cfg"])
    cfg = update_config_from_args(cfg, args)
    set_seed(cfg.seed, cfg.torch_deterministic)
    device = get_device(cfg.device)
    agent = M4POAgent.load(checkpoint, device=device, override_cfg=cfg)
    evaluation = evaluate_policy(
        agent,
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
        "env": agent.cfg.env,
        "tasks": agent.cfg.task_names,
        "embodiments": agent.cfg.embodiment_names,
        "seed": agent.cfg.seed,
        "deterministic": bool(args.deterministic),
        **evaluation,
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
            "per_pair",
            "per_task",
            "per_embodiment",
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
