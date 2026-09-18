from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from m4po.envs.mmbench_env import load_mmbench_catalog


NEWT_COMMIT = "1d3fc058b81ddf8d36a5457c29ea407dd374c1b8"
REFERENCE_URL = f"https://github.com/nicklashansen/newt/blob/{NEWT_COMMIT}"
FULL_BUDGET = 100_000_000
SOURCE_DIGESTS = {
    "csv/newt/newt_avg.csv": "91d90cfb7134168eedc91cd3e75557864866bf03dddac91a07fe766f8dff9565",
    "csv/newt/newt_by_domain.csv": "9956d5478e0ff9c943e78b497704bdc037b4a47a8c77a77db956bc4f8c7eefab",
    "csv/tdmpc2/tdmpc2_avg.csv": "b914ad3105f9837b373ccdf6d525dae0ac4ef611454ab050a680ac8f59174d84",
    "csv/tdmpc2/tdmpc2_by_domain.csv": "362bc52c8ac981c310d738808c690f2af9c49a52f8a5a927ac05ea2a8612cb83",
    "tdmpc2/common/__init__.py": "8cb642db53f68953dd701a8f49cb47754330107cb9161387af59477b8a11f659",
    "tasks.json": "b55dcdefc0ff6a1f96ad73f2c20fd72319440533a60992012435ce109f804590",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite normalized score")
    result = float(value)
    if not math.isfinite(result) or not -1e-6 <= result <= 1 + 1e-6:
        raise ValueError(f"{label} must be a finite normalized score in [0, 1]")
    return result


def _read_reference(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = {}
    for row in rows:
        raw_step = float(row.pop("eval/step"))
        if not math.isfinite(raw_step) or raw_step < 0 or not raw_step.is_integer():
            raise ValueError(f"Invalid reference step in {path}")
        step = int(raw_step)
        if step in result:
            raise ValueError(f"Duplicate reference step in {path}: {step}")
        result[step] = {key: _score(float(value), key) for key, value in row.items()}
    return result


def _validate_evaluation(payload: dict, task_sets: dict) -> dict[str, float]:
    coverage = payload.get("benchmark_coverage", {})
    if coverage.get("full_suite") is not True:
        raise ValueError("Comparison requires full-suite coverage, not a pilot")
    for field in (
        "canonical_task_count",
        "selected_task_count",
        "evaluated_task_count",
        "score_task_count",
    ):
        if coverage.get(field) != 200:
            raise ValueError(f"Full-suite coverage requires {field}=200")
    if coverage.get("domain_count") != 10:
        raise ValueError("Full-suite coverage requires all 10 domains")
    if payload.get("score_aggregation") != "mean_of_task_mean_scores":
        raise ValueError("Comparison requires task-macro-averaged native scores")
    per_task = payload.get("per_task", {})
    if set(per_task) != set(task_sets["soup"]):
        raise ValueError("Per-task scores must cover exactly the canonical 200 tasks")
    scores = {
        task: _score(per_task[task].get("score_mean"), f"{task}.score_mean")
        for task in task_sets["soup"]
    }
    mean = _score(payload.get("score_mean"), "score_mean")
    if not math.isclose(mean, fmean(scores.values()), rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("Aggregate score does not match the mean of 200 task scores")
    return scores


def _full_curve(
    path: Path | None,
    task_sets: dict,
    required_steps: list[int],
    expected_initialization: dict | None = None,
    expected_seed: int | None = None,
) -> tuple[dict[int, float] | None, str | None]:
    if path is None:
        return None, "No evaluation curve was supplied"
    values: dict[int, float] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            evaluation = row.get("eval")
            if evaluation is None:
                continue  # Training scores are not frozen-evaluation scores.
            step = _integer(row.get("step"), f"curve line {line_number} step")
            if step not in required_steps:
                continue
            if expected_initialization is not None and (
                row.get("initialization") != expected_initialization
                or row.get("seed") != expected_seed
            ):
                return (
                    None,
                    f"Evaluation initialization/seed provenance differs at step {step}",
                )
            if step in values:
                return None, f"Ambiguous duplicate frozen evaluation at step {step}"
            try:
                scores = _validate_evaluation(evaluation, task_sets)
            except ValueError as exc:
                return None, f"Invalid full-suite evaluation at step {step}: {exc}"
            values[step] = fmean(scores.values())
    missing = sorted(set(required_steps) - values.keys())
    if missing:
        return None, f"Missing full-suite evaluations on the 0..100M grid: {missing}"
    return values, None


def _auc(values: dict[int, float], steps: list[int]) -> float:
    area = sum(
        (right - left) * (values[left] + values[right]) / 2
        for left, right in zip(steps[:-1], steps[1:], strict=True)
    )
    return area / FULL_BUDGET


def compare_benchmarks(
    benchmarks: list[str | Path],
    *,
    newt_root: str | Path | None = None,
    curves: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Compare complete frozen evaluations to immutable released reference CSVs."""

    if not benchmarks:
        raise ValueError("At least one frozen benchmark JSON is required")
    if curves is not None and len(curves) != len(benchmarks):
        raise ValueError("Supply exactly one evaluation curve per benchmark seed")
    root = Path(
        newt_root
        or os.environ.get("M4PO_NEWT_ROOT")
        or Path(__file__).resolve().parents[1] / "external" / "newt"
    ).resolve()
    sources = []
    for relative, expected in SOURCE_DIGESTS.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise ValueError(
                f"Reference source differs from pinned Newt release: {relative}"
            )
        sources.append(
            {"path": relative, "sha256": actual, "url": f"{REFERENCE_URL}/{relative}"}
        )
    _, task_sets, catalog_digest = load_mmbench_catalog(root)
    domains = [name for name in task_sets if name != "soup"]
    references = {
        name: {
            "average": _read_reference(root / f"csv/{name}/{name}_avg.csv"),
            "domain": _read_reference(root / f"csv/{name}/{name}_by_domain.csv"),
        }
        for name in ("newt", "tdmpc2")
    }
    required_steps = sorted(references["newt"]["average"])
    if required_steps != list(range(0, FULL_BUDGET + 1, 2_000_000)):
        raise ValueError("Expected the complete released 0..100M evaluation grid")

    runs = []
    seed_scores = []
    seeds = set()
    protocol = None
    initialization_protocol = None
    auc_values = []
    auc_reasons = []
    for index, filename in enumerate(benchmarks):
        path = Path(filename)
        payload = json.loads(path.read_text(encoding="utf-8"))
        initialization = payload.get("initialization", {"kind": "random"})
        if not isinstance(initialization, Mapping):
            raise ValueError("Initialization provenance must be a mapping")
        kind = initialization.get("kind")
        if kind not in {"random", "demonstration_pretraining"}:
            raise ValueError("Unrecognized M4PO initialization provenance")
        demo = initialization.get("demonstration_pretraining", {})
        if not isinstance(demo, Mapping):
            raise ValueError("Demonstration provenance must be a mapping")
        if kind == "demonstration_pretraining" and (
            demo.get("stage_completed") is not True
            or type(demo.get("completed_updates")) is not int
            or type(demo.get("target_updates")) is not int
            or demo["completed_updates"] <= 0
            or demo.get("completed_updates") != demo.get("target_updates")
            or demo.get("task_count") != 200
            or demo.get("online_demo_mixing") is not False
            or demo.get("actor_objective") != "masked_behavior_cloning_plus_entropy"
            or not isinstance(demo.get("dataset_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", demo["dataset_sha256"]) is None
            or not isinstance(demo.get("protocol_sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", demo["protocol_sha256"]) is None
        ):
            raise ValueError("Incomplete demonstration initialization provenance")
        initialization_key = (
            kind,
            demo.get("dataset_sha256"),
            demo.get("protocol_sha256"),
            demo.get("completed_updates"),
            demo.get("actor_objective"),
            demo.get("online_demo_mixing"),
        )
        if (
            initialization_protocol is not None
            and initialization_key != initialization_protocol
        ):
            raise ValueError(
                "Do not aggregate different initialization/pretraining protocols"
            )
        initialization_protocol = initialization_key
        if (
            payload.get("env") != "mmbench"
            or payload.get("learning_mode") != "off_policy"
        ):
            raise ValueError("Expected an off-policy M4PO MMBench frozen evaluation")
        seed = _integer(payload.get("seed"), "seed")
        if seed in seeds:
            raise ValueError("Multi-run aggregation requires distinct training seeds")
        seeds.add(seed)
        budget = _integer(payload.get("checkpoint_step"), "checkpoint_step")
        episodes = _integer(
            payload.get("episodes_per_pair"), "episodes_per_pair", minimum=1
        )
        if not isinstance(payload.get("deterministic"), bool):
            raise ValueError(
                "Frozen evaluation must record its deterministic policy setting"
            )
        this_protocol = (budget, episodes, payload["deterministic"])
        if protocol is not None and protocol != this_protocol:
            raise ValueError(
                "Seeds must share checkpoint budget and evaluation protocol"
            )
        protocol = this_protocol
        for reference in references.values():
            if budget not in reference["average"] or budget not in reference["domain"]:
                raise ValueError(
                    f"No exact released reference at step {budget}; interpolation is disabled"
                )
        scores = _validate_evaluation(payload, task_sets)
        if set(payload.get("tasks", [])) != set(task_sets["soup"]):
            raise ValueError(
                "Frozen checkpoint task list must match the canonical 200 tasks"
            )
        seed_scores.append(scores)
        curve_path = Path(curves[index]) if curves is not None else None
        curve, reason = _full_curve(
            curve_path,
            task_sets,
            required_steps,
            payload.get("initialization"),
            seed,
        )
        if budget != FULL_BUDGET:
            curve, reason = None, "AUC requires a final checkpoint at 100M transitions"
        auc_values.append(_auc(curve, required_steps) if curve is not None else None)
        if reason:
            auc_reasons.append({"seed": seed, "reason": reason})
        runs.append(
            {
                "seed": seed,
                "initialization": initialization,
                "benchmark": str(path.resolve()),
                "benchmark_sha256": _sha256(path),
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_step": budget,
                "score_mean": fmean(scores.values()),
                "episodes_per_task": episodes,
                "deterministic": payload["deterministic"],
                "action_timing": payload.get("action_timing"),
                "curve": str(curve_path.resolve()) if curve_path else None,
                "curve_sha256": _sha256(curve_path) if curve_path else None,
                "auc_0_100m": auc_values[-1],
            }
        )

    budget = protocol[0]
    task_means = {
        task: fmean(scores[task] for scores in seed_scores)
        for task in task_sets["soup"]
    }
    mean = fmean(task_means.values())
    newt_score = references["newt"]["average"][budget]["eval/avg_score"]
    tdmpc2_score = references["tdmpc2"]["average"][budget]["eval/avg_score"]
    per_domain = {}
    for domain in domains:
        domain_mean = fmean(task_means[task] for task in task_sets[domain])
        newt_domain = references["newt"]["domain"][budget][f"eval/avg_score_{domain}"]
        tdmpc2_domain = references["tdmpc2"]["domain"][budget][
            f"eval/avg_score_{domain}"
        ]
        per_domain[domain] = {
            "task_count": len(task_sets[domain]),
            "m4po_score": domain_mean,
            "newt_score": newt_domain,
            "tdmpc2_score": tdmpc2_domain,
            "delta_vs_newt": domain_mean - newt_domain,
            "delta_vs_tdmpc2": domain_mean - tdmpc2_domain,
        }
    complete_auc = all(value is not None for value in auc_values)
    return {
        "benchmark": "NEWT MMBench: all 200 training tasks",
        "metric": "unweighted mean of native normalized task scores",
        "checkpoint_step": budget,
        "coverage": {"tasks": 200, "domains": 10, "full_suite": True},
        "m4po": {
            "label": (
                "M4PO demo-pretrained, then off-policy online RL (no online demo mixing)"
                if initialization_protocol[0] == "demonstration_pretraining"
                else "M4PO off-policy, online from scratch (no demonstrations)"
            ),
            "num_seeds": len(runs),
            "seeds": [run["seed"] for run in runs],
            "score_mean": mean,
            "score_std_across_seeds": stdev(run["score_mean"] for run in runs)
            if len(runs) > 1
            else None,
            "auc_0_100m": fmean(auc_values) if complete_auc else None,
            "action_timing": "See measured per-run values; no cross-hardware aggregation",
        },
        "references": {
            "newt": {
                "label": "Released Newt (demonstration-pretrained + online RL)",
                "score_mean": newt_score,
                "auc_0_100m": _auc(
                    {
                        step: row["eval/avg_score"]
                        for step, row in references["newt"]["average"].items()
                    },
                    required_steps,
                )
                if complete_auc
                else None,
                "action_timing": None,
                "seed_count": None,
            },
            "tdmpc2": {
                "label": "Released TD-MPC2 reference (not relabeled as Newt no-demos)",
                "score_mean": tdmpc2_score,
                "auc_0_100m": _auc(
                    {
                        step: row["eval/avg_score"]
                        for step, row in references["tdmpc2"]["average"].items()
                    },
                    required_steps,
                )
                if complete_auc
                else None,
                "action_timing": None,
                "seed_count": None,
            },
        },
        "delta_vs_newt": mean - newt_score,
        "delta_vs_tdmpc2": mean - tdmpc2_score,
        "per_domain": per_domain,
        "per_task": {
            task: {"m4po_score": score, "newt_score": None}
            for task, score in task_means.items()
        },
        "runs": runs,
        "auc": {
            "available": complete_auc,
            "definition": "trapezoidal normalized-score integral over identical 0..100M steps, divided by 100M",
            "required_steps": required_steps,
            "unavailable_reasons": auc_reasons,
        },
        "provenance": {
            "newt_commit": NEWT_COMMIT,
            "catalog_sha256": catalog_digest,
            "sources": sources,
        },
        "caveats": [
            (
                "Both methods use demonstrations for pretraining, but NEWT also mixes demonstrations during online RL; M4PO does not. Objectives and architectures also differ."
                if initialization_protocol[0] == "demonstration_pretraining"
                else "Newt uses expert demonstrations; M4PO is trained without demonstrations. This is not an equal-data comparison."
            ),
            "The release has no explicitly identified Newt no-demonstration CSV; TD-MPC2 retains its published directory label.",
            "Newt CSVs contain no per-seed uncertainty or Newt per-task values; these are not inferred.",
            "M4PO final evaluation uses the recorded fixed episodes per task; Newt's vector loop collects at least two per task and more for shorter tasks.",
            "Model size, training throughput, and action latency are not hardware- or compute-matched by this report.",
        ],
    }


def write_reports(report: dict[str, Any], prefix: str | Path) -> dict[str, str]:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {suffix: Path(f"{prefix}.{suffix}") for suffix in ("json", "csv", "md")}
    paths["json"].write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [
        "scope",
        "task_count",
        "checkpoint_step",
        "m4po_score",
        "newt_score",
        "tdmpc2_score",
        "delta_vs_newt",
        "delta_vs_tdmpc2",
    ]
    rows = [
        {
            "scope": "all",
            "task_count": 200,
            "checkpoint_step": report["checkpoint_step"],
            "m4po_score": report["m4po"]["score_mean"],
            "newt_score": report["references"]["newt"]["score_mean"],
            "tdmpc2_score": report["references"]["tdmpc2"]["score_mean"],
            "delta_vs_newt": report["delta_vs_newt"],
            "delta_vs_tdmpc2": report["delta_vs_tdmpc2"],
        }
    ]
    rows.extend(
        {"scope": domain, "checkpoint_step": report["checkpoint_step"], **values}
        for domain, values in report["per_domain"].items()
    )
    with paths["csv"].open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MMBench comparison",
        "",
        f"All 200 tasks; {report['checkpoint_step']:,} online transitions; {report['m4po']['num_seeds']} M4PO seed(s).",
        "Scores are normalized task averages, not success percentages.",
        f"Method: {report['m4po']['label']}.",
        "",
        "| Domain | Tasks | M4PO | Newt with demos | Released TD-MPC2 | Delta vs Newt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['task_count']} | {row['m4po_score']:.4f} | {row['newt_score']:.4f} | {row['tdmpc2_score']:.4f} | {row['delta_vs_newt']:+.4f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    if report["auc"]["available"]:
        lines.extend(
            [
                "",
                f"Normalized AUC (0–100M): M4PO {report['m4po']['auc_0_100m']:.4f}; Newt {report['references']['newt']['auc_0_100m']:.4f}; TD-MPC2 {report['references']['tdmpc2']['auc_0_100m']:.4f}.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "AUC: unavailable without complete full-suite evaluations on the common 0–100M grid.",
            ]
        )
    lines.extend(
        [
            "",
            "## Measured M4PO action latency",
            "",
            "Newt action latency is unavailable; no cross-hardware value is estimated.",
            "",
        ]
    )
    for run in report["runs"]:
        timing = run.get("action_timing") or {}
        mean_ms = timing.get("mean_ms")
        description = (
            "unavailable"
            if mean_ms is None
            else f"{mean_ms:.3f} ms per delivered action batch; device {timing.get('device')}; batch {timing.get('batch_size_min')}–{timing.get('batch_size_max')}"
        )
        lines.append(f"- Seed {run['seed']}: {description}.")
    lines.extend(
        [
            "",
            f"Reference: [pinned Newt CSVs]({REFERENCE_URL}/csv), revision `{NEWT_COMMIT}`. Source and input SHA-256 digests are recorded in the JSON report.",
            "",
        ]
    )
    paths["md"].write_text("\n".join(lines), encoding="utf-8")
    return {suffix: str(path) for suffix, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare complete MMBench frozen evaluations to released Newt references."
    )
    parser.add_argument(
        "benchmarks", nargs="+", help="One full-suite benchmark JSON per training seed."
    )
    parser.add_argument(
        "--newt-root", default=None, help="Pinned official Newt checkout."
    )
    parser.add_argument(
        "--curves",
        nargs="+",
        default=None,
        help="Optional metrics JSONL files in the same seed order.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output path prefix for JSON, CSV, and Markdown.",
    )
    args = parser.parse_args()
    report = compare_benchmarks(
        args.benchmarks, newt_root=args.newt_root, curves=args.curves
    )
    print(json.dumps(write_reports(report, args.out_prefix), indent=2))


if __name__ == "__main__":
    main()
