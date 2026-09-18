from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from m4po.compare_mmbench import compare_benchmarks, write_reports
from m4po.envs.mmbench_env import load_mmbench_catalog


NEWT_ROOT = Path(__file__).resolve().parents[1] / "external" / "newt"


@pytest.fixture
def benchmark():
    _, task_sets, _ = load_mmbench_catalog(NEWT_ROOT)
    tasks = task_sets["soup"]
    return {
        "env": "mmbench",
        "learning_mode": "off_policy",
        "seed": 0,
        "checkpoint_step": 100_000_000,
        "checkpoint": "/example/checkpoints/latest.pt",
        "episodes_per_pair": 10,
        "deterministic": True,
        "tasks": tasks,
        "score_aggregation": "mean_of_task_mean_scores",
        "score_mean": 0.4,
        "per_task": {task: {"score_mean": 0.4} for task in tasks},
        "benchmark_coverage": {
            "canonical_task_count": 200,
            "selected_task_count": 200,
            "evaluated_task_count": 200,
            "score_task_count": 200,
            "domain_count": 10,
            "full_suite": True,
        },
        "action_timing": {
            "device": "cuda",
            "mean_ms": 3.2,
            "batch_size_min": 1,
            "batch_size_max": 1,
        },
    }


def _save(tmp_path, payload, name="benchmark.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_full_suite_reference_values_domains_and_missing_baselines(tmp_path, benchmark):
    report = compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)
    assert report["m4po"]["score_mean"] == pytest.approx(0.4)
    assert report["references"]["newt"]["score_mean"] == pytest.approx(
        0.4378576328375057
    )
    assert report["references"]["tdmpc2"]["score_mean"] == pytest.approx(
        0.26244842269339813
    )
    assert report["delta_vs_newt"] == pytest.approx(0.4 - 0.4378576328375057)
    assert report["per_domain"]["pygame"]["task_count"] == 19
    assert report["per_domain"]["dmcontrol-ext"]["newt_score"] == pytest.approx(
        0.3656315305852331
    )
    assert all(row["newt_score"] is None for row in report["per_task"].values())
    assert report["references"]["newt"]["action_timing"] is None
    assert report["m4po"]["score_std_across_seeds"] is None
    assert report["m4po"]["auc_0_100m"] is None
    assert report["runs"][0]["action_timing"]["mean_ms"] == 3.2
    assert len(report["provenance"]["sources"]) == 6
    assert len(report["runs"][0]["benchmark_sha256"]) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("full_suite", False),
        ("score_task_count", 199),
        ("selected_task_count", 4),
        ("domain_count", 9),
    ],
)
def test_reject_incomplete_coverage(tmp_path, benchmark, field, value):
    benchmark["benchmark_coverage"][field] = value
    with pytest.raises(ValueError, match="coverage|domains"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), 1.1, True])
def test_reject_undefined_or_invalid_task_scores(tmp_path, benchmark, bad_score):
    benchmark["per_task"]["walker-stand"]["score_mean"] = bad_score
    with pytest.raises(ValueError, match="normalized score"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)


def test_reject_wrong_task_identity_and_wrong_macro_average(tmp_path, benchmark):
    wrong_task = deepcopy(benchmark)
    wrong_task["per_task"]["not-a-native-task"] = wrong_task["per_task"].pop(
        "walker-stand"
    )
    with pytest.raises(ValueError, match="canonical 200"):
        compare_benchmarks([_save(tmp_path, wrong_task)], newt_root=NEWT_ROOT)
    benchmark["score_mean"] = 0.5
    with pytest.raises(ValueError, match="Aggregate score"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)


def test_reference_budget_must_match_exactly(tmp_path, benchmark):
    benchmark["checkpoint_step"] = 99_999_999
    with pytest.raises(ValueError, match="exact released reference"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)
    benchmark["checkpoint_step"] = 20_000_000
    report = compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)
    assert report["references"]["newt"]["score_mean"] == pytest.approx(
        0.3468608894064103
    )
    assert report["m4po"]["auc_0_100m"] is None


def test_multiseed_requires_same_protocol_and_distinct_seeds(tmp_path, benchmark):
    first = _save(tmp_path, benchmark, "first.json")
    with pytest.raises(ValueError, match="distinct training seeds"):
        compare_benchmarks([first, first], newt_root=NEWT_ROOT)
    second = deepcopy(benchmark)
    second["seed"] = 1
    second["episodes_per_pair"] = 20
    with pytest.raises(ValueError, match="same|share"):
        compare_benchmarks(
            [first, _save(tmp_path, second, "second.json")], newt_root=NEWT_ROOT
        )
    second["episodes_per_pair"] = 10
    second["score_mean"] = 0.6
    for row in second["per_task"].values():
        row["score_mean"] = 0.6
    report = compare_benchmarks(
        [first, _save(tmp_path, second, "second.json")], newt_root=NEWT_ROOT
    )
    assert report["m4po"]["num_seeds"] == 2
    assert report["m4po"]["score_mean"] == pytest.approx(0.5)
    assert report["m4po"]["score_std_across_seeds"] == pytest.approx(0.1414213562)


def _write_curve(tmp_path, benchmark, steps):
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": step, "eval": benchmark} for step in steps]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_auc_uses_complete_equal_grid_without_extrapolation(tmp_path, benchmark):
    path = _save(tmp_path, benchmark)
    steps = list(range(0, 100_000_001, 2_000_000))
    curve = _write_curve(tmp_path, benchmark, steps)
    report = compare_benchmarks([path], newt_root=NEWT_ROOT, curves=[curve])
    assert report["auc"]["available"] is True
    assert report["m4po"]["auc_0_100m"] == pytest.approx(0.4)
    assert report["references"]["newt"]["auc_0_100m"] > 0
    curve = _write_curve(tmp_path, benchmark, steps[1:])
    report = compare_benchmarks([path], newt_root=NEWT_ROOT, curves=[curve])
    assert report["auc"]["available"] is False
    assert report["references"]["newt"]["auc_0_100m"] is None
    assert "[0]" in report["auc"]["unavailable_reasons"][0]["reason"]


def test_auc_rejects_partial_suite_and_duplicate_points(tmp_path, benchmark):
    path = _save(tmp_path, benchmark)
    steps = list(range(0, 100_000_001, 2_000_000))
    curve = _write_curve(tmp_path, benchmark, steps + [0])
    report = compare_benchmarks([path], newt_root=NEWT_ROOT, curves=[curve])
    assert "duplicate" in report["auc"]["unavailable_reasons"][0]["reason"]
    benchmark["benchmark_coverage"]["full_suite"] = False
    curve = _write_curve(tmp_path, benchmark, steps)
    report = compare_benchmarks([path], newt_root=NEWT_ROOT, curves=[curve])
    assert "Invalid full-suite" in report["auc"]["unavailable_reasons"][0]["reason"]


def test_outputs_are_json_csv_markdown_with_explicit_caveats(tmp_path, benchmark):
    report = compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)
    outputs = write_reports(report, tmp_path / "comparison")
    assert json.loads(Path(outputs["json"]).read_text())["coverage"]["tasks"] == 200
    with Path(outputs["csv"]).open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 11
    markdown = Path(outputs["md"]).read_text()
    assert "not success percentages" in markdown
    assert "not an equal-data comparison" in markdown
    assert "no-demonstration CSV" in markdown
    assert "3.200 ms" in markdown


def test_reference_files_are_digest_verified(tmp_path, benchmark, monkeypatch):
    from m4po import compare_mmbench

    monkeypatch.setitem(
        compare_mmbench.SOURCE_DIGESTS, "csv/newt/newt_avg.csv", "wrong"
    )
    with pytest.raises(ValueError, match="pinned Newt release"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)


def _demo_initialization():
    return {
        "kind": "demonstration_pretraining",
        "checkpoint_sha256": "b" * 64,
        "demonstration_pretraining": {
            "stage_completed": True,
            "completed_updates": 10_000,
            "target_updates": 10_000,
            "task_count": 200,
            "dataset_sha256": "a" * 64,
            "protocol_sha256": "e" * 64,
            "actor_objective": "masked_behavior_cloning_plus_entropy",
            "online_demo_mixing": False,
        },
    }


def test_demo_comparison_labels_data_protocol_accurately(tmp_path, benchmark):
    benchmark["initialization"] = _demo_initialization()
    report = compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)
    assert "demo-pretrained" in report["m4po"]["label"]
    assert "no online demo mixing" in report["m4po"]["label"]
    assert "Both methods use demonstrations" in report["caveats"][0]
    assert "NEWT also mixes demonstrations" in report["caveats"][0]
    assert report["runs"][0]["initialization"] == benchmark["initialization"]
    outputs = write_reports(report, tmp_path / "demo-comparison")
    markdown = Path(outputs["md"]).read_text()
    assert "demo-pretrained" in markdown
    assert "from scratch (no demonstrations)" not in markdown


@pytest.mark.parametrize(
    "different", ["random", "dataset", "protocol", "updates", "objective"]
)
def test_comparison_rejects_mixed_initialization_protocols(
    tmp_path, benchmark, different
):
    first = deepcopy(benchmark)
    first["initialization"] = _demo_initialization()
    second = deepcopy(first)
    second["seed"] = 1
    if different == "random":
        second["initialization"] = {"kind": "random"}
    else:
        demo = second["initialization"]["demonstration_pretraining"]
        if different == "dataset":
            demo["dataset_sha256"] = "c" * 64
        elif different == "protocol":
            demo["protocol_sha256"] = "f" * 64
        elif different == "updates":
            demo["completed_updates"] = demo["target_updates"] = 20_000
        else:
            demo["actor_objective"] = "different_objective"
    with pytest.raises(ValueError, match="initialization|pretraining"):
        compare_benchmarks(
            [
                _save(tmp_path, first, "first.json"),
                _save(tmp_path, second, "second.json"),
            ],
            newt_root=NEWT_ROOT,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage_completed", False),
        ("task_count", 199),
        ("completed_updates", 0),
        ("completed_updates", True),
        ("dataset_sha256", "short"),
        ("dataset_sha256", "z" * 64),
        ("protocol_sha256", None),
        ("protocol_sha256", "short"),
        ("protocol_sha256", "z" * 64),
        ("actor_objective", "q_max"),
        ("online_demo_mixing", True),
    ],
)
def test_comparison_rejects_incomplete_or_invalid_demo_provenance(
    tmp_path, benchmark, field, value
):
    benchmark["initialization"] = _demo_initialization()
    demo = benchmark["initialization"]["demonstration_pretraining"]
    demo[field] = value
    if field == "completed_updates":
        demo["target_updates"] = value
    with pytest.raises(ValueError, match="demonstration|initialization|pretraining"):
        compare_benchmarks([_save(tmp_path, benchmark)], newt_root=NEWT_ROOT)


def test_demo_runs_with_same_protocol_can_pool_distinct_training_seeds(
    tmp_path, benchmark
):
    first = deepcopy(benchmark)
    first["initialization"] = _demo_initialization()
    second = deepcopy(first)
    second["seed"] = 1
    second["initialization"]["checkpoint_sha256"] = "d" * 64
    report = compare_benchmarks(
        [_save(tmp_path, first, "first.json"), _save(tmp_path, second, "second.json")],
        newt_root=NEWT_ROOT,
    )
    assert report["m4po"]["num_seeds"] == 2
    assert "demo-pretrained" in report["m4po"]["label"]


@pytest.mark.parametrize("kind", ["random", "demonstration_pretraining"])
def test_auc_accepts_curve_bound_to_final_initialization_and_seed(
    tmp_path, benchmark, kind
):
    benchmark["initialization"] = (
        _demo_initialization()
        if kind == "demonstration_pretraining"
        else {"kind": "random"}
    )
    rows = [
        {
            "step": step,
            "eval": benchmark,
            "initialization": benchmark["initialization"],
            "seed": benchmark["seed"],
        }
        for step in range(0, 100_000_001, 2_000_000)
    ]
    curve = tmp_path / "metrics.jsonl"
    curve.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    report = compare_benchmarks(
        [_save(tmp_path, benchmark)], newt_root=NEWT_ROOT, curves=[curve]
    )
    assert report["auc"]["available"] is True
    assert report["m4po"]["auc_0_100m"] == pytest.approx(0.4)


@pytest.mark.parametrize("different", ["initialization", "dataset", "seed", "missing"])
def test_auc_rejects_curve_with_wrong_initialization_or_seed_provenance(
    tmp_path, benchmark, different
):
    benchmark["initialization"] = _demo_initialization()
    rows = [
        {
            "step": step,
            "eval": benchmark,
            "initialization": deepcopy(benchmark["initialization"]),
            "seed": benchmark["seed"],
        }
        for step in range(0, 100_000_001, 2_000_000)
    ]
    bad_row = rows[1]
    if different == "initialization":
        bad_row["initialization"] = {"kind": "random"}
    elif different == "dataset":
        bad_row["initialization"]["demonstration_pretraining"]["dataset_sha256"] = (
            "d" * 64
        )
    elif different == "seed":
        bad_row["seed"] = 1
    else:
        del bad_row["initialization"], bad_row["seed"]
    curve = tmp_path / "metrics.jsonl"
    curve.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    report = compare_benchmarks(
        [_save(tmp_path, benchmark)], newt_root=NEWT_ROOT, curves=[curve]
    )
    assert report["auc"]["available"] is False
    assert report["m4po"]["auc_0_100m"] is None
    assert "provenance differs" in report["auc"]["unavailable_reasons"][0]["reason"]
