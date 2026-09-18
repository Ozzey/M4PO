from copy import deepcopy
import hashlib
import json

import pytest
import torch

from m4po.common.config import M4POConfig
import m4po.common.demonstrations as demos
from m4po.envs.mmbench_env import load_mmbench_catalog


def _shard(episodes=2, length=2):
    count = episodes * (length + 1)
    obs = torch.zeros(count, 128)
    obs[:, :3] = torch.arange(count).view(-1, 1)
    action = torch.zeros(count, 16)
    action[:, :2] = 0.25
    reward = torch.arange(count).float()
    action[:: length + 1] = float("nan")
    reward[:: length + 1] = float("nan")
    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "terminated": torch.zeros(count, dtype=torch.bool),
        "episode": torch.arange(episodes).repeat_interleave(length + 1),
        "value": torch.full((count,), float("nan")),
        "feat": torch.zeros(count, 7),
    }


def _convert(shard, **kwargs):
    return list(
        demos.iter_demonstration_episodes(
            shard,
            task_id=4,
            embodiment_id=2,
            state_dim=3,
            action_dim=2,
            episode_horizon=kwargs.pop("episode_horizon", 2),
            **kwargs,
        )
    )


def test_newt_dummy_row_alignment_contexts_masks_and_timeouts():
    episodes = _convert(_shard())
    assert len(episodes) == 2
    first, second = episodes
    assert first.observations["state"][:, 0].tolist() == [0, 1, 2]
    assert first.rewards[:, 0].tolist() == [1, 2]
    assert second.observations["state"][:, 0].tolist() == [3, 4, 5]
    assert second.rewards[:, 0].tolist() == [4, 5]
    assert not first.terminated.any()
    assert first.task_ids.eq(4).all() and first.embodiment_ids.eq(2).all()
    assert first.action_masks.sum(-1).tolist() == [2, 2]
    assert first.observations["state_mask"].sum(-1).tolist() == [3, 3, 3]
    assert first.observations["image"].shape == (3, 0, 0, 0)


def test_native_early_termination_is_preserved():
    shard = _shard(1, 1)
    shard["terminated"][-1] = True
    (episode,) = _convert(shard)
    assert episode.terminated.item() == 1


def test_maniskill_cap_retains_exact_first_twenty_complete_episodes():
    episodes = _convert(_shard(22), max_episodes=20)
    assert len(episodes) == 20
    assert episodes[-1].observations["state"][-1, 0] == 59
    with pytest.raises(ValueError, match="fewer"):
        _convert(_shard(19), max_episodes=20)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda td: td.pop("reward"), "missing"),
        (lambda td: td["action"][0].zero_(), "dummy NaNs"),
        (lambda td: td["reward"].__setitem__(1, float("nan")), "finite"),
        (lambda td: td["action"].__setitem__((1, 0), 1.5), "\\[-1, 1\\]"),
        (lambda td: td["action"].__setitem__((1, 3), 0.2), "coordinates"),
        (lambda td: td["obs"].__setitem__((1, 7), 0.1), "padding"),
        (lambda td: td["obs"].__setitem__((1, 0), float("nan")), "finite"),
        (lambda td: td["terminated"].__setitem__(1, True), "final"),
        (lambda td: td["episode"].__setitem__(3, 0), "dummy NaNs|time limit"),
        (lambda td: td["episode"].__setitem__(1, 1), "contiguous"),
        (lambda td: td.__setitem__("terminated", td["terminated"].float()), "bool"),
        (lambda td: td.__setitem__("action", td["action"][:, :2]), "shapes"),
    ],
)
def test_invalid_shards_are_rejected(mutation, message):
    shard = _shard()
    mutation(shard)
    with pytest.raises(ValueError, match=message):
        _convert(shard)


def test_incomplete_nonterminal_trajectories_are_not_imported():
    with pytest.raises(ValueError, match="incomplete"):
        _convert(_shard(length=1))


@pytest.fixture
def dataset_files(tmp_path, monkeypatch):
    metadata, task_sets, catalog_digest = load_mmbench_catalog()
    metadata = deepcopy(metadata)
    tasks = task_sets["soup"]
    for task in tasks:
        metadata[task]["max_episode_steps"] = 2
    monkeypatch.setattr(
        demos, "load_mmbench_catalog", lambda *_: (metadata, task_sets, catalog_digest)
    )
    monkeypatch.setattr(demos, "_native_source_digest", lambda *_: "native-sha")
    report = {
        "complete": True,
        "full_suite_passed": True,
        "passed_task_count": 200,
        "failed_task_count": 0,
        "catalog_sha256": catalog_digest,
        "required_tasks": tasks,
        "per_task": {},
    }
    records = []
    for task in tasks:
        shard = _shard(20 if task in task_sets["maniskill"] else 1)
        shard["action"][:, 1:] = 0
        shard["action"][::3] = float("nan")
        path = tmp_path / f"{task}.pt"
        torch.save(shard, path)
        records.append(
            {
                "task": task,
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        report["per_task"][task] = {
            "status": "passed",
            "catalog_sha256": catalog_digest,
            "native_sources_sha256": "native-sha",
            "native_state_dim": 3,
            "native_action_dim": metadata[task]["action_dim"],
            "embodiment": metadata[task]["embodiment"],
            "metadata_horizon": 2,
        }
    manifest = {
        "schema_version": 1,
        "repo_id": demos.DATASET_REPOSITORY,
        "revision": demos.DATASET_REVISION,
        "catalog_sha256": catalog_digest,
        "task_count": 200,
        "total_bytes": sum(row["size"] for row in records),
        "files": records,
    }
    manifest["dataset_manifest_sha256"] = demos.canonical_digest(manifest)
    monkeypatch.setattr(
        demos, "DATASET_MANIFEST_SHA256", manifest["dataset_manifest_sha256"]
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(report))
    cfg = M4POConfig.from_yaml("m4po/configs/mmbench_all.yaml")
    cfg.replay_capacity = 1
    return cfg, tmp_path, report_path, manifest, report


def test_full_loader_validates_all_tasks_and_never_evicts(dataset_files):
    cfg, directory, report, manifest, _ = dataset_files
    data = demos.load_demonstrations(cfg, directory, report)
    expected = 200 + 19 * len(load_mmbench_catalog()[1]["maniskill"])
    assert data.replay.num_episodes == expected
    assert len(data.replay) == 2 * expected
    assert data.replay.capacity >= len(data.replay)
    assert data.provenance["task_count"] == 200
    assert data.provenance["dataset_sha256"] == manifest["dataset_manifest_sha256"]
    assert data.task_contexts.shape == (200, 512)
    assert {int(ep.task_ids[0]) for ep in data.replay._episodes} == set(range(200))


@pytest.mark.parametrize(
    "corruption", ["manifest", "missing_preflight", "native_source", "shard"]
)
def test_loader_rejects_mismatched_artifacts(dataset_files, corruption):
    cfg, directory, report_path, manifest, report = dataset_files
    if corruption == "manifest":
        manifest["revision"] = "not-pinned"
        (directory / "manifest.json").write_text(json.dumps(manifest))
    elif corruption == "missing_preflight":
        report["full_suite_passed"] = False
        report_path.write_text(json.dumps(report))
    elif corruption == "native_source":
        report["per_task"][cfg.task_names[0]]["native_sources_sha256"] = "other"
        report_path.write_text(json.dumps(report))
    else:
        path = directory / manifest["files"][0]["path"]
        value = bytearray(path.read_bytes())
        value[-1] ^= 1
        path.write_bytes(value)
    with pytest.raises(ValueError):
        demos.load_demonstrations(cfg, directory, report_path)
