import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from m4po.common.buffer import EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import M4POConfig
from m4po.common.demonstrations import DemonstrationDataset, iter_demonstration_episodes
from m4po.m4po import M4POAgent
import m4po.pretrain_demonstrations as offline
from test_demonstrations import _shard


@pytest.fixture
def training(tmp_path, monkeypatch):
    cfg = M4POConfig(
        use_images=False,
        proprio_dim=0,
        state_dim=128,
        num_envs=1,
        task_latent_dim=8,
        body_latent_dim=8,
        task_context_dim=8,
        embodiment_context_dim=8,
        encoder_dim=16,
        mlp_dim=16,
        num_encoder_layers=1,
        dropout=0,
        num_q=2,
        model_horizon=2,
        planning_horizon=2,
        planner_samples=2,
        batch_size=4,
        replay_capacity=16,
        device="cpu",
        seed=31,
        log_dir=str(tmp_path / "offline"),
    )

    def load(cfg, *_):
        spec = ObservationSpec((0, 0, 0), 0, 128)
        replay = EpisodeReplayBuffer(16, 2, 4, seed=cfg.seed, observation_spec=spec)
        for episode in iter_demonstration_episodes(
            _shard(3),
            task_id=0,
            embodiment_id=0,
            state_dim=3,
            action_dim=2,
            episode_horizon=2,
        ):
            replay.add(episode)
        return DemonstrationDataset(
            replay,
            spec,
            np.zeros((1, 8), dtype=np.float32),
            {
                "dataset_sha256": "demo-data",
                "catalog_sha256": "catalog",
                "native_sources_sha256": "native",
                "task_count": 1,
                "task_names": ["reach"],
                "online_demo_mixing": False,
            },
        )

    monkeypatch.setattr(offline, "load_demonstrations", load)
    monkeypatch.setattr(offline, "_implementation_digest", lambda: "code-sha")
    return cfg


def _train(cfg, **kwargs):
    return offline.pretrain(
        cfg,
        data_dir="verified-demo-directory",
        preflight_report="passed.json",
        save_every=2,
        log_every=1,
        **kwargs,
    )


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _equal_nested(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            _equal_nested(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _equal_nested(left, right)
    else:
        assert first == second


def test_offline_stage_updates_real_model_with_zero_online_steps_and_immutable_final(
    training,
):
    final = _train(training, updates=3)
    payload = _load(final)
    extra = payload["extra"]
    assert final.name == "pretrained_model.pt"
    assert final.read_bytes() == final.with_name("latest.pt").read_bytes()
    assert payload["step"] == 0 and extra["updates_completed"] == 3
    assert extra["training_complete"] and extra["resumable"]
    assert extra["stage"] == "demonstration_pretraining"
    assert "replay_state" not in extra and "update_count" not in extra
    provenance = extra["demonstration_pretraining"]
    assert provenance["completed_updates"] == 3 and provenance["stage_completed"]
    assert provenance["actor_objective"] == "masked_behavior_cloning_plus_entropy"
    assert provenance["online_demo_mixing"] is False
    status = json.loads(final.with_name("latest_status.json").read_text())
    stat = final.with_name("latest.pt").stat()
    assert status["checkpoint_size"] == stat.st_size
    assert status["checkpoint_mtime_ns"] == stat.st_mtime_ns
    assert status["dataset_sha256"] == "demo-data"
    assert status["training_complete"] is True
    assert (Path(training.log_dir) / "config_resolved.yaml").exists()
    metrics = [
        json.loads(line)
        for line in (Path(training.log_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["step"] == 0 and row["phase"] == offline.STAGE for row in metrics)
    assert all(
        row["actor_bc_enabled"] == 1 for row in metrics if "actor_bc_enabled" in row
    )
    first_stat = final.stat()
    assert _train(training, updates=3, resume=final.with_name("latest.pt")) == final
    assert final.stat() == first_stat
    with pytest.raises(FileExistsError, match="already"):
        _train(training, updates=3)


def test_safe_wall_stop_resume_exactly_matches_uninterrupted_optimization(
    training, tmp_path, monkeypatch
):
    cfg = training
    full = _load(_train(cfg, updates=5))
    cfg.log_dir = str(tmp_path / "resumed")
    clock = {"now": 0.0}
    monkeypatch.setattr(
        offline, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    update = M4POAgent.update

    def timed_update(self, replay, **kwargs):
        metrics = update(self, replay, **kwargs)
        clock["now"] += 1
        return metrics

    monkeypatch.setattr(M4POAgent, "update", timed_update)
    checkpoint = _train(cfg, updates=5, max_wall_time_seconds=2)
    partial = _load(checkpoint)
    assert partial["extra"]["updates_completed"] == 2
    assert not partial["extra"]["training_complete"]
    assert partial["extra"]["stop_reason"] == "wall_time_limit"
    assert not checkpoint.with_name("pretrained_model.pt").exists()
    resumed = _load(_train(cfg, updates=5, resume=checkpoint))
    for key in ("model", "actor", "world_model_optimizer", "actor_optimizer", "scale"):
        _equal_nested(full[key], resumed[key])
    _equal_nested(
        full["extra"]["replay_rng_state"], resumed["extra"]["replay_rng_state"]
    )
    assert torch.equal(
        full["extra"]["rng_state"]["torch_cpu"],
        resumed["extra"]["rng_state"]["torch_cpu"],
    )
    assert resumed["cfg"]["total_steps"] == cfg.total_steps


@pytest.mark.parametrize("field", ["updates", "dataset", "code", "config", "progress"])
def test_resume_rejects_changed_data_code_objective_budget_or_progress(
    training, monkeypatch, field
):
    final = _train(training, updates=2)
    kwargs = {"updates": 2, "resume": final.with_name("latest.pt")}
    if field == "updates":
        kwargs["updates"] = 3
    elif field == "dataset":
        loader = offline.load_demonstrations

        def changed(*args):
            dataset = loader(*args)
            dataset.provenance["dataset_sha256"] = "other-demo-data"
            return dataset

        monkeypatch.setattr(offline, "load_demonstrations", changed)
    elif field == "code":
        monkeypatch.setattr(offline, "_implementation_digest", lambda: "different-code")
    elif field == "config":
        training.entropy_coef *= 2
    else:
        payload = _load(kwargs["resume"])
        payload["extra"]["updates_completed"] = 1
        corrupt = final.with_name("corrupt.pt")
        torch.save(payload, corrupt)
        kwargs["resume"] = corrupt
    with pytest.raises(ValueError, match="incompatible"):
        _train(training, **kwargs)


def test_nonfinite_update_preserves_last_good_checkpoint(training, monkeypatch):
    original = M4POAgent.update
    count = {"updates": 0}

    def broken(self, replay, **kwargs):
        metrics = original(self, replay, **kwargs)
        count["updates"] += 1
        if count["updates"] == 3:
            metrics["model_loss"] = float("nan")
        return metrics

    monkeypatch.setattr(M4POAgent, "update", broken)
    with pytest.raises(FloatingPointError):
        _train(training, updates=5)
    checkpoint = Path(training.log_dir) / "checkpoints" / "latest.pt"
    assert _load(checkpoint)["extra"]["updates_completed"] == 2
    assert not checkpoint.with_name("pretrained_model.pt").exists()


def test_compute_allocation_is_verified_before_loading_data(training, monkeypatch):
    def refuse():
        raise RuntimeError("Not an allocated node")

    monkeypatch.setattr(offline, "verify_slurm_allocation", refuse)
    with pytest.raises(RuntimeError, match="allocated"):
        _train(training, updates=1, require_slurm=True)
    assert not Path(training.log_dir).exists()


def test_final_publication_never_overwrites_other_weights(tmp_path):
    source, target = tmp_path / "latest.pt", tmp_path / "pretrained_model.pt"
    source.write_bytes(b"new weights")
    target.write_bytes(b"original weights")
    with pytest.raises(FileExistsError, match="replace"):
        offline._publish_pretrained(source, target)
    assert target.read_bytes() == b"original weights"
