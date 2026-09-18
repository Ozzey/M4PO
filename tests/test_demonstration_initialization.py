from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from m4po.common.buffer import Episode, EpisodeReplayBuffer, ObservationSpec
from m4po.common.config import M4POConfig
from m4po.common.initialization import load_demonstration_initialization
from m4po.envs.mmbench_env import (
    MMBenchVectorEnv,
    _native_source_digest,
    load_mmbench_catalog,
)
from m4po.m4po import M4POAgent
from m4po.trainer import OnlineTrainer


NEWT_ROOT = Path(__file__).resolve().parents[1] / "external" / "newt"


def _assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, Mapping):
        assert first.keys() == second.keys()
        for key, value in first.items():
            _assert_equal(value, second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_equal(left, right)
    else:
        assert first == second


@pytest.fixture
def completed_pretraining(tmp_path):
    cfg = M4POConfig(
        env="mmbench",
        mmbench_root=str(NEWT_ROOT),
        use_images=False,
        num_envs=1,
        task_latent_dim=8,
        body_latent_dim=8,
        embodiment_context_dim=4,
        encoder_dim=16,
        mlp_dim=16,
        num_encoder_layers=1,
        simnorm_dim=4,
        dropout=0,
        model_horizon=2,
        planning_horizon=2,
        planner_samples=4,
        batch_size=2,
        replay_capacity=32,
        total_steps=2,
        seed_steps=100,
        pretrain_updates=2,
        rollout_steps=1,
        eval_every=0,
        save_every=0,
        log_every=1,
        save_replay=True,
        log_dir=str(tmp_path / "online"),
        device="cpu",
    )
    cfg.validate()
    metadata, _, catalog_hash = load_mmbench_catalog(NEWT_ROOT)
    contexts = np.asarray(
        [metadata[task]["text_embedding"] for task in cfg.task_names], dtype=np.float32
    )
    spec = ObservationSpec(image_shape=(0, 0, 0), proprio_dim=0, state_dim=128)
    torch.manual_seed(73)
    agent = M4POAgent(spec, 16, cfg, torch.device("cpu"), task_contexts=contexts)
    replay = EpisodeReplayBuffer(32, 2, 2, observation_spec=spec)
    observation = {name: torch.zeros(3, *shape) for name, shape in spec.shapes.items()}
    observation["state_mask"].fill_(1)
    action_mask = torch.zeros(2, 16)
    action_mask[:, :6] = 1
    replay.add(
        Episode(
            observations=observation,
            actions=torch.zeros(2, 16),
            rewards=torch.ones(2, 1),
            terminated=torch.tensor([[0.0], [1.0]]),
            task_ids=torch.zeros(3, dtype=torch.long),
            embodiment_ids=torch.zeros(3, dtype=torch.long),
            action_masks=action_mask,
        )
    )
    agent.update(replay, actor_mode="behavior_cloning")
    provenance = {
        "stage_completed": True,
        "completed_updates": 1,
        "target_updates": 1,
        "task_count": cfg.num_tasks,
        "task_names": cfg.task_names,
        "catalog_sha256": catalog_hash,
        "native_sources_sha256": _native_source_digest(NEWT_ROOT),
        "dataset_sha256": "a" * 64,
        "actor_objective": "masked_behavior_cloning_plus_entropy",
        "online_demo_mixing": False,
    }
    path = tmp_path / "demo.pt"
    agent.save(
        path,
        step=0,
        extra={
            "stage": "demonstration_pretraining",
            "training_complete": True,
            "updates_completed": 1,
            "demonstration_pretraining": provenance,
        },
    )
    return SimpleNamespace(
        path=path, cfg=cfg, spec=spec, contexts=contexts, agent=agent
    )


def _load(completed, *, cfg=None, spec=None, action_dim=16, contexts=None):
    return load_demonstration_initialization(
        completed.path,
        cfg or deepcopy(completed.cfg),
        spec or completed.spec,
        action_dim,
        completed.contexts if contexts is None else contexts,
        torch.device("cpu"),
    )


def _rewrite(completed, mutate):
    payload = torch.load(completed.path, weights_only=False)
    mutate(payload)
    torch.save(payload, completed.path)


def test_completed_m4po_initialization_retains_weights_and_optimizers(
    completed_pretraining,
):
    completed = completed_pretraining
    cfg = replace(
        completed.cfg,
        seed=101,
        log_dir="different-online-run",
        init_checkpoint=str(completed.path),
    )
    restored, provenance = _load(completed, cfg=cfg)
    _assert_equal(restored.state_dict(), completed.agent.state_dict())
    _assert_equal(
        restored.actor_optimizer.state_dict(),
        completed.agent.actor_optimizer.state_dict(),
    )
    _assert_equal(
        restored.world_model_optimizer.state_dict(),
        completed.agent.world_model_optimizer.state_dict(),
    )
    assert restored.cfg.seed == 101
    assert restored.cfg.log_dir == "different-online-run"
    assert provenance["kind"] == "demonstration_pretraining"
    assert len(provenance["checkpoint_sha256"]) == 64
    assert provenance["demonstration_pretraining"]["completed_updates"] == 1


def test_online_initialization_starts_fresh_replay_and_counters(
    completed_pretraining, monkeypatch
):
    import m4po.trainer.off_policy_trainer as trainer

    completed = completed_pretraining
    metadata, _, _ = load_mmbench_catalog(NEWT_ROOT)
    cfg = deepcopy(completed.cfg)
    cfg.init_checkpoint = str(completed.path)
    # These are deliberately not online-compatible; initialization must not
    # treat completed demonstration replay/counters as a resumed online run.
    _rewrite(
        completed,
        lambda payload: payload["extra"].update(
            {
                "step": 900,
                "update_count": 800,
                "completed_episodes": 700,
                "replay_state": {
                    "sentinel": "demonstrations must not enter online replay"
                },
            }
        ),
    )

    class NativeEnv:
        def __init__(self, task):
            dim = metadata[task]["action_dim"]
            self.action_space = SimpleNamespace(
                shape=(dim,), low=-np.ones(dim), high=np.ones(dim)
            )

        def reset(self):
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            return (
                np.ones(4, dtype=np.float32),
                0.5,
                True,
                False,
                {"success": 0.0, "score": 0.0},
            )

        def close(self):
            pass

    monkeypatch.setattr(
        trainer,
        "make_vector_env",
        lambda config: MMBenchVectorEnv(
            config, native_factory=lambda **kwargs: NativeEnv(kwargs["task"])
        ),
    )
    path = OnlineTrainer(cfg).train()
    payload = torch.load(path, weights_only=False)
    extra = payload["extra"]
    assert payload["step"] == 2
    assert extra["update_count"] == 0
    assert extra["pretrain_updates_completed"] == 0
    assert extra["completed_episodes"] == 2
    assert extra["replay_state"]["size"] == 2
    assert len(extra["replay_state"]["episodes"]) == 2
    assert extra["initialization"]["kind"] == "demonstration_pretraining"
    _assert_equal(
        payload["actor_optimizer"], completed.agent.actor_optimizer.state_dict()
    )
    _assert_equal(
        payload["world_model_optimizer"],
        completed.agent.world_model_optimizer.state_dict(),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_complete", False),
        ("stage", "online_training"),
        ("stage", ["demonstration_pretraining", "demonstration_pretraining"]),
        ("updates_completed", 0),
        ("updates_completed", True),
    ],
)
def test_incomplete_or_repeated_stage_metadata_is_rejected(
    completed_pretraining, field, value
):
    _rewrite(
        completed_pretraining,
        lambda payload: payload["extra"].__setitem__(field, value),
    )
    with pytest.raises(ValueError, match="completed M4PO"):
        _load(completed_pretraining)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage_completed", False),
        ("completed_updates", 0),
        ("completed_updates", True),
        ("target_updates", 2),
        ("target_updates", True),
        ("online_demo_mixing", True),
        ("actor_objective", "q_max"),
    ],
)
def test_invalid_pretraining_protocol_is_rejected(completed_pretraining, field, value):
    _rewrite(
        completed_pretraining,
        lambda payload: payload["extra"]["demonstration_pretraining"].__setitem__(
            field, value
        ),
    )
    with pytest.raises(ValueError, match="completed M4PO"):
        _load(completed_pretraining)


@pytest.mark.parametrize(
    "field,value",
    [
        ("catalog_sha256", "b" * 64),
        ("native_sources_sha256", "b" * 64),
        ("dataset_sha256", "short"),
        ("dataset_sha256", "z" * 64),
        ("task_count", 199),
        ("task_names", ["wrong-task"]),
    ],
)
def test_native_catalog_dataset_and_task_mismatches_are_rejected(
    completed_pretraining, field, value
):
    _rewrite(
        completed_pretraining,
        lambda payload: payload["extra"]["demonstration_pretraining"].__setitem__(
            field, value
        ),
    )
    with pytest.raises(ValueError, match="semantics differ|dataset"):
        _load(completed_pretraining)


def test_architecture_context_observation_and_action_mismatches_rejected(
    completed_pretraining,
):
    completed = completed_pretraining
    with pytest.raises(ValueError, match="configuration differs"):
        _load(completed, cfg=replace(completed.cfg, mlp_dim=32))
    contexts = completed.contexts.copy()
    contexts[0, 0] += 1
    with pytest.raises(ValueError, match="contexts differ"):
        _load(completed, contexts=contexts)
    with pytest.raises(ValueError, match="semantics differ"):
        _load(
            completed,
            spec=ObservationSpec(image_shape=(0, 0, 0), proprio_dim=0, state_dim=127),
        )
    with pytest.raises(ValueError, match="semantics differ"):
        _load(completed, action_dim=15)


@pytest.mark.parametrize(
    "field,value",
    [("algorithm", "newt"), ("implementation_id", "tdmpc2"), ("step", 100)],
)
def test_foreign_newt_weights_and_online_checkpoints_are_not_initializations(
    completed_pretraining, field, value
):
    _rewrite(completed_pretraining, lambda payload: payload.__setitem__(field, value))
    with pytest.raises(ValueError):
        _load(completed_pretraining)


@pytest.mark.parametrize("env,mode", [("mock", "off_policy"), ("mmbench", "on_policy")])
def test_initialization_requires_offpolicy_mmbench(completed_pretraining, env, mode):
    cfg = replace(
        completed_pretraining.cfg,
        env=env,
        learning_mode=mode,
        init_checkpoint=str(completed_pretraining.path),
    )
    with pytest.raises(ValueError, match="off-policy MMBench"):
        cfg.validate()
    with pytest.raises(ValueError, match="completed M4PO"):
        _load(completed_pretraining, cfg=cfg)
