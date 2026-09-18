from dataclasses import replace

import pytest
import torch

from m4po.common.buffer import Episode, EpisodeReplayBuffer


def _episode(length=3, marker=0.0):
    return Episode(
        observations={
            "state": torch.arange(length + 1, dtype=torch.float32).view(-1, 1) + marker
        },
        actions=torch.full((length, 2), 0.25) * torch.tensor([1.0, 0.0]),
        rewards=torch.arange(length, dtype=torch.float32).view(-1, 1),
        terminated=torch.cat((torch.zeros(length - 1, 1), torch.ones(1, 1))),
        task_ids=torch.zeros(length + 1, dtype=torch.long),
        embodiment_ids=torch.zeros(length + 1, dtype=torch.long),
        action_masks=torch.tensor([1.0, 0.0]).expand(length, 2).clone(),
    )


def test_replay_owns_detached_cpu_copy_and_samples_padded_tails():
    replay = EpisodeReplayBuffer(capacity=8, horizon=5, batch_size=64, seed=7)
    episode = _episode()
    episode.observations["state"].requires_grad_(True)
    replay.add(episode)
    with torch.no_grad():
        episode.observations["state"].fill_(-100)
    batch = replay.sample()
    assert batch.observations["state"].shape == (6, 64, 1)
    assert not batch.observations["state"].requires_grad
    assert batch.observations["state"].device.type == "cpu"
    assert batch.valid.shape == (5, 64, 1)
    assert set(batch.valid.sum(0).flatten().tolist()) == {1.0, 2.0, 3.0}
    for index in range(64):
        length = int(batch.valid[:, index].sum())
        assert batch.observations["state"][length:, index].eq(3).all()
        assert batch.actions[length:, index].eq(0).all()
        assert batch.rewards[length:, index].eq(0).all()
        assert batch.terminated[length - 1 :, index].eq(1).all()
    assert batch.actions[..., 1].eq(0).all()


def test_replay_fifo_capacity_and_rng_roundtrip():
    replay = EpisodeReplayBuffer(capacity=5, horizon=2, batch_size=16)
    replay.add(_episode(3, marker=10))
    replay.add(_episode(3, marker=20))
    assert len(replay) == 3
    assert replay.num_episodes == 1
    assert replay.sample().observations["state"].ge(20).all()
    state = replay.rng_state_dict()
    first = replay.sample()
    replay.load_rng_state_dict(state)
    second = replay.sample()
    assert torch.equal(first.observations["state"], second.observations["state"])
    with pytest.raises(ValueError, match="capacity"):
        replay.add(_episode(6))


def test_replay_uniform_over_transition_starts_not_episodes():
    replay = EpisodeReplayBuffer(capacity=20, horizon=2, batch_size=8000, seed=13)
    replay.add(_episode(1, marker=10))
    replay.add(_episode(3, marker=20))
    batch = replay.sample()
    proportion = float((batch.observations["state"][0] < 20).float().mean())
    assert 0.23 < proportion < 0.27


@pytest.mark.parametrize(
    "change,match",
    [
        ({"task_ids": torch.tensor([0, 0, 1, 1])}, "context/reset"),
        ({"embodiment_ids": torch.tensor([0, 1, 1, 1])}, "context/reset"),
        ({"terminated": torch.tensor([[1.0], [0.0], [1.0]])}, "internal terminal"),
        ({"actions": torch.ones(3, 2)}, "Invalid action"),
        ({"rewards": torch.full((3, 1), float("nan"))}, "finite"),
    ],
)
def test_replay_rejects_invalid_episodes(change, match):
    replay = EpisodeReplayBuffer(capacity=8, horizon=2, batch_size=2)
    with pytest.raises(ValueError, match=match):
        replay.add(replace(_episode(), **change))


def test_time_limit_episode_keeps_final_bootstrap_observation():
    replay = EpisodeReplayBuffer(capacity=8, horizon=3, batch_size=1)
    replay.add(replace(_episode(1), terminated=torch.zeros(1, 1)))
    batch = replay.sample()
    assert batch.terminated[:, 0, 0].tolist() == [0.0, 1.0, 1.0]
    assert batch.valid[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert batch.observations["state"][:, 0, 0].tolist() == [0.0, 1.0, 1.0, 1.0]
