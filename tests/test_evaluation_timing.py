from types import SimpleNamespace

import numpy as np
import pytest

from m4po.evaluate import _TimedAgent


def test_timing_excludes_warmup_and_reports_batch_latency(monkeypatch):
    readings = iter([0.0, 0.001, 0.01, 0.013])
    monkeypatch.setattr("m4po.evaluate.time.perf_counter", lambda: next(readings))
    agent = SimpleNamespace(
        device="cpu",
        cfg="forwarded",
        act_np=lambda *args, **kwargs: SimpleNamespace(action=np.zeros((4, 2))),
    )
    timed = _TimedAgent(agent, warmup_calls=1)
    timed.act_np(None)
    timed.act_np(None)
    summary = timed.summary()
    assert timed.cfg == "forwarded"
    assert summary["warmup_calls_excluded"] == 1
    assert summary["measured_calls"] == 1
    assert summary["batch_size_min"] == summary["batch_size_max"] == 4
    assert summary["mean_ms"] == pytest.approx(3.0)
    assert summary["p95_ms"] == pytest.approx(3.0)


def test_timing_with_only_warmup_does_not_invent_measurements():
    timed = _TimedAgent(SimpleNamespace(device="cpu"))
    assert timed.summary()["mean_ms"] is None
    assert timed.summary()["measured_calls"] == 0
