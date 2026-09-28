"""Capacity model checks for the production and historical dispatch policies."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from evodevo.campaign.dispatch import DispatchCore


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capacity_simulator", ROOT / "scripts/capacity_planning/simulate_campaign.py")
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


def test_continuous_model_executes_the_shared_dispatch_core(monkeypatch):
    calls = []

    class ObservedCore(DispatchCore):
        def begin_phase(self, phase, run_keys):
            calls.append((phase, tuple(run_keys)))
            return super().begin_phase(phase, calls[-1][1])

    monkeypatch.setattr(SIM, "DispatchCore", ObservedCore)
    durations = [np.ones((SIM.GENS, SIM.POP)) for _ in range(2)]
    masks = [np.ones_like(row, dtype=bool) for row in durations]
    costs = {"prepare_s": [0.01, 0.01], "finish_s": [0.01, 0.01],
             "journal_s": 0.001}
    SIM.run_continuous_dispatch(durations, [1.0] * 4, costs,
                                prefetch=2, inline_masks=masks)
    assert calls == [("simulation", ("0", "1"))]


def test_all_scheduler_work_is_serialized_by_one_blocking_server():
    durations = [np.ones((SIM.GENS, SIM.POP)) for _ in range(2)]
    masks = [np.zeros_like(row, dtype=bool) for row in durations]
    # Per run and generation: prepare 1s + finish 2s. With 2 runs x 20
    # generations and no worker tasks, a single server must take exactly 120s.
    costs = {"prepare_s": [1.0, 1.0], "finish_s": [2.0, 2.0], "journal_s": 7.0}
    makespan, _finish, metrics = SIM.run_continuous_dispatch(
        durations, [1.0] * 4, costs, prefetch=2, inline_masks=masks)
    assert makespan == 120.0
    assert metrics["worker_utilization"] == 0.0
    assert metrics["scheduler_busy_fraction"] == 1.0


def test_drained_global_waves_are_slower_and_less_utilized_on_same_tasks():
    durations = [np.ones((SIM.GENS, SIM.POP)),
                 np.full((SIM.GENS, SIM.POP), 2.0)]
    masks = [np.ones_like(row, dtype=bool) for row in durations]
    costs = {"prepare_s": [0.1, 0.2], "finish_s": [0.01, 0.01],
             "journal_s": 0.001}
    factors = [1.0] * 4
    continuous = SIM.run_continuous_dispatch(
        durations, factors, costs, prefetch=2, inline_masks=masks)
    waves = SIM.run_global_wave_dispatch(
        durations, factors, costs, inline_masks=masks,
        assignment_rng=np.random.default_rng(20260928))
    assert continuous[0] < waves[0]
    assert continuous[2]["worker_utilization"] > waves[2]["worker_utilization"]
