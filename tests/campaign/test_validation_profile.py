"""Validation-only campaign profile and timed worker contract."""
from __future__ import annotations

import math

from evodevo.campaign.plan import CampaignPlan, VALIDATION_STAMP
from evodevo.evaluation.ppo_runner import PPOTask, validation_task_metadata
from evodevo.experiments.benchmark import BenchmarkConfig, CAMPAIGN_VALIDATION
from evodevo.parallel import worker_pool


def test_validation_profile_is_full_grid_and_embeds_timing_inputs():
    config = BenchmarkConfig.from_yaml("evodevo/experiments/configs/campaign_validation.yaml")
    plan = CampaignPlan.from_benchmark(config)
    assert config.profile == CAMPAIGN_VALIDATION
    assert [phase.task for phase in plan.phases] == ["Walker-v0", "UpStepper-v0"]
    assert all(len(phase.runs) == 20 for phase in plan.phases)
    assert [phase.ppo_seeds for phase in plan.phases] == [1, 2]
    assert plan.validation_profile["stamp"] == VALIDATION_STAMP
    assert plan.validation_profile["time_scale"] == {
        "Walker-v0": 1 / 100, "UpStepper-v0": 1 / 250}
    assert len(plan.validation_profile["duration_distribution"]) == 8


def test_validation_duration_draw_is_identity_deterministic():
    profile = {
        "duration_distribution": {"direct|Walker-v0": [100.0, 10.0]},
        "time_scale": {"Walker-v0": 0.01}, "e_over_p_ratio": 2.3172,
        "e_core_ids": [16], "stamp": VALIDATION_STAMP,
    }
    kwargs = dict(run_key="direct/Walker-v0/seed_0", arm="direct", task="Walker-v0",
                  generation=3, eval_index=47, ppo_k=0, fitness=1.25)
    assert validation_task_metadata(profile, **kwargs) == validation_task_metadata(profile, **kwargs)
    changed = validation_task_metadata(profile, **{**kwargs, "eval_index": 48})
    assert changed["base_duration_s"] != validation_task_metadata(profile, **kwargs)["base_duration_s"]


def test_validation_worker_applies_scale_and_ecore_ratio(monkeypatch):
    monkeypatch.setitem(worker_pool._WORKER_STATE, "slot", 0)
    monkeypatch.setitem(worker_pool._WORKER_STATE, "cpu_ids", [16])
    task = PPOTask("r", "direct", "Walker-v0", 0, 0, 0, 0, 0,
                   None, None, 0, 0, None,
                   validation_meta={"base_duration_s": 0.01, "time_scale": 0.1,
                                    "e_over_p_ratio": 2.0, "e_core_ids": [16],
                                    "fitness": 1.5, "stamp": VALIDATION_STAMP})
    result = worker_pool._run_ppo_task(task, timeout_s=1)
    assert result.completed is True and result.fitness == 1.5
    assert math.isclose(result.validation_base_duration_s, 0.01)
    assert math.isclose(result.validation_planned_duration_s, 0.002)
    assert result.wall_clock_s >= result.validation_planned_duration_s
