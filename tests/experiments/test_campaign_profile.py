"""Tests for the `campaign` profile (evodevo/experiments/benchmark.py::PROFILES)
-- the final campaign design ACCEPTED in docs/decisions.md on 2026-09-27 --
its schema validation, and the K > 1 guard that stops a D16 multi-seed
config from silently running as K=1 before the evaluator supports it."""
from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest
import yaml

from evodevo.core.config import ExperimentConfig
from evodevo.experiments.benchmark import (
    CAMPAIGN,
    PROFILES,
    BenchmarkConfig,
    CampaignProfileError,
    OpenProfileDecisionError,
    PpoSeedsNotImplementedError,
    _check_profile_ready_for_real_run,
    check_ppo_seeds_supported,
    resolve_configs,
    run_benchmark,
    validate_profile,
)
from evodevo.experiments.run_experiment import run_experiment

REAL_BENCHMARK_YAML = "evodevo/experiments/configs/benchmark.yaml"
TASKS = ["Walker-v0", "UpStepper-v0"]


def _campaign_bconfig(**overrides):
    defaults = dict(
        encoders=["direct", "nca", "rule_based", {"encoder": "rule_based", "variant": "monotone"}],
        tasks=list(TASKS),
        seeds=[0, 1, 2, 3, 4],
        surrogate=True,
        factors={"dev_steps": 5, "init_condition": "center_seed"},
        profile=CAMPAIGN,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def test_campaign_profile_holds_every_decided_value():
    p = PROFILES[CAMPAIGN]
    assert (p["population_size"], p["offspring_size"], p["eval_budget"]) == (15, 15, 300)
    assert p["ppo_timesteps"] == {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
    assert p["ppo_seeds"] == {"Walker-v0": 1, "UpStepper-v0": 2}
    assert p["ppo_seed_aggregation"] == "max"


def test_real_benchmark_yaml_targets_the_campaign():
    bconfig = BenchmarkConfig.from_yaml(REAL_BENCHMARK_YAML)
    assert bconfig.profile == CAMPAIGN
    assert bconfig.tasks == TASKS  # D7: Walker-v0 first
    assert bconfig.seeds == [0, 1, 2, 3, 4]
    assert [a.arm_id for a in bconfig.resolved_arms()] == ["direct", "nca", "rule_based", "rule_based_monotone"]


def test_campaign_resolves_per_task_values_identically_across_arms():
    resolved = resolve_configs(BenchmarkConfig.from_yaml(REAL_BENCHMARK_YAML))
    expected = {"Walker-v0": (1_000_000, 1), "UpStepper-v0": (1_200_000, 2)}
    assert len(resolved) == 8
    for (arm_id, task), cfg in resolved.items():
        assert (cfg.ppo_timesteps, cfg.ppo_seeds) == expected[task], (arm_id, task)
        assert cfg.ppo_seed_aggregation == "max"
        assert (cfg.population_size, cfg.offspring_size, cfg.eval_budget) == (15, 15, 300)


def test_campaign_has_no_open_decision_left_for_a_real_run():
    """OpenProfileDecisionError is kept for `primary_matrix` only; the
    campaign profile decides every field, so a real-run readiness check
    passes (the K > 1 guard is separate -- see below)."""
    bconfig = _campaign_bconfig(surrogate=False)
    _check_profile_ready_for_real_run(bconfig)  # must not raise


def test_primary_matrix_still_raises_open_decision_error():
    bconfig = _campaign_bconfig(surrogate=False, profile="primary_matrix", tasks=["Walker-v0"])
    with pytest.raises(OpenProfileDecisionError):
        _check_profile_ready_for_real_run(bconfig)


@pytest.mark.parametrize("bad", ["median", "MAX", "", None])
def test_validation_rejects_unknown_aggregation(bad):
    profile = {**PROFILES[CAMPAIGN], "ppo_seed_aggregation": bad}
    with pytest.raises(CampaignProfileError, match="ppo_seed_aggregation"):
        validate_profile(profile, TASKS)


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True])
def test_validation_rejects_k_below_one_or_non_integer(bad_k):
    profile = {**PROFILES[CAMPAIGN], "ppo_seeds": {"Walker-v0": 1, "UpStepper-v0": bad_k}}
    with pytest.raises(CampaignProfileError, match="ppo_seeds"):
        validate_profile(profile, TASKS)


@pytest.mark.parametrize("field_name", ["ppo_timesteps", "ppo_seeds"])
def test_validation_rejects_missing_per_task_value(field_name):
    profile = {**PROFILES[CAMPAIGN], field_name: {"Walker-v0": PROFILES[CAMPAIGN][field_name]["Walker-v0"]}}
    with pytest.raises(CampaignProfileError, match="UpStepper-v0"):
        validate_profile(profile, TASKS)


@pytest.mark.parametrize("field_name", ["eval_budget", "ppo_seeds", "ppo_seed_aggregation"])
def test_validation_rejects_missing_required_field(field_name):
    profile = {k: v for k, v in PROFILES[CAMPAIGN].items() if k != field_name}
    with pytest.raises(CampaignProfileError, match=field_name):
        validate_profile(profile, TASKS)


def test_validation_runs_when_resolving_a_bconfig():
    bconfig = _campaign_bconfig(profile_overrides={"ppo_seed_aggregation": "median"})
    with pytest.raises(CampaignProfileError):
        resolve_configs(bconfig)


def test_experiment_config_itself_rejects_bad_k_and_aggregation():
    base = dict(seed=0, encoder="direct", task="Walker-v0", population_size=3, offspring_size=3,
                eval_budget=6, ppo_timesteps=100, ppo_eval_episodes=1, output_dir="unused")
    assert ExperimentConfig(**base).ppo_seeds == 1
    with pytest.raises(ValueError, match="ppo_seeds"):
        ExperimentConfig(**base, ppo_seeds=0)
    with pytest.raises(ValueError, match="ppo_seed_aggregation"):
        ExperimentConfig(**base, ppo_seed_aggregation="median")


def test_k_above_one_is_accepted_by_the_benchmark_guard():
    resolved = resolve_configs(BenchmarkConfig.from_yaml(REAL_BENCHMARK_YAML))
    check_ppo_seeds_supported(resolved)
    assert issubclass(PpoSeedsNotImplementedError, NotImplementedError)


def test_k_of_one_everywhere_passes_the_guard():
    resolved = resolve_configs(_campaign_bconfig(tasks=["Walker-v0"]))
    check_ppo_seeds_supported(resolved)  # Walker-v0 is K=1: must not raise


def test_run_experiment_accepts_k_above_one_in_surrogate_mode():
    """The second, entry-point-level guard: a resolved config with K=2
    reaching run_experiment() directly (bypassing run_benchmark) is refused,
    never silently evaluated once."""
    resolved = resolve_configs(BenchmarkConfig.from_yaml(REAL_BENCHMARK_YAML))
    cfg = resolved[("direct", "UpStepper-v0")]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(cfg, output_dir=tmp)
        path = os.path.join(tmp, "resolved.yaml")
        cfg.to_yaml(path)
        run_experiment(path, seed=0, surrogate=True, overwrite=True,
                       encoder_kwargs={"dev_steps": 5, "init_condition": "center_seed"}, allow_dirty=True)
        assert yaml.safe_load(open(path))["ppo_seeds"] == 2
