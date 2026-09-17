"""Tests for the fast_dev/primary_matrix profile mechanism
(evodevo/experiments/benchmark.py::PROFILES) -- see docs/decisions.md and
docs/PRE_PARALLELIZATION_READINESS.md for the context: ppo_timesteps=5000
was always a disclosed pipeline-validation-scale placeholder (never the
scientific comparison budget), and population_size/eval_budget for the real
primary matrix are a genuinely OPEN project decision that must never be
guessed.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest
import yaml

from evodevo.experiments.benchmark import (
    FAST_DEV,
    PRIMARY_MATRIX,
    PRIMARY_MATRIX_OPEN_FIELDS,
    PROFILES,
    BenchmarkConfig,
    FastDevRealRunGuardError,
    OpenProfileDecisionError,
    _check_profile_ready_for_real_run,
    resolve_configs,
    run_benchmark,
)

SHARED_BASE_YAML = "evodevo/experiments/configs/shared/base.yaml"


def _tiny_bconfig(**overrides):
    defaults = dict(
        encoders=["direct"],
        tasks=["Walker-v0"],
        seeds=[0],
        surrogate=True,
        factors={"dev_steps": 5, "init_condition": "center_seed"},
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def test_fast_dev_profile_matches_shared_base_yaml_bit_for_bit():
    """The acceptance criterion, made mechanical: fast_dev's declared
    values must literally equal shared/base.yaml's current committed
    values, not merely "look similar" -- if either ever drifts, this test
    catches it rather than silently going stale."""
    with open(SHARED_BASE_YAML) as f:
        base = yaml.safe_load(f)
    for field in ("ppo_timesteps", "population_size", "offspring_size", "eval_budget"):
        assert PROFILES[FAST_DEV][field] == base[field], (
            f"PROFILES['fast_dev']['{field}']={PROFILES[FAST_DEV][field]} no longer matches "
            f"{SHARED_BASE_YAML}'s {field}={base[field]}"
        )


def test_primary_matrix_ppo_timesteps_is_the_confirmed_target():
    assert PROFILES[PRIMARY_MATRIX]["ppo_timesteps"] == 1_000_000


def test_primary_matrix_has_no_invented_population_or_budget_values():
    """MUST NOT: Part 2 is explicit that these must never be guessed."""
    for field in PRIMARY_MATRIX_OPEN_FIELDS:
        assert field not in PROFILES[PRIMARY_MATRIX], (
            f"PROFILES['primary_matrix'] must not contain '{field}' -- "
            "this is a genuinely open project decision, not a value to default"
        )


def test_default_profile_is_fast_dev_for_every_pre_existing_construction():
    bconfig = _tiny_bconfig()
    assert bconfig.profile == FAST_DEV


def test_fast_dev_profile_is_a_no_op_on_resolved_configs():
    """A test's own tiny, custom shared-base fixture (or the real
    configs/shared/base.yaml) must resolve completely UNCHANGED under
    fast_dev -- this is what makes fast_dev a true no-op, not a forced
    re-application of PROFILES['fast_dev']'s own literal values, which
    would silently blow up any test using a deliberately smaller budget for
    speed."""
    with tempfile.TemporaryDirectory() as configs_root:
        shared = {
            "population_size": 3, "offspring_size": 3, "eval_budget": 9,
            "ppo_timesteps": 1, "ppo_eval_episodes": 1, "grid_shape": [5, 5],
            "sentinel_fitness": -10.0, "cache_enabled": False, "min_voxels": 3,
            "min_actuators": 1, "use_crossover": False,
        }
        os.makedirs(os.path.join(configs_root, "shared"), exist_ok=True)
        with open(os.path.join(configs_root, "shared", "base.yaml"), "w") as f:
            yaml.safe_dump(shared, f)
        os.makedirs(os.path.join(configs_root, "direct"), exist_ok=True)
        with open(os.path.join(configs_root, "direct", "walker.yaml"), "w") as f:
            yaml.safe_dump({"seed": 0, "encoder": "direct", "task": "Walker-v0", "output_dir": "unused"}, f)

        bconfig = BenchmarkConfig(
            encoders=["direct"], tasks=["Walker-v0"], seeds=[0],
            configs_root=configs_root, surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )
        resolved = resolve_configs(bconfig)
        config = resolved[("direct", "Walker-v0")]
        assert config.eval_budget == 9
        assert config.ppo_timesteps == 1
        assert config.population_size == 3


def test_primary_matrix_profile_overrides_ppo_timesteps_on_resolved_configs():
    bconfig = _tiny_bconfig(
        profile=PRIMARY_MATRIX,
        profile_overrides={"population_size": 2, "offspring_size": 2, "eval_budget": 4},
    )
    resolved = resolve_configs(bconfig)
    config = resolved[("direct", "Walker-v0")]
    assert config.ppo_timesteps == 1_000_000
    assert config.population_size == 2
    assert config.eval_budget == 4


def test_surrogate_run_is_exempt_from_both_guards():
    bconfig = _tiny_bconfig()  # fast_dev, surrogate=True, no override
    _check_profile_ready_for_real_run(bconfig)  # must not raise

    bconfig2 = _tiny_bconfig(profile=PRIMARY_MATRIX)  # missing fields, but surrogate=True
    _check_profile_ready_for_real_run(bconfig2)  # must not raise


def test_real_run_under_fast_dev_without_override_raises_loudly():
    bconfig = _tiny_bconfig(surrogate=False)
    with pytest.raises(FastDevRealRunGuardError, match="fast_dev"):
        _check_profile_ready_for_real_run(bconfig)


def test_real_run_under_fast_dev_with_explicit_override_is_allowed():
    bconfig = _tiny_bconfig(surrogate=False, allow_fast_dev_real_run=True)
    _check_profile_ready_for_real_run(bconfig)  # must not raise


def test_real_run_under_primary_matrix_without_scoping_decision_raises_loudly():
    bconfig = _tiny_bconfig(surrogate=False, profile=PRIMARY_MATRIX)
    with pytest.raises(OpenProfileDecisionError, match="population_size"):
        _check_profile_ready_for_real_run(bconfig)


def test_real_run_under_primary_matrix_with_explicit_scoping_decision_is_allowed():
    bconfig = _tiny_bconfig(
        surrogate=False,
        profile=PRIMARY_MATRIX,
        profile_overrides={"population_size": 24, "offspring_size": 24, "eval_budget": 720},
    )
    _check_profile_ready_for_real_run(bconfig)  # must not raise


def test_partial_primary_matrix_override_still_raises():
    """All of PRIMARY_MATRIX_OPEN_FIELDS must be supplied together -- they
    are one scoping decision, not three independently-guessable numbers."""
    bconfig = _tiny_bconfig(
        surrogate=False,
        profile=PRIMARY_MATRIX,
        profile_overrides={"population_size": 24},  # offspring_size/eval_budget still missing
    )
    with pytest.raises(OpenProfileDecisionError):
        _check_profile_ready_for_real_run(bconfig)


def test_run_benchmark_refuses_real_fast_dev_run_before_touching_disk():
    """The end-to-end proof: run_benchmark() itself refuses, and does so
    before writing anything (no manifest, no output_dir contents) --
    mirrors this project's other "abort before anything runs" guards
    (ConfigOverrideError/FairnessViolation)."""
    with tempfile.TemporaryDirectory() as output_dir:
        bconfig = _tiny_bconfig(surrogate=False, output_dir=output_dir)
        with pytest.raises(FastDevRealRunGuardError):
            run_benchmark(bconfig, dry_run=False, allow_dirty=True)
        assert os.listdir(output_dir) == []


def test_run_benchmark_refuses_real_primary_matrix_run_before_touching_disk():
    with tempfile.TemporaryDirectory() as output_dir:
        bconfig = _tiny_bconfig(surrogate=False, profile=PRIMARY_MATRIX, output_dir=output_dir)
        with pytest.raises(OpenProfileDecisionError):
            run_benchmark(bconfig, dry_run=False, allow_dirty=True)
        assert os.listdir(output_dir) == []
