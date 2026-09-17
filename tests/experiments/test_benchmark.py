import json
import os
import tempfile

import pytest
import yaml

from evodevo.core.config import ConfigOverrideError
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.encoders.nca.resolved_config import PRIMARY as NCA_PRIMARY
from evodevo.encoders.rule_based.resolved_config import (
    MONOTONE_ABLATION as RULE_BASED_MONOTONE,
    PRIMARY as RULE_BASED_PRIMARY,
)
from evodevo.experiments.benchmark import (
    ArmSpec,
    BenchmarkConfig,
    MANIFEST_NAME,
    _resolved_arm_kwargs,
    check_fairness,
    resolve_configs,
    run_benchmark,
)
from evodevo.experiments.fairness import FairnessViolation

TINY_SHARED_BASE = {
    "population_size": 3,
    "offspring_size": 3,
    "eval_budget": 9,
    "ppo_timesteps": 1,
    "ppo_eval_episodes": 1,
    "grid_shape": [5, 5],
    "sentinel_fitness": -10.0,
    "cache_enabled": False,
    "min_voxels": 3,
    "min_actuators": 1,
    "use_crossover": False,
}


def _write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _build_configs_tree(root, shared_overrides=None, per_method_overrides=None):
    """Builds a temp configs_root with shared/base.yaml + one dir per
    method, mirroring the real project's evodevo/experiments/configs/
    layout but with a tiny, fast budget."""
    shared = dict(TINY_SHARED_BASE)
    shared.update(shared_overrides or {})
    _write_yaml(os.path.join(root, "shared", "base.yaml"), shared)

    per_method_overrides = per_method_overrides or {}
    for encoder in ("direct", "nca", "rule_based"):
        override = {
            "seed": 0,
            "encoder": encoder,
            "task": "Walker-v0",
            "output_dir": "unused",  # benchmark.py always forces its own output_dir
        }
        override.update(per_method_overrides.get(encoder, {}))
        _write_yaml(os.path.join(root, encoder, "walker.yaml"), override)

    return root


def test_end_to_end_surrogate_benchmark_identical_evaluation_counts_and_manifest():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as output_dir:
        _build_configs_tree(configs_root)
        bconfig = BenchmarkConfig(
            encoders=["direct", "nca", "rule_based"],
            tasks=["Walker-v0"],
            seeds=[0, 1],
            configs_root=configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        manifest = run_benchmark(bconfig, dry_run=False, allow_dirty=True)

        # manifest written
        manifest_path = os.path.join(output_dir, MANIFEST_NAME)
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            on_disk = json.load(f)
        assert on_disk["run_matrix"] == manifest["run_matrix"]
        assert len(manifest["completed_this_invocation"]) == 6  # 3 encoders x 2 seeds
        assert manifest["skipped_already_done"] == []

        # identical evaluation counts per arm (same eval_budget for all three,
        # verified by inspecting each run's run_log.jsonl directly)
        eval_counts = {}
        for entry in manifest["run_matrix"]:
            log_path = os.path.join(entry["output_dir"], "run_log.jsonl")
            with open(log_path) as f:
                n_lines = sum(1 for line in f if line.strip())
            eval_counts.setdefault(entry["encoder"], []).append(n_lines)

        assert set(eval_counts) == {"direct", "nca", "rule_based"}
        all_counts = {c for counts in eval_counts.values() for c in counts}
        assert all_counts == {TINY_SHARED_BASE["eval_budget"]}


def test_resume_skips_completed_runs():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as output_dir:
        _build_configs_tree(configs_root)
        bconfig = BenchmarkConfig(
            encoders=["direct", "nca", "rule_based"],
            tasks=["Walker-v0"],
            seeds=[0],
            configs_root=configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        first = run_benchmark(bconfig, dry_run=False, allow_dirty=True)
        assert len(first["completed_this_invocation"]) == 3
        assert first["skipped_already_done"] == []

        second = run_benchmark(bconfig, dry_run=False, allow_dirty=True)
        assert second["completed_this_invocation"] == []
        assert len(second["skipped_already_done"]) == 3


def test_dry_run_writes_manifest_and_timing_but_runs_nothing():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as output_dir:
        _build_configs_tree(configs_root)
        bconfig = BenchmarkConfig(
            encoders=["direct", "nca", "rule_based"],
            tasks=["Walker-v0"],
            seeds=[0],
            configs_root=configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        manifest = run_benchmark(bconfig, dry_run=True, allow_dirty=True)
        assert "dry_run_timing_estimate" in manifest
        assert set(manifest["dry_run_timing_estimate"]["per_encoder_measured_cost_s"]) == {
            "direct",
            "nca",
            "rule_based",
        }

        for entry in manifest["run_matrix"]:
            assert not os.path.exists(os.path.join(entry["output_dir"], "run_meta.json"))


def test_mismatched_config_aborts_before_any_run_starts_via_config_override_error():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as output_dir:
        # nca's config illegally overrides a shared-base key with a
        # different value -- caught at config-load time.
        _build_configs_tree(
            configs_root,
            per_method_overrides={"nca": {"eval_budget": 999}},
        )
        bconfig = BenchmarkConfig(
            encoders=["direct", "nca", "rule_based"],
            tasks=["Walker-v0"],
            seeds=[0],
            configs_root=configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        with pytest.raises(ConfigOverrideError, match="eval_budget"):
            run_benchmark(bconfig, dry_run=False, allow_dirty=True)

        # nothing was written -- not even the manifest.
        assert not os.path.exists(os.path.join(output_dir, MANIFEST_NAME))
        assert os.listdir(output_dir) == []


def test_mismatched_config_aborts_via_assert_comparable_when_bypassing_override_detection():
    """Constructs a scenario where check_fairness's own assert_comparable
    call is what catches the mismatch, rather than ConfigOverrideError --
    i.e. two configs that were each individually valid (no shared/base.yaml
    involved) but disagree on a required-equal field."""
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as output_dir:
        # No shared/base.yaml at all: every per-method config is a fully
        # self-contained flat ExperimentConfig, so from_method_yaml() falls
        # back to plain from_yaml() for each -- no override detection
        # applies, only the runtime assert_comparable check can catch this.
        for encoder, eval_budget in (("direct", 9), ("nca", 999), ("rule_based", 9)):
            _write_yaml(
                os.path.join(configs_root, encoder, "walker.yaml"),
                {
                    "seed": 0,
                    "encoder": encoder,
                    "task": "Walker-v0",
                    "output_dir": "unused",
                    "population_size": 3,
                    "offspring_size": 3,
                    "eval_budget": eval_budget,
                    "ppo_timesteps": 1,
                    "ppo_eval_episodes": 1,
                },
            )

        bconfig = BenchmarkConfig(
            encoders=["direct", "nca", "rule_based"],
            tasks=["Walker-v0"],
            seeds=[0],
            configs_root=configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        with pytest.raises(FairnessViolation, match="eval_budget"):
            check_fairness(bconfig)

        assert not os.path.exists(os.path.join(output_dir, MANIFEST_NAME))


def test_real_benchmark_yaml_wires_rule_based_resolved_readout_policy():
    """This is the test that would have caught the original gap (docs/
    decisions.md's "Wire the resolved readout policy into the primary
    benchmark path" row): builds the encoder for the PRIMARY rule_based arm
    through the actual production config-building path (BenchmarkConfig.
    from_yaml on the real evodevo/experiments/configs/benchmark.yaml,
    resolve_configs, _resolved_arm_kwargs, get_encoder) -- not a
    hand-constructed RuleBasedEncoder(...) -- and asserts its readout_mode/
    readout_window match evodevo.encoders.rule_based.resolved_config.PRIMARY
    exactly."""
    bconfig = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
    resolved = resolve_configs(bconfig)
    config = resolved[("rule_based", "Walker-v0")]

    arm = ArmSpec(encoder="rule_based")
    arm_kwargs = _resolved_arm_kwargs(bconfig, arm)
    factors, remaining_kwargs = extract_factors(arm_kwargs, seed=config.seed)
    encoder = get_encoder("rule_based", factors=factors, grid_shape=config.grid_shape, **remaining_kwargs)

    assert encoder.readout_mode == RULE_BASED_PRIMARY.readout_mode == "mode_of_window"
    assert encoder.readout_window == RULE_BASED_PRIMARY.readout_window == 6


def test_real_benchmark_yaml_wires_nca_resolved_hidden_channels():
    """The NCA counterpart of the readout-policy wiring test above: docs/
    decisions.md's "Hidden channel count (H) -- final re-resolution" row
    dropped H to 0 (resolved_config.py::PRIMARY.hidden_channels=0), but
    ExperimentFactors.hidden_channels' own default (3, evodevo/core/
    factors.py) was never wired to track it -- unlike update_rule/decay/
    state_bound/init_gain/readout_mode/readout_window, which already reach
    the real benchmark path via NCAEncoder's own class defaults, H reaches
    it only via this arm-specific _BUILTIN_ARM_DEFAULT_KWARGS entry. Builds
    the encoder for the real, production `nca` arm through the actual
    production config-building path and asserts hidden_channels matches
    resolved_config.PRIMARY exactly, not ExperimentFactors' own stale
    class-level default."""
    bconfig = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
    resolved = resolve_configs(bconfig)
    config = resolved[("nca", "Walker-v0")]

    arm = ArmSpec(encoder="nca")
    arm_kwargs = _resolved_arm_kwargs(bconfig, arm)
    factors, remaining_kwargs = extract_factors(arm_kwargs, seed=config.seed)
    encoder = get_encoder("nca", factors=factors, grid_shape=config.grid_shape, **remaining_kwargs)

    assert encoder.hidden_channels == NCA_PRIMARY.hidden_channels == 0


def test_benchmark_yaml_registers_rule_based_monotone_as_a_distinct_arm():
    """docs/decisions.md's "Monotone vs unrestricted: an explicit decision"
    row elevated MONOTONE_ABLATION to a co-primary PPO-matrix candidate; the
    "Wire the resolved readout policy into the primary benchmark path" row
    is now resolved to state this arm is registered. This asserts the real
    benchmark.yaml declares it (as a {encoder, variant} entry, not a bare
    string) and that its resolved encoder_kwargs match
    resolved_config.MONOTONE_ABLATION exactly -- the real-config-path
    counterpart to test_real_benchmark_yaml_wires_rule_based_resolved_readout_policy
    above."""
    bconfig = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
    arms = {arm.arm_id: arm for arm in bconfig.resolved_arms()}
    assert "rule_based_monotone" in arms
    monotone_arm = arms["rule_based_monotone"]
    assert monotone_arm.encoder == "rule_based"
    assert monotone_arm.variant == "monotone"

    resolved = resolve_configs(bconfig)
    config = resolved[("rule_based_monotone", "Walker-v0")]

    arm_kwargs = _resolved_arm_kwargs(bconfig, monotone_arm)
    factors, remaining_kwargs = extract_factors(arm_kwargs, seed=config.seed)
    encoder = get_encoder("rule_based", factors=factors, grid_shape=config.grid_shape, **remaining_kwargs)

    assert encoder.action_monotonicity == RULE_BASED_MONOTONE.action_monotonicity == "monotone"
    assert encoder.readout_mode == RULE_BASED_MONOTONE.readout_mode == "mode_of_window"
    assert encoder.readout_window == RULE_BASED_MONOTONE.readout_window == 6

    # And the PRIMARY arm is untouched -- still unrestricted.
    primary_config = resolved[("rule_based", "Walker-v0")]
    primary_kwargs = _resolved_arm_kwargs(bconfig, arms["rule_based"])
    p_factors, p_remaining = extract_factors(primary_kwargs, seed=primary_config.seed)
    primary_encoder = get_encoder("rule_based", factors=p_factors, grid_shape=primary_config.grid_shape, **p_remaining)
    assert primary_encoder.action_monotonicity == RULE_BASED_PRIMARY.action_monotonicity == "unrestricted"


def test_resolved_arms_parses_mixed_string_and_dict_encoders_generically():
    """A fourth (throwaway) variant on top of an unrelated encoder ("nca")
    should work with zero code changes to benchmark.py -- proving the
    ArmSpec/resolved_arms() mechanism generalizes beyond rule_based_monotone,
    not just special-cases it."""
    bconfig = BenchmarkConfig(
        encoders=[
            "direct",
            {"encoder": "nca"},
            {"encoder": "nca", "variant": "throwaway_fourth_variant", "encoder_kwargs": {"dev_steps": 3}},
        ],
        tasks=["Walker-v0"],
        seeds=[0],
    )
    arms = bconfig.resolved_arms()
    arm_ids = [a.arm_id for a in arms]
    assert arm_ids == ["direct", "nca", "nca_throwaway_fourth_variant"]
    assert arms[2].encoder == "nca"
    assert arms[2].is_variant
    assert arms[2].encoder_kwargs == {"dev_steps": 3}
    assert not arms[0].is_variant and not arms[1].is_variant


def test_rule_based_and_rule_based_monotone_produce_distinct_non_colliding_artifacts():
    """The test that proves the collision is actually fixed, not merely
    declared fixed: constructs both rule_based (PRIMARY) and
    rule_based_monotone via their real benchmark-config path (the production
    evodevo/experiments/configs/benchmark.yaml), runs a short surrogate
    benchmark, and asserts distinct output directories, no file overwrites,
    and no shared state between the two runs."""
    with tempfile.TemporaryDirectory() as output_dir:
        real_bconfig = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
        bconfig = BenchmarkConfig(
            encoders=["rule_based", {"encoder": "rule_based", "variant": "monotone"}],
            tasks=["Walker-v0"],
            seeds=[0],
            configs_root=real_bconfig.configs_root,
            output_dir=output_dir,
            surrogate=True,
            factors={"dev_steps": 5, "init_condition": "center_seed"},
        )

        manifest = run_benchmark(bconfig, dry_run=False, allow_dirty=True)

        entries_by_arm = {e["arm_id"]: e for e in manifest["run_matrix"]}
        assert set(entries_by_arm) == {"rule_based", "rule_based_monotone"}

        primary_dir = entries_by_arm["rule_based"]["output_dir"]
        monotone_dir = entries_by_arm["rule_based_monotone"]["output_dir"]
        assert primary_dir != monotone_dir

        # Both actually ran (not one skipped/overwritten by the other).
        assert len(manifest["completed_this_invocation"]) == 2
        assert manifest["skipped_already_done"] == []

        for arm_id, expected_monotonicity in (("rule_based", "unrestricted"), ("rule_based_monotone", "monotone")):
            entry = entries_by_arm[arm_id]
            meta_path = os.path.join(entry["output_dir"], "run_meta.json")
            log_path = os.path.join(entry["output_dir"], "run_log.jsonl")
            assert os.path.exists(meta_path)
            assert os.path.exists(log_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["config"]["arm_id"] == arm_id
            assert meta["config"]["encoder"] == "rule_based"  # same registry class for both
            assert meta["encoder"]["action_monotonicity"] == expected_monotonicity
            with open(log_path) as f:
                n_lines = sum(1 for line in f if line.strip())
            assert n_lines == meta["config"]["eval_budget"]
