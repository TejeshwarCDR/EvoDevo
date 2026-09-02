import json
import os
import tempfile

import pytest
import yaml

from evodevo.core.config import ConfigOverrideError
from evodevo.experiments.benchmark import (
    BenchmarkConfig,
    MANIFEST_NAME,
    check_fairness,
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
        )

        manifest = run_benchmark(bconfig, dry_run=False)

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
        )

        first = run_benchmark(bconfig, dry_run=False)
        assert len(first["completed_this_invocation"]) == 3
        assert first["skipped_already_done"] == []

        second = run_benchmark(bconfig, dry_run=False)
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
        )

        manifest = run_benchmark(bconfig, dry_run=True)
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
        )

        with pytest.raises(ConfigOverrideError, match="eval_budget"):
            run_benchmark(bconfig, dry_run=False)

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
        )

        with pytest.raises(FairnessViolation, match="eval_budget"):
            check_fairness(bconfig)

        assert not os.path.exists(os.path.join(output_dir, MANIFEST_NAME))
