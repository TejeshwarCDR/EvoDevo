"""Acceptance test: a short surrogate three-arm benchmark records
evals_spent/evals_valid (and cache/invalid-reason bookkeeping), the
counters are monotonic, and evals_valid <= evals_spent for every arm --
see evodevo/evolution/ea.py and evodevo/analysis/budget_curves.py."""
import os
import tempfile

import pandas as pd
import pytest

from evodevo.analysis.aggregate import discover_runs, load_runs
from evodevo.experiments.benchmark import BenchmarkConfig, run_benchmark
from tests.experiments.test_benchmark import TINY_SHARED_BASE, _build_configs_tree


def _run_tiny_three_arm_benchmark(configs_root, output_dir):
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
    return run_benchmark(bconfig, dry_run=False, allow_dirty=True)


@pytest.fixture(scope="module")
def three_arm_df():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        _run_tiny_three_arm_benchmark(configs_root, runs_root)
        df = load_runs(discover_runs(runs_root))
    return df


def test_budget_counters_present(three_arm_df):
    required_columns = {
        "evals_spent",
        "evals_valid_cumulative",
        "evals_invalid_cumulative",
        "evals_cache_hit_cumulative",
        "best_fitness_so_far",
        "reason_category",
    }
    assert required_columns <= set(three_arm_df.columns)


def test_evals_valid_never_exceeds_evals_spent(three_arm_df):
    assert (three_arm_df["evals_valid_cumulative"] <= three_arm_df["evals_spent"]).all()


def test_evals_spent_equals_valid_plus_invalid(three_arm_df):
    assert (
        three_arm_df["evals_spent"]
        == three_arm_df["evals_valid_cumulative"] + three_arm_df["evals_invalid_cumulative"]
    ).all()


def test_counters_are_monotonic_per_encoder_per_seed(three_arm_df):
    for (_encoder, _seed), group in three_arm_df.groupby(["encoder", "seed"]):
        group = group.sort_values("eval_index")
        for column in ("evals_spent", "evals_valid_cumulative", "evals_invalid_cumulative",
                       "evals_cache_hit_cumulative", "best_fitness_so_far"):
            values = group[column].to_numpy()
            assert (values[1:] >= values[:-1]).all(), f"{column} is not monotonic for this (encoder, seed)"


def test_evals_spent_matches_eval_index_plus_one(three_arm_df):
    # evals_spent is THE budget-governing counter -- it must agree exactly
    # with the pre-existing eval_index bookkeeping (eval_index is 0-based).
    assert (three_arm_df["evals_spent"] == three_arm_df["eval_index"] + 1).all()


def test_all_three_arms_present(three_arm_df):
    assert set(three_arm_df["encoder"].unique()) == {"direct", "nca", "rule_based"}


def test_generation_log_written_per_run():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        _run_tiny_three_arm_benchmark(configs_root, runs_root)
        run_dirs = discover_runs(runs_root)
        assert run_dirs
        for run_dir in run_dirs:
            gen_log_path = os.path.join(run_dir, "generation_log.jsonl")
            assert os.path.exists(gen_log_path)
            with open(gen_log_path) as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) >= 1


def test_run_meta_records_resolved_policy():
    import json

    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        _run_tiny_three_arm_benchmark(configs_root, runs_root)
        run_dirs = discover_runs(runs_root)
        policies = []
        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "run_meta.json")) as f:
                meta = json.load(f)
            assert meta["resolved_policy"] is not None
            policies.append(meta["resolved_policy"])
        # Identical across every arm.
        assert all(p == policies[0] for p in policies)
