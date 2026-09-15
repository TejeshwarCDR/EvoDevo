"""Acceptance test: {ea, random_search} x {direct, nca, rule_based} -- six
cells -- runs end-to-end via a surrogate benchmark, and every cell records
both P6 budget counters."""
import tempfile

from evodevo.analysis.aggregate import discover_runs, load_runs
from evodevo.experiments.benchmark import BenchmarkConfig, run_benchmark
from tests.experiments.test_benchmark import _build_configs_tree


def _run_two_strategy_benchmark(configs_root, output_dir):
    _build_configs_tree(configs_root)
    bconfig = BenchmarkConfig(
        encoders=["direct", "nca", "rule_based"],
        tasks=["Walker-v0"],
        seeds=[0],
        strategies=["ea", "random_search"],
        configs_root=configs_root,
        output_dir=output_dir,
        surrogate=True,
        factors={"dev_steps": 5, "init_condition": "center_seed"},
    )
    return run_benchmark(bconfig, dry_run=False, allow_dirty=True)


def test_six_cells_produced_with_both_counters():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        manifest = _run_two_strategy_benchmark(configs_root, runs_root)

        cells = {(e["encoder"], e["strategy"]) for e in manifest["run_matrix"]}
        assert cells == {
            ("direct", "ea"), ("direct", "random_search"),
            ("nca", "ea"), ("nca", "random_search"),
            ("rule_based", "ea"), ("rule_based", "random_search"),
        }
        assert len(manifest["completed_this_invocation"]) == 6

        df = load_runs(discover_runs(runs_root))
        assert set(df["encoder"].unique()) == {"direct", "nca", "rule_based"}
        assert {"evals_spent", "evals_valid_cumulative"} <= set(df.columns)
        assert (df["evals_valid_cumulative"] <= df["evals_spent"]).all()


def test_random_search_output_dir_disambiguated_from_ea():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        manifest = _run_two_strategy_benchmark(configs_root, runs_root)
        by_cell = {(e["encoder"], e["strategy"]): e["output_dir"] for e in manifest["run_matrix"]}
        assert by_cell[("direct", "ea")] != by_cell[("direct", "random_search")]
        assert "random_search" in by_cell[("direct", "random_search")]
        assert "random_search" not in by_cell[("direct", "ea")]


def test_run_meta_records_strategy():
    import json
    import os

    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root:
        _run_two_strategy_benchmark(configs_root, runs_root)
        for run_dir in discover_runs(runs_root):
            with open(os.path.join(run_dir, "run_meta.json")) as f:
                meta = json.load(f)
            assert meta["config"]["strategy"] in {"ea", "random_search"}
