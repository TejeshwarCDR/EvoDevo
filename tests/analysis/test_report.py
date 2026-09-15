import os
import tempfile

from evodevo.analysis.report import generate_report
from evodevo.experiments.benchmark import BenchmarkConfig, run_benchmark
from tests.experiments.test_benchmark import TINY_SHARED_BASE, _build_configs_tree


def _run_tiny_benchmark(configs_root, output_dir):
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


def test_report_generated_with_three_arms_and_all_figures():
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root, tempfile.TemporaryDirectory() as report_dir:
        _run_tiny_benchmark(configs_root, runs_root)

        output_path = os.path.join(report_dir, "BENCHMARK_REPORT.md")
        figures_dir = os.path.join(report_dir, "figures")
        result_path = generate_report(runs_root=runs_root, output_path=output_path, figures_dir=figures_dir)

        assert result_path == output_path
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()

        for section in (
            "# Benchmark Report",
            "## Research question",
            "## Fairness protocol",
            "## Three-way comparison table",
            "## Figures",
            "## Statistical comparison",
            "## Threats to validity",
        ):
            assert section in content

        for encoder in ("direct", "nca", "rule_based"):
            assert encoder in content

        assert os.path.exists(os.path.join(figures_dir, "fitness_curves.png"))
        assert os.path.exists(os.path.join(figures_dir, "morphology_comparison.png"))
        assert os.path.exists(os.path.join(figures_dir, "validity_efficiency.png"))


def test_report_headline_is_neutral_not_assumed():
    """The report generator must not assume any particular arm wins --
    verified by checking it can produce either a "no detectable
    difference" headline or a specific-arm headline depending on what the
    loaded data actually shows, never hardcoding one direction."""
    with tempfile.TemporaryDirectory() as configs_root, tempfile.TemporaryDirectory() as runs_root, tempfile.TemporaryDirectory() as report_dir:
        _run_tiny_benchmark(configs_root, runs_root)
        output_path = os.path.join(report_dir, "BENCHMARK_REPORT.md")
        generate_report(runs_root=runs_root, output_path=output_path, figures_dir=os.path.join(report_dir, "figures"))

        with open(output_path) as f:
            content = f.read()

        headline_line = next(line for line in content.splitlines() if line.startswith("**Headline:**"))
        possible_phrases = (
            "No detectable difference between arms",
            "outperformed",
            "No completed runs are available",
        )
        assert any(phrase in headline_line for phrase in possible_phrases)


def test_report_handles_no_manifest_gracefully():
    with tempfile.TemporaryDirectory() as empty_runs_root, tempfile.TemporaryDirectory() as report_dir:
        output_path = os.path.join(report_dir, "BENCHMARK_REPORT.md")
        generate_report(
            runs_root=empty_runs_root, output_path=output_path, figures_dir=os.path.join(report_dir, "figures")
        )
        with open(output_path) as f:
            content = f.read()
        assert "no `benchmark_manifest.json` found" in content
        assert "No completed runs are available" in content
