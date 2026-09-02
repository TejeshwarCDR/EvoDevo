"""Generates results/BENCHMARK_REPORT.md: the N-way benchmark report
combining aggregate.py's summary table, statistics.py's pairwise
comparisons, and plots.py's figures.

Reports the result direction NEUTRALLY: this module makes NO assumption
that a developmental encoding (NCA/Rule-Based) beats the Direct baseline,
or vice versa. The headline sentence is derived from whichever the loaded
data actually shows -- including "no detectable difference between arms"
when that is what the statistics say.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from evodevo.analysis.aggregate import discover_runs, load_runs, per_seed_finals, three_way_table
from evodevo.analysis.plots import (
    plot_fitness_curves,
    plot_morphology_comparison,
    plot_validity_and_efficiency,
)
from evodevo.analysis.statistics import three_way_comparison
from evodevo.experiments.benchmark import MANIFEST_NAME

DEFAULT_RUNS_ROOT = "results/runs"
DEFAULT_OUTPUT_PATH = "results/BENCHMARK_REPORT.md"
DEFAULT_FIGURES_DIR = "results/figures"

# From results/nca_sanity.md / docs/decisions.md (N=2000, seed=0): NCA's
# known developmental-instability limitation, cited verbatim in the threats
# section regardless of which arms are present in a given benchmark run.
NCA_NEVER_SETTLED_FRACTION = 0.6835
NCA_STATE_MAGNITUDE_MEAN = 40.7
NCA_INIT_SCALE = 0.1


def _load_manifest(runs_root: str) -> dict:
    path = os.path.join(runs_root, MANIFEST_NAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal, dependency-free (no `tabulate`) markdown table renderer."""
    if df.empty:
        return "*(empty)*"

    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def _headline(three_way_df: pd.DataFrame, comparisons: list) -> str:
    if three_way_df.empty:
        return "No completed runs are available to report on."

    best_row = three_way_df.loc[three_way_df["final_best_fitness_mean"].idxmax()]
    best_encoder = str(best_row["encoder"])

    detectable = [c for c in comparisons if not c.overlapping_ci_zero]
    if not detectable:
        return (
            "No detectable difference between arms: across every pairwise comparison, "
            "the bootstrap confidence interval for the mean-fitness difference includes zero."
        )

    other_encoders = sorted(e for e in three_way_df["encoder"] if e != best_encoder)
    if best_encoder == "direct":
        return (
            f"The direct baseline outperformed the developmental encoding(s) "
            f"({', '.join(other_encoders)}) on final fitness, with a detectable "
            "difference in at least one pairwise comparison."
        )
    return (
        f"The {best_encoder} encoding outperformed the other arm(s) "
        f"({', '.join(other_encoders)}) on final fitness, with a detectable "
        "difference in at least one pairwise comparison."
    )


def _write_figures(df: pd.DataFrame, table: pd.DataFrame, figures_dir: str) -> dict:
    os.makedirs(figures_dir, exist_ok=True)
    paths = {}

    import matplotlib.pyplot as plt  # Agg backend already set by evodevo.analysis.plots

    if not df.empty:
        fig, ax = plt.subplots()
        plot_fitness_curves(df, ax=ax)
        ax.set_title("Fitness vs. evaluations (per-seed min/max band)")
        fig.tight_layout()
        path = os.path.join(figures_dir, "fitness_curves.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths["fitness_curves"] = path

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_morphology_comparison(df, axes=axes)
        fig.tight_layout()
        path = os.path.join(figures_dir, "morphology_comparison.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths["morphology_comparison"] = path

    if not table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        plot_validity_and_efficiency(table, axes=axes)
        fig.tight_layout()
        path = os.path.join(figures_dir, "validity_efficiency.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths["validity_efficiency"] = path

    return paths


def generate_report(
    runs_root: str = DEFAULT_RUNS_ROOT,
    output_path: str = DEFAULT_OUTPUT_PATH,
    figures_dir: str = DEFAULT_FIGURES_DIR,
) -> str:
    manifest = _load_manifest(runs_root)
    df = load_runs(discover_runs(runs_root))
    table = three_way_table(df)

    finals_by_encoder = per_seed_finals(df)
    comparisons = three_way_comparison(finals_by_encoder) if len(finals_by_encoder) >= 2 else []

    figure_paths = _write_figures(df, table, figures_dir)
    output_dir_for_relpath = os.path.dirname(output_path) or "."

    headline = _headline(table, comparisons)

    lines = []
    lines.append("# Benchmark Report: Direct vs. NCA vs. Rule-Based Morphology Encodings")
    lines.append("")
    lines.append(f"**Headline:** {headline}")
    lines.append("")

    lines.append("## Research question")
    lines.append("")
    lines.append(
        "Does the choice of genome-to-morphology encoding (a direct per-voxel encoding, "
        "a Neural Cellular Automaton, or a symbolic developmental rule system) change "
        "evolutionary search efficiency or final locomotion fitness on a fixed 5x5 voxel "
        "soft robot, when every other part of the pipeline (task, PPO budget, population "
        "size, evaluation budget, validation policy, invalid-genome accounting) is held "
        "identical across arms? The novelty claimed here is the controlled, "
        "identical-conditions three-way comparison itself -- not the invention of any "
        "one encoding (NCA-for-soft-robots and symbolic developmental rules both predate "
        "this project)."
    )
    lines.append("")

    lines.append("## Fairness protocol (as actually resolved from the benchmark manifest)")
    lines.append("")
    if manifest:
        lines.append(f"- Git commit: `{manifest.get('git_commit', 'unknown')}`")
        lines.append(f"- Surrogate mode: `{manifest.get('surrogate')}`")
        lines.append("- Shared config (verified identical across every arm before any run started):")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for key, value in sorted(manifest.get("shared_config", {}).items()):
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("- Per-encoder configuration (`describe()`):")
        lines.append("")
        for encoder, describe in sorted(manifest.get("encoders", {}).items()):
            lines.append(f"  - **{encoder}**: `{describe}`")
        lines.append("")
        lines.append(f"- Library versions: `{manifest.get('library_versions', {})}`")
    else:
        lines.append(
            f"*(no `{MANIFEST_NAME}` found under `{runs_root}` -- run "
            "`python -m evodevo.experiments.benchmark --config ...` first)*"
        )
    lines.append("")

    lines.append("## Three-way comparison table")
    lines.append("")
    lines.append(_df_to_markdown(table))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    if "fitness_curves" in figure_paths:
        lines.append(f"![fitness curves]({os.path.relpath(figure_paths['fitness_curves'], output_dir_for_relpath)})")
        lines.append("")
    if "morphology_comparison" in figure_paths:
        lines.append(
            f"![morphology comparison]({os.path.relpath(figure_paths['morphology_comparison'], output_dir_for_relpath)})"
        )
        lines.append("")
    if "validity_efficiency" in figure_paths:
        lines.append(
            f"![validity and search efficiency]({os.path.relpath(figure_paths['validity_efficiency'], output_dir_for_relpath)})"
        )
        lines.append("")
    if not figure_paths:
        lines.append("*(no figures -- no completed runs yet)*")
        lines.append("")

    lines.append("## Statistical comparison")
    lines.append("")
    lines.append(
        "With only a handful of seeds per arm, no statistical test here can support a "
        "strong claim -- see `evodevo/analysis/statistics.py`'s module docstring. "
        "Overlapping confidence intervals are reported as **no detectable difference**, "
        "never as evidence of equivalence."
    )
    lines.append("")
    if comparisons:
        lines.append(
            "| Arm A | Arm B | Metric | Mean diff (A-B) | 95% CI | Cliff's delta | "
            "Mann-Whitney U | p (approx) | Verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in comparisons:
            lines.append(
                f"| {c.encoder_a} | {c.encoder_b} | {c.metric} | {c.mean_difference:.4f} "
                f"| [{c.ci_low:.4f}, {c.ci_high:.4f}] | {c.cliffs_delta:.3f} | {c.mann_whitney_u:.1f} "
                f"| {c.mann_whitney_p_approx:.4f} | {c.verdict} |"
            )
    else:
        lines.append("*(fewer than two arms have completed runs -- nothing to compare yet)*")
    lines.append("")

    lines.append("## Threats to validity")
    lines.append("")
    n_seeds = int(table["n_seeds"].max()) if not table.empty else 0
    lines += [
        f"- **Seed count**: {n_seeds} seed(s) per arm in this run. This is far too few to "
        "support strong statistical claims; treat every comparison above as indicative, "
        "not conclusive.",
        "- **PPO budget**: this comparison's `ppo_timesteps` is far below EvoGym's own "
        ">1e6-timestep recommendation for a serious PPO run on a soft-robot locomotion "
        "task; fitness numbers here reflect a heavily truncated training budget, not "
        "converged policies.",
        "- **Single task family**: this run covers only the task(s) listed in the fairness "
        "protocol section above; results may not generalize to other EvoGym environments.",
        "- **Per-arm mutation-strength tuning asymmetry**: mutation rate/sigma cannot be "
        "equalized across a categorical genome (direct), a continuous weight vector "
        "(NCA), and a variable-length rule list (Rule-Based); each arm's mutation "
        "strength was set independently, not via a shared tuning procedure, which is a "
        "potential confound in any efficiency comparison.",
        f"- **NCA's known non-settling behaviour**: at default hyperparameters, "
        f"{NCA_NEVER_SETTLED_FRACTION:.2%} of random NCA genomes are still changing state "
        "on the final developmental step (never observably settled within `dev_steps`), "
        f"and residual-state magnitude at t=T averages ~{NCA_STATE_MAGNITUDE_MEAN:.1f} "
        f"against an initial-state scale of {NCA_INIT_SCALE} -- the residual update is not "
        "bounded/self-stabilizing (see `results/nca_sanity.md`).",
        "- **Grid size**: a 5x5 voxel grid may be too small for either developmental "
        "encoding (NCA or Rule-Based) to express any advantage over direct per-voxel "
        "encoding -- developmental/generative representations are generally expected to "
        "pay off at larger genome/phenotype sizes where direct encoding's search space "
        "becomes intractable, which this project's fixed 5x5 grid does not test.",
    ]
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_RUNS_ROOT, help="results/runs root to load completed runs from")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="path to write the report markdown to")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR, help="directory to write figures into")
    args = parser.parse_args(argv)

    path = generate_report(args.root, args.output, args.figures_dir)
    print(f"report written to {path}")


if __name__ == "__main__":
    main()
