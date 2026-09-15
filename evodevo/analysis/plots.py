"""Figures for the encoding comparison. Matplotlib only, no display side effects."""
from __future__ import annotations

import json
import math
import os

import matplotlib

# Must be set before `matplotlib.pyplot` is imported anywhere in the process.
# This module (not nca_sanity.py's import-time side effect) is the
# authoritative place this is enforced for the analysis package, so plotting
# works headlessly regardless of import order.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_INSPECTION_DIR = "results/inspection"


def _best_so_far_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form frame with columns encoder, seed, eval_index, best_so_far."""
    out = df.sort_values(["encoder", "seed", "eval_index"]).copy()
    out["best_so_far"] = out.groupby(["encoder", "seed"])["fitness"].cummax()
    return out[["encoder", "seed", "eval_index", "best_so_far"]]


def plot_fitness_curves(df: pd.DataFrame, ax=None):
    """Best-so-far fitness vs. evaluations, one line per encoder, shaded
    min/max band across seeds."""
    if ax is None:
        _fig, ax = plt.subplots()

    curves = _best_so_far_by_seed(df)

    for encoder, group in curves.groupby("encoder"):
        pivot = group.pivot(index="eval_index", columns="seed", values="best_so_far")
        pivot = pivot.ffill()
        mean = pivot.mean(axis=1)
        lo = pivot.min(axis=1)
        hi = pivot.max(axis=1)

        ax.plot(mean.index, mean.values, label=encoder)
        ax.fill_between(lo.index, lo.values, hi.values, alpha=0.2)

    ax.set_xlabel("evaluations")
    ax.set_ylabel("best-so-far fitness")
    ax.legend()
    return ax


MORPHOLOGY_METRICS = ("actuator_fraction", "occupied_count", "horizontal_symmetry_score")


def _best_body_metrics_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (encoder, seed): the record with the highest fitness."""
    idx = df.groupby(["encoder", "seed"])["fitness"].idxmax()
    return df.loc[idx]


def plot_morphology_comparison(df: pd.DataFrame, metrics=MORPHOLOGY_METRICS, axes=None):
    """Box-plot comparison of the best body's morphology metrics per encoder,
    one subplot per metric, one box per encoder (across seeds)."""
    best = _best_body_metrics_by_seed(df)
    encoders = sorted(best["encoder"].unique())

    if axes is None:
        _fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = [best.loc[best["encoder"] == enc, metric].values for enc in encoders]
        ax.boxplot(data, tick_labels=encoders)
        ax.set_title(metric)

    return axes


def load_prior_validity_rates(encoders, inspection_dir: str = DEFAULT_INSPECTION_DIR) -> dict:
    """Read each encoder's true pre-run random-genome validity prior from its
    PPO-free inspection output (`{inspection_dir}/{encoder}/{encoder}_sanity.json`,
    written by evodevo.experiments.nca_sanity / inspect_encoder.py), keyed by
    `data[encoder]["valid_rate"]`. Encoders with no inspection file on disk map
    to NaN rather than raising, so callers can still render a figure (e.g. in
    tests that don't run the inspection step)."""
    rates = {}
    for encoder in encoders:
        path = os.path.join(inspection_dir, encoder, f"{encoder}_sanity.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            rates[encoder] = float(data[encoder]["valid_rate"])
        except (FileNotFoundError, KeyError, ValueError):
            rates[encoder] = math.nan
    return rates


def plot_validity_and_efficiency(three_way_df: pd.DataFrame, prior_validity_by_encoder: dict, axes=None):
    """Three-panel bar chart: the TRUE pre-run random-genome validity prior
    per encoder (from PPO-free inspection output, via
    `load_prior_validity_rates()` -- NOT derived from any evolutionary run),
    the realized in-run validity rate per encoder (from
    evodevo.analysis.aggregate.three_way_table()'s `validity_rate` column --
    the fraction of genomes that validated DURING evolution, which is
    typically much higher than the random-genome prior because search finds
    valid regions of genome space), and evaluations needed to reach 50% of
    the best arm's final fitness per encoder (lower = more sample-efficient
    search)."""
    if axes is None:
        _fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    encoders = three_way_df["encoder"].tolist()

    axes[0].bar(encoders, [prior_validity_by_encoder.get(e, math.nan) for e in encoders])
    axes[0].set_ylabel("validity rate")
    axes[0].set_title("Random-genome validity prior (pre-run)")
    axes[0].set_ylim(0, 1)

    axes[1].bar(encoders, three_way_df["validity_rate"])
    axes[1].set_ylabel("validity rate")
    axes[1].set_title("Realized validity rate (during evolution)")
    axes[1].set_ylim(0, 1)

    axes[2].bar(encoders, three_way_df["evaluations_to_50pct_of_best_arm_mean"])
    axes[2].set_ylabel("evaluations to reach 50% of best arm's final fitness")
    axes[2].set_title("Search efficiency")

    return axes
