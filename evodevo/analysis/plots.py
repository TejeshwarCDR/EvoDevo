"""Figures for the encoding comparison. Matplotlib only, no display side effects."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


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
