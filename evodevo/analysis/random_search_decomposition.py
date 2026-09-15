"""Decomposes each encoding's final performance into what its
representational PRIOR gives for free (random_search) and what the
evolutionary algorithm's SEARCH adds on top (ea minus random_search) --
under BOTH P6 budget counters (evals_spent, evals_valid_cumulative), so an
EA-vs-random advantage attributable purely to the prior (fewer wasted
evaluations) can be told apart from a genuine search-dynamics advantage.

These are different claims and must never be reported as one number:
  - what random search achieves = what the encoding's prior gives for free
  - what the EA adds on top       = what SEARCH contributes

GUARD CLAUSE: if an encoding's EA-minus-random advantage does not exceed
the seed-to-seed variance (approximated here as the larger of the two
strategies' across-seed standard deviation of final best-so-far), that
encoding's search-efficiency numbers are NOT interpretable and must be
flagged in bold, not quietly reported alongside interpretable ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evodevo.analysis.budget_curves import _best_so_far_by_seed

X_FIELDS = {"evals_spent": "evals_spent", "evals_valid": "evals_valid_cumulative"}


def _finals_per_seed(df: pd.DataFrame, encoder: str, x_field: str) -> list:
    subset = df[df["encoder"] == encoder]
    if subset.empty:
        return []
    curves = _best_so_far_by_seed(subset, x_field)
    finals = []
    for _seed, g in curves.groupby("seed"):
        g = g.sort_values(x_field)
        if len(g):
            finals.append(float(g["best_so_far"].iloc[-1]))
    return finals


def compute_decomposition(df: pd.DataFrame) -> dict:
    """Returns {encoder: {counter_label: {random_mean, random_std, ea_mean,
    ea_std, advantage, noise_floor, interpretable, n_seeds_ea,
    n_seeds_random}}}. Only encoders with BOTH strategies present in `df`
    are included."""
    if df.empty or "strategy" not in df.columns:
        return {}

    result = {}
    for encoder in sorted(df["encoder"].unique()):
        ea_df = df[(df["encoder"] == encoder) & (df["strategy"] == "ea")]
        rs_df = df[(df["encoder"] == encoder) & (df["strategy"] == "random_search")]
        if ea_df.empty or rs_df.empty:
            continue

        per_counter = {}
        for counter_label, x_field in X_FIELDS.items():
            ea_finals = _finals_per_seed(ea_df, encoder, x_field)
            rs_finals = _finals_per_seed(rs_df, encoder, x_field)
            if not ea_finals or not rs_finals:
                continue

            ea_mean, ea_std = float(np.mean(ea_finals)), float(np.std(ea_finals))
            rs_mean, rs_std = float(np.mean(rs_finals)), float(np.std(rs_finals))
            advantage = ea_mean - rs_mean
            # The seed-to-seed variance guard: the noise floor is the LARGER
            # of the two strategies' across-seed standard deviations (a
            # single-strategy std with only 1 seed is 0 -- in that
            # degenerate case the guard cannot protect against noise it
            # never measured, which is itself worth flagging).
            noise_floor = max(ea_std, rs_std)
            interpretable = advantage > noise_floor

            per_counter[counter_label] = {
                "random_mean": rs_mean,
                "random_std": rs_std,
                "ea_mean": ea_mean,
                "ea_std": ea_std,
                "advantage": advantage,
                "noise_floor": noise_floor,
                "interpretable": interpretable,
                "n_seeds_ea": len(ea_finals),
                "n_seeds_random": len(rs_finals),
            }

        if per_counter:
            result[encoder] = per_counter

    return result


def _fmt(value, digits=4):
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def generate_decomposition_report(df: pd.DataFrame) -> str:
    """Markdown: the framing paragraph, a per-encoder/per-counter table, and
    the guard-clause statement for any encoding whose EA-minus-random
    advantage doesn't clear the seed-to-seed noise floor."""
    decomposition = compute_decomposition(df)
    if not decomposition:
        return (
            "*(no random_search control data available -- run the benchmark with "
            "`strategies: [ea, random_search]` to enable this decomposition)*"
        )

    lines = []
    lines.append(
        "**Decomposition framing**: `random_search` (no selection, no mutation, no population -- fresh "
        "i.i.d. samples from the encoder's own genome distribution) measures what the encoding's "
        "representational PRIOR achieves for free. `ea - random_search` measures what SEARCH (selection "
        "+ mutation) adds on top. These are different claims: a large `random_search` result with a small "
        "`ea - random_search` gap means the encoding's prior is doing the work, not the algorithm."
    )
    lines.append("")
    lines.append(
        "| encoder | counter | random_search (mean +/- std) | ea (mean +/- std) | EA-minus-random advantage | "
        "seed-to-seed noise floor | interpretable? |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    not_interpretable = []
    for encoder, per_counter in decomposition.items():
        for counter_label, stats in per_counter.items():
            interpretable = stats["interpretable"]
            lines.append(
                f"| {encoder} | {counter_label} | {_fmt(stats['random_mean'])} +/- {_fmt(stats['random_std'])} | "
                f"{_fmt(stats['ea_mean'])} +/- {_fmt(stats['ea_std'])} | {_fmt(stats['advantage'])} | "
                f"{_fmt(stats['noise_floor'])} | {'yes' if interpretable else '**NO**'} |"
            )
            if not interpretable:
                not_interpretable.append((encoder, counter_label, stats))

    lines.append("")

    if not_interpretable:
        for encoder, counter_label, stats in not_interpretable:
            lines.append(
                f"**{encoder}'s search-efficiency result under `{counter_label}` is NOT interpretable**: "
                f"its EA-minus-random advantage ({_fmt(stats['advantage'])}) does not exceed the "
                f"seed-to-seed noise floor ({_fmt(stats['noise_floor'])}, from {stats['n_seeds_ea']} EA "
                f"seed(s) / {stats['n_seeds_random']} random_search seed(s)) -- do not report this "
                "encoding's search-efficiency claim under this counter alongside interpretable ones "
                "without this caveat."
            )
        lines.append("")

    return "\n".join(lines)
