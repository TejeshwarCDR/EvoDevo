"""The DEFAULT fitness-comparison figure for this project: two panels,
side by side --

  (a) best-so-far fitness vs. evaluations SPENT (the budget-governing counter)
  (b) best-so-far fitness vs. evaluations VALID (real, decode-and-validate-
      passing search attempts)

-- plus an automatically generated interpretation paragraph. A single-panel
fitness curve is NOT acceptable output from this project: with validity
priors differing by an order of magnitude across arms, panel (a) alone
conflates "the encoding searches better" with "the encoding wastes fewer
evaluations on disconnected junk" (a property of its representational
prior, not of its search dynamics). Panel (b) is the correction.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fraction-of-final-best-fitness convergence threshold used to decide
# whether two arms' panel-(b) curves have "converged" (see
# generate_interpretation()) -- deliberately coarse, since this is a
# qualitative call ("did the prior-driven gap mostly close"), not a
# statistical test.
CONVERGENCE_RELATIVE_TOLERANCE = 0.15


def _best_so_far_by_seed(df: pd.DataFrame, x_field: str) -> pd.DataFrame:
    """Long-form frame: encoder, seed, <x_field>, best_so_far -- best_so_far
    is the SAME running value used for both panels (evodevo.evolution.ea
    already computes it once, keyed against evals_spent by construction
    since it's written as each evaluation happens); only the x-axis field
    used to plot it differs between the two panels."""
    out = df.sort_values(["encoder", "seed", "eval_index"]).copy()
    if "best_fitness_so_far" in out.columns:
        out["best_so_far"] = out.groupby(["encoder", "seed"])["best_fitness_so_far"].cummax()
    else:
        # Backward compatibility with run logs predating this prompt (no
        # evals_valid_cumulative/best_fitness_so_far fields): panel (b)
        # cannot be built from those logs; callers should expect a KeyError
        # instead of a silently wrong plot.
        out["best_so_far"] = out.groupby(["encoder", "seed"])["fitness"].cummax()
    return out[["encoder", "seed", x_field, "best_so_far"]]


def _plot_panel(df: pd.DataFrame, x_field: str, ax, xlabel: str) -> None:
    """Median line + IQR (25th/75th percentile) band across seeds -- the
    locked convergence-curve convention (see docs/analysis-protocol.md and
    evodevo.analysis.stats_protocol.median_iqr_curve, the same summary this
    panel now delegates to). NOT mean +/- SD and NOT mean/min-max: median/IQR
    is robust to the occasional outlier seed (e.g. one seed that never finds
    a valid genome) dominating the displayed band, which either of those
    alternatives would not be."""
    from evodevo.analysis.stats_protocol import median_iqr_curve

    curves = _best_so_far_by_seed(df, x_field)
    for encoder, group in curves.groupby("encoder"):
        pivot = group.pivot_table(index=x_field, columns="seed", values="best_so_far", aggfunc="last")
        pivot = pivot.ffill()
        median, q25, q75 = median_iqr_curve(pivot.to_numpy())
        ax.plot(pivot.index, median, label=encoder)
        ax.fill_between(pivot.index, q25, q75, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("best-so-far fitness (median, IQR band)")
    ax.legend()


def plot_two_panel_comparison(df: pd.DataFrame, axes=None):
    """The default fitness-comparison figure: (axes[0]) best-so-far fitness
    vs. evals_spent, (axes[1]) best-so-far fitness vs. evals_valid_cumulative.
    Returns (fig, axes)."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig = axes[0].figure

    if df.empty:
        for ax in axes:
            ax.set_axis_off()
        return fig, axes

    _plot_panel(df, "evals_spent", axes[0], "evaluations spent (budget-governing)")
    axes[0].set_title("(a) vs. evaluations spent")

    _plot_panel(df, "evals_valid_cumulative", axes[1], "evaluations valid (real search attempts)")
    axes[1].set_title("(b) vs. evaluations valid")

    return fig, axes


# ---------------------------------------------------------------------------
# Automatic interpretation
# ---------------------------------------------------------------------------


def _final_best_by_encoder(df: pd.DataFrame, x_field: str) -> dict:
    """{encoder: mean-across-seeds final best_so_far value}, using x_field
    as the within-seed ordering (evals_spent or evals_valid_cumulative)."""
    curves = _best_so_far_by_seed(df, x_field)
    result = {}
    for encoder, group in curves.groupby("encoder"):
        finals = [g.sort_values(x_field)["best_so_far"].iloc[-1] for _seed, g in group.groupby("seed")]
        result[encoder] = float(np.mean(finals))
    return result


def generate_interpretation(df: pd.DataFrame) -> str:
    """Automatically derives, from the data alone (never hand-written per
    figure), which of the three interpretations applies:

      - leads on (a) but converges on (b): the advantage is attributable to
        the representational prior (validity rate), not search dynamics.
      - leads on both: the advantage survives correction for the prior.
      - leads on (b) but not (a): searches better but wastes its budget.
    """
    if df.empty or "evals_valid_cumulative" not in df.columns:
        return "No data (or pre-budget-accounting run logs) available to generate an interpretation."

    finals_spent = _final_best_by_encoder(df, "evals_spent")
    finals_valid = _final_best_by_encoder(df, "evals_valid_cumulative")
    if len(finals_spent) < 2:
        return "Only one arm present -- no between-arm comparison to interpret."

    leader_spent = max(finals_spent, key=finals_spent.get)
    leader_valid = max(finals_valid, key=finals_valid.get)

    spent_values = list(finals_spent.values())
    valid_values = list(finals_valid.values())
    spent_spread = (max(spent_values) - min(spent_values)) / max(abs(max(spent_values)), 1e-9)
    valid_spread = (max(valid_values) - min(valid_values)) / max(abs(max(valid_values)), 1e-9)
    converged_on_valid = valid_spread <= CONVERGENCE_RELATIVE_TOLERANCE

    spent_str = ", ".join(f"{enc}={v:.4f}" for enc, v in sorted(finals_spent.items()))
    valid_str = ", ".join(f"{enc}={v:.4f}" for enc, v in sorted(finals_valid.items()))

    if converged_on_valid:
        # Whatever gap panel (a) shows, panel (b) shows the arms converging
        # once each is only credited for evaluations that reached a real
        # candidate -- the gap is attributable to the representational
        # prior (validity rate), not to search dynamics.
        return (
            f"**{leader_spent}** leads on evals_spent (final best-so-far: {spent_str}), but the arms "
            f"converge once re-indexed against evals_valid (final best-so-far: {valid_str}, relative "
            f"spread {valid_spread:.1%} <= {CONVERGENCE_RELATIVE_TOLERANCE:.0%} convergence threshold) "
            f"-- **{leader_spent}'s advantage is attributable to its representational prior (fewer "
            "evaluations wasted on invalid genomes), not to better search dynamics once every arm is "
            "credited only for evaluations that reached a real candidate.**"
        )

    if leader_spent == leader_valid:
        # Same leader on both panels, and panel (b) did NOT converge -- the
        # advantage survives correction for the prior.
        return (
            f"**{leader_spent}** leads on both panels (final best-so-far vs. evals_spent: {spent_str}; "
            f"vs. evals_valid: {valid_str}) -- its advantage SURVIVES correction for the validity prior: "
            "this is a real search-dynamics advantage, not merely a byproduct of wasting fewer evaluations "
            "on invalid genomes."
        )

    # Different leaders on the two panels, and panel (b) did not converge:
    # the evals_valid leader searches better on a per-real-attempt basis,
    # but that doesn't show up in per-budget-slot terms -- it wastes a
    # larger share of its fixed budget on invalid genomes than the
    # evals_spent leader.
    return (
        f"**{leader_valid}** leads on evals_valid (final best-so-far vs. evals_valid: {valid_str}) but NOT "
        f"on evals_spent (vs. evals_spent: {spent_str}, led by **{leader_spent}**) -- {leader_valid} "
        "searches better once given a real candidate, but wastes a larger share of its fixed budget on "
        "invalid genomes than the evals_spent leader, so its per-real-attempt search advantage does not "
        "translate into a per-budget-slot advantage."
    )


def plot_two_panel_comparison_with_interpretation(df: pd.DataFrame, axes=None):
    """Convenience wrapper: (fig, axes, interpretation_text)."""
    fig, axes = plot_two_panel_comparison(df, axes=axes)
    return fig, axes, generate_interpretation(df)
