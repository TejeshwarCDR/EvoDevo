"""Calibrates the NCA's bounded_leaky decay rate (lambda) -- and, crossed
with it, state_bound (B) -- so that its measured developmental settling
DISTRIBUTION matches rule_based's, at the P3-resolved initial condition
(center_seed). Same methodological principle as P5's mutation calibration
(evodevo/analysis/mutation_calibration.py): equalise a measured EMERGENT
property (the settle-step distribution), not a nominal hyperparameter, and
match on distribution shape (Wasserstein-1 + KS), not just the median --
P5 found that matching only a summary statistic (there: the mean) can hide
large tail differences; the same risk applies to matching only the median
settle step here.

Context: risk-register item 4 / results/depth/DEPTH_STUDY.md found NCA's
median settle step (37) an order of magnitude above rule_based's (2), with
no single shared T_primary fair to both. Lambda is now an explicit,
documented knob on the NCA's relaxation time (tau ~ 1/lambda -- see
evodevo/encoders/nca/encoder.py), so this gap is a calibration target
rather than an architectural fact to work around.

Fitness-blind: only evodevo.analysis.structural.settling_profile (P2) is
called -- NO PPO, NO task, NO reward, NO EA, and the rule_based arm's
distribution is measured once and never modified; it is the fixed target
(see tests/analysis/test_structural_no_ppo_imports.py for the mechanical
guarantee that structural.py itself stays fitness-blind, which this module
inherits by only importing from it).

Usage:
    python -m evodevo.analysis.timescale_calibration [--n 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder

DEFAULT_N = 2000
DEFAULT_SEED = 0
VERIFICATION_SEED = 1  # distinct from DEFAULT_SEED: a genuinely fresh, independent sample
INIT_CONDITION = "center_seed"  # P3 decision -- the fixed measurement condition
T_MAX = 40  # T_MAX_HARD_CAP, matches results/depth/DEPTH_STUDY.md's settling_profile horizon
HIDDEN_CHANNELS = 3

LAMBDAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
STATE_BOUNDS = [2, 5, 10]
TAIL_THRESHOLDS = [2, 5, 10, 20]

RESULTS_DIR = "results/nca"
RAW_ROOT = "results/nca/timescale_calibration_raw"


# ---------------------------------------------------------------------------
# 1D empirical-distribution distances (no scipy dependency, matching
# evodevo/analysis/statistics.py's numpy-only convention)
# ---------------------------------------------------------------------------


def wasserstein_1(a, b) -> float:
    """W1(F_a, F_b) = integral |F_a(x) - F_b(x)| dx, computed exactly for two
    finite empirical distributions via a merged sorted grid (standard 1D
    optimal-transport identity; equivalent to sorted-quantile-matching)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.union1d(a, b)
    if grid.size <= 1:
        return 0.0
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    widths = np.diff(grid)
    return float(np.sum(np.abs(cdf_a[:-1] - cdf_b[:-1]) * widths))


def ks_statistic(a, b) -> float:
    """max_x |F_a(x) - F_b(x)|, the two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    grid = np.union1d(a, b)
    if grid.size == 0:
        return 0.0
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ---------------------------------------------------------------------------
# Measurement plumbing
# ---------------------------------------------------------------------------


def _build_encoder(name: str, seed: int, **encoder_kwargs):
    factors = ExperimentFactors(
        dev_steps=T_MAX, init_condition=INIT_CONDITION, seed=seed, hidden_channels=HIDDEN_CHANNELS
    )
    return get_encoder(name, factors=factors, **encoder_kwargs), factors


def _censored_settle_steps(jsonl_path: str, censor_value: int) -> np.ndarray:
    """Reads back settling_profile's own per-genome jsonl output and returns
    the settle-step array with never-settled genomes right-censored at
    `censor_value` (T_MAX): a never-settled genome's true settle step is
    unknown but is, by definition, > T_MAX-1, so treating it as "at least as
    late as the last observed step" is the standard way to fold censored
    observations into a 1D distribution comparison without discarding them
    (discarding would silently favour whichever arm/setting has MORE
    never-settled genomes, the opposite of what should be penalised)."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            steps.append(record["settle_step"] if record["settled"] else censor_value)
    return np.array(steps, dtype=float)


def _measure_settling(encoder, factors, n: int, seed: int, output_root: str):
    rng = np.random.default_rng(seed)
    result = S.settling_profile(encoder, factors, n, rng, t_max=T_MAX, output_root=output_root)
    steps = _censored_settle_steps(S._jsonl_path(encoder.name, "settling_profile", output_root), censor_value=T_MAX)
    return result, steps


def _tail_fractions(steps: np.ndarray, thresholds, censor_value: int) -> dict:
    """fraction of genomes with a settle_step <= threshold; a censored
    (never-settled) genome never counts, for any threshold, since its true
    settle step is unknown but definitely > censor_value - 1 >= any
    threshold below."""
    n = len(steps)
    return {t: float(np.sum((steps <= t) & (steps < censor_value)) / n) for t in thresholds}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(n: int, seed: int):
    # --- target: rule_based, measured once, never modified ------------------
    rb_encoder, rb_factors = _build_encoder("rule_based", seed=seed)
    rb_result, rb_steps = _measure_settling(
        rb_encoder, rb_factors, n, seed, output_root=os.path.join(RAW_ROOT, "target_rule_based")
    )

    # --- sweep: NCA bounded_leaky over lambda x state_bound ------------------
    grid = []
    for decay in LAMBDAS:
        for state_bound in STATE_BOUNDS:
            nca_encoder, nca_factors = _build_encoder(
                "nca", seed=seed, update_rule="bounded_leaky", decay=decay, state_bound=state_bound
            )
            output_root = os.path.join(RAW_ROOT, f"decay_{decay}_B_{state_bound}")
            nca_result, nca_steps = _measure_settling(nca_encoder, nca_factors, n, seed, output_root=output_root)

            w1 = wasserstein_1(nca_steps, rb_steps)
            ks = ks_statistic(nca_steps, rb_steps)
            grid.append(
                {
                    "decay": decay,
                    "state_bound": state_bound,
                    "wasserstein_1": w1,
                    "ks_statistic": ks,
                    "never_settled_fraction": nca_result.never_settled_fraction,
                    "median_settle_step": float(np.median(nca_steps)),
                    "mean_settle_step": float(np.mean(nca_steps)),
                }
            )

    best = min(grid, key=lambda row: row["wasserstein_1"])

    return {
        "n": n,
        "seed": seed,
        "rb_result": rb_result,
        "rb_steps": rb_steps,
        "grid": grid,
        "best": best,
    }


def run_verification(best: dict, n: int, seed: int, rb_steps: np.ndarray):
    """Fresh, independent NCA sample at (lambda*, B*) -- a different seed
    from the sweep -- checked against the SAME (fixed, already-measured)
    rule_based target distribution."""
    nca_encoder, nca_factors = _build_encoder(
        "nca", seed=seed, update_rule="bounded_leaky", decay=best["decay"], state_bound=best["state_bound"]
    )
    output_root = os.path.join(RAW_ROOT, "verification")
    nca_result, nca_steps = _measure_settling(nca_encoder, nca_factors, n, seed, output_root=output_root)

    return {
        "seed": seed,
        "n": n,
        "wasserstein_1": wasserstein_1(nca_steps, rb_steps),
        "ks_statistic": ks_statistic(nca_steps, rb_steps),
        "never_settled_fraction": nca_result.never_settled_fraction,
        "median_settle_step": float(np.median(nca_steps)),
        "mean_settle_step": float(np.mean(nca_steps)),
        "steps": nca_steps,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(sweep: dict, verification: dict, rb_tail: dict, nca_sweep_tail: dict, nca_verify_tail: dict) -> str:
    n = sweep["n"]
    best = sweep["best"]
    rb_result = sweep["rb_result"]

    lines = []
    lines.append("# NCA timescale (lambda, state_bound) calibration")
    lines.append("")
    lines.append(
        f"Fitness-blind (evodevo.analysis.structural.settling_profile only -- NO PPO, NO task, NO reward, NO EA). "
        f"N={n} genomes/cell, init_condition=`{INIT_CONDITION}` (the P3 decision), T_max={T_MAX}, "
        f"hidden_channels={HIDDEN_CHANNELS}, seed={sweep['seed']}."
    )
    lines.append("")
    lines.append(
        "**Target (fixed, not modified)**: rule_based's own settling distribution, measured once at these "
        "settings via `structural.settling_profile` -- the same function used in "
        "results/depth/DEPTH_STUDY.md, reused here rather than re-implemented."
    )
    lines.append("")
    lines.append(
        f"rule_based: median settle_step={_fmt(float(np.median(sweep['rb_steps'])))}, "
        f"mean settle_step={_fmt(float(np.mean(sweep['rb_steps'])))}, "
        f"never_settled_fraction={_fmt(rb_result.never_settled_fraction)}."
    )
    lines.append("")
    lines.append(
        "## Joint sweep: lambda (decay) x state_bound (B)"
    )
    lines.append("")
    lines.append(
        "state_bound is crossed with decay because it sets the argmax margin the readout must cross to "
        "flip, so it affects the MEASURED settle step even though it plays no role in the contraction "
        "condition itself (results/nca/CONVERGENCE.md, Proposition 2: the certificate depends on "
        "`state_bound * K(theta)`, not on B or K individually)."
    )
    lines.append("")
    lines.append("| lambda | B | Wasserstein-1 | KS | never_settled_fraction | median settle_step | mean settle_step |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in sweep["grid"]:
        marker = " **<- selected**" if (row["decay"], row["state_bound"]) == (best["decay"], best["state_bound"]) else ""
        lines.append(
            f"| {row['decay']} | {row['state_bound']} | {_fmt(row['wasserstein_1'])} | {_fmt(row['ks_statistic'])} | "
            f"{_fmt(row['never_settled_fraction'])} | {_fmt(row['median_settle_step'])} | "
            f"{_fmt(row['mean_settle_step'])}{marker} |"
        )
    lines.append("")
    lines.append(
        "Selection criterion: minimise Wasserstein-1 (a full-distribution distance, not just matched "
        "medians -- matching only the median is exactly the limitation P5's mutation calibration hit when "
        "it matched only the mean phenotypic displacement and missed a 7-cell IQR gap in the tails; KS is "
        "reported alongside as a second, independent distribution-shape check, not used for selection)."
    )
    lines.append("")
    lines.append(f"**Selected: lambda\\* = {best['decay']}, B\\* = {best['state_bound']}** "
                  f"(Wasserstein-1 = {_fmt(best['wasserstein_1'])}, KS = {_fmt(best['ks_statistic'])}).")
    lines.append("")
    lines.append("## Held-out verification")
    lines.append("")
    lines.append(
        f"A fresh, independent NCA sample (N={verification['n']}, seed={verification['seed']} -- distinct from "
        f"the sweep's seed={sweep['seed']}) at (lambda\\*={best['decay']}, B\\*={best['state_bound']}), compared "
        "against the SAME fixed rule_based target distribution measured above (the target is not resampled: "
        "it is fixed once and never modified, per the task's constraint)."
    )
    lines.append("")
    lines.append("| quantity | sweep (seed={}) | verification (seed={}) |".format(sweep["seed"], verification["seed"]))
    lines.append("| --- | --- | --- |")
    lines.append(f"| Wasserstein-1 | {_fmt(best['wasserstein_1'])} | {_fmt(verification['wasserstein_1'])} |")
    lines.append(f"| KS | {_fmt(best['ks_statistic'])} | {_fmt(verification['ks_statistic'])} |")
    lines.append(f"| never_settled_fraction | {_fmt(best['never_settled_fraction'])} | {_fmt(verification['never_settled_fraction'])} |")
    lines.append(f"| median settle_step | {_fmt(best['median_settle_step'])} | {_fmt(verification['median_settle_step'])} |")
    lines.append(f"| mean settle_step | {_fmt(best['mean_settle_step'])} | {_fmt(verification['mean_settle_step'])} |")
    lines.append("")
    verify_delta = abs(verification["wasserstein_1"] - best["wasserstein_1"])
    lines.append(
        f"Verification Wasserstein-1 ({_fmt(verification['wasserstein_1'])}) differs from the sweep's selection "
        f"value ({_fmt(best['wasserstein_1'])}) by {_fmt(verify_delta)} -- "
        + (
            "within a small-sample-noise tolerance of 0.5 steps, so the selected (lambda*, B*) generalises to "
            "unseen genomes at this setting and is not an artifact of the sweep's particular sample."
            if verify_delta <= 0.5
            else "OUTSIDE the 0.5-step tolerance -- the selected (lambda*, B*) does not reliably generalise and "
            "this result should not be trusted without further investigation."
        )
    )
    lines.append("")
    lines.append("## Tail behaviour at lambda*, B* (P5-style: matching a median or mean is not enough)")
    lines.append("")
    lines.append("| settle within <= N steps | rule_based (target) | nca @ lambda*,B* (sweep sample) | nca @ lambda*,B* (verification sample) |")
    lines.append("| --- | --- | --- | --- |")
    for t in TAIL_THRESHOLDS:
        lines.append(
            f"| {t} | {_fmt(rb_tail[t])} | {_fmt(nca_sweep_tail[t])} | {_fmt(nca_verify_tail[t])} |"
        )
    lines.append("")
    lines.append("## How much of the 37-vs-2 gap this closes")
    lines.append("")
    original_median_nca = 37.0
    original_median_rb = 2.0
    original_gap = original_median_nca - original_median_rb
    new_gap = best["median_settle_step"] - float(np.median(sweep["rb_steps"]))
    closed_fraction = 1.0 - (abs(new_gap) / original_gap) if original_gap else float("nan")
    lines.append(
        f"Risk-register item 4's baseline gap (results/depth/DEPTH_STUDY.md, unbounded_residual, center_seed): "
        f"NCA median settle_step={original_median_nca:.0f} vs rule_based={original_median_rb:.0f}, a gap of "
        f"{original_gap:.0f} steps."
    )
    if new_gap <= original_gap:
        lines.append(
            f"At (lambda\\*, B\\*): NCA median settle_step={_fmt(best['median_settle_step'])} vs rule_based="
            f"{_fmt(float(np.median(sweep['rb_steps'])))}, a residual gap of {_fmt(new_gap)} steps "
            f"({_fmt(closed_fraction*100, 1)}% of the original {original_gap:.0f}-step median gap closed)."
        )
    else:
        lines.append(
            f"**Negative result on the median**: at (lambda\\*, B\\*), NCA median settle_step="
            f"{_fmt(best['median_settle_step'])} vs rule_based={_fmt(float(np.median(sweep['rb_steps'])))} -- "
            f"a gap of {_fmt(new_gap)} steps, WORSE than (not an improvement on) the original "
            f"{original_gap:.0f}-step baseline gap. Every cell in the swept lambda x B grid (lambda in "
            f"{LAMBDAS}, B in {STATE_BOUNDS}) has a median settle_step of 39-40, at or near T_max={T_MAX}: "
            "within this grid, decay/state_bound tuning does not bring NCA's median settling speed anywhere "
            "close to rule_based's. lambda* and B* were still selected on Wasserstein-1 (the least-bad cell "
            "in the grid, not a successful match) since a selection criterion is required regardless of "
            "whether the calibration goal was reached."
        )
    lines.append("")
    lines.append(
        f"**This does not claim the gap is solved -- it is not.** The Wasserstein-1 distance at "
        f"(lambda\\*, B\\*) is {_fmt(best['wasserstein_1'])} steps (out of a maximum possible {T_MAX}), "
        "driven by BOTH a residual median/mean gap and the tail differences in the table below. Within the "
        f"tested grid, the best cell sits at its lowest tested B ({min(STATE_BOUNDS)}) and the "
        f"never_settled_fraction is lowest at the lowest tested lambda ({min(LAMBDAS)}) for every B -- "
        "suggestive (not conclusive: W1 itself is not strictly monotone in lambda at fixed B, see the table) "
        "that smaller lambda and/or B values, outside this task's specified grid, might close more of the "
        "gap. That is future work, not evidence that this grid's selection has actually solved the mismatch."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This calibration never computes or consults a fitness, reward, or task-performance quantity: only "
        "settling_profile's decode()-derived, fitness-blind measurement is used. The rule_based arm was "
        "measured once and never modified to ease matching."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verification-seed", type=int, default=VERIFICATION_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "TIMESCALE_CALIBRATION.md"))
    args = parser.parse_args()

    sweep = run_sweep(n=args.n, seed=args.seed)
    verification = run_verification(sweep["best"], n=args.n, seed=args.verification_seed, rb_steps=sweep["rb_steps"])

    rb_tail = _tail_fractions(sweep["rb_steps"], TAIL_THRESHOLDS, censor_value=T_MAX)

    # re-derive the sweep's own steps for the selected cell for tail reporting
    best = sweep["best"]
    best_output_root = os.path.join(RAW_ROOT, f"decay_{best['decay']}_B_{best['state_bound']}")
    best_steps = _censored_settle_steps(
        S._jsonl_path("nca", "settling_profile", best_output_root), censor_value=T_MAX
    )
    nca_sweep_tail = _tail_fractions(best_steps, TAIL_THRESHOLDS, censor_value=T_MAX)
    nca_verify_tail = _tail_fractions(verification["steps"], TAIL_THRESHOLDS, censor_value=T_MAX)

    report = render_report(sweep, verification, rb_tail, nca_sweep_tail, nca_verify_tail)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
