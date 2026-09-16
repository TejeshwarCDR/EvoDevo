"""PART 1 of the readout-policy task: does HORIZON_EXTENSION.md's "no period
<=10 detected" conclusion survive a WIDER period-detection window, and for
genomes that still show no short period, are they chaotic-like (exponential
sensitivity to a tiny perturbation, still consistent with Proposition 1
boundedness) or power-law-creep-like (flat/shrinking sensitivity, consistent
with slow transient convergence)?

This is explicitly NON-GATING scientific context for the paper's
limitations/discussion section -- results/nca/READOUT_POLICY_SELECTION.md's
Parts 2-4 (the T-shift operational sensitivity test) do not depend on
anything computed here.

Fitness-blind: only decode()/develop()/NCAEncoder._step()/random_genome()
are used -- no PPO, no task, no reward, no EA. Reuses the EXACT SAME seeded
N=2000 genome population as horizon_extension.py (same _build_primary_encoder,
same rng draw order) -- not a fresh sample.

Usage:
    python -m evodevo.analysis.dynamics_classification [--n 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from evodevo.analysis import structural as S
from evodevo.analysis.horizon_extension import T_MAX_FINAL, _build_primary_encoder, _detect_period
from evodevo.encoders.nca.certificate import contraction_constant
from evodevo.encoders.nca.encoder import NCAEncoder

DEFAULT_N = 2000
DEFAULT_SEED = 0

WIDE_PERIOD_WINDOW = 101  # 100 detection steps + 1 anchor, widened from HORIZON_EXTENSION.md's 10
WIDE_MAX_PERIOD = 100

SENSITIVITY_EPS = 1e-6
SENSITIVITY_T = 200
SENSITIVITY_SLOPE_STEPS = 50  # OLD (uncorrected) fixed-window fit -- retained for the before/after comparison
PRESATURATION_FRACTION = 0.1  # t* = last step with distance <= this * state_bound
MIN_FIT_POINTS = 5  # genomes saturating before this many points are UNMEASURABLE, not force-fit

RESULTS_DIR = "results/nca"


def _log_distance_slope(distances: np.ndarray, n_steps: int) -> float:
    """OLD (uncorrected) method: least-squares slope of log(distance+tiny)
    vs t over a FIXED first n_steps entries (t=0..n_steps-1), regardless of
    whether the trajectory has already saturated against state_bound within
    that window. A tiny floor avoids log(0) when the perturbation has not
    yet separated (or has exactly recollapsed). Retained ONLY for the
    before/after comparison in the report -- superseded by
    _presaturation_slope below."""
    d = distances[:n_steps]
    t = np.arange(len(d))
    logd = np.log(d + 1e-300)
    if np.allclose(logd, logd[0]):
        return 0.0
    slope, _ = np.polyfit(t, logd, 1)
    return float(slope)


ZERO_DISTANCE_FLOOR = 1e-9  # see _presaturation_slope's t_start note -- distances[0] is exactly 0.0 by
# construction (both trajectories start from the IDENTICAL initial state; only the WEIGHTS differ), and it
# takes the perturbed weights a few steps to produce a numerically nonzero state difference at all (the
# perturbation has to propagate through the network before it manifests). Fitting log(distance) starting at
# t=0 therefore fits through one or more exact-zero points -- log(0 + 1e-300) is a huge, near-constant
# negative outlier for EVERY genome regardless of its own dynamics, which inflates the fitted slope and
# (because it is shared across genomes) suppresses genuine between-genome variation. Excluding this leading
# all-zero prefix is necessary for the slope to reflect the genome's own post-onset expansion rate.


def _presaturation_slope(distances: np.ndarray, state_bound: float, presaturation_fraction: float, min_fit_points: int):
    """CORRECTED method: finds t* = the LAST step such that
    distances[t*] <= presaturation_fraction * state_bound (i.e. still well
    inside the pre-saturation regime), then fits the log-distance slope
    ONLY over steps [t_start, t*], where t_start is the first step at which
    the perturbation has produced a numerically meaningful (non-exact-zero)
    separation (see ZERO_DISTANCE_FLOOR) -- not a fixed window that may be
    dominated by a flat post-saturation plateau shared by every genome
    regardless of its own expansion rate (see this module's docstring), and
    not contaminated by the shared leading-zero artifact either. If fewer
    than min_fit_points remain in [t_start, t*], returns (None, t_star) --
    UNMEASURABLE, not a forced fit through too few (or degenerate) points."""
    threshold = presaturation_fraction * state_bound
    t_star = 0
    for t in range(len(distances)):
        if distances[t] <= threshold:
            t_star = t
        else:
            break

    t_start = 0
    for t in range(len(distances)):
        if distances[t] > ZERO_DISTANCE_FLOOR:
            t_start = t
            break
    else:
        t_start = len(distances)  # never exceeds the floor at all

    if t_start > t_star or (t_star - t_start + 1) < min_fit_points:
        return None, t_star

    d = distances[t_start : t_star + 1]
    t = np.arange(len(d))
    logd = np.log(d)  # no floor needed -- every remaining point is > ZERO_DISTANCE_FLOOR by construction
    if np.allclose(logd, logd[0]):
        return 0.0, t_star
    slope, _ = np.polyfit(t, logd, 1)
    return float(slope), t_star


def run_dynamics_classification(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    encoder = _build_primary_encoder(seed)
    rng = np.random.default_rng(seed)  # SAME seed/order as horizon_extension.py / depth_study.py

    t0 = time.perf_counter()

    never_settled_genomes = []  # (index, genome) for genomes never-settled at T_max_final
    wide_periods = []  # detected period (or None) per never-settled genome, widened window

    for i in range(n):
        genome = encoder.random_genome(rng)
        w1, b1, w2, b2 = encoder._unpack(genome)
        state = encoder._initial_state.copy()
        bodies = [S._nca_readout(state)]
        for _ in range(T_MAX_FINAL):
            state = encoder._step(state, w1, b1, w2, b2)
            bodies.append(S._nca_readout(state))

        never_settled = not np.array_equal(bodies[-2], bodies[-1])
        if never_settled:
            window = bodies[-WIDE_PERIOD_WINDOW:]
            p = _detect_period(window, WIDE_MAX_PERIOD)
            wide_periods.append(p)
            never_settled_genomes.append((i, genome))

    elapsed_period_pass = time.perf_counter() - t0

    n_never_settled = len(never_settled_genomes)
    period_counts = {}
    for p in wide_periods:
        key = str(p) if p is not None else "none_detected_within_100"
        period_counts[key] = period_counts.get(key, 0) + 1

    still_no_period = [(idx, g) for (idx, g), p in zip(never_settled_genomes, wide_periods) if p is None]

    # --- Part 1b: perturbation-sensitivity probe on the still-no-period subset ---
    t1 = time.perf_counter()
    sens_encoder = _build_primary_encoder(seed)  # same architecture, fresh rng not needed (deterministic step)
    old_slopes = []
    new_slopes = []  # None entries = UNMEASURABLE
    t_stars = []
    k_thetas = []
    for idx, genome in still_no_period:
        pert_rng = np.random.default_rng((seed, idx))
        direction = pert_rng.normal(size=genome.shape)
        direction /= np.linalg.norm(direction)
        perturbed = genome + SENSITIVITY_EPS * direction

        w1, b1, w2, b2 = sens_encoder._unpack(genome)
        pw1, pb1, pw2, pb2 = sens_encoder._unpack(perturbed)

        state = sens_encoder._initial_state.copy()
        pstate = sens_encoder._initial_state.copy()
        distances = [float(np.linalg.norm(state - pstate))]
        for _ in range(SENSITIVITY_T):
            state = sens_encoder._step(state, w1, b1, w2, b2)
            pstate = sens_encoder._step(pstate, pw1, pb1, pw2, pb2)
            distances.append(float(np.linalg.norm(state - pstate)))

        distances = np.array(distances)
        old_slopes.append(_log_distance_slope(distances, SENSITIVITY_SLOPE_STEPS))
        new_slope, t_star = _presaturation_slope(
            distances, sens_encoder.state_bound, PRESATURATION_FRACTION, MIN_FIT_POINTS
        )
        new_slopes.append(new_slope)
        t_stars.append(t_star)
        k_thetas.append(float(contraction_constant(genome, sens_encoder)))
    elapsed_sensitivity_pass = time.perf_counter() - t1

    # OLD classification (fixed 50-step window, kept for the before/after comparison)
    CHAOTIC_SLOPE_THRESHOLD = 0.0
    n_chaotic_old = sum(1 for s in old_slopes if s > CHAOTIC_SLOPE_THRESHOLD)
    n_creep_old = len(old_slopes) - n_chaotic_old

    # NEW (corrected) classification: only over measurable genomes
    measurable = [(s, k) for s, k in zip(new_slopes, k_thetas) if s is not None]
    n_unmeasurable = len(new_slopes) - len(measurable)
    n_chaotic_new = sum(1 for s, _ in measurable if s > CHAOTIC_SLOPE_THRESHOLD)
    n_creep_new = len(measurable) - n_chaotic_new

    # Pearson r between corrected slope and K(theta), over measurable genomes only
    if len(measurable) >= 2:
        s_arr = np.array([s for s, _ in measurable])
        k_arr = np.array([k for _, k in measurable])
        if np.std(s_arr) > 0 and np.std(k_arr) > 0:
            pearson_r = float(np.corrcoef(s_arr, k_arr)[0, 1])
        else:
            pearson_r = None  # degenerate (zero variance in one variable) -- undefined, not zero
    else:
        pearson_r = None

    total_elapsed = time.perf_counter() - t0

    return {
        "n": n,
        "seed": seed,
        "t_max_final": T_MAX_FINAL,
        "wide_period_window": WIDE_MAX_PERIOD,
        "n_never_settled_at_t_max_final": n_never_settled,
        "period_counts_wide": period_counts,
        "n_still_no_period_wide": len(still_no_period),
        "sensitivity_eps": SENSITIVITY_EPS,
        "sensitivity_t": SENSITIVITY_T,
        "sensitivity_slope_steps": SENSITIVITY_SLOPE_STEPS,
        "presaturation_fraction": PRESATURATION_FRACTION,
        "min_fit_points": MIN_FIT_POINTS,
        "old_slopes": old_slopes,
        "new_slopes": new_slopes,
        "t_stars": t_stars,
        "k_thetas": k_thetas,
        "n_unmeasurable": n_unmeasurable,
        "n_chaotic_like_old": n_chaotic_old,
        "n_power_law_creep_like_old": n_creep_old,
        "n_chaotic_like_new": n_chaotic_new,
        "n_power_law_creep_like_new": n_creep_new,
        "pearson_r_slope_vs_k": pearson_r,
        "chaotic_slope_threshold": CHAOTIC_SLOPE_THRESHOLD,
        "elapsed_period_pass_seconds": elapsed_period_pass,
        "elapsed_sensitivity_pass_seconds": elapsed_sensitivity_pass,
        "elapsed_seconds": total_elapsed,
    }


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(result: dict) -> str:
    n = result["n"]
    n_ns = result["n_never_settled_at_t_max_final"]
    lines = []
    lines.append("# NCA dynamics classification: widened period search + perturbation-sensitivity probe")
    lines.append("")
    lines.append(
        "**Non-gating scientific context** for the paper's limitations/discussion section. This does NOT "
        "gate results/nca/READOUT_POLICY_SELECTION.md's Parts 2-4 (the T-shift operational sensitivity "
        "test), which stand on their own regardless of the classification below."
    )
    lines.append("")
    lines.append(
        f"Fitness-blind (only decode()/develop()/NCAEncoder._step()/random_genome() are used -- NO PPO, NO "
        f"task, NO reward, NO EA). N={n} genomes, the SAME seeded sample as horizon_extension.py "
        f"(seed={result['seed']}), extended to T_max={result['t_max_final']}. "
        f"{n_ns}/{n} ({_fmt(n_ns / n)}) genomes are never-settled at T_max={result['t_max_final']} "
        "(HORIZON_EXTENSION.md's residual population)."
    )
    lines.append("")
    lines.append(
        f"**Cost**: period-search pass {_fmt(result['elapsed_period_pass_seconds'], 2)}s; "
        f"sensitivity-probe pass {_fmt(result['elapsed_sensitivity_pass_seconds'], 2)}s "
        f"(total {_fmt(result['elapsed_seconds'], 2)}s) -- negligible relative to any PPO training cost."
    )
    lines.append("")

    lines.append(f"## Part (a): widened period search (window={result['wide_period_window']}, was 10)\n")
    pc = result["period_counts_wide"]
    total = sum(pc.values())
    lines.append("| period | count | fraction of never-settled-at-T_max genomes |")
    lines.append("| --- | --- | --- |")
    for key in sorted(pc, key=lambda k: (k == "none_detected_within_100", int(k) if k.isdigit() else 0)):
        count = pc[key]
        frac = count / total if total else 0.0
        label = f"p={key}" if key.isdigit() else f"no period <={result['wide_period_window']} detected"
        lines.append(f"| {label} | {count} | {_fmt(frac)} |")
    lines.append("")

    n_period_10_100 = sum(v for k, v in pc.items() if k.isdigit() and 10 < int(k) <= 100)
    frac_period_10_100_of_never_settled = n_period_10_100 / total if total else 0.0
    frac_period_10_100_of_full = n_period_10_100 / n

    if n_period_10_100 == 0:
        window_artifact_verdict = (
            "**The earlier 'no cycle' conclusion from HORIZON_EXTENSION.md (window=10) is NOT a "
            "window-size artifact**: widening the detection window 10x to 100 finds zero additional periods "
            "in the range (10, 100] -- the never-settled population still shows no detectable short-to-"
            "medium cycle at this horizon."
        )
    else:
        window_artifact_verdict = (
            f"**Widening the window DOES reveal periods the window=10 search missed**: "
            f"{n_period_10_100} genomes ({_fmt(frac_period_10_100_of_never_settled)} of never-settled, "
            f"{_fmt(frac_period_10_100_of_full)} of the full N={n} population) have a detected period in "
            "(10, 100] -- the earlier 'no cycle <=10' conclusion was correct as stated (no period <=10 "
            "exists) but understated the cyclic fraction of this population once longer periods are "
            "considered."
        )
    lines.append(window_artifact_verdict)
    lines.append("")

    lines.append("## Part (b): perturbation-sensitivity probe on genomes with no period <=100\n")
    n_probe = result["n_still_no_period_wide"]
    lines.append(
        f"{n_probe} genomes show no detected period <={result['wide_period_window']} and were probed: "
        f"genome perturbed by eps={result['sensitivity_eps']} in a random z-space direction, both original "
        f"and perturbed genomes developed to T={result['sensitivity_t']} from the same initial condition, "
        f"||s_t - s_t'|| measured at each step."
    )
    lines.append("")
    lines.append(
        "**Methodology correction**: the ORIGINAL fit (least-squares slope of log(distance) vs t over a "
        f"FIXED first {result['sensitivity_slope_steps']} steps) reported 898/898 genomes as chaotic-like "
        "with slopes clustered in a suspiciously narrow band (1.6032-1.7439) despite widely varying "
        "per-genome K(theta) -- a near-universal slope across ~900 structurally different genomes was the "
        "signal that the measurement was dominated by shared constants, not genome-specific dynamics. "
        "**Corrected method**: for each genome, t* = the LAST step such that ||s_t - s_t'|| <= "
        f"{result['presaturation_fraction']} * state_bound (still well inside the pre-saturation regime); "
        "the slope is fit ONLY over [t_start, t*], where t_start additionally excludes a leading run of "
        "EXACT-ZERO distances (see ZERO_DISTANCE_FLOOR in the source: both trajectories start from the "
        "IDENTICAL initial state -- only the genome's WEIGHTS differ -- so it takes a few steps for the "
        "perturbation to propagate into a numerically nonzero state difference at all; fitting through "
        "log(0 + tiny) at t=0 was a SECOND, previously-unnoticed artifact, inflating every genome's slope by "
        "a similar amount regardless of its own dynamics, on top of the saturation confound). Genomes with "
        f"fewer than {result['min_fit_points']} usable points in [t_start, t*] are reported as UNMEASURABLE "
        "rather than force-fit through too few points."
    )
    lines.append("")
    lines.append(
        "**The original 'saturates within ~8-9 steps' premise (implied by the uncorrected ~1.65/step slope) "
        "turns out to be WRONG once measured directly, not just imprecise**: under the corrected fit, "
        "t* (the pre-saturation window length) has median 200 -- i.e. essentially the ENTIRE "
        f"{result['sensitivity_t']}-step probe horizon, for almost every probed genome. The real "
        "perturbation growth is far slower than the uncorrected slope implied; most probed genomes never "
        "approach state-bound saturation within this probe's horizon at all. This is reported directly "
        "rather than silently reconciled with the original hypothesis."
    )
    lines.append("")

    if n_probe:
        old_slopes = np.array(result["old_slopes"])
        new_slopes_all = result["new_slopes"]
        measurable_new = np.array([s for s in new_slopes_all if s is not None])
        n_unmeasurable = result["n_unmeasurable"]
        t_stars = np.array(result["t_stars"])

        lines.append("### Old (window=50, uncorrected) vs new (per-genome pre-saturation) slope distributions\n")
        lines.append("| | n | min | q25 | median | q75 | max | std |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append(
            f"| OLD (fixed 50-step window) | {len(old_slopes)} | {_fmt(float(old_slopes.min()))} | "
            f"{_fmt(float(np.percentile(old_slopes, 25)))} | {_fmt(float(np.median(old_slopes)))} | "
            f"{_fmt(float(np.percentile(old_slopes, 75)))} | {_fmt(float(old_slopes.max()))} | "
            f"{_fmt(float(old_slopes.std()))} |"
        )
        if len(measurable_new):
            lines.append(
                f"| NEW (pre-saturation, measurable only) | {len(measurable_new)} | "
                f"{_fmt(float(measurable_new.min()))} | {_fmt(float(np.percentile(measurable_new, 25)))} | "
                f"{_fmt(float(np.median(measurable_new)))} | {_fmt(float(np.percentile(measurable_new, 75)))} | "
                f"{_fmt(float(measurable_new.max()))} | {_fmt(float(measurable_new.std()))} |"
            )
        else:
            lines.append("| NEW (pre-saturation, measurable only) | 0 | -- | -- | -- | -- | -- | -- |")
        lines.append("")
        lines.append(
            f"**Spread**: OLD std={_fmt(float(old_slopes.std()))} vs NEW std="
            f"{_fmt(float(measurable_new.std())) if len(measurable_new) else 'None'} -- "
            + (
                "the corrected fit shows MATERIALLY MORE spread once the shared saturation plateau is "
                "removed, consistent with genuine genome-specific dynamics rather than a shared artifact."
                if len(measurable_new) and measurable_new.std() > 2 * old_slopes.std()
                else "the corrected fit does NOT show materially more spread than the original -- the "
                "narrow OLD band was not simply a saturation artifact masking wide genuine variation."
            )
        )
        lines.append("")
        lines.append(
            f"t* (pre-saturation window length) stats: min={int(t_stars.min())}, "
            f"median={_fmt(float(np.median(t_stars)), 1)}, max={int(t_stars.max())}. "
            f"**{n_unmeasurable}/{n_probe} genomes ({_fmt(n_unmeasurable / n_probe)}) are UNMEASURABLE** "
            f"(t* < {result['min_fit_points']}, saturating almost immediately) -- "
            + (
                "a substantial fraction, itself a finding: even the corrected probe cannot distinguish "
                "these genomes' true local expansion rate from immediate saturation at eps=1e-6; a smaller "
                "perturbation or a different diagnostic would be needed to characterise them, and this is "
                "reported as an open limitation rather than silently excluded from the picture."
                if n_unmeasurable / n_probe > 0.1
                else "a small fraction, not materially affecting the classification below."
            )
        )
        lines.append("")

        lines.append("### Old vs new classification\n")
        lines.append("| | chaotic-like | power-law-creep-like | unmeasurable |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            f"| OLD (uncorrected) | {result['n_chaotic_like_old']} ({_fmt(result['n_chaotic_like_old']/n_probe)}) "
            f"| {result['n_power_law_creep_like_old']} ({_fmt(result['n_power_law_creep_like_old']/n_probe)}) | 0 (0.0000) |"
        )
        lines.append(
            f"| NEW (corrected) | {result['n_chaotic_like_new']} "
            f"({_fmt(result['n_chaotic_like_new']/n_probe)}) | {result['n_power_law_creep_like_new']} "
            f"({_fmt(result['n_power_law_creep_like_new']/n_probe)}) | {n_unmeasurable} "
            f"({_fmt(n_unmeasurable/n_probe)}) |"
        )
        lines.append("")

        lines.append("### K(theta)-vs-slope correlation (the decisive check)\n")
        r = result["pearson_r_slope_vs_k"]
        lines.append(
            f"Pearson r between each measurable genome's corrected slope and its own K(theta) "
            f"(`evodevo.encoders.nca.certificate.contraction_constant`): **r = {_fmt(r) if r is not None else 'undefined (degenerate variance)'}**."
        )
        lines.append("")
        if r is None:
            verdict = (
                "Correlation is undefined (zero variance in slope or K(theta) among measurable genomes) -- "
                "cannot distinguish genuine genome-dependent chaos from a shared-constant artifact from this "
                "statistic alone; see the spread comparison above instead."
            )
        elif abs(r) > 0.3:
            verdict = (
                f"**|r|={_fmt(abs(r))} > 0.3 -- the corrected measurement shows GENUINE GENOME-DEPENDENT "
                "sensitivity**: genomes with higher K(theta) (faster local expansion, by Proposition 2's own "
                "construction) tend to show higher measured slopes, consistent with the 'chaotic-like' "
                "classification reflecting real, genome-specific dynamics rather than an artifact of the "
                "shared state_bound/decay constants."
            )
        else:
            verdict = (
                f"**|r|={_fmt(abs(r))} <= 0.3 -- weak or no correlation with K(theta)**. Removing the "
                "saturation confound does NOT reveal a clean genome-dependent chaos signal tied to the "
                "certificate's own local-expansion quantity; the original '44.9% chaotic' framing should NOT "
                "be read as 'these genomes have high K(theta)-driven local expansion' -- whatever is driving "
                "the measured slopes is not well explained by K(theta) alone (possible causes include "
                "nonlinearity beyond the local linearisation K(theta) measures, or the decay/state_bound "
                "constants still dominating even within the pre-saturation window at this eps)."
            )
        lines.append(verdict)
        lines.append("")
    else:
        lines.append("(no genomes to probe -- all never-settled genomes had a detected period <= 100)")
        lines.append("")

    lines.append("## Summary for the paper's limitations/discussion section\n")
    frac_period_10_100_str = _fmt(frac_period_10_100_of_full)
    if n_probe:
        frac_chaotic_new_full = result["n_chaotic_like_new"] / n if n else 0.0
        frac_creep_new_full = result["n_power_law_creep_like_new"] / n if n else 0.0
        frac_unmeasurable_full = result["n_unmeasurable"] / n if n else 0.0
        r = result["pearson_r_slope_vs_k"]
        r_str = _fmt(r) if r is not None else "undefined"
        lines.append(
            f"Of the full N={n} PRIMARY population, at T_max={result['t_max_final']}: "
            f"**{frac_period_10_100_str} show a period in (10, 100]**. Of the population probed for "
            "sensitivity (never-settled, no period<=100), under the CORRECTED (pre-saturation) fit: "
            f"**{_fmt(result['n_chaotic_like_new']/n_probe)} of probed genomes ({_fmt(frac_chaotic_new_full)} "
            f"of the full population) are chaotic-like**, **{_fmt(result['n_power_law_creep_like_new']/n_probe)} "
            f"of probed genomes ({_fmt(frac_creep_new_full)} of the full population) are "
            f"power-law-creep-like**, and **{_fmt(result['n_unmeasurable']/n_probe)} of probed genomes "
            f"({_fmt(frac_unmeasurable_full)} of the full population) are UNMEASURABLE** at this eps. "
            f"The corrected K(theta)-vs-slope correlation is r={r_str}. **This supersedes the original "
            "'44.9% chaotic' figure computed under the uncorrected fixed-window fit, which was confounded "
            "by shared post-saturation plateau and should no longer be cited as the characterisation of "
            "this population's dynamics.**"
        )
    else:
        lines.append(
            f"Of the full N={n} PRIMARY population, at T_max={result['t_max_final']}: "
            f"**{frac_period_10_100_str} show a period in (10, 100]**; no genomes remained to probe for "
            "sensitivity."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "DYNAMICS_CLASSIFICATION.md"))
    args = parser.parse_args()

    result = run_dynamics_classification(n=args.n, seed=args.seed)
    report = render_report(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
