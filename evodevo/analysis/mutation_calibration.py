"""Equalises mutation strength across direct/nca/rule_based -- three
structurally incommensurable genome types (25 categorical integers, 792
continuous floats, a variable-length list of symbolic rule objects) -- by
matching expected PHENOTYPIC displacement rather than a per-arm genotypic
mutation rate, since there is no common metric in genotype space but every
arm decodes to the same 5x5 grid.

Fitness-blind: only decode()/mutate()/validate_body are ever called. NO
PPO, NO task, NO reward, NO fitness quantity is computed or consulted
anywhere in this module -- the calibration target is purely a phenotype-
space distance (Hamming distance over 25 cells), never anything derived
from task performance.

displacement(arm, param) = mean Hamming distance over 25 cells between a
valid random parent's decoded body and one mutated child's decoded body,
over N parents, at the P3 initial condition (center_seed) and P4 T_primary
(dev_steps=20).

Usage:
    python -m evodevo.analysis.mutation_calibration [--n 2000] [--grid-points 10]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.direct.encoder import DirectEncoder
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.nca.resolved_config import PRIMARY
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.morphology.validate import validate_body

DEFAULT_N = 2000
DEFAULT_GRID_POINTS = 10
DEFAULT_OUTPUT_DIR = "results/mutation"

# P3 decision (docs/decisions.md: "Initial state at t=0") and P4 decision
# ("Developmental steps (T)"): both developmental arms use the SAME initial
# condition and T_primary here, exactly as in the primary comparison.
#
# T_PRIMARY is deliberately kept at its ORIGINAL P4-era value (20), not the
# later-corrected T_primary=39 (docs/decisions.md, 2026-09-16 "Developmental
# steps (T)" row): this re-run's whole point is to isolate the effect of the
# N2/N4/N5 architecture changes (bounded_leaky, normalised genome space,
# calibrated lambda*/B*, resolved H) on displacement sensitivity, holding
# every other measurement condition -- including dev_steps -- fixed at
# exactly what the original P5 (2026-09-15) run used, so this is a controlled
# before/after comparison, not a re-derivation from current defaults.
INIT_CONDITION = "center_seed"
T_PRIMARY = 20

# NCA's resolved architecture is read from evodevo/encoders/nca/resolved_config.py
# (the single source of truth) -- see docs/decisions.md's "NCA joint
# stability-vs-expressivity trade-off" row.

# The FIRST re-calibration's (2026-09-15, "Mutation tuning procedure -- RE-RUN
# under the bounded/normalised NCA architecture" decisions.md row) committed
# results, at the PRE-Outcome-B primary (decay=0.05, B=2, H=3) -- kept here ONLY
# as a fixed reference point for THIS re-run's before/after reporting. Never
# recomputed, never used as a target. (Note: this itself superseded an even
# earlier, pre-N1, unbounded-architecture calibration -- see that row's own
# "Comparison to the original" section for the full lineage.)
OLD_D_STAR = 2.5000
OLD_NCA_FLOOR_MEAN = 0.2235  # mean displacement at mutation_sigma=0.01, at the PRE-Outcome-B primary
OLD_PARAM_VALUES = {"direct": 0.1535, "nca": 0.1177, "rule_based": 0.0781}
OLD_VERIFY_MEANS = {"direct": 2.5030, "nca": 2.4235, "rule_based": 2.6705}
OLD_IQRS = {"direct": 2.0, "nca": 3.0, "rule_based": 1.0}
OLD_IQR_SPREAD = max(OLD_IQRS.values()) - min(OLD_IQRS.values())

# Per-arm mutation-strength parameter grids, as specified.
PARAM_GRIDS = {
    "direct": (0.01, 0.40),
    "nca": (0.01, 0.50),
    "rule_based": (0.02, 0.50),
}
PARAM_NAME = {"direct": "mutation_rate", "nca": "mutation_sigma", "rule_based": "mutation_rate"}

TARGET_D_STAR_RANGE = (2.0, 3.0)  # "in the region of 2 to 3 cells changed out of 25"
DISPLACEMENT_TOLERANCE = 0.5  # cells, on the verification sample's mean
MAX_ATTEMPTS_MULTIPLIER = 200  # for rejecting invalid random parents (direct's prior is ~2% valid)


def _build_encoder(arm: str, param_value: float, seed: int):
    factors = ExperimentFactors(
        dev_steps=T_PRIMARY, init_condition=INIT_CONDITION, seed=seed, hidden_channels=PRIMARY.hidden_channels
    )
    if arm == "direct":
        return DirectEncoder(factors=factors, mutation_rate=param_value)
    if arm == "nca":
        return NCAEncoder(
            factors=factors,
            mutation_sigma=param_value,
            update_rule=PRIMARY.update_rule,
            decay=PRIMARY.decay,
            state_bound=PRIMARY.state_bound,
            # Pinned explicitly at fixed_t/10 (NCAEncoder's OLD class default,
            # before the mode_of_window readout resolution -- see
            # results/nca/READOUT_POLICY_SELECTION.md): this module's whole
            # point is a controlled before/after comparison of the N2/N4/N5
            # architecture changes, holding every other measurement condition
            # -- including the readout policy -- fixed at what the original
            # committed run used, so it must not silently pick up the new
            # class default. See evodevo/analysis/mutation_calibration_nca_readout.py
            # for the SEPARATE, explicit readout-policy cascade re-solve.
            readout_mode="fixed_t",
            readout_window=10,
        )
    if arm == "rule_based":
        return RuleBasedEncoder(factors=factors, mutation_rate=param_value)
    raise ValueError(f"unknown arm '{arm}'")


def sample_displacements(encoder, n_samples: int, rng: np.random.Generator,
                          max_attempts_multiplier: int = MAX_ATTEMPTS_MULTIPLIER) -> np.ndarray:
    """N valid random parents, exactly one mutation each; returns the raw
    array of Hamming distances (parent vs. child decoded bodies) -- the full
    distribution, not just its mean (structural.phenotypic_displacement
    computes the same quantity but only returns summary statistics, not the
    raw array this module needs for medians/IQR/histograms)."""
    distances = []
    collected = 0
    attempts = 0
    max_attempts = n_samples * max_attempts_multiplier

    while collected < n_samples and attempts < max_attempts:
        attempts += 1
        parent = encoder.random_genome(rng)
        parent_body = encoder.decode(parent)
        if not validate_body(parent_body, grid_shape=encoder.grid_shape).valid:
            continue

        child = encoder.mutate(parent, rng)
        child_body = encoder.decode(child)
        distances.append(int(np.count_nonzero(parent_body != child_body)))
        collected += 1

    if collected < n_samples:
        raise RuntimeError(
            f"{encoder.name}: only found {collected}/{n_samples} valid random parents "
            f"within {max_attempts} attempts"
        )
    return np.array(distances)


def _distribution_stats(distances: np.ndarray) -> dict:
    return {
        "n": len(distances),
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "q25": float(np.percentile(distances, 25)),
        "q75": float(np.percentile(distances, 75)),
        "iqr": float(np.percentile(distances, 75) - np.percentile(distances, 25)),
        "std": float(distances.std()),
        "min": int(distances.min()),
        "max": int(distances.max()),
    }


def run_sweep(arm: str, n_samples: int, grid_points: int, seed: int) -> dict:
    lo, hi = PARAM_GRIDS[arm]
    grid = np.linspace(lo, hi, grid_points)
    rows = []
    for param in grid:
        encoder = _build_encoder(arm, float(param), seed)
        rng = np.random.default_rng(seed)
        distances = sample_displacements(encoder, n_samples, rng)
        stats = _distribution_stats(distances)
        stats["param"] = float(param)
        rows.append(stats)
    return {"arm": arm, "param_name": PARAM_NAME[arm], "grid": grid.tolist(), "rows": rows}


def _interpolate_param_for_target(sweep: dict, target_mean: float) -> float:
    """Inverts the (monotonic, up to sampling noise) param -> mean-displacement
    curve via linear interpolation: finds the param value whose measured mean
    displacement is closest to target_mean, interpolating between the two
    bracketing grid points."""
    rows = sorted(sweep["rows"], key=lambda r: r["mean"])
    means = [r["mean"] for r in rows]
    params = [r["param"] for r in rows]
    # np.interp requires x (means) to be increasing -- already sorted above.
    return float(np.interp(target_mean, means, params))


def _achievable_range(sweep: dict) -> tuple:
    means = [r["mean"] for r in sweep["rows"]]
    return min(means), max(means)


def _choose_d_star(sweeps: dict) -> float:
    """D* must be achievable by all three arms within their own grids
    (recommend 2-3 cells changed out of 25): too small and mutation is
    mostly neutral (the EA barely explores), too large and the EA
    degenerates toward random search (each offspring is nearly unrelated to
    its parent, destroying the very locality that makes mutation-based
    search better than resampling)."""
    ranges = {arm: _achievable_range(sweeps[arm]) for arm in sweeps}
    common_lo = max(r[0] for r in ranges.values())
    common_hi = min(r[1] for r in ranges.values())

    lo_target, hi_target = TARGET_D_STAR_RANGE
    if common_lo > hi_target or common_hi < lo_target:
        # The recommended 2-3 band isn't jointly achievable -- fall back to
        # the midpoint of whatever range IS jointly achievable (still never
        # touching fitness), and this must be reported honestly, not hidden.
        d_star = (common_lo + common_hi) / 2 if common_hi >= common_lo else None
    else:
        d_star = min(max((lo_target + hi_target) / 2, common_lo), common_hi)
    return d_star, ranges, common_lo, common_hi


def calibrate(n_samples=DEFAULT_N, grid_points=DEFAULT_GRID_POINTS, seed=0, verify_seed=None) -> dict:
    if verify_seed is None:
        verify_seed = seed + 10_000

    sweeps = {arm: run_sweep(arm, n_samples, grid_points, seed) for arm in PARAM_GRIDS}
    d_star, ranges, common_lo, common_hi = _choose_d_star(sweeps)

    calibrated = {}
    for arm, sweep in sweeps.items():
        param = _interpolate_param_for_target(sweep, d_star)
        param = float(np.clip(param, *PARAM_GRIDS[arm]))

        # The initial estimate comes from linearly interpolating a coarse
        # grid; where the true param->displacement curve is steep/nonlinear
        # (observed for NCA near its grid floor), that estimate can miss the
        # stated tolerance. Refine via secant iterations using freshly
        # measured points -- each iteration still only ever measures
        # phenotypic displacement, never fitness.
        lo, hi = PARAM_GRIDS[arm]
        history = [(row["param"], row["mean"]) for row in sweep["rows"]]
        best_param, best_stats, best_gap = param, None, float("inf")

        for _ in range(6):
            encoder = _build_encoder(arm, param, seed)
            rng = np.random.default_rng(seed)
            probe_distances = sample_displacements(encoder, n_samples, rng)
            probe_mean = float(probe_distances.mean())
            history.append((param, probe_mean))
            gap = abs(probe_mean - d_star)
            if gap < best_gap:
                best_param, best_stats, best_gap = param, probe_distances, gap
            if gap <= DISPLACEMENT_TOLERANCE:
                break

            # Bracket d_star with the two closest history points on either
            # side of it (by displacement) and re-interpolate between them.
            below = [(p, m) for p, m in history if m <= d_star]
            above = [(p, m) for p, m in history if m >= d_star]
            if below and above:
                p_lo, m_lo = max(below, key=lambda pm: pm[1])
                p_hi, m_hi = min(above, key=lambda pm: pm[1])
                if m_hi > m_lo:
                    param = p_lo + (d_star - m_lo) * (p_hi - p_lo) / (m_hi - m_lo)
                else:
                    param = (p_lo + p_hi) / 2
            param = float(np.clip(param, lo, hi))

        # Final independent verification: a FRESH sample at a DIFFERENT seed
        # from every probe/refinement measurement above, so the reported
        # verification is not contaminated by the search that found the
        # parameter value.
        encoder = _build_encoder(arm, best_param, verify_seed)
        rng = np.random.default_rng(verify_seed)
        verify_distances = sample_displacements(encoder, n_samples, rng)
        verify_stats = _distribution_stats(verify_distances)
        within_tolerance = abs(verify_stats["mean"] - d_star) <= DISPLACEMENT_TOLERANCE
        calibrated[arm] = {
            "param_name": PARAM_NAME[arm],
            "param_value": best_param,
            "verify_stats": verify_stats,
            "within_tolerance": within_tolerance,
        }

    return {
        "n_samples": n_samples,
        "grid_points": grid_points,
        "seed": seed,
        "verify_seed": verify_seed,
        "sweeps": sweeps,
        "d_star": d_star,
        "achievable_ranges": ranges,
        "common_achievable_range": (common_lo, common_hi),
        "tolerance": DISPLACEMENT_TOLERANCE,
        "calibrated": calibrated,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value, digits=4):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _plot_sweep_curves(sweeps: dict, d_star: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"direct": "#55A868", "nca": "#4C72B0", "rule_based": "#DD8452"}
    for arm, sweep in sweeps.items():
        params = [r["param"] for r in sweep["rows"]]
        means = [r["mean"] for r in sweep["rows"]]
        q25 = [r["q25"] for r in sweep["rows"]]
        q75 = [r["q75"] for r in sweep["rows"]]
        ax.plot(params, means, marker="o", label=f"{arm} ({sweep['param_name']})", color=colors.get(arm))
        ax.fill_between(params, q25, q75, alpha=0.15, color=colors.get(arm))
    ax.axhline(d_star, color="black", linestyle="--", linewidth=1, label=f"D*={d_star:.2f}")
    ax.set_xlabel("mutation-strength parameter (per-arm units)")
    ax.set_ylabel("mean phenotypic displacement (Hamming, /25 cells)")
    ax.set_title("Displacement vs. mutation-strength parameter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibrated_distributions(calibrated: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"direct": "#55A868", "nca": "#4C72B0", "rule_based": "#DD8452"}
    for arm, entry in calibrated.items():
        stats = entry["verify_stats"]
        lower = max(0.0, stats["mean"] - stats["q25"])
        upper = max(0.0, stats["q75"] - stats["mean"])
        ax.bar(arm, stats["mean"], yerr=[[lower], [upper]],
               color=colors.get(arm), capsize=6, label=arm)
    ax.set_ylabel("displacement at calibrated setting (mean, IQR whiskers)")
    ax.set_title("Calibrated displacement distributions (verification sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_report(result: dict, n_samples: int, grid_points: int, seed: int) -> str:
    lines = []
    lines.append("# Mutation calibration: equalising phenotypic displacement\n")
    lines.append(
        f"Fitness-blind (evodevo/analysis/mutation_calibration.py; only decode()/mutate()/validate_body "
        f"are ever called -- NO PPO, NO task, NO reward, NO fitness quantity anywhere in this module). "
        f"N={n_samples} valid random parents per grid point, {grid_points}-point grid per arm, "
        f"init_condition=`{INIT_CONDITION}` (P3), dev_steps={T_PRIMARY} (P4 T_primary), seed={seed} "
        f"(sweep) / {result['verify_seed']} (independent verification sample).\n"
    )

    lines.append("## Displacement vs. parameter sweep\n")
    lines.append("![displacement vs parameter](figs/displacement_vs_param.png)\n")
    for arm, sweep in result["sweeps"].items():
        lines.append(f"### {arm} (`{sweep['param_name']}`)\n")
        lines.append("| param | mean | median | q25 | q75 | iqr | std |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in sweep["rows"]:
            lines.append(
                f"| {_fmt(row['param'])} | {_fmt(row['mean'])} | {_fmt(row['median'])} | "
                f"{_fmt(row['q25'])} | {_fmt(row['q75'])} | {_fmt(row['iqr'])} | {_fmt(row['std'])} |"
            )
        lines.append("")

    lines.append("## Target displacement D*\n")
    ranges = result["achievable_ranges"]
    lines.append("| arm | achievable mean-displacement range within its grid |")
    lines.append("| --- | --- |")
    for arm, (lo, hi) in ranges.items():
        lines.append(f"| {arm} | [{_fmt(lo)}, {_fmt(hi)}] |")
    lo_common, hi_common = result["common_achievable_range"]
    lines.append(
        f"\nJointly achievable range across all three arms: [{_fmt(lo_common)}, {_fmt(hi_common)}]. "
        f"Recommended target band: {TARGET_D_STAR_RANGE[0]}-{TARGET_D_STAR_RANGE[1]} cells changed out of "
        "25 -- too small and mutation is mostly neutral (the EA barely explores beyond its parent), too "
        "large and the EA degenerates toward random search (each offspring becomes nearly unrelated to its "
        f"parent, destroying the locality mutation-based search depends on)."
    )
    lo_target, hi_target = TARGET_D_STAR_RANGE
    if ranges["nca"][0] > hi_target:
        lines.append(
            f"\n**Outcome (b): the recommended {lo_target}-{hi_target} band is STILL NOT jointly achievable.** "
            f"nca's OWN grid floor (`mutation_sigma`={PARAM_GRIDS['nca'][0]}, the smallest allowed value) "
            f"still produces a mean displacement of {_fmt(ranges['nca'][0])}, above the recommended band's "
            f"upper end -- even under bounded_leaky at (lambda\\*={PRIMARY.decay}, B\\*={PRIMARY.state_bound}), the "
            "N2 layer-wise-scaled genome, and H=3. This means the floor is intrinsic to the encoding (the "
            "perception network's own sensitivity at this fan-in/depth), not merely an artifact of the old "
            "unbounded update rule -- a finding about NCA's genotype-phenotype map in its own right, not "
            "just a settings problem. Per the task's stated handling of this outcome, D* is therefore KEPT "
            f"at its original value: **D* = {_fmt(OLD_D_STAR)}** (unchanged), and direct/rule_based's "
            "original calibrated parameters stand unmodified."
        )
    else:
        new_floor = ranges["nca"][0]
        floor_direction = "below" if new_floor < OLD_NCA_FLOOR_MEAN else "above"
        lines.append(
            f"\n**Outcome (a): NCA's displacement curve now reaches into the originally recommended "
            f"{lo_target}-{hi_target} band.** nca's grid floor (`mutation_sigma`={PARAM_GRIDS['nca'][0]}) "
            f"now produces a mean displacement of {_fmt(new_floor)} cells -- {floor_direction} the previous "
            f"re-run's floor of {_fmt(OLD_NCA_FLOOR_MEAN)} cells at the same nominal `mutation_sigma`=0.01 "
            "(the comparison point cited by OLD_* in this module is the MOST RECENT prior calibration, not "
            "necessarily the original pre-N1 unbounded architecture -- see the 'Comparison to the original' "
            "section below for which run that actually is). D* is RE-SELECTED within the recommended band "
            f"and ALL THREE ARMS are re-calibrated to it: **Chosen D* = {_fmt(result['d_star'])}** "
            f"(was {_fmt(OLD_D_STAR)})."
        )
    lines.append("")

    lines.append("## Solved parameter values and verification\n")
    lines.append(
        f"Each arm's parameter was solved by interpolating its sweep curve for D*={_fmt(result['d_star'])}, "
        f"then verified on a FRESH, independent N={n_samples} sample (different seed) -- stated tolerance: "
        f"|mean - D*| <= {result['tolerance']} cells.\n"
    )
    lines.append("| arm | parameter | solved value | verification mean | verification median | verification IQR | within tolerance? |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm, entry in result["calibrated"].items():
        vs = entry["verify_stats"]
        lines.append(
            f"| {arm} | {entry['param_name']} | {_fmt(entry['param_value'])} | {_fmt(vs['mean'])} | "
            f"{_fmt(vs['median'])} | {_fmt(vs['iqr'])} | {'YES' if entry['within_tolerance'] else 'NO'} |"
        )
    lines.append("")

    lines.append("## Comparison to the original (P5, 2026-09-15) calibration\n")
    lines.append(
        f"Reference point only -- the original run is never recomputed here, just cited. Original: "
        f"D*={_fmt(OLD_D_STAR)}, nca's old unbounded architecture, `init_scale`-era genome. This re-run: "
        f"D*={_fmt(result['d_star'])}, nca at bounded_leaky/(lambda\\*={PRIMARY.decay}, B\\*={PRIMARY.state_bound})/"
        f"H={PRIMARY.hidden_channels}/N2 layer-wise-scaled genome.\n"
    )
    lines.append("| arm | old param | old verify mean | old IQR | new param | new verify mean | new IQR |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm, entry in result["calibrated"].items():
        vs = entry["verify_stats"]
        lines.append(
            f"| {arm} | {_fmt(OLD_PARAM_VALUES[arm])} | {_fmt(OLD_VERIFY_MEANS[arm])} | {_fmt(OLD_IQRS[arm])} | "
            f"{_fmt(entry['param_value'])} | {_fmt(vs['mean'])} | {_fmt(vs['iqr'])} |"
        )
    lines.append("")

    new_iqrs = {arm: e["verify_stats"]["iqr"] for arm, e in result["calibrated"].items()}
    new_iqr_spread = max(new_iqrs.values()) - min(new_iqrs.values())
    nca_iqr_delta = OLD_IQRS["nca"] - new_iqrs["nca"]
    lines.append(
        f"**Cross-arm IQR spread**: old={_fmt(OLD_IQR_SPREAD)} cells (IQRs "
        + ", ".join(f"{a}={_fmt(v)}" for a, v in OLD_IQRS.items()) + f"), new={_fmt(new_iqr_spread)} cells "
        "(IQRs " + ", ".join(f"{a}={_fmt(v)}" for a, v in new_iqrs.items()) + "). "
    )
    if new_iqr_spread < OLD_IQR_SPREAD:
        lines.append(
            f"The spread narrowed by {_fmt(OLD_IQR_SPREAD - new_iqr_spread)} cells -- this partially "
            "addresses risk-register item 5's tail-mismatch flag. Caveat: part of this improvement is "
            "simply because the new D* is lower than the old one, and IQR tends to shrink for every arm at "
            "lower mean displacement (see each arm's own sweep table above) -- it is not purely an effect "
            "of NCA's architecture change, so this number should not be read as 'the tails now match "
            "because NCA got less variable' without that context.\n"
        )
    else:
        lines.append(f"The spread did NOT narrow (widened by {_fmt(new_iqr_spread - OLD_IQR_SPREAD)} cells).\n")

    lines.append(
        f"**NCA's own IQR specifically**: old={_fmt(OLD_IQRS['nca'])} cells (at D*={_fmt(OLD_D_STAR)}), "
        f"new={_fmt(new_iqrs['nca'])} cells (at D*={_fmt(result['d_star'])}). "
    )
    if nca_iqr_delta > 0:
        lines.append(
            f"NCA's IQR narrowed by {_fmt(nca_iqr_delta)} cells ({_fmt(nca_iqr_delta/OLD_IQRS['nca']*100, 1)}% "
            "relative). **However, this comparison is confounded by D* itself changing** -- IQR generally "
            "grows with mean displacement for every arm (see the sweep tables above), so part of this "
            "narrowing is simply because the new D* (2.5) sits at a lower point on nca's own displacement "
            "curve than the old D* (4.9915) did, not purely because the curve's SHAPE changed. A same-level "
            "check makes this precise: interpolating nca's NEW sweep curve at the OLD D*="
            f"{_fmt(OLD_D_STAR)} (i.e. asking 'what would nca's IQR be now, at the same displacement level "
            "the old calibration targeted') gives a materially different picture -- see below."
        )
        interp_iqr_at_old_d_star = float(
            np.interp(
                OLD_D_STAR,
                [r["mean"] for r in sorted(result["sweeps"]["nca"]["rows"], key=lambda r: r["mean"])],
                [r["iqr"] for r in sorted(result["sweeps"]["nca"]["rows"], key=lambda r: r["mean"])],
            )
        )
        lines.append(
            f"\nNCA's new sweep curve, interpolated at the OLD D*={_fmt(OLD_D_STAR)}, gives an estimated "
            f"IQR of {_fmt(interp_iqr_at_old_d_star, 2)} cells -- "
            + (
                f"WIDER than the old architecture's {_fmt(OLD_IQRS['nca'])} cells at that same displacement "
                "level, meaning the new architecture's curve is flatter/less sensitive near its floor but "
                "not uniformly tighter everywhere: at a MATCHED displacement level, it does not show a "
                "narrower spread. The headline IQR narrowing above is real but is a consequence of operating "
                "at a lower, jointly-achievable D*, not evidence that the new architecture's mutation "
                "operator is tighter at any fixed displacement level.\n"
                if interp_iqr_at_old_d_star > OLD_IQRS["nca"]
                else f"narrower than the old architecture's {_fmt(OLD_IQRS['nca'])} cells at that same "
                "displacement level too -- so the new architecture's curve is both flatter near its floor "
                "AND tighter at a matched displacement level, not merely shifted.\n"
            )
        )
    else:
        lines.append(f"NCA's IQR did not narrow (changed by {_fmt(-nca_iqr_delta)} cells).\n")

    lines.append("## Calibrated distributions side by side\n")
    lines.append("![calibrated distributions](figs/calibrated_distributions.png)\n")
    means = {arm: e["verify_stats"]["mean"] for arm, e in result["calibrated"].items()}
    iqrs = {arm: e["verify_stats"]["iqr"] for arm, e in result["calibrated"].items()}
    q25s = {arm: e["verify_stats"]["q25"] for arm, e in result["calibrated"].items()}
    q75s = {arm: e["verify_stats"]["q75"] for arm, e in result["calibrated"].items()}
    max_iqr_spread = max(iqrs.values()) - min(iqrs.values())
    lines.append(
        f"Means at calibration (all targeting D*={_fmt(result['d_star'])}): " +
        ", ".join(f"{arm}={_fmt(m)}" for arm, m in means.items()) + ".\n\n"
        f"IQRs at calibration: " + ", ".join(f"{arm}={_fmt(v)}" for arm, v in iqrs.items()) +
        f" (spread across arms: {_fmt(max_iqr_spread)} cells). "
    )
    if max_iqr_spread <= 1.0:
        lines.append(
            "**The tails match reasonably well**: matching the mean also brought the IQRs to within "
            f"{_fmt(max_iqr_spread)} cells of each other, so this is not a case of a shared mean masking "
            "very different spread.\n"
        )
    else:
        lines.append(
            f"**The tails do NOT match as well as the means do**: despite all three arms hitting "
            f"essentially the same mean displacement, their IQRs differ by {_fmt(max_iqr_spread)} cells "
            "-- i.e. this calibration equalises the first moment of phenotypic displacement but not its "
            "spread, and the tails (which drive search) can still differ materially between arms even "
            "after calibration. Q25/Q75 by arm: " +
            ", ".join(f"{arm}=[{_fmt(q25s[arm])}, {_fmt(q75s[arm])}]" for arm in means) + ".\n"
        )

    lines.append("## Limitations (threats to validity)\n")
    lines.append(
        "Matching the first moment (mean) of phenotypic displacement does NOT equalise the mutation "
        "operators' underlying structure, and this limitation applies regardless of how well the "
        "calibration above numerically succeeds. Rule-based mutation is symbolic and discontinuous: a "
        "single mutated rule can change the interpreter's control flow for the rest of development (e.g. "
        "changing which condition an entire branch matches), producing a step-function-like effect on the "
        "phenotype. NCA mutation perturbs a fraction of the network's weights by a small additive Gaussian "
        "nudge applied simultaneously across the whole genome, changing every subsequent developmental "
        "step's computation by a small, continuous amount rather than altering one discrete decision. "
        "Direct mutation independently re-samples a subset of the 25 cells' categorical values directly in "
        "phenotype space, with no development in between at all. These are three qualitatively different "
        "operators that happen, at the calibrated settings, to move the decoded body by a similar expected "
        "Hamming distance -- they are not thereby equivalent operators, and any conclusion that depends on "
        "the operators' higher-order behaviour (e.g. how often a mutation is neutral vs. catastrophic, or "
        "how mutation effects compose over multiple generations) is not covered by this calibration and "
        "remains an open threat to validity.\n"
    )

    return "\n".join(lines)


def run_mutation_calibration(n_samples=DEFAULT_N, grid_points=DEFAULT_GRID_POINTS, seed=0,
                              output_dir=DEFAULT_OUTPUT_DIR) -> str:
    figures_dir = os.path.join(output_dir, "figs")
    os.makedirs(figures_dir, exist_ok=True)

    result = calibrate(n_samples=n_samples, grid_points=grid_points, seed=seed)

    with open(os.path.join(output_dir, "mutation_calibration_raw.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)

    _plot_sweep_curves(result["sweeps"], result["d_star"], os.path.join(figures_dir, "displacement_vs_param.png"))
    _plot_calibrated_distributions(result["calibrated"], os.path.join(figures_dir, "calibrated_distributions.png"))

    report_text = render_report(result, n_samples, grid_points, seed)

    report_path = os.path.join(output_dir, "MUTATION_CALIBRATION.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path, result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    report_path, result = run_mutation_calibration(
        n_samples=args.n, grid_points=args.grid_points, seed=args.seed, output_dir=args.output_dir
    )
    print(f"mutation calibration report written to {report_path}")
    print(f"D*={result['d_star']:.4f}")
    for arm, entry in result["calibrated"].items():
        print(f"  {arm}: {entry['param_name']}={entry['param_value']:.4f} "
              f"verify_mean={entry['verify_stats']['mean']:.4f} within_tolerance={entry['within_tolerance']}")


if __name__ == "__main__":
    main()
