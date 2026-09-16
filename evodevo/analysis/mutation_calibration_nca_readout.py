"""PART 5 cascade: re-solves NCA's `mutation_sigma` ONLY, under the new
readout policy (`resolved_config.py::PRIMARY.readout_mode="mode_of_window"`),
holding D* FIXED at the value the original joint N6 calibration already
committed to (`results/mutation/mutation_calibration_raw.json::d_star`).

Does NOT re-run the joint three-arm D* derivation
(`evodevo.analysis.mutation_calibration.calibrate`) -- that procedure picks
D* from the INTERSECTION of all three arms' achievable displacement ranges,
so re-running it would also change direct's and rule_based's already-solved
`mutation_rate` values purely because NCA's own readout changed, which
directly violates this task's "MUST NOT touch direct's or rule_based's
encoders or calibrations" constraint. Holding D* fixed and re-solving only
NCA's own parameter isolates the readout-policy effect exactly the same way
`mutation_calibration.py`'s own T_PRIMARY=20 freeze isolates the N2/N4/N5
architecture effect (see that module's docstring).

Fitness-blind: only decode()/mutate()/validate_body are used -- no PPO, no
task, no reward, no EA.

Usage:
    python -m evodevo.analysis.mutation_calibration_nca_readout
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evodevo.analysis.mutation_calibration import (
    DEFAULT_GRID_POINTS,
    DEFAULT_N,
    DISPLACEMENT_TOLERANCE,
    INIT_CONDITION,
    PARAM_GRIDS,
    T_PRIMARY,
    _distribution_stats,
    _interpolate_param_for_target,
    sample_displacements,
)
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.nca.resolved_config import PRIMARY

RESULTS_DIR = "results/nca"
OLD_RAW_PATH = "results/mutation/mutation_calibration_raw.json"


def _build_windowed_nca_encoder(param_value: float, seed: int) -> NCAEncoder:
    """Same construction as mutation_calibration.py's own `_build_encoder`
    for arm='nca', except readout_mode/readout_window come from PRIMARY (now
    mode_of_window) instead of NCAEncoder's fixed_t default -- kept as a
    SEPARATE builder rather than modifying `_build_encoder` itself, because
    that function is shared by the original joint N6 calibration, which
    must stay byte-identical to its committed fixed_t-based result
    (verified by tests/analysis/test_mutation_calibration.py)."""
    factors = ExperimentFactors(
        dev_steps=T_PRIMARY, init_condition=INIT_CONDITION, seed=seed, hidden_channels=PRIMARY.hidden_channels
    )
    return NCAEncoder(
        factors=factors,
        mutation_sigma=param_value,
        update_rule=PRIMARY.update_rule,
        decay=PRIMARY.decay,
        state_bound=PRIMARY.state_bound,
        readout_mode=PRIMARY.readout_mode,
        readout_window=PRIMARY.readout_window,
    )


def _run_sweep_windowed(n_samples: int, grid_points: int, seed: int) -> dict:
    lo, hi = PARAM_GRIDS["nca"]
    grid = np.linspace(lo, hi, grid_points)
    rows = []
    for param in grid:
        encoder = _build_windowed_nca_encoder(float(param), seed)
        rng = np.random.default_rng(seed)
        distances = sample_displacements(encoder, n_samples, rng)
        stats = _distribution_stats(distances)
        stats["param"] = float(param)
        rows.append(stats)
    return {"arm": "nca", "param_name": "mutation_sigma", "grid": grid.tolist(), "rows": rows}


def _load_old_d_star_and_sigma() -> tuple:
    with open(OLD_RAW_PATH) as f:
        old = json.load(f)
    return old["d_star"], old["calibrated"]["nca"]["param_value"], old["calibrated"]["nca"]["verify_stats"]["mean"]


def resolve_nca_sigma_for_fixed_d_star(
    d_star: float, n_samples: int = DEFAULT_N, grid_points: int = DEFAULT_GRID_POINTS, seed: int = 0
) -> dict:
    verify_seed = seed + 10_000
    arm = "nca"
    sweep = _run_sweep_windowed(n_samples, grid_points, seed)
    param = _interpolate_param_for_target(sweep, d_star)
    param = float(np.clip(param, *PARAM_GRIDS[arm]))

    lo, hi = PARAM_GRIDS[arm]
    history = [(row["param"], row["mean"]) for row in sweep["rows"]]
    best_param, best_stats, best_gap = param, None, float("inf")

    for _ in range(6):
        encoder = _build_windowed_nca_encoder(param, seed)
        rng = np.random.default_rng(seed)
        probe_distances = sample_displacements(encoder, n_samples, rng)
        probe_mean = float(probe_distances.mean())
        history.append((param, probe_mean))
        gap = abs(probe_mean - d_star)
        if gap < best_gap:
            best_param, best_stats, best_gap = param, probe_distances, gap
        if gap <= DISPLACEMENT_TOLERANCE:
            break
        below = [(p, m) for p, m in history if m <= d_star]
        above = [(p, m) for p, m in history if m >= d_star]
        if below and above:
            p_lo, m_lo = max(below, key=lambda pm: pm[1])
            p_hi, m_hi = min(above, key=lambda pm: pm[1])
            param = p_lo + (d_star - m_lo) * (p_hi - p_lo) / (m_hi - m_lo) if m_hi > m_lo else (p_lo + p_hi) / 2
        param = float(np.clip(param, lo, hi))

    encoder = _build_windowed_nca_encoder(best_param, verify_seed)
    rng = np.random.default_rng(verify_seed)
    verify_distances = sample_displacements(encoder, n_samples, rng)
    verify_stats = _distribution_stats(verify_distances)
    within_tolerance = abs(verify_stats["mean"] - d_star) <= DISPLACEMENT_TOLERANCE

    return {
        "arm": arm,
        "readout_window": PRIMARY.readout_window,
        "d_star": d_star,
        "param_name": "mutation_sigma",
        "param_value": best_param,
        "verify_stats": verify_stats,
        "within_tolerance": within_tolerance,
        "sweep": sweep,
    }


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(result: dict, old_sigma: float, old_verify_mean: float) -> str:
    lines = []
    lines.append("# NCA mutation_sigma re-solve under the mode_of_window readout policy (N6 readout cascade)")
    lines.append("")
    lines.append(
        "Fitness-blind (only decode()/mutate()/validate_body are used -- NO PPO, NO task, NO reward, NO EA). "
        "Re-solves NCA's `mutation_sigma` ONLY, holding D* FIXED at the value the original joint N6 "
        f"calibration already committed to (D*={_fmt(result['d_star'])}, "
        "`results/mutation/mutation_calibration_raw.json`, UNCHANGED). direct's and rule_based's own solved "
        "parameters are NOT re-derived here and remain exactly as committed."
    )
    lines.append("")
    lines.append("## Before / after (readout-level, NOT the N10 architecture-level table)\n")
    lines.append("| | mutation_sigma | verify_stats.mean | within_tolerance |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| fixed-T-argmax-at-T=20 (previous, `readout_mode=fixed_t`) | {_fmt(old_sigma)} | "
        f"{_fmt(old_verify_mean)} | True |"
    )
    lines.append(
        f"| mode-of-window-at-T=20,W={result.get('readout_window', 50)} (new, "
        f"`readout_mode=mode_of_window`) | {_fmt(result['param_value'])} | "
        f"{_fmt(result['verify_stats']['mean'])} | {result['within_tolerance']} |"
    )
    lines.append("")
    delta = result["param_value"] - old_sigma
    lines.append(
        f"delta mutation_sigma = {_fmt(delta)} ({_fmt(100*delta/old_sigma, 2)}% relative to the old value). "
        "direct's mutation_rate and rule_based's mutation_rate are UNCHANGED (not re-run) -- this table is "
        "readout-level for nca only, dev_steps held at T_PRIMARY=20 exactly as the original N6 module's own "
        "controlled-comparison convention specifies (see mutation_calibration.py's T_PRIMARY docstring)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default=os.path.join(RESULTS_DIR, "MUTATION_CALIBRATION_READOUT_CASCADE.md")
    )
    args = parser.parse_args()

    d_star, old_sigma, old_verify_mean = _load_old_d_star_and_sigma()
    result = resolve_nca_sigma_for_fixed_d_star(d_star, n_samples=args.n, grid_points=args.grid_points, seed=args.seed)
    report = render_report(result, old_sigma, old_verify_mean)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
