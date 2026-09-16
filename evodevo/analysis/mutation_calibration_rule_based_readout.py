"""PART 3 cascade: quantifies whether rule_based's mutation-displacement
calibration (results/mutation/MUTATION_CALIBRATION.md, D*=4.9915) actually
MOVES under the new mode_of_window readout, before deciding whether a full
re-calibration (like N13's mutation_calibration_nca_readout.py) is
warranted.

For the 88.45% clean-settling population, decoded bodies are IDENTICAL
under fixed_t and mode_of_window (results/rule_based/READOUT_POLICY_
SELECTION.md's equivalence check) -- so displacement can only change for
parent/child pairs that land in the 11.55% cycling population. This module
measures the ACTUAL magnitude of that movement directly rather than
assuming either "negligible" or "large".

Fitness-blind: only decode()/mutate()/random_genome() are used -- no PPO,
no task, no reward, no EA. Does not modify rule_based's rule vocabulary,
interpreter, or genome representation.

Usage:
    python -m evodevo.analysis.mutation_calibration_rule_based_readout --w 2
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evodevo.analysis.mutation_calibration import (
    DEFAULT_N,
    DISPLACEMENT_TOLERANCE,
    INIT_CONDITION,
    PARAM_GRIDS,
    T_PRIMARY,
    _distribution_stats,
    sample_displacements,
)
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

RESULTS_DIR = "results/rule_based"
OLD_D_STAR = 4.9915  # results/mutation/mutation_calibration_raw.json, UNCHANGED by this module
OLD_RULE_BASED_RATE = 0.1707  # results/mutation/mutation_calibration_raw.json's committed solved value


def _build_encoder(readout_mode: str, readout_window: int, seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=T_PRIMARY, init_condition=INIT_CONDITION, seed=seed)
    return RuleBasedEncoder(
        factors=factors,
        mutation_rate=OLD_RULE_BASED_RATE,
        readout_mode=readout_mode,
        readout_window=readout_window,
    )


def measure_displacement_movement(w: int, n: int = DEFAULT_N, seed: int = 0, verify_seed: int = 10_000) -> dict:
    fixed_encoder = _build_encoder("fixed_t", 5, seed)
    windowed_encoder = _build_encoder("mode_of_window", w, seed)

    rng_fixed = np.random.default_rng(verify_seed)
    rng_windowed = np.random.default_rng(verify_seed)

    fixed_distances = sample_displacements(fixed_encoder, n, rng_fixed)
    windowed_distances = sample_displacements(windowed_encoder, n, rng_windowed)

    fixed_stats = _distribution_stats(fixed_distances)
    windowed_stats = _distribution_stats(windowed_distances)

    mean_delta = windowed_stats["mean"] - fixed_stats["mean"]
    return {
        "w": w,
        "n": n,
        "t_primary": T_PRIMARY,
        "fixed_stats": fixed_stats,
        "windowed_stats": windowed_stats,
        "mean_delta": mean_delta,
        "mean_delta_within_tolerance": abs(mean_delta) <= DISPLACEMENT_TOLERANCE,
        "old_d_star": OLD_D_STAR,
    }


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(result: dict) -> str:
    lines = []
    lines.append("# rule_based mutation-displacement: does D* move under mode_of_window?")
    lines.append("")
    lines.append(
        "Fitness-blind (only decode()/mutate()/random_genome() are used -- NO PPO, NO task, NO reward, NO "
        f"EA). N={result['n']}, dev_steps={result['t_primary']} (mutation_calibration.py's own frozen "
        f"T_PRIMARY, unchanged), W={result['w']}, mutation_rate held at rule_based's already-committed "
        f"solved value ({OLD_RULE_BASED_RATE}) so this ISOLATES the readout-mode effect on displacement, "
        "not a re-solve of the rate itself."
    )
    lines.append("")
    lines.append("| readout | mean | median | std | iqr |")
    lines.append("| --- | --- | --- | --- | --- |")
    fs, ws = result["fixed_stats"], result["windowed_stats"]
    lines.append(f"| fixed_t | {_fmt(fs['mean'])} | {_fmt(fs['median'])} | {_fmt(fs['std'])} | {_fmt(fs['iqr'])} |")
    lines.append(f"| mode_of_window (W={result['w']}) | {_fmt(ws['mean'])} | {_fmt(ws['median'])} | {_fmt(ws['std'])} | {_fmt(ws['iqr'])} |")
    lines.append("")
    lines.append(
        f"**mean displacement delta = {_fmt(result['mean_delta'])} cells** "
        f"({'within' if result['mean_delta_within_tolerance'] else 'EXCEEDS'} the calibration's own "
        f"{DISPLACEMENT_TOLERANCE}-cell tolerance)."
    )
    lines.append("")
    if result["mean_delta_within_tolerance"]:
        lines.append(
            "**Verdict: no full re-calibration needed.** The measured movement is within the calibration's "
            f"own tolerance band -- D*={OLD_D_STAR} and rule_based's solved `mutation_rate="
            f"{OLD_RULE_BASED_RATE}` remain valid under mode_of_window without re-solving; the 11.55% "
            "cycling population's displacement contribution is not large enough to shift the mean "
            "meaningfully at N=2000."
        )
    else:
        lines.append(
            "**Verdict: displacement moves enough to warrant a re-calibration**, mirroring N13's "
            "`mutation_calibration_nca_readout.py` (re-solve rule_based's `mutation_rate` ONLY, holding D* "
            f"fixed at {OLD_D_STAR}, direct's and NCA's solved values untouched)."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, required=True, help="rule_based's resolved W (from Part 1's sweep)")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "MUTATION_DISPLACEMENT_CHECK.md"))
    args = parser.parse_args()

    result = measure_displacement_movement(w=args.w, n=args.n)
    report = render_report(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
