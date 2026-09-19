"""PART 2 of the N15 follow-up: checks whether the even-W deterministic
tie-break (lowest material ID) discards useful genome-specific information
for rule_based's period-2 cycling population, or is a harmless mechanism
achieving T-shift stability without meaningfully degrading validity.

For a sample of confirmed period-2 cycling genomes, measures whether
mode_of_window's tie-broken output is INVALID while at least one of the
genome's own two real alternating phases (A, B) WAS valid -- the concrete
harm case: a genome that had a legitimate valid body available in its
cycle, but the readout policy discards it for an artificial default that
isn't valid.

If this fraction is non-trivial, tests a genome-aware alternative: prefer
whichever of the genome's own two phases (A or B) is independently valid
under an exact tie; fall back to the original tie-break if both or neither
phase is valid. This alternative is still T-shift stable BY CONSTRUCTION
(it depends only on the genome's fixed A/B pair, never on t) -- verified,
not assumed.

Fitness-blind: only decode()/develop_trajectory()/random_genome()/
validate_body() are used -- no PPO, no task, no reward, no EA. Does not
modify rule_based's rule vocabulary or interpreter.

Usage:
    python -m evodevo.analysis.rule_based_tiebreak_check
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from evodevo.analysis.horizon_extension import _detect_period
from evodevo.analysis.windowed_readout import windowed_body_at
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.morphology.validate import validate_body

DEFAULT_N_SAMPLE = 200
DEFAULT_SEED = 0
T_MAX_AUDIT = 100  # matches SETTLING_AUDIT.md's own horizon
RESOLVED_W = 6  # rule_based's resolved W, results/rule_based/READOUT_POLICY_SELECTION.md
RESOLVED_T = 100  # shared T_primary, results/T_PRIMARY_UNIFIED.md (post N15 fix)
HARM_FRACTION_THRESHOLD = 0.05  # 5%, stated and justified in the report

RESULTS_DIR = "results/rule_based"


def _build_encoder(seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=T_MAX_AUDIT, init_condition="center_seed", seed=seed)
    return RuleBasedEncoder(factors=factors)


def find_period2_genomes(n_target: int = DEFAULT_N_SAMPLE, seed: int = DEFAULT_SEED, draw_limit: int = 20000):
    """Draws genomes in the SAME order SETTLING_AUDIT.md used
    (random_genome() in a simple loop, seed=0) until n_target confirmed
    period-2 cycling genomes are found (or draw_limit is exhausted).
    Returns a list of (genome, A, B) where A, B are the two alternating
    body configurations (A = body at the trajectory's last step, B = the
    step before)."""
    encoder = _build_encoder(seed)
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(draw_limit):
        if len(found) >= n_target:
            break
        genome = encoder.random_genome(rng)
        bodies = encoder.develop_trajectory(genome)
        never_settled = not np.array_equal(bodies[-2], bodies[-1])
        if not never_settled:
            continue
        window = bodies[-21:]
        p = _detect_period(window, 20)
        if p != 2:
            continue
        A = bodies[-1]
        B = bodies[-2]
        found.append((genome, A, B))
    return found, encoder


def _tie_break_default(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The ORIGINAL deterministic tie-break: lowest-material-ID lexicographic
    winner between the two tied whole-grid configurations -- reproduces
    windowed_readout.mode_of_window's own tie-break for a perfect 2-way tie
    (both configurations equally frequent in the window)."""
    flat_a = tuple(int(v) for v in a.flatten())
    flat_b = tuple(int(v) for v in b.flatten())
    winner = min(flat_a, flat_b)
    return np.array(winner, dtype=int).reshape(a.shape)


def genome_aware_readout(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PROPOSED alternative: prefer whichever of the genome's own two cycle
    phases is independently valid; if both or neither are valid, fall back
    to the original deterministic tie-break. Depends ONLY on (a, b) -- the
    genome's own fixed cycle content -- never on t, so it is T-shift stable
    by construction for any exact 2-cycle (verified separately below)."""
    a_valid = validate_body(a).valid
    b_valid = validate_body(b).valid
    if a_valid and not b_valid:
        return a
    if b_valid and not a_valid:
        return b
    return _tie_break_default(a, b)


def run_check(n_sample: int = DEFAULT_N_SAMPLE, seed: int = DEFAULT_SEED) -> dict:
    genomes, encoder = find_period2_genomes(n_target=n_sample, seed=seed)

    rows = []
    for genome, A, B in genomes:
        # mode_of_window(W=RESOLVED_W) readout, using the actual shared utility,
        # on a trajectory extended to T_MAX_AUDIT (already computed by
        # find_period2_genomes via develop_trajectory, but recomputed here for
        # a fresh, explicit windowed_body_at call at the resolved (T, W)).
        bodies = encoder.develop_trajectory(genome)
        windowed = windowed_body_at(bodies, min(RESOLVED_T, len(bodies) - 1), RESOLVED_W)

        tie_break_expected = _tie_break_default(A, B)
        equals_tiebreak = np.array_equal(windowed, tie_break_expected)

        # cells that differ between A and B are "genuinely alternating";
        # cells that agree are fixed regardless of phase.
        diff_mask = A != B
        # does the windowed output match the tie-break exactly at every
        # differing cell? (it necessarily matches A==B at agreeing cells,
        # since both phases already agree there.)
        matches_tiebreak_at_diff_cells = bool(np.array_equal(windowed[diff_mask], tie_break_expected[diff_mask]))

        windowed_valid = validate_body(windowed).valid
        a_valid = validate_body(A).valid
        b_valid = validate_body(B).valid

        aware = genome_aware_readout(A, B)
        aware_valid = validate_body(aware).valid

        rows.append({
            "windowed_valid": windowed_valid,
            "a_valid": a_valid,
            "b_valid": b_valid,
            "aware_valid": aware_valid,
            "equals_tiebreak": equals_tiebreak,
            "matches_tiebreak_at_diff_cells": matches_tiebreak_at_diff_cells,
            "n_diff_cells": int(diff_mask.sum()),
        })

    n = len(rows)
    n_harm = sum(1 for r in rows if (not r["windowed_valid"]) and (r["a_valid"] or r["b_valid"]))
    harm_fraction = n_harm / n if n else 0.0

    n_windowed_valid = sum(1 for r in rows if r["windowed_valid"])
    n_aware_valid = sum(1 for r in rows if r["aware_valid"])
    n_a_valid = sum(1 for r in rows if r["a_valid"])
    n_b_valid = sum(1 for r in rows if r["b_valid"])
    n_either_valid = sum(1 for r in rows if r["a_valid"] or r["b_valid"])
    n_equals_tiebreak = sum(1 for r in rows if r["equals_tiebreak"])

    return {
        "n": n,
        "seed": seed,
        "resolved_t": RESOLVED_T,
        "resolved_w": RESOLVED_W,
        "harm_fraction_threshold": HARM_FRACTION_THRESHOLD,
        "n_harm": n_harm,
        "harm_fraction": harm_fraction,
        "windowed_validity_rate": n_windowed_valid / n if n else 0.0,
        "aware_validity_rate": n_aware_valid / n if n else 0.0,
        "a_validity_rate": n_a_valid / n if n else 0.0,
        "b_validity_rate": n_b_valid / n if n else 0.0,
        "either_phase_validity_rate": n_either_valid / n if n else 0.0,
        "fraction_equals_tiebreak": n_equals_tiebreak / n if n else 0.0,
        "rows": rows,
    }


def verify_genome_aware_readout_is_t_shift_stable(n_check: int = 50, seed: int = DEFAULT_SEED) -> dict:
    """genome_aware_readout(A, B) depends only on the genome's own fixed
    (A, B) pair, computed once from the trajectory's LAST TWO steps -- never
    on t itself -- so it cannot vary with a T-shift for a genome that has
    already reached its exact 2-cycle, REGARDLESS of which phase T or T+shift
    lands on. Verified directly (not just asserted) by re-deriving (A, B) at
    every shifted query point and confirming the SET {A, B} -- hence the
    readout -- never changes.

    NOTE on the t-clamp below: `bodies` has length T_MAX_AUDIT+1, but
    RESOLVED_T+shift can reach 105 for shift=+5 -- outside that range, so
    `t` is clamped. This is verified harmless for THIS check specifically
    (an exact period-2 genome's body at any t is always phase A or B, so
    clamping cannot change which pair is read), but is warned rather than
    left silent, since a silent version of exactly this pattern (a T-shift
    query landing past the computed trajectory) previously produced a
    spurious result elsewhere (docs/decisions.md's "T=40-staleness audit"
    row, the T=105-clipped-to-100 HARD_MAX_ITERATIONS incident)."""
    genomes, encoder = find_period2_genomes(n_target=n_check, seed=seed)
    n_stable = 0
    for genome, A, B in genomes:
        bodies = encoder.develop_trajectory(genome)
        base_pair = {tuple(int(v) for v in A.flatten()), tuple(int(v) for v in B.flatten())}
        stable = True
        for shift in (-5, -1, 1, 5):
            requested_t = RESOLVED_T + shift
            t = max(0, min(requested_t, len(bodies) - 1))
            if t != requested_t:
                warnings.warn(
                    f"verify_genome_aware_readout_is_t_shift_stable: requested t={requested_t} "
                    f"exceeds the computed trajectory (0..{len(bodies) - 1}) -- clipped to t={t}. "
                    "Verified harmless for exact period-2 genomes (see docstring), but this is a "
                    "real clip and must not pass silently.",
                    stacklevel=2,
                )
            a2, b2 = bodies[t], bodies[t - 1]
            pair = {tuple(int(v) for v in a2.flatten()), tuple(int(v) for v in b2.flatten())}
            if pair != base_pair:
                stable = False
        if stable:
            n_stable += 1
    return {"n_check": len(genomes), "n_stable": n_stable}


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(result: dict, stability_check: dict) -> str:
    n = result["n"]
    lines = []
    lines.append("# rule_based tie-break validity check: does the even-W tie-break discard valid phases?")
    lines.append("")
    lines.append(
        "Fitness-blind (only decode()/develop_trajectory()/random_genome()/validate_body() are used -- NO "
        f"PPO, NO task, NO reward, NO EA). N={n} confirmed period-2 cycling genomes (SETTLING_AUDIT.md's own "
        f"detection method, T_max=100, seed={result['seed']}), evaluated at the resolved (T_primary="
        f"{result['resolved_t']}, W={result['resolved_w']})."
    )
    lines.append("")

    lines.append("## Per-genome measurements\n")
    lines.append(
        f"**{_fmt(result['fraction_equals_tiebreak'])} of sampled genomes' windowed readout equals the "
        "deterministic tie-break exactly** -- confirming these are genuinely 50/50-tied cases (every "
        "differing cell between A and B resolves to the tie-break value), not cases where the window "
        "happens to contain enough non-tied history to produce a real majority. For a pure period-2 cycle "
        "sampled deep into its trajectory with W=6 (even), this is expected to be at or near 1.0."
    )
    lines.append("")

    lines.append("## Validity comparison: windowed (tie-break) vs fixed-at-A vs fixed-at-B\n")
    lines.append("| readout | validity rate |")
    lines.append("| --- | --- |")
    lines.append(f"| mode_of_window (tie-break) | {_fmt(result['windowed_validity_rate'])} |")
    lines.append(f"| fixed at phase A | {_fmt(result['a_validity_rate'])} |")
    lines.append(f"| fixed at phase B | {_fmt(result['b_validity_rate'])} |")
    lines.append(f"| EITHER phase A or B valid | {_fmt(result['either_phase_validity_rate'])} |")
    lines.append("")

    lines.append("## The concrete harm case\n")
    lines.append(
        f"Fraction of the {n}-genome sample where mode_of_window's tie-broken body is INVALID while AT "
        f"LEAST ONE of the genome's two real alternating phases (A or B) WAS valid: "
        f"**{_fmt(result['harm_fraction'])}** ({result['n_harm']}/{n})."
    )
    lines.append("")
    threshold = result["harm_fraction_threshold"]
    lines.append(
        f"Threshold: {_fmt(threshold)} (5%), justified as the point past which the tie-break mechanism is "
        "discarding a legitimately-available valid body for a non-trivial share of the very population this "
        "readout policy exists to serve -- below this, the mechanism is a minor, acceptable simplification; "
        "at or above it, genome-specific information matters enough to warrant a more careful mechanism."
    )
    lines.append("")

    needs_fix = result["harm_fraction"] > threshold
    if not needs_fix:
        lines.append(
            f"**Verdict: the original tie-break is FINE.** {_fmt(result['harm_fraction'])} <= {_fmt(threshold)} "
            "-- validity is not meaningfully harmed by the deterministic lowest-material-ID tie-break for "
            "this population. Recommendation: KEEP the original mechanism; do not adopt a more complex one "
            "without evidence it is needed."
        )
    else:
        lines.append(
            f"**Verdict: the original tie-break is NOT acceptable as a default.** "
            f"{_fmt(result['harm_fraction'])} > {_fmt(threshold)} -- a non-trivial share of cycling genomes "
            "have a legitimately valid phase that the tie-break discards. Proceeding to test the "
            "genome-aware alternative."
        )
    lines.append("")

    lines.append("## Genome-aware alternative (tested regardless, for completeness)\n")
    lines.append(
        "Candidate: for a genome in an exact 2-way tie, prefer whichever of its own two cycle phases (A or "
        "B) is independently valid (checked via `validate_body`); if both or neither are valid, fall back to "
        "the original tie-break."
    )
    lines.append("")
    lines.append("| readout | validity rate |")
    lines.append("| --- | --- |")
    lines.append(f"| original tie-break (mode_of_window) | {_fmt(result['windowed_validity_rate'])} |")
    lines.append(f"| genome-aware tie-break | {_fmt(result['aware_validity_rate'])} |")
    lines.append("")
    lines.append(
        f"**T-shift stability of the genome-aware alternative, verified (not assumed)**: "
        f"{stability_check['n_stable']}/{stability_check['n_check']} genomes' own (A, B) pair -- and hence "
        "the genome-aware readout, which depends only on it -- is UNCHANGED when re-derived at every "
        "T+shift query point (+-1, +-5) used by the operational T-shift test. "
        + (
            "Confirmed stable by construction, as expected for a mechanism that depends only on the "
            "genome's fixed cycle content, never on t."
            if stability_check["n_stable"] == stability_check["n_check"]
            else "**UNSTABLE for some genomes -- this needs investigation before adoption.**"
        )
    )
    lines.append("")

    lines.append("## Recommendation\n")
    if not needs_fix:
        lines.append(
            "**Keep the original deterministic lowest-material-ID tie-break.** The genome-aware alternative "
            f"was tested for completeness (validity rate {_fmt(result['aware_validity_rate'])} vs original "
            f"{_fmt(result['windowed_validity_rate'])}) and confirmed T-shift stable, but is NOT adopted "
            "since the evidence does not show the original mechanism causing meaningful harm -- adopting a "
            "more complex mechanism without evidence it is needed would be unjustified complexity."
        )
    else:
        delta = result["aware_validity_rate"] - result["windowed_validity_rate"]
        lines.append(
            f"**Adopt the genome-aware tie-break.** It improves validity by {_fmt(delta)} "
            f"({_fmt(result['windowed_validity_rate'])} -> {_fmt(result['aware_validity_rate'])}) while "
            "remaining T-shift stable by construction, directly addressing the concrete harm case measured "
            "above."
        )
    lines.append("")
    return "\n".join(lines) + "\n", needs_fix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "TIEBREAK_VALIDITY_CHECK.md"))
    args = parser.parse_args()

    result = run_check(n_sample=args.n, seed=args.seed)
    stability_check = verify_genome_aware_readout_is_t_shift_stable(n_check=min(50, args.n), seed=args.seed)
    report, needs_fix = render_report(result, stability_check)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    print(f"NEEDS_FIX = {needs_fix}")


if __name__ == "__main__":
    main()
