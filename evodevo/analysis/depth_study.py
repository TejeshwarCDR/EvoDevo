"""Establishes what developmental depth T means across the NCA and
rule-based substrates, and selects T on measured evidence -- see
docs/decisions.md's "Developmental steps (T)" row for the resulting policy.

Three layers, all fitness-blind (evodevo.analysis.structural only; NO PPO,
NO task, NO reward, NO EA -- see
tests/analysis/test_structural_no_ppo_imports.py, which also covers this
module):

  Layer 1 -- VERIFY THE INVARIANT: is "one step = one cell of reach" true
  for both arms (light_cone)? If not, equal T is not equal reach.

  Layer 2 -- SELECT T FROM MEASURED CONVERGENCE: settling_profile at a hard
  T_max=40 gives each arm's own convergence horizon; T_primary is the max
  over arms (asymmetric-loss argument, reproduced verbatim in the report),
  clipped only to a floor of 4 (the geometric minimum) and a ceiling of
  T_max=40 (developmental steps are forward passes evaluated before PPO
  training starts, so there is no compute justification for any tighter
  ceiling).

  Layer 3 -- T AS A CROSSED FACTOR: re-run the full P2 structural suite at
  T in {1, 4, T_primary, min(2*T_primary, 40)} for both arms.

Usage:
    python -m evodevo.analysis.depth_study [--n 2000] [--init-condition center_seed] [--seed 0]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

ARMS = ["nca", "rule_based"]
DEFAULT_N = 2000
DEFAULT_INIT_CONDITION = "center_seed"  # the P3 decision (docs/decisions.md)
DEFAULT_OUTPUT_DIR = "results/depth"
T_MAX_HARD_CAP = 40
GEOMETRIC_BOUND_FROM_CENTER = 4  # max Manhattan distance, 5x5 grid centre -> any cell
GEOMETRIC_BOUND_CORNER_TO_CORNER = 8  # max Manhattan distance, any cell -> any cell
NEVER_SETTLED_PROVISIONAL_THRESHOLD = 0.5  # ">half never settle" counts as "does not settle"
SPEED_DIFFERENCE_THRESHOLD = 0.20  # 20%

ASYMMETRIC_LOSS_PARAGRAPH = """\
The cost of mis-specifying T is asymmetric. Granting an arm more steps than it
needs is semantically free, because once development reaches a fixed point further
steps are idempotent and cannot change the phenotype. Granting an arm fewer steps
than it needs is semantically destructive, because it truncates development and
evaluates an unfinished phenotype. Under an asymmetric loss, the optimum is the
maximum of the arms' convergence horizons, not a compromise between them.

This argument holds only for arms whose development CONVERGES. For an arm whose
state diverges rather than settling, extra steps are not free, and the argument
does not apply until that divergence is resolved."""


def _fmt(value, digits=4):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_encoder(arm: str, dev_steps: int, init_condition: str, seed: int):
    factors = ExperimentFactors(dev_steps=dev_steps, init_condition=init_condition, seed=seed)
    return get_encoder(arm, factors=factors), factors


# ---------------------------------------------------------------------------
# LAYER 1: propagation speed / the reach invariant
# ---------------------------------------------------------------------------


def _light_cone_raw_pairs(encoder, n_samples, rng, t_max):
    """Replicates structural.light_cone's inner loop but returns every
    (distance, step) pair actually observed (reached only), for a full
    distribution rather than only the mean-by-distance/fitted-slope
    aggregate structural.light_cone() itself returns."""
    rows, cols = encoder.grid_shape
    perturb_cell = (rows // 2, cols // 2)
    distances = {
        (r, c): abs(r - perturb_cell[0]) + abs(c - perturb_cell[1])
        for r in range(rows) for c in range(cols)
    }

    pairs = []  # (distance, step)
    total_other_cells = 0
    never_reached = 0

    for _ in range(n_samples):
        genome = encoder.random_genome(rng)

        if isinstance(encoder, NCAEncoder):
            ref_bodies, _, _ = S._nca_trajectory_from_state(encoder, genome, encoder._initial_state, t_max)
            pert_state0 = S._perturbed_initial_state_nca(encoder, perturb_cell)
            pert_bodies, _, _ = S._nca_trajectory_from_state(encoder, genome, pert_state0, t_max)
        elif isinstance(encoder, RuleBasedEncoder):
            ref_grid0 = encoder._initial_condition.as_discrete_grid(encoder.grid_shape)
            ref_bodies, _ = S._rule_based_trajectory_from_grid(encoder, genome, ref_grid0, t_max)
            pert_grid0 = S._perturbed_initial_grid_rule_based(encoder, perturb_cell)
            pert_bodies, _ = S._rule_based_trajectory_from_grid(encoder, genome, pert_grid0, t_max)
        else:
            ref_bodies = [encoder.decode(genome)]
            pert_bodies = [encoder.decode(S._perturbed_genome_direct(encoder, genome, perturb_cell))]

        first_diff = S._first_diff_steps(ref_bodies, pert_bodies)
        for (r, c), dist in distances.items():
            if (r, c) == perturb_cell:
                continue
            total_other_cells += 1
            step = first_diff[r, c]
            if step is None:
                never_reached += 1
            else:
                pairs.append((dist, int(step)))

    return pairs, never_reached, total_other_cells


def run_layer1(n_samples, init_condition, seed, output_dir):
    results = {}
    for arm in ARMS:
        encoder, factors = _build_encoder(arm, dev_steps=T_MAX_HARD_CAP, init_condition=init_condition, seed=seed)
        rng = np.random.default_rng(seed)
        agg = S.light_cone(encoder, factors, n_samples, rng, t_max=T_MAX_HARD_CAP,
                            output_root=os.path.join(output_dir, "raw", "layer1"))

        rng2 = np.random.default_rng(seed)
        pairs, never_reached, total = _light_cone_raw_pairs(encoder, n_samples, rng2, T_MAX_HARD_CAP)
        steps = np.array([s for _, s in pairs])
        distances = np.array([d for d, _ in pairs])
        per_pair_speed = distances / np.maximum(steps, 1)  # cells per step, per observed (cell, genome) pair

        results[arm] = {
            "aggregate": agg,
            "n_reached_pairs": len(pairs),
            "never_reached_fraction": never_reached / total,
            "step_distribution": {
                "min": int(steps.min()) if len(steps) else None,
                "p10": float(np.percentile(steps, 10)) if len(steps) else None,
                "median": float(np.median(steps)) if len(steps) else None,
                "p90": float(np.percentile(steps, 90)) if len(steps) else None,
                "max": int(steps.max()) if len(steps) else None,
                "mean": float(steps.mean()) if len(steps) else None,
                "std": float(steps.std()) if len(steps) else None,
            },
            "per_pair_speed_distribution": {
                "min": float(per_pair_speed.min()) if len(per_pair_speed) else None,
                "p10": float(np.percentile(per_pair_speed, 10)) if len(per_pair_speed) else None,
                "median": float(np.median(per_pair_speed)) if len(per_pair_speed) else None,
                "p90": float(np.percentile(per_pair_speed, 90)) if len(per_pair_speed) else None,
                "max": float(per_pair_speed.max()) if len(per_pair_speed) else None,
                "mean": float(per_pair_speed.mean()) if len(per_pair_speed) else None,
                "std": float(per_pair_speed.std()) if len(per_pair_speed) else None,
            },
        }
    return results


def _layer1_section(layer1: dict, n_samples: int, init_condition: str) -> str:
    lines = ["## Layer 1: verifying the reach invariant\n"]
    lines.append(
        f"`light_cone` measurement at the P3 initial condition (`{init_condition}` -- a single "
        f"seed at the grid centre), N={n_samples} genomes/arm, T_max={T_MAX_HARD_CAP}.\n"
    )
    lines.append(
        "| arm | fitted speed (cells/step) | never_reached_fraction | step p10 | step median | step p90 | "
        "per-pair speed p10 | per-pair speed median | per-pair speed p90 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        r = layer1[arm]
        sd = r["step_distribution"]
        pd = r["per_pair_speed_distribution"]
        lines.append(
            f"| {arm} | {_fmt(r['aggregate'].propagation_speed_cells_per_step)} | "
            f"{_fmt(r['never_reached_fraction'])} | {_fmt(sd['p10'])} | {_fmt(sd['median'])} | "
            f"{_fmt(sd['p90'])} | {_fmt(pd['p10'])} | {_fmt(pd['median'])} | {_fmt(pd['p90'])} |"
        )
    lines.append("")

    speed_nca = layer1["nca"]["aggregate"].propagation_speed_cells_per_step
    speed_rb = layer1["rule_based"]["aggregate"].propagation_speed_cells_per_step

    lines.append(
        f"**Hypothesis: NCA's effective speed is below 1.0 cell/step** (a small residual delta arriving "
        f"from a neighbour may need several steps to accumulate enough magnitude to flip an argmax). "
        f"Measured: nca fitted speed = {_fmt(speed_nca)} cells/step "
        f"(median per-pair speed {_fmt(layer1['nca']['per_pair_speed_distribution']['median'])}), rule_based "
        f"fitted speed = {_fmt(speed_rb)} cells/step "
        f"(median per-pair speed {_fmt(layer1['rule_based']['per_pair_speed_distribution']['median'])}). "
    )

    if speed_nca is not None and speed_nca < 1.0:
        lines.append(
            f"**Confirmed**: nca's fitted speed ({_fmt(speed_nca)}) is below 1.0, consistent with the "
            "accumulate-then-flip hypothesis -- residual deltas from a single neighbouring cell rarely "
            "flip an argmax in exactly one step.\n"
        )
    else:
        lines.append(
            f"**Not confirmed**: nca's fitted speed ({_fmt(speed_nca)}) is not below 1.0 at this initial "
            "condition/sample.\n"
        )

    ratio = None
    if speed_nca and speed_rb:
        ratio = speed_nca / speed_rb
        pct_diff = abs(ratio - 1.0) * 100
        lines.append(
            f"Ratio nca/rule_based = {_fmt(ratio)} ({pct_diff:.1f}% difference)."
        )

    lines.append(
        f"\n**Geometric lower bound.** On a 5x5 grid, the maximum Manhattan distance from the centre to "
        f"any cell is {GEOMETRIC_BOUND_FROM_CENTER}, and corner-to-corner is {GEOMETRIC_BOUND_CORNER_TO_CORNER}. "
        f"The P3 initial condition (`{DEFAULT_INIT_CONDITION}`) seeds development from the grid CENTRE, and "
        f"`light_cone`'s own perturbation is also applied at the centre cell -- so the bound that applies "
        f"here is **T >= {GEOMETRIC_BOUND_FROM_CENTER}** (centre-to-any-cell), not the corner-to-corner bound "
        f"of {GEOMETRIC_BOUND_CORNER_TO_CORNER}, which would only be the relevant bound for a perturbation "
        f"or seed placed at a corner (not the case under the chosen P3 condition or this measurement)."
    )

    speed_correction_needed = ratio is not None and abs(ratio - 1.0) > SPEED_DIFFERENCE_THRESHOLD
    if speed_correction_needed:
        lines.append(
            f"\n\n**Speeds differ by more than {SPEED_DIFFERENCE_THRESHOLD*100:.0f}% -- a speed-corrected "
            "depth sensitivity analysis is required (see Layer 2's T_primary derivation section below, "
            "once T_primary is known).**"
        )
    return "\n".join(lines), speed_correction_needed, speed_nca, speed_rb


# ---------------------------------------------------------------------------
# LAYER 2: select T from measured convergence
# ---------------------------------------------------------------------------


def run_layer2(n_samples, init_condition, seed, output_dir):
    results = {}
    for arm in ARMS:
        encoder, factors = _build_encoder(arm, dev_steps=T_MAX_HARD_CAP, init_condition=init_condition, seed=seed)
        rng = np.random.default_rng(seed)

        settle_steps = []
        never_settled = 0
        magnitude_traces = []  # NCA only
        idempotence_violations = 0
        idempotence_checked = 0

        for _ in range(n_samples):
            genome = encoder.random_genome(rng)
            bodies, magnitudes = S._full_trajectory(encoder, genome, T_MAX_HARD_CAP)
            settle_step, settled = S._settle_info(bodies)

            if settled:
                settle_steps.append(settle_step)
                # Empirical idempotence check: re-verify, from the SAME
                # recorded trajectory, that the body at every step from
                # settle_step to T_max is identical -- not assumed from how
                # settle_step was derived.
                idempotence_checked += 1
                reference = bodies[settle_step]
                for t in range(settle_step, len(bodies)):
                    if not np.array_equal(bodies[t], reference):
                        idempotence_violations += 1
                        break
            else:
                never_settled += 1

            if isinstance(encoder, NCAEncoder):
                magnitude_traces.append(magnitudes)

        settle_arr = np.array(settle_steps) if settle_steps else np.array([])
        never_settled_fraction = never_settled / n_samples

        entry = {
            "n": n_samples,
            "t_max": T_MAX_HARD_CAP,
            "never_settled_count": never_settled,
            "never_settled_fraction": never_settled_fraction,
            "settled_count": n_samples - never_settled,
            "median_settle_step": float(np.median(settle_arr)) if len(settle_arr) else None,
            "p95_settle_step": float(np.percentile(settle_arr, 95)) if len(settle_arr) else None,
            "mean_settle_step": float(settle_arr.mean()) if len(settle_arr) else None,
            "idempotence_checked": idempotence_checked,
            "idempotence_violations": idempotence_violations,
            "stabilize_by_step_1": float(np.mean(settle_arr <= 1)) if len(settle_arr) else None,
            "stabilize_by_step_2": float(np.mean(settle_arr <= 2)) if len(settle_arr) else None,
        }

        if isinstance(encoder, NCAEncoder) and magnitude_traces:
            init_ref_scale = float(np.max(np.abs(encoder._initial_state)))
            steps_to_10x = []
            steps_to_100x = []
            for trace in magnitude_traces:
                s10 = next((i + 1 for i, m in enumerate(trace) if m > 10 * init_ref_scale), None)
                s100 = next((i + 1 for i, m in enumerate(trace) if m > 100 * init_ref_scale), None)
                steps_to_10x.append(s10)
                steps_to_100x.append(s100)
            reached_10x = [s for s in steps_to_10x if s is not None]
            reached_100x = [s for s in steps_to_100x if s is not None]
            final_magnitudes = [trace[-1] if trace else init_ref_scale for trace in magnitude_traces]
            entry["magnitude_analysis"] = {
                "initial_reference_scale": init_ref_scale,
                "fraction_exceeding_10x": len(reached_10x) / len(magnitude_traces),
                "fraction_exceeding_100x": len(reached_100x) / len(magnitude_traces),
                "median_step_exceeding_10x": float(np.median(reached_10x)) if reached_10x else None,
                "median_step_exceeding_100x": float(np.median(reached_100x)) if reached_100x else None,
                "mean_final_magnitude": float(np.mean(final_magnitudes)),
                "median_final_magnitude": float(np.median(final_magnitudes)),
            }

        results[arm] = entry
    return results


def _select_t_primary(layer2: dict):
    nca_never_settled = layer2["nca"]["never_settled_fraction"]
    provisional = nca_never_settled > NEVER_SETTLED_PROVISIONAL_THRESHOLD

    if provisional:
        rb_q95 = layer2["rule_based"]["p95_settle_step"]
        t_primary_raw = max(math.ceil(rb_q95) if rb_q95 is not None else GEOMETRIC_BOUND_FROM_CENTER,
                             GEOMETRIC_BOUND_FROM_CENTER)
    else:
        q95_values = [layer2[arm]["p95_settle_step"] for arm in ARMS if layer2[arm]["p95_settle_step"] is not None]
        t_primary_raw = math.ceil(max(q95_values)) if q95_values else GEOMETRIC_BOUND_FROM_CENTER

    # Floor only, no artificial ceiling: developmental steps are forward
    # passes evaluated before PPO training even starts, so the marginal
    # cost of a larger T_primary is microseconds, not a resource
    # constraint -- there is no compute justification for capping
    # T_primary below T_MAX_HARD_CAP. (An earlier version of this function
    # clipped to [4, 20], which capped T_primary BELOW nca's own q95/median
    # settle_step and contradicted the asymmetric-loss argument the whole
    # selection procedure rests on -- see docs/decisions.md's corrected
    # "Developmental steps (T)" row for the full account of why that was
    # wrong.)
    t_primary = int(min(max(t_primary_raw, GEOMETRIC_BOUND_FROM_CENTER), T_MAX_HARD_CAP))
    return t_primary, provisional


def _layer2_section(layer2: dict, t_primary: int, provisional: bool, n_samples: int, init_condition: str,
                     speed_correction_needed: bool = False, speed_nca=None, speed_rb=None) -> str:
    lines = ["## Layer 2: selecting T from measured convergence\n"]
    lines.append(f"`settling_profile` at T_max={T_MAX_HARD_CAP}, N={n_samples}/arm, init_condition=`{init_condition}`.\n")
    lines.append(
        "| arm | median settle_step | p95 settle_step | never_settled_count | never_settled_fraction | "
        "stabilize_by_step<=1 | stabilize_by_step<=2 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        r = layer2[arm]
        lines.append(
            f"| {arm} | {_fmt(r['median_settle_step'])} | {_fmt(r['p95_settle_step'])} | "
            f"{r['never_settled_count']} | {_fmt(r['never_settled_fraction'])} | "
            f"{_fmt(r['stabilize_by_step_1'])} | {_fmt(r['stabilize_by_step_2'])} |"
        )
    lines.append(
        "\nnever_settled_count is reported explicitly for every arm above -- it is never dropped or "
        "coerced to a default.\n"
    )

    lines.append("### Asymmetric-loss justification (verbatim)\n")
    lines.append("```\n" + ASYMMETRIC_LOSS_PARAGRAPH + "\n```\n")

    lines.append("### Idempotence verification\n")
    total_checked = sum(layer2[arm]["idempotence_checked"] for arm in ARMS)
    total_violations = sum(layer2[arm]["idempotence_violations"] for arm in ARMS)
    for arm in ARMS:
        r = layer2[arm]
        lines.append(
            f"- {arm}: checked {r['idempotence_checked']} settled genomes (of {n_samples}); "
            f"{r['idempotence_violations']} violation(s) (body changed again after its recorded settle_step, "
            f"before T_max)."
        )
    if total_violations == 0:
        lines.append(
            f"\n**Idempotence premise empirically confirmed**: 0/{total_checked} settled genomes across both "
            "arms showed any further body change between their recorded settle_step and T_max. Development "
            "that settles genuinely stays settled within the measured horizon -- extra steps beyond "
            "settle_step are confirmed free for these genomes, not merely assumed.\n"
        )
    else:
        lines.append(
            f"\n**IDEMPOTENCE VIOLATED**: {total_violations}/{total_checked} settled genomes changed again "
            "after their recorded settle_step. This means development can oscillate rather than converge, "
            "which invalidates the max-over-arms selection criterion as stated -- this must be resolved "
            "before trusting T_primary below.\n"
        )

    lines.append("### NCA divergence detail\n")
    nca_mag = layer2["nca"].get("magnitude_analysis")
    if nca_mag:
        lines.append(
            f"- initial reference scale (max\\|state\\| at t=0): {_fmt(nca_mag['initial_reference_scale'])}\n"
            f"- fraction of genomes whose max-abs state ever exceeds 10x that scale within T_max: "
            f"{_fmt(nca_mag['fraction_exceeding_10x'])} (median step: {_fmt(nca_mag['median_step_exceeding_10x'])})\n"
            f"- fraction exceeding 100x: {_fmt(nca_mag['fraction_exceeding_100x'])} "
            f"(median step: {_fmt(nca_mag['median_step_exceeding_100x'])})\n"
            f"- mean/median final (t=T_max) max-abs state magnitude: "
            f"{_fmt(nca_mag['mean_final_magnitude'])} / {_fmt(nca_mag['median_final_magnitude'])}\n"
        )

    lines.append("### Argmax-stabilisation distribution (headline number for NCA)\n")
    nca = layer2["nca"]
    lines.append(
        f"The argmax-stabilisation step IS `settle_step` (defined as the last step at which the decoded "
        f"argmax body changed) -- reused directly, not a separate measurement. For NCA: "
        f"{_fmt(nca['stabilize_by_step_1'])} of genomes have their final phenotype decided by step 1, "
        f"{_fmt(nca['stabilize_by_step_2'])} by step 2, median settle_step={_fmt(nca['median_settle_step'])}, "
        f"while the state magnitude analysis above shows the continuous state continues growing "
        f"(often past 10x/100x the initial scale) long after that. "
    )
    if nca["stabilize_by_step_2"] is not None and nca["stabilize_by_step_2"] > 0.5:
        lines.append(
            "**More than half of NCA genomes have their phenotype decided within the first 2 steps while "
            "the underlying state keeps changing for dozens more -- NCA development is largely vestigial: "
            "the phenotype is decided almost immediately and the remaining steps are theatre for the "
            "MAJORITY of genomes**, even though a minority genuinely keep re-deciding their argmax for "
            "much longer (see median/p95 settle_step above, which are pulled up by that minority).\n"
        )
    else:
        lines.append(
            "This sample does not show the majority of NCA genomes stabilising within the first 2 steps; "
            "see the full median/p95 above for the actual argmax-stabilisation horizon.\n"
        )

    lines.append("### T_primary derivation\n")
    if provisional:
        lines.append(
            f"**NCA's never_settled_fraction ({_fmt(layer2['nca']['never_settled_fraction'])}) exceeds the "
            f"{NEVER_SETTLED_PROVISIONAL_THRESHOLD:.0%} provisional-fallback threshold** -- the NCA arm does "
            f"not settle within T_max={T_MAX_HARD_CAP} for the majority of genomes, so its own q95 cannot be "
            f"trusted as a convergence horizon (it is a lower bound on 'still developing', not a settled "
            f"value). Per the pre-registered fallback: T_primary is set from the rule-based arm's q95 "
            f"({_fmt(layer2['rule_based']['p95_settle_step'])}) and the geometric bound "
            f"(T >= {GEOMETRIC_BOUND_FROM_CENTER}), clipped to [{GEOMETRIC_BOUND_FROM_CENTER}, "
            f"{T_MAX_HARD_CAP}] -> **T_primary = {t_primary}**. "
            f"**T selection is PROVISIONAL pending a fix to NCA's unbounded residual update.**"
        )
    else:
        q95s = ", ".join(f"{arm}={_fmt(layer2[arm]['p95_settle_step'])}" for arm in ARMS)
        lines.append(
            f"T_primary = ceil(max over developmental arms of q95(settle_step)), clipped to "
            f"[{GEOMETRIC_BOUND_FROM_CENTER}, {T_MAX_HARD_CAP}] (floor = geometric minimum; ceiling = "
            f"T_MAX_HARD_CAP -- developmental steps are cheap forward passes evaluated before PPO training "
            f"starts, so there is no compute justification for any tighter ceiling; a previous version of "
            f"this derivation used an unjustified [4, 20] ceiling that capped T_primary below nca's own "
            f"q95/median settle_step -- see docs/decisions.md for the correction). "
            f"q95 values: {q95s}. max = {_fmt(max(layer2[arm]['p95_settle_step'] for arm in ARMS))} -> "
            f"**T_primary = {t_primary}**."
        )

    if speed_correction_needed and speed_nca and speed_rb:
        speed_reference = max(speed_nca, speed_rb)
        reference_arm = "nca" if speed_nca >= speed_rb else "rule_based"
        lines.append("\n\n### Speed-corrected depth sensitivity analysis\n")
        lines.append(
            f"Layer 1 found nca and rule_based's fitted propagation speeds differ by more than "
            f"{SPEED_DIFFERENCE_THRESHOLD*100:.0f}%. **Equal-T and equal-reach are NOT the same protocol "
            f"under these measurements.** As a sensitivity analysis (not a replacement for the primary "
            f"equal-T protocol above), a speed-corrected depth T_arm = round(T_primary * speed_reference "
            f"/ speed_arm), with speed_reference = max(nca, rule_based) = {_fmt(speed_reference)} "
            f"(the `{reference_arm}` arm):\n"
        )
        lines.append("| arm | fitted speed | T_arm = round(T_primary * speed_reference / speed_arm) |")
        lines.append("| --- | --- | --- |")
        for arm, speed in (("nca", speed_nca), ("rule_based", speed_rb)):
            t_arm = round(t_primary * speed_reference / speed) if speed else None
            lines.append(f"| {arm} | {_fmt(speed)} | {t_arm} |")
        lines.append(
            "\nThis sensitivity analysis is reported for transparency; it is a SEPARATE, explicitly "
            "labelled protocol from the primary equal-T comparison, per the requirement that any use of "
            "different T per arm must be justified by this exact correction and labelled as such -- it is "
            "not applied to the primary comparison or the depth sweep in Layer 3 below."
        )

    lines.append("")
    return "\n".join(lines)


NON_SETTLING_CATEGORIES = ("STILL_GROWING", "PLATEAUED_OSCILLATING", "SLOW_BUT_MONOTONE")
PLATEAU_TOLERANCE = 0.05  # 5%


def _classify_non_settling_genome(bodies, magnitudes):
    """One of NON_SETTLING_CATEGORIES for a single never-settled NCA
    genome's full T_max=40 trajectory. magnitudes has one entry per step
    1..T_max (see structural._full_trajectory); mag[-1] is step T_max,
    mag[-6] is T_max-5 (step 35 when T_max=40)."""
    mag_now = magnitudes[-1]
    mag_5_ago = magnitudes[-6]
    still_growing = mag_5_ago > 0 and (mag_now > (1 + PLATEAU_TOLERANCE) * mag_5_ago)
    if still_growing:
        return "STILL_GROWING"

    last_window = magnitudes[-5:]
    window_mean = np.mean(last_window)
    plateaued = window_mean > 0 and ((max(last_window) - min(last_window)) / window_mean) <= PLATEAU_TOLERANCE
    if plateaued:
        return "PLATEAUED_OSCILLATING"

    return "SLOW_BUT_MONOTONE"


def _flip_steps(bodies):
    return [t + 1 for t in range(len(bodies) - 1) if not np.array_equal(bodies[t], bodies[t + 1])]


def diagnose_non_settling_nca(n_samples, init_condition, seed):
    """Classifies every NCA genome recorded as never_settled at T_max=40
    into exactly one of NON_SETTLING_CATEGORIES. Regenerates the SAME
    deterministic (encoder, seed, T_max) genome stream run_layer2 already
    used -- not a new/different sample, the identical draw -- so the
    never-settled subset and its per-genome magnitude traces are
    reproduced bit-for-bit rather than re-sampled."""
    encoder, factors = _build_encoder("nca", dev_steps=T_MAX_HARD_CAP, init_condition=init_condition, seed=seed)
    rng = np.random.default_rng(seed)

    counts = {c: 0 for c in NON_SETTLING_CATEGORIES}
    examples = {c: [] for c in NON_SETTLING_CATEGORIES}
    total_never_settled = 0

    for _ in range(n_samples):
        genome = encoder.random_genome(rng)
        bodies, magnitudes = S._full_trajectory(encoder, genome, T_MAX_HARD_CAP)
        settle_step, settled = S._settle_info(bodies)
        if settled:
            continue

        total_never_settled += 1
        category = _classify_non_settling_genome(bodies, magnitudes)
        counts[category] += 1
        if len(examples[category]) < 3:
            flips = _flip_steps(bodies)
            examples[category].append({
                "magnitude_last5": [float(m) for m in magnitudes[-5:]],
                "magnitude_step35": float(magnitudes[-6]),
                "magnitude_step40": float(magnitudes[-1]),
                "flip_steps_last5": [s for s in flips if s > T_MAX_HARD_CAP - 5],
                "total_flips": len(flips),
            })

    fractions = {c: (counts[c] / total_never_settled if total_never_settled else None) for c in NON_SETTLING_CATEGORIES}
    return {
        "total_never_settled": total_never_settled,
        "counts": counts,
        "fractions": fractions,
        "examples": examples,
    }


def _diagnosis_section(diagnosis: dict) -> str:
    lines = ["## Diagnosis of non-settling genomes (T_max=40)\n"]
    lines.append(
        "Every NCA genome recorded as never_settled in Layer 2 (T_max=40, center_seed) is classified into "
        "exactly one of three categories from its full state_magnitude_trace and argmax flip history -- "
        "diagnosis only, no update-rule change made here regardless of the answer.\n"
    )
    lines.append(
        "- **STILL_GROWING**: max-abs state magnitude at step 40 exceeds step 35's by more than "
        f"{PLATEAU_TOLERANCE:.0%} -- genuinely still diverging, not close to any fixed point.\n"
        "- **PLATEAUED_OSCILLATING**: magnitude has stabilised (within "
        f"{PLATEAU_TOLERANCE:.0%} over the last 5 steps) but the argmax readout is still flipping -- two or "
        "more material channels are persistently near-tied.\n"
        "- **SLOW_BUT_MONOTONE**: neither of the above -- magnitude is still changing, but not fast enough "
        "to count as STILL_GROWING and not stable enough to count as PLATEAUED_OSCILLATING; consistent with "
        "\"would settle given more T\".\n"
    )

    total = diagnosis["total_never_settled"]
    lines.append(f"**Breakdown** (n={total} never-settled genomes):\n")
    lines.append("| category | count | fraction |")
    lines.append("| --- | --- | --- |")
    for cat in NON_SETTLING_CATEGORIES:
        lines.append(f"| {cat} | {diagnosis['counts'][cat]} | {_fmt(diagnosis['fractions'][cat])} |")
    lines.append("")

    largest = max(NON_SETTLING_CATEGORIES, key=lambda c: diagnosis["counts"][c])
    lines.append(f"**Largest category: {largest}** ({diagnosis['counts'][largest]}/{total}).\n")

    if largest in ("STILL_GROWING", "PLATEAUED_OSCILLATING"):
        lines.append(
            f"**Verdict**: the dominant failure mode ({largest}) is a genuine instability, not merely "
            "T_max being too short for an honest slow minority -- this is evidence for the NCA "
            "architecture/update-rule work, not something a larger T_max alone would necessarily resolve. "
            "No architecture change is made in this prompt.\n"
        )
    else:
        lines.append(
            "**Verdict**: the dominant failure mode (SLOW_BUT_MONOTONE) is consistent with an honest slow "
            "minority that T_max=40 simply did not give enough time to settle -- this does NOT by itself "
            "indicate an update-rule instability, and can be carried as an accepted tail in "
            "threats-to-validity rather than requiring an architecture fix.\n"
        )

    lines.append(f"### Example trajectories from `{largest}` (largest category)\n")
    for i, example in enumerate(diagnosis["examples"][largest], 1):
        lines.append(
            f"{i}. magnitude at step 35 = {_fmt(example['magnitude_step35'])}, step 40 = "
            f"{_fmt(example['magnitude_step40'])}, last-5-step magnitudes = "
            f"{[round(m, 3) for m in example['magnitude_last5']]}; argmax flip steps in the last 5 steps = "
            f"{example['flip_steps_last5']} (total flips across the full T=40 trajectory: "
            f"{example['total_flips']})."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LAYER 3: T as a crossed factor
# ---------------------------------------------------------------------------


def run_layer3(n_samples, init_condition, seed, t_primary, output_dir):
    depths = sorted(set([1, 4, t_primary, min(2 * t_primary, T_MAX_HARD_CAP)]))
    cells = {}
    for arm in ARMS:
        for T in depths:
            encoder, factors = _build_encoder(arm, dev_steps=T, init_condition=init_condition, seed=seed)
            rng = np.random.default_rng(seed)
            jsonl_root = os.path.join(output_dir, "raw", "layer3", arm, f"T{T}")

            metrics = {}
            metrics["validity_rate"] = S.validity_rate(encoder, factors, n_samples, rng, output_root=jsonl_root)
            metrics["settling_profile"] = S.settling_profile(encoder, factors, n_samples, rng, t_max=T, output_root=jsonl_root)
            metrics["light_cone"] = S.light_cone(encoder, factors, n_samples, rng, t_max=T, output_root=jsonl_root)
            metrics["phenotypic_displacement"] = S.phenotypic_displacement(encoder, factors, n_samples, rng, output_root=jsonl_root)
            metrics["redundancy"] = S.redundancy(encoder, factors, n_samples, rng, output_root=jsonl_root)
            metrics["diversity"] = S.diversity(encoder, factors, n_samples, rng, output_root=jsonl_root)
            metrics["perturbation_robustness"] = S.perturbation_robustness(encoder, factors, n_samples, rng, output_root=jsonl_root)
            cells[(arm, T)] = metrics
    return depths, cells


def _plot_metric_vs_depth(cells, depths, metric_name, field, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = {"nca": "#4C72B0", "rule_based": "#DD8452"}
    for arm in ARMS:
        values = [getattr(cells[(arm, T)][metric_name], field) for T in depths]
        ax.plot(depths, values, marker="o", label=arm, color=colors.get(arm))
    ax.set_xlabel("developmental depth T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _layer3_section(depths, cells, figures_dir, n_samples, init_condition, t_primary) -> str:
    lines = ["## Layer 3: T as a crossed factor\n"]
    lines.append(
        f"Depth sweep T in {depths} (T=1: shared non-developmental control, marked NON-DROPPABLE under "
        "any compute cut; T=4: geometric minimum; T_primary; min(2*T_primary, T_MAX_HARD_CAP): "
        f"stability-past-horizon check), N={n_samples}/cell, init_condition=`{init_condition}`.\n"
    )
    doubled = 2 * t_primary
    if doubled > T_MAX_HARD_CAP:
        lines.append(
            f"**Note**: 2*T_primary = 2*{t_primary} = {doubled} exceeds T_MAX_HARD_CAP={T_MAX_HARD_CAP}, so "
            f"the sweep's fourth cell is clamped to min(2*T_primary, T_MAX_HARD_CAP) = {T_MAX_HARD_CAP} "
            f"itself, NOT a genuine 2x-T_primary probe -- the 'stability past 2x the convergence horizon' "
            f"cell and the 'developmental hard ceiling' cell coincide for this run. There is no depth point "
            f"in this sweep that tests behaviour strictly beyond T_MAX_HARD_CAP=40.\n"
        )

    metric_specs = [
        ("validity_rate", "valid_rate", "Validity rate vs T", "valid_rate", "validity_rate_vs_T.png"),
        ("diversity", "unique_valid_body_fraction", "Diversity vs T", "unique_valid_body_fraction", "diversity_vs_T.png"),
        ("redundancy", "genomes_per_unique_phenotype", "Redundancy vs T", "genomes_per_unique_phenotype", "redundancy_vs_T.png"),
        ("settling_profile", "never_settled_fraction", "Never-settled fraction vs T", "never_settled_fraction", "settling_vs_T.png"),
        ("light_cone", "propagation_speed_cells_per_step", "Propagation speed vs T", "cells/step", "light_cone_vs_T.png"),
        ("phenotypic_displacement", "mean_hamming_distance", "Phenotypic displacement vs T", "mean Hamming distance", "displacement_vs_T.png"),
        ("perturbation_robustness", "mean_hamming_distance", "Perturbation robustness vs T", "mean Hamming distance", "robustness_vs_T.png"),
    ]

    lines.append("| metric | arm | " + " | ".join(f"T={T}" for T in depths) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in depths) + " |")
    for metric_name, field, _title, _ylabel, fname in metric_specs:
        _plot_metric_vs_depth(cells, depths, metric_name, field, _title, _ylabel,
                               os.path.join(figures_dir, fname))
        for arm in ARMS:
            values = [_fmt(getattr(cells[(arm, T)][metric_name], field)) for T in depths]
            lines.append(f"| {metric_name}.{field} | {arm} | " + " | ".join(values) + " |")
    lines.append("")

    for _metric_name, _field, _title, _ylabel, fname in metric_specs:
        lines.append(f"![{_title}](figs/{fname})\n")

    # Stability-across-sweep paragraph.
    stability_notes = []
    for metric_name, field, _title, _ylabel, _fname in metric_specs:
        arm_ranges = {}
        for arm in ARMS:
            values = [getattr(cells[(arm, T)][metric_name], field) for T in depths]
            values = [v for v in values if v is not None]
            if len(values) >= 2:
                arm_ranges[arm] = max(values) - min(values)
        if arm_ranges:
            worst_arm = max(arm_ranges, key=arm_ranges.get)
            stability_notes.append((f"{metric_name}.{field}", worst_arm, arm_ranges[worst_arm]))

    lines.append("### Which conclusions are stable across the depth sweep\n")
    lines.append(
        "For each metric, the largest range observed across T (within the more-variable arm) is reported "
        "below; a small range means that metric's conclusion is stable regardless of exactly which T in "
        "the sweep is used, and a large range means T materially changes the picture and any single-T "
        "comparison risks being an artefact of that choice:\n"
    )
    for name, worst_arm, spread in sorted(stability_notes, key=lambda x: -x[2]):
        lines.append(f"- `{name}`: max range across T = {_fmt(spread)} (largest in `{worst_arm}`)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_depth_study(n_samples=DEFAULT_N, init_condition=DEFAULT_INIT_CONDITION, seed=0, output_dir=DEFAULT_OUTPUT_DIR):
    figures_dir = os.path.join(output_dir, "figs")
    os.makedirs(figures_dir, exist_ok=True)

    layer1 = run_layer1(n_samples, init_condition, seed, output_dir)
    layer1_text, speed_correction_needed, speed_nca, speed_rb = _layer1_section(layer1, n_samples, init_condition)

    layer2 = run_layer2(n_samples, init_condition, seed, output_dir)
    t_primary, provisional = _select_t_primary(layer2)
    layer2_text = _layer2_section(layer2, t_primary, provisional, n_samples, init_condition,
                                   speed_correction_needed, speed_nca, speed_rb)

    diagnosis = diagnose_non_settling_nca(n_samples, init_condition, seed)
    diagnosis_text = _diagnosis_section(diagnosis)

    depths, layer3_cells = run_layer3(n_samples, init_condition, seed, t_primary, output_dir)
    layer3_text = _layer3_section(depths, layer3_cells, figures_dir, n_samples, init_condition, t_primary)

    sections = []
    if provisional:
        sections.append(
            "# DEPTH STUDY -- T SELECTION IS PROVISIONAL\n\n"
            f"**NCA's development does not settle within T_max={T_MAX_HARD_CAP} for the majority of "
            f"genomes (never_settled_fraction={_fmt(layer2['nca']['never_settled_fraction'])}). T_primary "
            f"below was derived from the rule-based arm and the geometric bound alone, NOT from NCA's own "
            "convergence, and must be revisited once NCA's unbounded residual update is fixed.**\n"
        )
    sections.append("# Depth study (developmental depth T)\n")
    sections.append(
        f"Fitness-blind (evodevo/analysis/structural.py, evodevo/analysis/depth_study.py). "
        f"N={n_samples} genomes/cell, init_condition=`{init_condition}` (the P3 decision), seed={seed}. "
        "No PPO, no task, no reward were run to produce any number on this page.\n"
    )
    sections.append(layer1_text)
    sections.append(layer2_text)
    sections.append(diagnosis_text)
    sections.append(layer3_text)

    report_path = os.path.join(output_dir, "DEPTH_STUDY.md")
    with open(report_path, "w") as f:
        f.write("\n".join(sections))

    raw = {
        "layer1": {arm: {k: (dataclasses.asdict(v) if k == "aggregate" else v) for k, v in r.items()} for arm, r in layer1.items()},
        "layer2": layer2,
        "non_settling_diagnosis": diagnosis,
        "t_primary": t_primary,
        "provisional": provisional,
        "depths": depths,
        "layer3": {f"{arm}__T{T}": {name: dataclasses.asdict(res) for name, res in metrics.items()}
                   for (arm, T), metrics in layer3_cells.items()},
    }
    with open(os.path.join(output_dir, "depth_study_raw.json"), "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True, default=str)

    return report_path, t_primary, provisional, depths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--init-condition", default=DEFAULT_INIT_CONDITION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    report_path, t_primary, provisional, depths = run_depth_study(
        n_samples=args.n, init_condition=args.init_condition, seed=args.seed, output_dir=args.output_dir
    )
    print(f"depth study written to {report_path}")
    print(f"T_primary={t_primary} provisional={provisional} depth_sweep={depths}")


if __name__ == "__main__":
    main()
