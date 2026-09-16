"""Maps the stability-vs-expressivity trade-off N4 should have swept: N4
(evodevo/analysis/timescale_calibration.py) swept (decay, state_bound) alone,
holding init_gain fixed at N2's value (1.0, chosen for near-unit
pre-activation variance) -- but the certificate condition
state_bound * K(theta) < 1 depends on K(theta) = sqrt(5) * sigma_max(W1) *
sigma_max(W2), which init_gain controls and decay/state_bound do not (they
only rescale/shift, they don't change the network's own operator norm).
This module sweeps all three jointly and asks the decision-relevant
question directly: does a setting exist with non-trivial certification AND
acceptable validity/diversity, or is this a genuine, structural stability-
vs-expressivity trade-off (the NCA analogue of the "edge of chaos" tension
in RNN/reservoir-computing dynamics)?

Fitness-blind throughout: only decode()/develop()/validate_body and
evodevo.analysis.structural.*/evodevo.encoders.nca.certificate.* are used --
NO PPO, NO task, NO reward, NO EA.

Usage:
    python -m evodevo.analysis.joint_calibration [--n 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.nca.certificate import contraction_constant, is_certified_contractive
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body

DEFAULT_N = 1000
DEFAULT_SEED = 0
INIT_CONDITION = "center_seed"  # P3 decision
T_MAX = 40  # T_MAX_HARD_CAP, the primary developmental horizon used throughout the N-series
HIDDEN_CHANNELS = 3  # N5-resolved H (re-checked at the decided primary settings separately)

INIT_GAINS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
DECAYS = [0.05, 0.1, 0.2, 0.5]
STATE_BOUNDS = [1, 2, 5]

# Pre-redesign (N1..N9-era) baselines this trade-off is judged against --
# cited from results/NCA_READINESS.md Section 8 (itself sourced from
# results/nca/ARCHITECTURE_DELTA.md), not re-derived here.
BASELINE_VALIDITY_RATE = 0.6540
BASELINE_UNIFORM_BODY_FRACTION = 0.0140

RESULTS_DIR = "results/nca"
RAW_ROOT = "results/nca/joint_calibration_raw"


def _build_encoder(init_gain: float, decay: float, state_bound: float, seed: int) -> tuple:
    factors = ExperimentFactors(
        dev_steps=T_MAX, init_condition=INIT_CONDITION, seed=seed, hidden_channels=HIDDEN_CHANNELS
    )
    encoder = NCAEncoder(
        factors=factors, update_rule="bounded_leaky", decay=decay, state_bound=state_bound, init_gain=init_gain
    )
    return encoder, factors


def _measure_cell(init_gain: float, decay: float, state_bound: float, n: int, seed: int) -> dict:
    encoder, factors = _build_encoder(init_gain, decay, state_bound, seed)

    rng = np.random.default_rng(seed)
    certified_flags = []
    k_values = []
    settle_steps = []
    never_settled = 0
    valid_flags = []
    body_hashes = set()
    uniform_count = 0
    near_tie_cell_flags = []
    near_tie_body_count = 0

    for _ in range(n):
        genome = encoder.random_genome(rng)
        certified_flags.append(is_certified_contractive(genome, encoder))
        k_values.append(contraction_constant(genome, encoder))

        bodies, _magnitudes, states = S._nca_trajectory_from_state(
            encoder, genome, encoder._initial_state, T_MAX
        )
        settle_step, settled = S._settle_info(bodies)
        if settled:
            settle_steps.append(settle_step)
        else:
            never_settled += 1

        final_body = bodies[-1]
        result = validate_body(final_body, grid_shape=encoder.grid_shape)
        valid_flags.append(result.valid)
        body_hashes.add(body_hash(final_body))
        unique_vals = np.unique(final_body)
        is_uniform = unique_vals.size == 1
        if is_uniform:
            uniform_count += 1

        final_logits = states[-1][:, :, :N_MATERIALS]
        sorted_logits = np.sort(final_logits, axis=-1)
        margin = (sorted_logits[..., -1] - sorted_logits[..., -2]).flatten()
        near_tie_cell_flags.append(margin < 0.01)
        if np.any(margin < 0.01):
            near_tie_body_count += 1

    k_arr = np.array(k_values)
    settle_arr = np.array(settle_steps) if settle_steps else np.array([])
    near_tie_cells_arr = np.concatenate(near_tie_cell_flags)

    # light_cone: separate sampling loop (structural.light_cone's own
    # perturbation methodology), same n and seed for a like-for-like cell.
    rng2 = np.random.default_rng(seed)
    lc = S.light_cone(encoder, factors, n, rng2, t_max=T_MAX, output_root=os.path.join(RAW_ROOT, f"g{init_gain}_d{decay}_B{state_bound}"))

    return {
        "init_gain": init_gain,
        "decay": decay,
        "state_bound": state_bound,
        "n": n,
        "certified_fraction": float(np.mean(certified_flags)),
        "k_median": float(np.median(k_arr)),
        "k_p90": float(np.quantile(k_arr, 0.9)),
        "never_settled_fraction": never_settled / n,
        "median_settle_step": float(np.median(settle_arr)) if len(settle_arr) else None,
        "light_cone_speed": lc.propagation_speed_cells_per_step,
        "validity_rate": float(np.mean(valid_flags)),
        "unique_valid_body_fraction": len(body_hashes) / n,  # NOTE: unique among ALL bodies, not only valid ones -- see module docstring caveat below
        "uniform_body_fraction": uniform_count / n,
        "near_tie_cell_fraction": float(np.mean(near_tie_cells_arr)),
        "near_tie_body_fraction": near_tie_body_count / n,
    }


def run_sweep(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list:
    cells = []
    for init_gain in INIT_GAINS:
        for decay in DECAYS:
            for state_bound in STATE_BOUNDS:
                cells.append(_measure_cell(init_gain, decay, state_bound, n, seed))
    return cells


# ---------------------------------------------------------------------------
# Pareto frontier / decision-relevant question
# ---------------------------------------------------------------------------


def _dominates(a: dict, b: dict) -> bool:
    """a dominates b if a is at least as good on every objective and
    strictly better on at least one. Objectives (all "higher/closer to
    target is better" after transformation):
      - settling_score = max(certified_fraction, 1 - never_settled_fraction/1.0-ish)
        -- simplified to a direct joint: certified_fraction OR settling speed
           (1 - never_settled_fraction), taking the max of the two per the
           task's "(higher certified_fraction OR faster settling)" wording.
      - validity_rate (higher better)
      - -uniform_body_fraction (lower better)
    """
    def settling_score(c):
        return max(c["certified_fraction"], 1.0 - c["never_settled_fraction"])

    a_settle, b_settle = settling_score(a), settling_score(b)
    a_valid, b_valid = a["validity_rate"], b["validity_rate"]
    a_uniform, b_uniform = -a["uniform_body_fraction"], -b["uniform_body_fraction"]

    at_least_as_good = a_settle >= b_settle and a_valid >= b_valid and a_uniform >= b_uniform
    strictly_better = a_settle > b_settle or a_valid > b_valid or a_uniform > b_uniform
    return at_least_as_good and strictly_better


def extract_pareto_frontier(cells: list) -> list:
    """Cells passing the task's explicit joint filter (validity near/above
    baseline, uniform_body_fraction not above baseline) that are additionally
    not DOMINATED by another cell also passing that filter."""
    eligible = [
        c for c in cells
        if c["validity_rate"] >= BASELINE_VALIDITY_RATE - 0.05  # "close to or above" -- 5pp slack, stated explicitly
        and c["uniform_body_fraction"] <= BASELINE_UNIFORM_BODY_FRACTION
    ]
    frontier = [c for c in eligible if not any(_dominates(other, c) for other in eligible if other is not c)]
    return frontier


def find_decision_relevant_cells(cells: list) -> list:
    """The task's explicit decision question: certified_fraction > 0.1 AND
    validity_rate >= 0.5 AND uniform_body_fraction <= 0.05."""
    return [
        c for c in cells
        if c["certified_fraction"] > 0.1 and c["validity_rate"] >= 0.5 and c["uniform_body_fraction"] <= 0.05
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _correlation(xs, ys) -> float:
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def render_report(cells: list, n: int, seed: int) -> str:
    lines = []
    lines.append("# NCA stability-vs-expressivity trade-off surface")
    lines.append("")
    lines.append(
        f"Fitness-blind (decode()/develop()/validate_body plus evodevo.analysis.structural.*/"
        f"evodevo.encoders.nca.certificate.* only -- NO PPO, NO task, NO reward, NO EA). N={n} genomes/cell, "
        f"T_max={T_MAX}, init_condition=`{INIT_CONDITION}` (P3), hidden_channels={HIDDEN_CHANNELS}, seed={seed}. "
        f"3D grid: init_gain in {INIT_GAINS} x decay in {DECAYS} x state_bound in {STATE_BOUNDS} "
        f"({len(INIT_GAINS)}x{len(DECAYS)}x{len(STATE_BOUNDS)} = {len(cells)} cells)."
    )
    lines.append("")
    lines.append(
        "**Why this grid, not N4's**: N4 (results/nca/TIMESCALE_CALIBRATION.md) swept (decay, state_bound) "
        "alone at init_gain=1.0 (N2's value, chosen for near-unit pre-activation variance -- a signal-"
        "propagation criterion, never checked against the certificate condition state_bound*K(theta)<1). "
        "K(theta) = sqrt(5)*sigma_max(W1)*sigma_max(W2) is controlled by init_gain, not by decay/state_bound "
        "(they rescale/shift the update, they do not change the network's own operator norm) -- so N4's "
        "sweep never actually varied the quantity the certificate depends on. This is the axis N4 should "
        "have swept."
    )
    lines.append("")

    lines.append("## The full 72-cell grid\n")
    lines.append(
        "| init_gain | decay | B | certified_fraction | K median | K p90 | never_settled | median settle | "
        "light_cone speed | validity | unique_valid_body | uniform_body | near-tie cells | near-tie bodies |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in cells:
        lines.append(
            f"| {c['init_gain']} | {c['decay']} | {c['state_bound']} | {_fmt(c['certified_fraction'])} | "
            f"{_fmt(c['k_median'])} | {_fmt(c['k_p90'])} | {_fmt(c['never_settled_fraction'])} | "
            f"{_fmt(c['median_settle_step'])} | {_fmt(c['light_cone_speed'])} | {_fmt(c['validity_rate'])} | "
            f"{_fmt(c['unique_valid_body_fraction'])} | {_fmt(c['uniform_body_fraction'])} | "
            f"{_fmt(c['near_tie_cell_fraction'])} | {_fmt(c['near_tie_body_fraction'])} |"
        )
    lines.append("")

    lines.append("## certified_fraction vs validity_rate, across all 72 cells\n")
    corr = _correlation([c["certified_fraction"] for c in cells], [c["validity_rate"] for c in cells])
    lines.append(f"Pearson correlation across the full grid: **{_fmt(corr, 3)}**.\n")
    lines.append("| init_gain | decay | B | certified_fraction | validity_rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in sorted(cells, key=lambda c: -c["certified_fraction"])[:15]:
        lines.append(f"| {c['init_gain']} | {c['decay']} | {c['state_bound']} | {_fmt(c['certified_fraction'])} | {_fmt(c['validity_rate'])} |")
    lines.append("\n(top 15 cells by certified_fraction, showing their validity_rate alongside)\n")

    lines.append("## The decision-relevant question\n")
    decision_cells = find_decision_relevant_cells(cells)
    lines.append(
        f"Does ANY cell achieve certified_fraction > 0.1 while keeping validity_rate >= 0.5 and "
        f"uniform_body_fraction <= 0.05? **{'YES' if decision_cells else 'NO'}** "
        f"({len(decision_cells)}/{len(cells)} cells qualify)."
    )
    lines.append("")
    if decision_cells:
        lines.append("| init_gain | decay | B | certified_fraction | validity_rate | uniform_body_fraction |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in decision_cells:
            lines.append(
                f"| {c['init_gain']} | {c['decay']} | {c['state_bound']} | {_fmt(c['certified_fraction'])} | "
                f"{_fmt(c['validity_rate'])} | {_fmt(c['uniform_body_fraction'])} |"
            )
        lines.append("")

    lines.append("## Pareto frontier\n")
    lines.append(
        f"Cells with validity_rate >= {BASELINE_VALIDITY_RATE - 0.05:.4f} (baseline {BASELINE_VALIDITY_RATE} "
        f"minus a stated 5-percentage-point slack for \"close to or above\") AND uniform_body_fraction <= "
        f"{BASELINE_UNIFORM_BODY_FRACTION} (the pre-redesign baseline, not to be exceeded), that are not "
        "dominated by another such cell on (settling_score = max(certified_fraction, 1-never_settled_fraction), "
        "validity_rate, -uniform_body_fraction):"
    )
    lines.append("")
    frontier = extract_pareto_frontier(cells)
    if frontier:
        lines.append("| init_gain | decay | B | certified_fraction | never_settled_fraction | validity_rate | uniform_body_fraction |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for c in sorted(frontier, key=lambda c: -c["validity_rate"]):
            lines.append(
                f"| {c['init_gain']} | {c['decay']} | {c['state_bound']} | {_fmt(c['certified_fraction'])} | "
                f"{_fmt(c['never_settled_fraction'])} | {_fmt(c['validity_rate'])} | {_fmt(c['uniform_body_fraction'])} |"
            )
    else:
        lines.append("**No cell satisfies both baseline constraints simultaneously.**")
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "`unique_valid_body_fraction` here is unique-body-hash fraction among ALL sampled bodies (matching "
        "this module's own single combined measurement loop), not restricted to valid bodies only -- unlike "
        "evodevo.analysis.structural.diversity's field of the same name, which restricts to valid bodies. "
        "Both are legitimate diversity measures; this module's column name is kept for readability but the "
        "denominator differs from structural.diversity's -- do not directly cross-cite the two without this "
        "caveat."
    )
    lines.append("")
    lines.append(
        "This module never computes or consults a fitness, reward, or task-performance quantity: every "
        "number above is derived from decode()/develop()/validate_body alone."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "TRADEOFF_SURFACE.md"))
    args = parser.parse_args()

    cells = run_sweep(n=args.n, seed=args.seed)
    report = render_report(cells, n=args.n, seed=args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"wrote {args.output}")
    print(f"decision-relevant cells: {len(find_decision_relevant_cells(cells))}")
    print(f"pareto frontier size: {len(extract_pareto_frontier(cells))}")


if __name__ == "__main__":
    main()
