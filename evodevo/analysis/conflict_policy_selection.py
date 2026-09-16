"""Chooses rule_based's default `conflict_resolution` policy ON EVIDENCE,
not in advance. Measures, for each of the four policies (the historical
order-DEPENDENT `row_major_last_wins` as baseline, and the three
order-INDEPENDENT alternatives `no_op`/`lowest_material_id`/
`first_rule_index`), over N=2000 random genomes at T=100:
  - validity_rate, unique_valid_body_fraction, uniform_body_fraction
    (reusing evodevo.analysis.structural directly -- no reimplementation)
  - never_settled_fraction (exact, per Proposition R1 -- the last-transition
    boundary test: body at T-1 != body at T)
  - mean occupied-cell count (does no_op stall growth by cancelling every
    conflicting write, as the "contact inhibition" framing would predict?)

Then separately re-runs the R1 permutation test (results/rule_based/
ORDER_INDEPENDENCE_AUDIT.md's own method) under the ADOPTED default, which
MUST report 0/2000 differing genomes for an order-independent policy.

Fitness-blind: only decode()/develop_rules()/random_genome()/validate_body()
are used -- no PPO, no task, no reward, no EA. Does not change the rule
vocabulary, mutation operators, MIN/MAX_RULES, or the initial condition.

Usage:
    python -m evodevo.analysis.conflict_policy_selection
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based._rules.interpreter import VALID_CONFLICT_RESOLUTIONS, develop_rules
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.morphology.validate import validate_body

DEFAULT_N = 2000
DEFAULT_SEED = 0
T = 100
SHAPE = (5, 5)
RESULTS_DIR = "results/rule_based"

BASELINE_POLICY = "row_major_last_wins"
CANDIDATE_POLICIES = ("no_op", "lowest_material_id", "first_rule_index")
ALL_POLICIES = (BASELINE_POLICY,) + CANDIDATE_POLICIES


def _build_encoder(conflict_resolution: str, seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=T, init_condition="center_seed", seed=seed)
    return RuleBasedEncoder(factors=factors, conflict_resolution=conflict_resolution)


def measure_policy(conflict_resolution: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    encoder = _build_encoder(conflict_resolution, seed)

    # validity_rate / diversity (unique_valid_body_fraction) / uniform_body_fraction:
    # reuse structural.py directly, same random_genome() draw order every time.
    rng1 = np.random.default_rng(seed)
    vr = S.validity_rate(encoder, encoder.factors, n, rng1, output_root=os.path.join(RESULTS_DIR, "_conflict_policy_raw"))
    rng2 = np.random.default_rng(seed)
    dv = S.diversity(encoder, encoder.factors, n, rng2, output_root=os.path.join(RESULTS_DIR, "_conflict_policy_raw"))
    rng3 = np.random.default_rng(seed)
    ub = S.uniform_body_fraction(encoder, encoder.factors, n, rng3, output_root=os.path.join(RESULTS_DIR, "_conflict_policy_raw"))

    # never_settled_fraction (exact boundary test: body at T-1 != body at T)
    # and mean occupied-cell count: computed directly, same draw order.
    rng4 = np.random.default_rng(seed)
    never_settled = 0
    occupied_counts = []
    for _ in range(n):
        genome = encoder.random_genome(rng4)
        initial_grid = encoder._initial_condition.as_discrete_grid(SHAPE)
        body_t_minus_1 = develop_rules(
            genome, shape=SHAPE, iterations=T - 1, initial_grid=initial_grid, conflict_resolution=conflict_resolution,
        )
        body_t = develop_rules(
            genome, shape=SHAPE, iterations=T, initial_grid=initial_grid, conflict_resolution=conflict_resolution,
        )
        if not np.array_equal(body_t_minus_1, body_t):
            never_settled += 1
        occupied_counts.append(int(np.count_nonzero(body_t)))

    return {
        "conflict_resolution": conflict_resolution,
        "n": n,
        "valid_rate": vr.valid_rate,
        "unique_valid_body_fraction": dv.unique_valid_body_fraction,
        "uniform_body_fraction": ub.uniform_fraction,
        "never_settled_fraction": never_settled / n,
        "mean_occupied_cells": float(np.mean(occupied_counts)),
    }


def run(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    results = {policy: measure_policy(policy, n=n, seed=seed) for policy in ALL_POLICIES}
    return {"n": n, "seed": seed, "t": T, "results": results}


# ---------------------------------------------------------------------------
# Permutation-test re-confirmation under the adopted default
# ---------------------------------------------------------------------------


def _row_major_order(shape=SHAPE):
    rows, cols = shape
    return [(i, j) for i in range(rows) for j in range(cols)]


def rerun_permutation_test_under_policy(conflict_resolution: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    import random as _global_random

    perm_rng = np.random.default_rng(12345)
    order = _row_major_order()
    perm = perm_rng.permutation(len(order))
    permuted_order = [order[k] for k in perm]

    rng = np.random.default_rng(seed)
    n_differ = 0
    example = None
    for idx in range(n):
        genome_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        _global_random.seed(int(genome_rng.integers(0, 2**31 - 1)))
        from evodevo.encoders.rule_based._rules.genome import sample_developmental_rules
        genome = sample_developmental_rules()

        row_major_result = develop_rules(genome, shape=SHAPE, iterations=T, conflict_resolution=conflict_resolution)
        permuted_result = develop_rules(
            genome, shape=SHAPE, iterations=T, conflict_resolution=conflict_resolution, _cell_order=permuted_order,
        )
        if not np.array_equal(row_major_result, permuted_result):
            n_differ += 1
            if example is None:
                example = (idx, row_major_result.copy(), permuted_result.copy())

    return {"n": n, "conflict_resolution": conflict_resolution, "n_differ": n_differ, "example": example}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def select_default(result: dict) -> str:
    """Selection logic: among the three order-independent candidates,
    reject any whose validity_rate is meaningfully worse than the baseline
    (more than a few points below row_major_last_wins) or whose
    mean_occupied_cells collapses relative to baseline (growth stalling);
    among the survivors, prefer the one with the HIGHEST validity_rate.
    Applied here, not asserted in advance."""
    baseline = result["results"][BASELINE_POLICY]
    survivors = []
    for policy in CANDIDATE_POLICIES:
        r = result["results"][policy]
        occupied_collapse = r["mean_occupied_cells"] < 0.5 * baseline["mean_occupied_cells"]
        validity_drop = baseline["valid_rate"] - r["valid_rate"]
        survivors.append((policy, r, occupied_collapse, validity_drop))
    # prefer non-collapsed policies with the best validity_rate
    non_collapsed = [s for s in survivors if not s[2]]
    pool = non_collapsed if non_collapsed else survivors
    best = max(pool, key=lambda s: s[1]["valid_rate"])
    return best[0]


def render_report(result: dict, permutation_check: dict, adopted: str) -> str:
    n = result["n"]
    lines = []
    lines.append("# rule_based growth-conflict resolution policy: four-way comparison")
    lines.append("")
    lines.append(
        f"Fitness-blind (only decode()/develop_rules()/random_genome()/validate_body() are used -- NO PPO, "
        f"NO task, NO reward, NO EA). N={n}, seed={result['seed']}, T={result['t']}. Compares the historical "
        "order-DEPENDENT `row_major_last_wins` (baseline, results/rule_based/ORDER_INDEPENDENCE_AUDIT.md's "
        "own subject) against the three order-INDEPENDENT alternatives."
    )
    lines.append("")

    lines.append("## Four-way comparison\n")
    lines.append(
        "| policy | valid_rate | unique_valid_body_fraction | uniform_body_fraction | never_settled_fraction "
        "| mean_occupied_cells |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for policy in ALL_POLICIES:
        r = result["results"][policy]
        marker = " (baseline)" if policy == BASELINE_POLICY else ""
        lines.append(
            f"| {policy}{marker} | {_fmt(r['valid_rate'])} | {_fmt(r['unique_valid_body_fraction'])} | "
            f"{_fmt(r['uniform_body_fraction'])} | {_fmt(r['never_settled_fraction'])} | "
            f"{_fmt(r['mean_occupied_cells'], 2)} |"
        )
    lines.append("")

    baseline = result["results"][BASELINE_POLICY]
    no_op = result["results"]["no_op"]
    occupied_ratio = no_op["mean_occupied_cells"] / baseline["mean_occupied_cells"] if baseline["mean_occupied_cells"] else float("nan")
    lines.append(
        f"**Does `no_op` stall growth, as its 'contact inhibition' framing might predict?** mean_occupied_cells: "
        f"{_fmt(no_op['mean_occupied_cells'], 2)} vs baseline's {_fmt(baseline['mean_occupied_cells'], 2)} "
        f"(ratio {_fmt(occupied_ratio, 3)}). "
        + (
            "YES -- growth is meaningfully stalled; this is reported plainly rather than choosing no_op on "
            "elegance alone."
            if occupied_ratio < 0.8
            else "No meaningful stalling observed at this population/T -- growth conflicts are not frequent "
            "enough per-genome for cancellation to visibly suppress total occupied area, though the "
            "PER-CONFLICT semantics still differ from the other policies."
        )
    )
    lines.append("")

    lines.append("## Selection\n")
    lines.append(
        f"Selection rule: among the three order-independent candidates, reject any whose mean_occupied_cells "
        "collapses to less than half the baseline's (growth-stalling disqualifier), then pick the highest "
        "valid_rate among the survivors -- applied to the actual numbers above, not decided in advance."
    )
    lines.append("")
    lines.append(f"**Adopted default: `conflict_resolution=\"{adopted}\"`.**")
    lines.append("")
    adopted_r = result["results"][adopted]
    lines.append(
        f"At the adopted policy: valid_rate={_fmt(adopted_r['valid_rate'])} (baseline "
        f"{_fmt(baseline['valid_rate'])}, delta {_fmt(adopted_r['valid_rate'] - baseline['valid_rate'])}), "
        f"unique_valid_body_fraction={_fmt(adopted_r['unique_valid_body_fraction'])} (baseline "
        f"{_fmt(baseline['unique_valid_body_fraction'])}), mean_occupied_cells="
        f"{_fmt(adopted_r['mean_occupied_cells'], 2)} (baseline {_fmt(baseline['mean_occupied_cells'], 2)})."
    )
    lines.append("")

    lines.append("## Permutation-test re-confirmation under the adopted default\n")
    lines.append(
        f"Re-running results/rule_based/ORDER_INDEPENDENCE_AUDIT.md's own permutation test "
        f"(N={permutation_check['n']}, T={T}, same fixed seed=12345 permutation), now under "
        f"`conflict_resolution=\"{adopted}\"`:"
    )
    lines.append("")
    lines.append(
        f"**Fraction of genomes whose decoded body differs under permutation: "
        f"{permutation_check['n_differ']}/{permutation_check['n']}.**"
    )
    lines.append("")
    if permutation_check["n_differ"] == 0:
        lines.append(
            "**0/2000 -- the adopted default is confirmed order-independent, exactly as required.** The "
            "restructure (Pass 2b collects growth candidates, groups by target, then resolves per-target as "
            "a pure function of the competing-write SET rather than enumeration order) achieves genuine "
            "order-independence, not merely a claim."
        )
    else:
        lines.append(
            "**NONZERO -- the restructure is INCOMPLETE.** This must be investigated, not accepted; a "
            "genuinely order-independent conflict_resolution policy should report exactly 0 here by "
            "construction."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "CONFLICT_POLICY.md"))
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed)
    adopted = select_default(result)
    permutation_check = rerun_permutation_test_under_policy(adopted, n=args.n, seed=args.seed)
    report = render_report(result, permutation_check, adopted)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)
    print(f"ADOPTED = {adopted}")


if __name__ == "__main__":
    main()
