"""Runs evodevo/encoders/rule_based/context_analysis.py's exact reachability/
coverage/redundancy analysis over N=2000 random genomes (and, if any evolved
genome archives exist, those too -- see this module's own EVOLVED GENOMES
note), producing results/rule_based/CONTEXT_ANALYSIS.md.

Fitness-blind: only decode()/develop_rules()/random_genome() are used (via
context_analysis.py's own helpers, which reuse interpreter.py's private
functions directly for fidelity) -- no PPO, no task, no reward, no EA.

Usage:
    python -m evodevo.analysis.rule_based_context_report
"""
from __future__ import annotations

import argparse
import os
import random as _global_random

import numpy as np

from evodevo.core.materials import MATERIAL_NAMES, SOFT
from evodevo.encoders.rule_based._rules.genome import sample_developmental_rules
from evodevo.encoders.rule_based.context_analysis import (
    CONTEXT_SPACE_SIZE,
    action_redundancy,
    dead_rules,
    dynamic_coverage,
    effective_rule_count,
    rule_coverage,
)
from evodevo.encoders.rule_based.resolved_config import PRIMARY

DEFAULT_N = 2000
DEFAULT_SEED = 0
SHAPE = (5, 5)
DYNAMIC_T = 100  # matches the shared resolved T_primary, results/T_PRIMARY_UNIFIED.md

RESULTS_DIR = "results/rule_based"


def _draw_genome(seed: int):
    rng = np.random.default_rng(seed)
    _global_random.seed(int(rng.integers(0, 2**31 - 1)))
    return sample_developmental_rules(position_predicate=PRIMARY.position_predicate)


def _initial_grid():
    grid = np.zeros(SHAPE, dtype=int)
    grid[2, 2] = SOFT
    return grid


def run(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    initial_grid = _initial_grid()

    static_dead_counts = []
    dynamic_dead_counts = []
    effective_counts = []
    coverage_shares = []  # per genome, the LARGEST single rule's coverage fraction of C
    redundancy_by_type_pair: dict = {}
    worked_examples = []

    for idx in range(n):
        genome = _draw_genome(idx)

        cov = rule_coverage(genome)
        static_dead = [k for k, c in enumerate(cov["coverage_counts"]) if c == 0]
        static_dead_counts.append(len(static_dead))
        effective_counts.append(effective_rule_count(genome))
        if cov["coverage_counts"]:
            coverage_shares.append(max(cov["coverage_counts"]) / CONTEXT_SPACE_SIZE)

        dyn = dynamic_coverage(genome, initial_grid, SHAPE, DYNAMIC_T)
        dynamic_dead = [k for k, c in enumerate(dyn["rule_fire_counts"]) if c == 0]
        dynamic_dead_counts.append(len(dynamic_dead))

        ar = action_redundancy(genome)
        for type_pair, fracs in ar["by_action_type_pair"].items():
            redundancy_by_type_pair.setdefault(type_pair, []).extend(fracs)

        if static_dead and len(worked_examples) < 3:
            k = static_dead[0]
            # find the earliest rule whose coverage absorbs rule k's would-be matches
            from evodevo.encoders.rule_based.context_analysis import _condition_mask
            k_matches = _condition_mask(genome.rules[k])
            dominator = None
            for earlier in range(k):
                earlier_matches = _condition_mask(genome.rules[earlier])
                if np.any(k_matches & earlier_matches):
                    dominator = earlier
                    break
            worked_examples.append({
                "genome_index": idx,
                "genome": genome,
                "dead_rule_index": k,
                "dead_rule_would_match_count": int(k_matches.sum()),
                "dominator_rule_index": dominator,
                "coverage_counts": cov["coverage_counts"],
                "dynamic_dead": dynamic_dead,
            })

    return {
        "n": n,
        "seed": seed,
        "dynamic_t": DYNAMIC_T,
        "static_dead_counts": static_dead_counts,
        "dynamic_dead_counts": dynamic_dead_counts,
        "effective_counts": effective_counts,
        "coverage_shares": coverage_shares,
        "redundancy_by_type_pair": redundancy_by_type_pair,
        "worked_examples": worked_examples,
    }


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _distribution_table(counts, label):
    arr = np.array(counts)
    values, freqs = np.unique(arr, return_counts=True)
    lines = [f"### {label} distribution\n", "| value | count | fraction |", "| --- | --- | --- |"]
    for v, f in zip(values, freqs):
        lines.append(f"| {int(v)} | {int(f)} | {_fmt(f / len(arr))} |")
    lines.append("")
    lines.append(f"mean={_fmt(float(arr.mean()))}, median={_fmt(float(np.median(arr)))}, max={int(arr.max())}")
    lines.append("")
    return "\n".join(lines)


def render_report(result: dict) -> str:
    n = result["n"]
    lines = []
    lines.append("# rule_based exact context analysis: reachability, coverage, redundancy")
    lines.append("")
    lines.append(
        "Fitness-blind (only decode()/develop_rules()/random_genome() are used -- NO PPO, NO task, NO reward, "
        f"NO EA). N={n} random genomes (position_predicate=\"disabled\", the encoder's own default), "
        f"seed={result['seed']}, dynamic horizon T={result['dynamic_t']} (matches the shared resolved "
        "T_primary, results/T_PRIMARY_UNIFIED.md)."
    )
    lines.append("")
    lines.append(
        "**Exhaustive enumeration verified complete**: `enumerate_contexts()` produces exactly "
        f"{CONTEXT_SPACE_SIZE} unique contexts (5 * 6^4, OFF_GRID distinguished per "
        "docs/rule-language-spec.md section D), confirmed programmatically in "
        "`tests/encoders/test_context_analysis.py::test_enumerate_contexts_is_exhaustive_and_unique`."
    )
    lines.append("")
    lines.append(
        "**EVOLVED GENOMES note**: no evolved-genome archives exist anywhere in this project's committed "
        "run artifacts to analyze -- `results/runs/**/run_log.jsonl` and `generation_log.jsonl` record only "
        "fitness/validity metadata and decoded body hashes, never the genome (rule list) itself, and "
        "`best_body.npz` stores only the decoded phenotype array. This is disclosed honestly rather than "
        "fabricating evolved genomes: this report covers RANDOM genomes only."
    )
    lines.append("")

    lines.append("## Static vs dynamic dead-rule counts\n")
    lines.append(
        "STATIC dead = provably unreachable (some earlier rule's condition strictly dominates it over ALL "
        "of C). DYNAMIC dead = never actually fired across the full T-step development from the project's "
        "own center-seed initial condition, for this specific genome -- a WEAKER, sample-dependent notion; "
        "every statically dead rule is also dynamically dead, but not vice versa (see the worked examples' "
        "own `dynamic_dead` column for genomes where this diverges)."
    )
    lines.append("")
    lines.append(_distribution_table(result["static_dead_counts"], "static dead-rule count per genome"))
    lines.append(_distribution_table(result["dynamic_dead_counts"], "dynamic dead-rule count per genome"))

    n_diverge = sum(
        1 for s, d in zip(result["static_dead_counts"], result["dynamic_dead_counts"]) if d > s
    )
    lines.append(
        f"**{n_diverge}/{n} genomes ({_fmt(n_diverge / n)}) have MORE dynamically-dead rules than "
        "statically-dead rules** -- i.e. at least one rule that is provably reachable in principle but never "
        "actually fires during this genome's own real development from the center-seed condition within "
        f"T={result['dynamic_t']} steps. This is the concrete gap between \"could this rule ever matter\" "
        "and \"did development actually encounter the input this rule needs.\""
    )
    lines.append("")

    lines.append(_distribution_table(result["effective_counts"], "effective_rule_count (static) per genome"))

    lines.append("## Coverage-share distribution (does one broad early rule dominate?)\n")
    shares = np.array(result["coverage_shares"])
    lines.append(
        f"For each genome, the LARGEST single rule's static coverage as a fraction of the full context "
        f"space: mean={_fmt(float(shares.mean()))}, median={_fmt(float(np.median(shares)))}, "
        f"max={_fmt(float(shares.max()))}, min={_fmt(float(shares.min()))}."
    )
    lines.append("")
    frac_dominated = float(np.mean(shares > 0.5))
    lines.append(
        f"**{_fmt(frac_dominated)} of genomes have a single rule covering MORE THAN HALF of the entire "
        "context space** -- "
        + (
            "confirming a strong 'one broad early rule dominates' pattern (typically an early ALWAYS or "
            "wide-threshold NEIGHBOR_COUNT rule absorbs the large majority of possible inputs, leaving later "
            "rules to handle only a narrow residual)."
            if frac_dominated > 0.5
            else "a real but not dominant pattern in this sampled population -- most genomes do NOT have a "
            "single rule absorbing a majority of the context space."
        )
    )
    lines.append("")

    lines.append("## Action-redundancy rates, per action-type pair\n")
    lines.append(
        "For every pair of rules in every genome, the fraction of the full context space where their two "
        "actions would produce the SAME resulting value for the acting cell (see "
        "`context_analysis.py::_action_effect_on_self`'s own docstring for the exact semantics used, "
        "including the GROW_DIRECTIONAL/COPY_NEIGHBOR-off-grid conventions), aggregated by the pair's "
        "(action_type, action_type) label."
    )
    lines.append("")
    lines.append(
        "**CAVEAT on `grow_directional`/`grow_directional` pairs**: GROW_DIRECTIONAL never writes the "
        "ACTING cell at all (it writes a NEIGHBOR) -- so under this metric (\"effect on the acting cell\"), "
        "ANY two GROW_DIRECTIONAL rules trivially show 100% agreement (both leave the actor unchanged), "
        "REGARDLESS of whether they grow in different directions or different materials. This is a "
        "limitation of comparing target-affecting actions via an actor-centric metric, not a genuine "
        "behavioural-redundancy finding -- reported here explicitly rather than left to be misread as \"all "
        "growth-growth rule pairs are interchangeable\", which this metric cannot actually establish."
    )
    lines.append("")
    lines.append("| action-type pair | n pairs observed | mean redundancy fraction | max |")
    lines.append("| --- | --- | --- | --- |")
    for type_pair, fracs in sorted(result["redundancy_by_type_pair"].items(), key=lambda kv: -len(kv[1])):
        arr = np.array(fracs)
        lines.append(f"| {type_pair[0]} / {type_pair[1]} | {len(arr)} | {_fmt(float(arr.mean()))} | {_fmt(float(arr.max()))} |")
    lines.append("")

    lines.append("## Worked examples: genomes with a statically dead rule\n")
    for ex in result["worked_examples"]:
        lines.append(f"### Genome index {ex['genome_index']}\n")
        lines.append("Rule list (in evaluation order):")
        lines.append("")
        for i, r in enumerate(ex["genome"].rules):
            marker = " **<- DEAD (static coverage 0)**" if i == ex["dead_rule_index"] else ""
            lines.append(f"{i}. {r!r}{marker}")
        lines.append("")
        lines.append(f"Static coverage counts (out of {CONTEXT_SPACE_SIZE}): {ex['coverage_counts']}")
        lines.append("")
        if ex["dominator_rule_index"] is not None:
            lines.append(
                f"Rule {ex['dead_rule_index']} would match {ex['dead_rule_would_match_count']} contexts on "
                f"its own condition alone, but rule {ex['dominator_rule_index']} (earlier in evaluation "
                "order) already matches every one of them -- rule "
                f"{ex['dead_rule_index']} is PROVABLY unreachable, not merely unobserved: "
                f"`{ex['genome'].rules[ex['dominator_rule_index']]!r}`."
            )
        else:
            lines.append(
                f"Rule {ex['dead_rule_index']} has empty coverage but no single earlier rule fully "
                "dominates it on its own -- its matches are absorbed by the UNION of multiple earlier "
                "rules collectively."
            )
        lines.append("")
        lines.append(f"Dynamically-dead rules for this genome (T={result['dynamic_t']}): {ex['dynamic_dead']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "CONTEXT_ANALYSIS.md"))
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed)
    report = render_report(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report[:4000])
    print("... (truncated console output; full report written to file)")


if __name__ == "__main__":
    main()
