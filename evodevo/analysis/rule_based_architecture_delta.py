"""Builds the rule_based before/after evidence table: results/rule_based/
ARCHITECTURE_DELTA.md -- the rule_based counterpart of
evodevo/analysis/architecture_delta.py (NCA's own table).

"OLD" = the pre-R2/R3 defaults: `conflict_resolution="row_major_last_wins"`
(order-DEPENDENT tie-break, the historical behaviour) and
`position_predicate="enabled"` (ConditionType.POSITION generated in random
genomes, later found to break translation-equivariance -- see
docs/decisions.md's "Restore translation-equivariance (Proposition R3)..."
row).

"NEW" = the FINAL resolved primary rule_based config, per
evodevo/encoders/rule_based/resolved_config.py::PRIMARY:
`conflict_resolution="first_rule_index"` (docs/decisions.md's "Make the
rule_based interpreter provably order-independent by default" row) and
`position_predicate="disabled"` (the R3 row above). `boundary="no_op"` and
`neighbourhood="von_neumann"` are held at the interpreter's entire
historical behaviour in BOTH old and new (they are not part of either
R2 or R3's semantics change), so this table isolates exactly the two axes
that changed.

Both configurations are measured FRESH, at IDENTICAL settings (T_primary=100
-- the shared, unified T-shift-stability-selected value, results/
T_PRIMARY_UNIFIED.md -- init_condition=center_seed, seed=0, N=2000), across
the full structural suite (validity_rate, diversity, redundancy,
settling_profile, phenotypic_displacement) -- a self-contained paper figure,
not a patchwork of citations from different N/T settings.

Fitness-blind throughout: only decode()/develop_rules()/random_genome()/
validate_body()/mutate() and evodevo.analysis.structural.* are used -- NO
PPO, NO task, NO reward, NO EA.

Usage:
    python -m evodevo.analysis.rule_based_architecture_delta [--n 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.encoders.rule_based.resolved_config import PRIMARY

DEFAULT_N = 2000
DEFAULT_SEED = 0
INIT_CONDITION = "center_seed"  # P3 decision
T_PRIMARY = 100  # shared T_primary, results/T_PRIMARY_UNIFIED.md

RESULTS_DIR = "results/rule_based"
RAW_ROOT = "results/rule_based/architecture_delta_raw"

OLD_CONFLICT_RESOLUTION = "row_major_last_wins"
OLD_POSITION_PREDICATE = "enabled"


def _build_encoder(conflict_resolution: str, position_predicate: str, seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=T_PRIMARY, init_condition=INIT_CONDITION, seed=seed)
    return RuleBasedEncoder(
        factors=factors,
        conflict_resolution=conflict_resolution,
        position_predicate=position_predicate,
        boundary=PRIMARY.boundary,
        neighbourhood=PRIMARY.neighbourhood,
        action_monotonicity=PRIMARY.action_monotonicity,
    )


def _measure(encoder: RuleBasedEncoder, n: int, seed: int, label: str) -> dict:
    factors = encoder.factors
    output_root = os.path.join(RAW_ROOT, label)

    validity = S.validity_rate(encoder, factors, n, np.random.default_rng(seed), output_root=output_root)
    diversity = S.diversity(encoder, factors, n, np.random.default_rng(seed), output_root=output_root)
    redundancy = S.redundancy(encoder, factors, n, np.random.default_rng(seed), output_root=output_root)
    settling = S.settling_profile(
        encoder, factors, n, np.random.default_rng(seed), t_max=T_PRIMARY, output_root=output_root
    )
    displacement = S.phenotypic_displacement(encoder, factors, n, np.random.default_rng(seed), output_root=output_root)

    return {
        "valid_rate": validity.valid_rate,
        "degenerate_fraction": validity.degenerate_fraction,
        "unique_valid_body_fraction": diversity.unique_valid_body_fraction,
        "material_entropy_bits": diversity.material_entropy_bits,
        "genomes_per_unique_phenotype": redundancy.genomes_per_unique_phenotype,
        "largest_neutral_class_fraction": redundancy.largest_neutral_class_fraction,
        "never_settled_fraction": settling.never_settled_fraction,
        "mean_settle_step": settling.mean_settle_step,
        "mean_hamming_displacement": displacement.mean_hamming_distance,
        "median_hamming_displacement": displacement.median_hamming_distance,
        "fraction_zero_displacement": displacement.fraction_zero_displacement,
    }


def run_architecture_delta(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    old_encoder = _build_encoder(OLD_CONFLICT_RESOLUTION, OLD_POSITION_PREDICATE, seed)
    new_encoder = _build_encoder(PRIMARY.conflict_resolution, PRIMARY.position_predicate, seed)

    old_metrics = _measure(old_encoder, n, seed, "old")
    new_metrics = _measure(new_encoder, n, seed, "new")

    rows = {key: (old_metrics[key], new_metrics[key]) for key in old_metrics}

    return {"n": n, "seed": seed, "t_primary": T_PRIMARY, "rows": rows}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x, digits=4):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


ROW_LABELS = {
    "valid_rate": "validity rate",
    "degenerate_fraction": "degenerate fraction",
    "unique_valid_body_fraction": "unique-valid-body fraction",
    "material_entropy_bits": "material entropy (bits)",
    "genomes_per_unique_phenotype": "genomes per unique phenotype (redundancy)",
    "largest_neutral_class_fraction": "largest neutral-class fraction",
    "never_settled_fraction": "never_settled_fraction",
    "mean_settle_step": "mean settle_step",
    "mean_hamming_displacement": "mean mutation displacement (Hamming)",
    "median_hamming_displacement": "median mutation displacement (Hamming)",
    "fraction_zero_displacement": "fraction zero displacement",
}

# Rows where a smaller value is the intuitively "better"/expected direction,
# used only to phrase the narrative -- the table itself always shows raw
# UP/DOWN/unchanged with no editorializing.
LOWER_IS_EXPECTED_WORSE = {"never_settled_fraction", "degenerate_fraction", "largest_neutral_class_fraction"}


def render_report(result: dict) -> str:
    n = result["n"]
    rows = result["rows"]
    lines = []
    lines.append("# rule_based architecture before/after: the R2/R3 evidence table")
    lines.append("")
    lines.append(
        f"Fitness-blind throughout (decode()/develop_rules()/random_genome()/validate_body()/mutate() plus "
        f"evodevo.analysis.structural.* only -- NO PPO, NO task, NO reward, NO EA). N={n}, "
        f"T_primary={result['t_primary']} (the shared, unified T-shift-stability-selected value, results/"
        f"T_PRIMARY_UNIFIED.md), init_condition=`{INIT_CONDITION}` (P3), seed={result['seed']}, both "
        "configurations measured at IDENTICAL settings."
    )
    lines.append("")
    lines.append(
        "**OLD** = the pre-R2/R3 defaults: `conflict_resolution=\"row_major_last_wins\"` (order-DEPENDENT "
        "tie-break -- 12.95% of genomes affected by rule ORDER alone, results/rule_based/"
        "ORDER_INDEPENDENCE_AUDIT.md) and `position_predicate=\"enabled\"` (ConditionType.POSITION generated "
        "in random genomes, later found to break translation-equivariance -- 0.2140 fraction of genomes "
        "violate D(translate(s))==translate(D(s)) on a torus, results/rule_based/LOCALITY_STUDY.md)."
    )
    lines.append("")
    lines.append(
        "**NEW** = the FINAL resolved primary (evodevo/encoders/rule_based/resolved_config.py::PRIMARY): "
        "`conflict_resolution=\"first_rule_index\"` (the only order-independent candidate with EQUAL-OR-"
        "BETTER validity than the historical baseline, docs/decisions.md's \"Make the rule_based interpreter "
        "provably order-independent by default\" row) and `position_predicate=\"disabled\"` (R3 restored, "
        "0.0000 violations on the same torus test, docs/decisions.md's \"Restore translation-equivariance "
        "(Proposition R3)...\" row). `boundary=\"no_op\"` and `neighbourhood=\"von_neumann\"` are held fixed "
        "at the interpreter's entire historical behaviour in BOTH columns -- neither R2 nor R3 changed those "
        "axes, so this table isolates exactly the two axes that did change."
    )
    lines.append("")

    lines.append("## The table")
    lines.append("")
    lines.append("| quantity | OLD (row_major_last_wins, position enabled) | NEW (first_rule_index, position disabled) | direction |")
    lines.append("| --- | --- | --- | --- |")
    for key, (old_v, new_v) in rows.items():
        label = ROW_LABELS.get(key, key)
        if old_v is None or new_v is None:
            direction = "N/A"
        elif new_v > old_v:
            direction = "UP"
        elif new_v < old_v:
            direction = "DOWN"
        else:
            direction = "unchanged"
        lines.append(f"| {label} | {_fmt(old_v)} | {_fmt(new_v)} | {direction} |")
    lines.append("")

    lines.append("## What changed and why")
    lines.append("")
    validity_old, validity_new = rows["valid_rate"]
    diversity_old, diversity_new = rows["unique_valid_body_fraction"]
    never_settled_old, never_settled_new = rows["never_settled_fraction"]
    disp_old, disp_new = rows["mean_hamming_displacement"]
    lines.append(
        f"**Validity**: {_fmt(validity_old)} -> {_fmt(validity_new)}. R2's own four-way evidence "
        "(results/rule_based/CONFLICT_POLICY.md) already showed `first_rule_index` beats the historical "
        "`row_major_last_wins` baseline on validity alone; R3's own cost disclosure (results/rule_based/"
        "LOCALITY_STUDY.md) showed disabling `position_predicate` COSTS 0.0250 validity at THOSE settings "
        "(N=2000, T=100, conflict_resolution already first_rule_index). The net row above is the two effects "
        "composed together, measured fresh rather than summed from the two separate reports, since those "
        "reports used different intermediate baselines."
    )
    lines.append(
        f"**Diversity** (unique-valid-body fraction): {_fmt(diversity_old)} -> {_fmt(diversity_new)}, the "
        "same composition of R2's improvement and R3's cost."
    )
    lines.append(
        f"**never_settled_fraction**: {_fmt(never_settled_old)} -> {_fmt(never_settled_new)}. Per "
        "docs/decisions.md's R2 row, conflict resolution changes WHICH body a cycling/growing genome ends up "
        "with, not WHETHER it settles, and R3's own cost disclosure states never_settled_fraction is "
        "essentially unchanged by disabling POSITION -- so this row is expected to move little, and any "
        "large observed shift here would be a genuine, unexpected finding worth flagging."
    )
    lines.append(
        f"**Mutation displacement** (mean Hamming): {_fmt(disp_old)} -> {_fmt(disp_new)}. The mutation "
        "operator itself (`mutate_developmental_rules`) is untouched by R2/R3 -- only DECODE's tie-break and "
        "predicate-generation changed -- so any shift here is purely a downstream consequence of which body "
        "a given (parent, mutated-child) genome pair now decodes to, not a change in mutation strength itself."
    )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This module never computes or consults a fitness, reward, or task-performance quantity: every "
        "number above is derived from decode()/develop_rules()/random_genome()/validate_body()/mutate() or "
        "evodevo.analysis.structural.* alone."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "ARCHITECTURE_DELTA.md"))
    args = parser.parse_args()

    result = run_architecture_delta(n=args.n, seed=args.seed)
    report = render_report(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
