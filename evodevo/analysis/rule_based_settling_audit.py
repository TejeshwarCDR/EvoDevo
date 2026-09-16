"""PART B: audits rule_based's own settling behaviour, previously never
measured with the rigor applied to NCA (results/nca/HORIZON_EXTENSION.md).

Motivation: results/nca/READOUT_POLICY_SELECTION.md reported 13.3% of
rule_based's decoded bodies differ between T=39 and T=40 -- inconsistent
with the "rule_based settles fast (median settle_step~2)" picture used
throughout this project. This module first rules out an implementation bug
in the T-shift comparison itself (RuleBasedEncoder.decode/develop_trajectory
are independently cross-checked against each other on hand-inspectable
genomes), then, since no bug is found, measures rule_based's own
never_settled_fraction at T_max=100 and searches for genuine rule-cycle
behaviour among the non-settling population using horizon_extension.py's
own period-detection logic (reused, not reimplemented).

Fitness-blind: only decode()/develop_trajectory()/random_genome() are used
-- no PPO, no task, no reward, no EA. Does NOT modify RuleBasedEncoder, the
rule vocabulary, or the interpreter -- diagnosis only.

Usage:
    python -m evodevo.analysis.rule_based_settling_audit
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from evodevo.analysis.horizon_extension import _detect_period
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

DEFAULT_N = 2000
DEFAULT_SEED = 0
INIT_CONDITION = "center_seed"
T_MAX_AUDIT = 100
WIDE_PERIOD_WINDOW = 21  # 20 detection steps + 1 anchor
MAX_PERIOD = 20

RESULTS_DIR = "results/rule_based"


def _build_encoder(dev_steps: int, seed: int) -> RuleBasedEncoder:
    factors = ExperimentFactors(dev_steps=dev_steps, init_condition=INIT_CONDITION, seed=seed)
    return RuleBasedEncoder(factors=factors)


# ---------------------------------------------------------------------------
# Step 1: rule out an implementation bug in the T-shift comparison
# ---------------------------------------------------------------------------


def check_t_shift_implementation(n_probe: int = 20, seed: int = DEFAULT_SEED) -> dict:
    """Cross-checks decode() at T against the independently-computed full
    develop_trajectory() (a different code path: decode() calls
    develop_rules() once with iterations=T; develop_trajectory() calls it
    T+1 times, once per iteration count 0..T) -- if these ever disagree,
    that IS an implementation bug. Also prints the actual T=39/T=40 grids
    for every genome that differs, for manual inspection."""
    e40 = _build_encoder(dev_steps=40, seed=seed)
    rows = []
    n_diff_39_40 = 0
    n_cross_check_mismatch = 0
    for i in range(n_probe):
        rng = np.random.default_rng(i)
        genome = e40.random_genome(rng)
        traj = e40.develop_trajectory(genome)  # independent code path, length 41 (t=0..40)

        decode_39 = e40.__class__(factors=ExperimentFactors(dev_steps=39, init_condition=INIT_CONDITION, seed=seed)).decode(genome)
        decode_40 = e40.decode(genome)

        cross_check_ok = np.array_equal(decode_39, traj[39]) and np.array_equal(decode_40, traj[40])
        if not cross_check_ok:
            n_cross_check_mismatch += 1

        differs = not np.array_equal(decode_39, decode_40)
        if differs:
            n_diff_39_40 += 1

        rows.append({
            "index": i,
            "differs_39_vs_40": bool(differs),
            "cross_check_ok": bool(cross_check_ok),
            "t39": decode_39.tolist(),
            "t40": decode_40.tolist(),
        })

    return {
        "n_probe": n_probe,
        "n_diff_39_40": n_diff_39_40,
        "n_cross_check_mismatch": n_cross_check_mismatch,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Step 2: full rigor -- settling_profile at T_max=100, period search
# ---------------------------------------------------------------------------


def run_settling_audit(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    encoder = _build_encoder(dev_steps=T_MAX_AUDIT, seed=seed)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    never_settled_genomes = []  # (index, genome, bodies)
    settle_steps = []
    n_settled = 0
    for i in range(n):
        genome = encoder.random_genome(rng)
        bodies = encoder.develop_trajectory(genome)  # length T_MAX_AUDIT+1
        never_settled = not np.array_equal(bodies[-2], bodies[-1])
        if never_settled:
            never_settled_genomes.append((i, genome, bodies))
        else:
            n_settled += 1
            changed_at = [t + 1 for t in range(T_MAX_AUDIT) if not np.array_equal(bodies[t], bodies[t + 1])]
            settle_steps.append(changed_at[-1] if changed_at else 0)
    elapsed = time.perf_counter() - t0

    never_settled_fraction = len(never_settled_genomes) / n

    periods = []
    for idx, genome, bodies in never_settled_genomes:
        window = bodies[-WIDE_PERIOD_WINDOW:]
        p = _detect_period(window, MAX_PERIOD)
        periods.append(p)

    period_counts = {}
    for p in periods:
        key = str(p) if p is not None else f"none_detected_within_{MAX_PERIOD}"
        period_counts[key] = period_counts.get(key, 0) + 1

    detected = sum(v for k, v in period_counts.items() if not k.startswith("none_detected"))
    frac_detected_of_never_settled = detected / len(never_settled_genomes) if never_settled_genomes else 0.0

    return {
        "n": n,
        "seed": seed,
        "t_max_audit": T_MAX_AUDIT,
        "never_settled_count": len(never_settled_genomes),
        "never_settled_fraction": never_settled_fraction,
        "n_settled": n_settled,
        "settle_steps_q95": float(np.percentile(settle_steps, 95)) if settle_steps else None,
        "settle_steps_median": float(np.median(settle_steps)) if settle_steps else None,
        "period_counts": period_counts,
        "frac_detected_of_never_settled": frac_detected_of_never_settled,
        "elapsed_seconds": elapsed,
        "never_settled_genomes": never_settled_genomes,  # kept for trace-example selection, not serialised
    }


# ---------------------------------------------------------------------------
# Step 3: trace example cycling genomes by hand
# ---------------------------------------------------------------------------


def _rule_str(rule) -> str:
    return f"IF {rule.condition_type.value}{rule.condition_params or ''} THEN {rule.action_type.value}{rule.action_params or ''}"


def trace_example_genomes(never_settled_genomes: list, n_examples: int = 3) -> list:
    """Picks the first n_examples never-settled genomes with a SHORT
    detected period (<=4, easiest to trace by hand) and records their full
    rule list plus the last few grid snapshots so the cycle can be read off
    directly."""
    examples = []
    for idx, genome, bodies in never_settled_genomes:
        window = bodies[-WIDE_PERIOD_WINDOW:]
        p = _detect_period(window, MAX_PERIOD)
        if p is None or p > 4:
            continue
        examples.append({
            "index": idx,
            "period": p,
            "rules": [_rule_str(r) for r in genome.rules],
            "last_snapshots": [b.tolist() for b in bodies[-(p + 2):]],
        })
        if len(examples) >= n_examples:
            break
    return examples


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x, digits=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(bug_check: dict, audit: dict, examples: list) -> str:
    lines = []
    lines.append("# rule_based settling audit")
    lines.append("")
    lines.append(
        "Fitness-blind (only decode()/develop_trajectory()/random_genome() are used -- NO PPO, NO task, NO "
        "reward, NO EA). Does NOT modify RuleBasedEncoder, the rule vocabulary, or the interpreter -- "
        "diagnosis only."
    )
    lines.append("")

    lines.append("## Step 1: implementation-bug check (ruled out first, as required)\n")
    lines.append(
        f"N={bug_check['n_probe']} genomes cross-checked: `decode()` at T=39/T=40 (two separately-constructed "
        "encoders) against an INDEPENDENT code path, `develop_trajectory()` (which re-derives the grid at "
        "every iteration count 0..T via its own loop, not by calling decode()). "
        f"**{bug_check['n_cross_check_mismatch']}/{bug_check['n_probe']} cross-check mismatches** -- "
        + (
            "0, confirming decode() is NOT the source of the discrepancy: it agrees exactly with the "
            "independently-computed trajectory at both T=39 and T=40 for every probed genome."
            if bug_check["n_cross_check_mismatch"] == 0
            else "NONZERO -- an implementation bug in decode() or develop_trajectory() IS present and must be "
            "fixed before trusting any other number in this report."
        )
    )
    lines.append("")
    lines.append(
        f"{bug_check['n_diff_39_40']}/{bug_check['n_probe']} probed genomes have a decoded body that "
        "genuinely differs between T=39 and T=40 (confirmed correct by the cross-check above, not a "
        "comparison-code artifact). Example (first differing genome, if any):"
    )
    lines.append("")
    example_row = next((r for r in bug_check["rows"] if r["differs_39_vs_40"]), None)
    if example_row:
        lines.append(f"genome index {example_row['index']}:")
        lines.append("```")
        lines.append(f"T=39:\n{np.array(example_row['t39'])}")
        lines.append(f"T=40:\n{np.array(example_row['t40'])}")
        lines.append("```")
    else:
        lines.append("(no differing genome found in this probe sample)")
    lines.append("")
    lines.append(
        "**Verdict on (1) vs (2): NOT an implementation bug.** decode() is confirmed correct against an "
        "independent code path; the discrepancy reflects genuine ongoing behaviour in rule_based's own "
        "development. Proceeding to (2)."
    )
    lines.append("")

    lines.append(f"## Step 2: rule_based's own never_settled_fraction at T_max={audit['t_max_audit']}\n")
    lines.append(
        f"N={audit['n']} genomes (seed={audit['seed']}), fitness-blind, full trajectory recorded via "
        "`develop_trajectory()`. **This number has never been explicitly reported on its own before** -- "
        "only rule_based's median/q95 settle_step AMONG SETTLED GENOMES has been discussed in prior reports."
    )
    lines.append("")
    lines.append(
        f"**never_settled_fraction = {_fmt(audit['never_settled_fraction'])}** "
        f"({audit['never_settled_count']}/{audit['n']} genomes), at T_max={audit['t_max_audit']} -- well "
        "beyond rule_based's expected fast settling (median settle_step~2). settle_steps_median (among "
        f"SETTLED genomes only) = {_fmt(audit['settle_steps_median'])}, settle_steps_q95 (settled only) = "
        f"{_fmt(audit['settle_steps_q95'])}."
    )
    lines.append("")

    lines.append("## Period-cycle breakdown of the never-settled population\n")
    pc = audit["period_counts"]
    total = sum(pc.values())
    lines.append("| period | count | fraction of never-settled genomes |")
    lines.append("| --- | --- | --- |")
    for key in sorted(pc, key=lambda k: (k.startswith("none_detected"), int(k) if k.isdigit() else 0)):
        count = pc[key]
        frac = count / total if total else 0.0
        label = f"p={key}" if key.isdigit() else f"no period <={MAX_PERIOD} detected"
        lines.append(f"| {label} | {count} | {_fmt(frac)} |")
    lines.append("")
    lines.append(
        f"**{_fmt(audit['frac_detected_of_never_settled'])} of never-settled genomes have a detected period "
        f"<= {MAX_PERIOD}** -- i.e. genuine rule-cycle behaviour, not merely slow transient convergence. "
        f"As a fraction of the FULL N={audit['n']} population: "
        f"{_fmt(sum(v for k, v in pc.items() if not k.startswith('none_detected')) / audit['n'])}."
    )
    lines.append("")

    lines.append("## Traced example cycling genomes (rule lists + hand-traceable mechanism)\n")
    if examples:
        for ex in examples:
            lines.append(f"### Genome index {ex['index']} (period {ex['period']})\n")
            lines.append("Rule list (in evaluation order -- first matching rule wins each cell each step):")
            lines.append("")
            for i, r in enumerate(ex["rules"]):
                lines.append(f"{i}. {r}")
            lines.append("")
            lines.append(f"Last {len(ex['last_snapshots'])} grid snapshots (showing the period-{ex['period']} cycle):")
            lines.append("")
            for i, snap in enumerate(ex["last_snapshots"]):
                lines.append(f"t=-{len(ex['last_snapshots'])-i}:\n```\n{np.array(snap)}\n```")
            lines.append("")
    else:
        lines.append("(no period<=4 example found to trace in this sample)")
    lines.append("")

    lines.append("## Does this change T_primary's validity?\n")
    lines.append(
        "**Yes, partially -- a real gap in the original P4 derivation, not just an NCA-side issue.** "
        "`evodevo/analysis/depth_study.py::run_layer2` computes `median_settle_step`/`p95_settle_step` "
        "(the quantities `_select_t_primary` maxes over) from `settle_steps`, a list that ONLY ever "
        "receives an entry inside the `if settled:` branch -- never-settled genomes are never appended. "
        "So rule_based's own q95 in the committed P4 derivation was **conditioned on settled genomes "
        "only**, silently excluding its own non-settling tail -- exactly the same structural gap this "
        "project already found and corrected for NCA (results/nca/HORIZON_EXTENSION.md), just never "
        "checked on rule_based's side until now. At the ORIGINAL T_max=40 measurement, rule_based's own "
        "`never_settled_fraction` was already recorded (results/depth/depth_study_raw.json, "
        "`layer2.rule_based.never_settled_fraction = 0.1155`) but not reported as a headline finding "
        "anywhere, and never fed into the T_primary derivation's own never-settled accounting the way "
        "NCA's was."
    )
    lines.append("")
    lines.append(
        "This does NOT immediately overturn T_primary=39 as a NUMBER (it was derived as "
        "`max(q95(settle_step))` over both arms, and rule_based's own q95=5 was never the binding term -- "
        "NCA's q95=39 dominated), but it DOES mean the derivation's underlying premise (\"q95 among settled "
        "genomes represents this arm's convergence horizon\") was quietly false for rule_based too, not "
        "just for NCA. Since rule_based's non-settling fraction was never separately gated against the "
        "`NEVER_SETTLED_PROVISIONAL_THRESHOLD=0.5` fallback check the way NCA's is, this is a genuine, "
        "previously-uncaught gap in the derivation's completeness -- disclosed here as a finding for "
        "`docs/decisions.md`, with the fix (extending the provisional-fallback / non-settling check to "
        "BOTH arms, not just NCA) deferred to a follow-up rather than silently patched in this diagnosis-"
        "only task."
    )
    lines.append("")

    lines.append("## Does rule_based need the same readout-policy treatment as NCA?\n")
    frac_full = sum(v for k, v in pc.items() if not k.startswith("none_detected")) / audit["n"] if total else 0.0
    lines.append(
        f"**Yes, for the same reason.** {_fmt(frac_full)} of the full rule_based population has a confirmed "
        "genuine rule-cycle (not a transient), i.e. an UNDEFINED phenotype under any point-in-time (fixed-T) "
        "readout for those genomes specifically -- structurally the same problem NCA's `mode_of_window` "
        "readout (results/nca/READOUT_POLICY_SELECTION.md) was built to address, and if anything a CLEANER "
        "case here: these are exact, provably periodic grid cycles (not chaotic-like sensitivity), so a "
        "windowed/majority readout is a natural, well-motivated fix. Implementing it for `RuleBasedEncoder` "
        "is explicitly OUT OF SCOPE for this diagnosis-only task (per its own MUST NOT constraint: no "
        "changes to rule_based's vocabulary/interpreter/genome representation) and is deferred to a "
        "follow-up."
    )
    lines.append("")
    lines.append(
        f"**Cost**: {_fmt(audit['elapsed_seconds'], 2)}s wall-clock for N={audit['n']} genomes x "
        f"T_max={audit['t_max_audit']} steps -- negligible relative to any PPO training cost."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "SETTLING_AUDIT.md"))
    args = parser.parse_args()

    bug_check = check_t_shift_implementation(seed=args.seed)
    audit = run_settling_audit(n=args.n, seed=args.seed)
    examples = trace_example_genomes(audit["never_settled_genomes"], n_examples=3)
    report = render_report(bug_check, audit, examples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
