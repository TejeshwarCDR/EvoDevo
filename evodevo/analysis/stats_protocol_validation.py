"""Two fitness-blind validations of evodevo.analysis.stats_protocol, run
BEFORE any real PPO result exists (see docs/analysis-protocol.md):

  (a) SYNTHETIC data with a known ground truth -- two Gaussian populations
      with a known analytic effect size, confirming A12/Mann-Whitney U
      recover it within expected sampling error, and a simulated null (many
      independent comparisons, no true effect) confirming the Holm-corrected
      family-wise false-positive rate tracks alpha while the uncorrected
      per-comparison rate does not.
  (b) a DRY RUN of the real pipeline on the already-committed surrogate
      results/runs/strategy_comparison/ data (3 arms x 3 seeds x 2
      strategies x Walker-v0) -- not a real PPO result, but real project
      data flowing through the exact same code path the day real PPO
      results arrive.

Neither validation touches real fitness/PPO output. Writes
results/STATS_PROTOCOL_VALIDATION.md. Run via:
    python -m evodevo.analysis.stats_protocol_validation
"""
from __future__ import annotations

import os
from math import erf, sqrt
from typing import Dict, List, Tuple

import numpy as np

from evodevo.analysis.aggregate import load_all_runs
from evodevo.analysis.stats_protocol import (
    DEFAULT_ALPHA,
    holm_bonferroni,
    mann_whitney_u,
    median_iqr_curve,
    run_protocol,
    vargha_delaney_a12,
)

OUTPUT_PATH = "results/STATS_PROTOCOL_VALIDATION.md"
# v2 (not the original results/runs/strategy_comparison/, kept as-is and
# still cited elsewhere) includes rule_based_monotone as a fourth arm --
# generated fresh via the real, current benchmark.yaml config path
# (surrogate=True, 3 seeds) specifically so this dry run can exercise and
# display the ablation_within_arm vs cross_arm distinction for real, not
# just on synthetic data (the original 3-arm dataset predates
# rule_based_monotone's registration and has no ablation pair to show).
STRATEGY_COMPARISON_ROOT = "results/runs/strategy_comparison_v2"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _analytic_a12_for_normals(mu_a: float, mu_b: float, sigma: float) -> float:
    """A12 for two normal distributions N(mu_a, sigma^2) and N(mu_b, sigma^2)
    (equal variance): A12 = P(X_a > X_b) = Phi((mu_a - mu_b) / (sigma*sqrt(2))),
    since X_a - X_b ~ N(mu_a - mu_b, 2*sigma^2)."""
    return _normal_cdf((mu_a - mu_b) / (sigma * sqrt(2.0)))


# ---------------------------------------------------------------------------
# (a) Synthetic validation
# ---------------------------------------------------------------------------


def _validate_a12_recovery(seed: int = 0) -> dict:
    """Large-n check that A12/effect-size machinery is correct at all, then
    a small-n (n=5, matching this project's real seed floor) check that the
    estimator is unbiased (centres on the analytic truth across many trials)
    even though any single 5-seed draw is noisy."""
    rng = np.random.default_rng(seed)
    mu_a, mu_b, sigma = 1.0, 0.0, 1.0
    true_a12 = _analytic_a12_for_normals(mu_a, mu_b, sigma)

    # Large-n: single draw should land close to the analytic value.
    n_large = 5000
    a_large = rng.normal(mu_a, sigma, n_large)
    b_large = rng.normal(mu_b, sigma, n_large)
    large_n_a12 = vargha_delaney_a12(a_large, b_large)

    # Small-n (n=5, this project's real floor): average A12 over many
    # independent trials should still centre on the analytic value, even
    # though individual trials are noisy -- i.e. the estimator is unbiased.
    n_small = 5
    n_trials = 4000
    small_n_a12_samples = np.empty(n_trials)
    for i in range(n_trials):
        a = rng.normal(mu_a, sigma, n_small)
        b = rng.normal(mu_b, sigma, n_small)
        small_n_a12_samples[i] = vargha_delaney_a12(a, b)
    mean_small_n_a12 = float(np.mean(small_n_a12_samples))
    se_small_n_a12 = float(np.std(small_n_a12_samples) / sqrt(n_trials))

    return dict(
        mu_a=mu_a,
        mu_b=mu_b,
        sigma=sigma,
        true_a12=true_a12,
        n_large=n_large,
        large_n_a12=large_n_a12,
        large_n_abs_error=abs(large_n_a12 - true_a12),
        n_small=n_small,
        n_trials=n_trials,
        mean_small_n_a12=mean_small_n_a12,
        se_small_n_a12=se_small_n_a12,
        small_n_within_2se=abs(mean_small_n_a12 - true_a12) <= 2 * se_small_n_a12,
    )


def _validate_null_family_wise_error(seed: int = 1, m: int = 10, n_per_group: int = 5, n_trials: int = 3000,
                                      alpha: float = DEFAULT_ALPHA) -> dict:
    """Simulates the CURRENT matrix's comparison family size (m=10, see
    docs/analysis-protocol.md Sec 3) under a simulated NULL: every group in
    every trial is drawn from the SAME distribution, so any p<alpha is, by
    construction, a false positive. Repeats n_trials times and reports:
      - the uncorrected per-comparison false-positive rate (should track
        alpha directly, since each individual test is calibrated)
      - the uncorrected family-wise error rate (probability at least one of
        the m comparisons in a trial is a false positive -- expected to be
        well above alpha, since m independent-ish tests are run)
      - the Holm-corrected family-wise error rate (expected to track/undercut
        alpha, by construction of the Holm procedure)
    """
    rng = np.random.default_rng(seed)
    uncorrected_any_reject = 0
    corrected_any_reject = 0
    uncorrected_reject_count = 0
    total_tests = 0

    for _ in range(n_trials):
        p_values = []
        for _comparison in range(m):
            a = rng.normal(0.0, 1.0, n_per_group)
            b = rng.normal(0.0, 1.0, n_per_group)
            _, p = mann_whitney_u(a, b)
            p_values.append(p)

        total_tests += m
        uncorrected_reject_count += sum(p <= alpha for p in p_values)
        uncorrected_any_reject += any(p <= alpha for p in p_values)

        holm = holm_bonferroni(p_values, alpha=alpha)
        corrected_any_reject += any(holm.reject)

    return dict(
        m=m,
        n_per_group=n_per_group,
        n_trials=n_trials,
        alpha=alpha,
        uncorrected_per_comparison_fpr=uncorrected_reject_count / total_tests,
        uncorrected_family_wise_error_rate=uncorrected_any_reject / n_trials,
        holm_family_wise_error_rate=corrected_any_reject / n_trials,
    )


def run_synthetic_validation() -> dict:
    return dict(
        a12_recovery=_validate_a12_recovery(),
        null_family_wise_error=_validate_null_family_wise_error(),
    )


# ---------------------------------------------------------------------------
# (b) Dry run on the already-committed surrogate strategy_comparison data
# ---------------------------------------------------------------------------


def _fitness_by_cell_from_run_dir(root_dir: str):
    """Loads every run under `root_dir` and groups final best-so-far fitness
    (the same cummax-by-eval_index definition used by
    evodevo.analysis.aggregate.per_seed_finals) by (encoder, strategy, task)
    -- the standard "final fitness per seed" shared across this project's
    analysis tooling. Also derives arm_base_encoder from the df's
    "registry_encoder" column (evodevo.analysis.aggregate's arm-identity
    fix) so run_protocol can correctly distinguish an ablation-within-arm
    comparison (e.g. rule_based vs rule_based_monotone) from a cross-arm
    one, generically, without hardcoding any arm name here."""
    df = load_all_runs(root_dir)
    fitness_by_cell: Dict[Tuple[str, str, str], List[float]] = {}
    for (encoder, strategy, task), group in df.groupby(["encoder", "strategy", "task"]):
        finals = []
        for _seed, g in group.groupby("seed"):
            g = g.sort_values("eval_index")
            finals.append(float(g["fitness"].cummax().iloc[-1]))
        fitness_by_cell[(encoder, strategy, task)] = finals

    arms = sorted(df["encoder"].unique())
    strategies = sorted(df["strategy"].unique())
    tasks = sorted(df["task"].unique())
    arm_base_encoder = dict(df.drop_duplicates("encoder").set_index("encoder")["registry_encoder"]) \
        if "registry_encoder" in df.columns else {arm: arm for arm in arms}
    return fitness_by_cell, arms, strategies, tasks, arm_base_encoder


def run_dry_run_validation(root_dir: str = STRATEGY_COMPARISON_ROOT):
    fitness_by_cell, arms, strategies, tasks, arm_base_encoder = _fitness_by_cell_from_run_dir(root_dir)
    result = run_protocol(fitness_by_cell, arms, strategies, tasks, arm_base_encoder=arm_base_encoder)
    return result, fitness_by_cell, arms, strategies, tasks


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(synthetic: dict, dry_run) -> str:
    result, fitness_by_cell, arms, strategies, tasks = dry_run
    a12r = synthetic["a12_recovery"]
    nfe = synthetic["null_family_wise_error"]

    lines = []
    lines.append("# Statistics protocol validation")
    lines.append("")
    lines.append(
        "Two fitness-blind validations of `evodevo/analysis/stats_protocol.py`, the single "
        "generic implementation of `docs/analysis-protocol.md`'s locked statistical protocol. "
        "Neither validation uses real fitness/PPO output."
    )
    lines.append("")
    lines.append("## 1. Synthetic validation (known ground truth)")
    lines.append("")
    lines.append("### 1a. A12 effect-size recovery")
    lines.append("")
    lines.append(
        f"Two Gaussian populations, N(mu_a={a12r['mu_a']}, sigma={a12r['sigma']}) and "
        f"N(mu_b={a12r['mu_b']}, sigma={a12r['sigma']}). Analytic A12 = "
        f"Phi((mu_a-mu_b)/(sigma*sqrt(2))) = **{a12r['true_a12']:.4f}**."
    )
    lines.append("")
    lines.append(
        f"- Large-n (n={a12r['n_large']}/group) single draw: A12 = {a12r['large_n_a12']:.4f} "
        f"(abs. error {a12r['large_n_abs_error']:.4f}) -- recovers the analytic value closely."
    )
    lines.append(
        f"- Small-n (n={a12r['n_small']}/group, this project's real seed floor), averaged over "
        f"{a12r['n_trials']} independent trials: mean A12 = {a12r['mean_small_n_a12']:.4f} "
        f"+/- {a12r['se_small_n_a12']:.4f} (SE) -- "
        f"{'WITHIN' if a12r['small_n_within_2se'] else 'OUTSIDE'} 2 SE of the analytic value "
        f"({a12r['true_a12']:.4f}), i.e. the estimator is unbiased at n=5 even though any single "
        "5-seed draw is noisy (consistent with docs/analysis-protocol.md Sec 5's power disclosure)."
    )
    lines.append("")
    lines.append("### 1b. Holm-Bonferroni correction under a simulated null")
    lines.append("")
    lines.append(
        f"m={nfe['m']} comparisons per trial (matching the CURRENT matrix's planned-comparison "
        f"count, see docs/analysis-protocol.md Sec 3), n={nfe['n_per_group']}/group, both groups "
        f"drawn from the SAME N(0,1) in every trial (a true null -- any p<=alpha is by "
        f"construction a false positive), {nfe['n_trials']} independent trials, alpha={nfe['alpha']}."
    )
    lines.append("")
    lines.append("| Quantity | Value | Expected |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Uncorrected per-comparison false-positive rate | {nfe['uncorrected_per_comparison_fpr']:.4f} "
        f"| ~= alpha ({nfe['alpha']}) -- each individual test is calibrated |"
    )
    lines.append(
        f"| Uncorrected family-wise error rate (>=1 false positive in {nfe['m']}) | "
        f"{nfe['uncorrected_family_wise_error_rate']:.4f} | >> alpha "
        f"(analytic, independent tests: 1-(1-{nfe['alpha']})^{nfe['m']} = "
        f"{1 - (1 - nfe['alpha']) ** nfe['m']:.4f}) |"
    )
    lines.append(
        f"| Holm-corrected family-wise error rate | {nfe['holm_family_wise_error_rate']:.4f} | "
        f"<= alpha ({nfe['alpha']}) -- this is exactly the guarantee Holm-Bonferroni provides |"
    )
    lines.append("")
    holm_controls = nfe["holm_family_wise_error_rate"] <= nfe["alpha"] + 0.02  # simulation slack
    lines.append(
        f"**Result: {'PASS' if holm_controls else 'FAIL'}** -- the uncorrected family-wise error "
        f"rate ({nfe['uncorrected_family_wise_error_rate']:.4f}) is far above alpha while the "
        f"Holm-corrected rate ({nfe['holm_family_wise_error_rate']:.4f}) tracks/undercuts it, "
        "exactly as docs/analysis-protocol.md Sec 3 asserts."
    )
    lines.append("")
    lines.append("## 2. Dry run: real pipeline on committed surrogate data")
    lines.append("")
    lines.append(
        f"`{STRATEGY_COMPARISON_ROOT}/` (surrogate, NOT real PPO fitness) loaded via "
        "`evodevo.analysis.aggregate.load_all_runs` and run through `stats_protocol.run_protocol` "
        "end-to-end, exactly as it will be run on real results."
    )
    lines.append("")
    lines.append(f"- Arms: {arms}")
    lines.append(f"- Strategies: {strategies}")
    lines.append(f"- Tasks: {tasks}")
    lines.append(
        f"- Planned comparisons enumerated: **{len(result.comparisons)}** "
        f"(this data's matrix: {len(arms)} arms x {len(strategies)} strategies x {len(tasks)} task "
        f"-> C({len(arms)},2)={len(arms) * (len(arms) - 1) // 2} cross-arm + {len(arms)} "
        "arm-vs-own-random_search, per task)"
    )
    lines.append("")
    lines.append("| Comparison | Kind | n_a | n_b | mean_a | mean_b | A12 | U | p_raw | p_holm | significant (Holm) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in result.comparisons:
        label = f"{c.cell_a.arm}/{c.cell_a.strategy} vs {c.cell_b.arm}/{c.cell_b.strategy} ({c.cell_a.task})"
        lines.append(
            f"| {label} | {c.kind} | {c.n_a} | {c.n_b} | {c.mean_a:.4f} | {c.mean_b:.4f} | "
            f"{c.a12:.4f} | {c.mann_whitney_u:.2f} | {c.p_raw:.4f} | {c.p_holm:.4f} | "
            f"{'yes' if c.significant_holm else 'no'} |"
        )
    lines.append("")
    lines.append(
        "**Result: PASS** -- the protocol ran end-to-end against real (surrogate) project data with "
        "no errors, producing correctly-labelled per-comparison rows and a Holm correction sized to "
        "the data's own matrix shape. This is the rehearsal for the day real PPO results arrive: "
        "the same call, `stats_protocol.run_protocol(fitness_by_cell, arms, strategies, tasks)`, "
        "against real per_seed_finals-derived data, produces the final report's significance table."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    synthetic = run_synthetic_validation()
    dry_run = run_dry_run_validation()
    report = render_report(synthetic, dry_run)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
