# Statistics protocol validation

Two fitness-blind validations of `evodevo/analysis/stats_protocol.py`, the single generic implementation of `docs/analysis-protocol.md`'s locked statistical protocol. Neither validation uses real fitness/PPO output.

## 1. Synthetic validation (known ground truth)

### 1a. A12 effect-size recovery

Two Gaussian populations, N(mu_a=1.0, sigma=1.0) and N(mu_b=0.0, sigma=1.0). Analytic A12 = Phi((mu_a-mu_b)/(sigma*sqrt(2))) = **0.7602**.

- Large-n (n=5000/group) single draw: A12 = 0.7544 (abs. error 0.0059) -- recovers the analytic value closely.
- Small-n (n=5/group, this project's real seed floor), averaged over 4000 independent trials: mean A12 = 0.7597 +/- 0.0025 (SE) -- WITHIN 2 SE of the analytic value (0.7602), i.e. the estimator is unbiased at n=5 even though any single 5-seed draw is noisy (consistent with docs/analysis-protocol.md Sec 5's power disclosure).

### 1b. Holm-Bonferroni correction under a simulated null

m=10 comparisons per trial (matching the CURRENT matrix's planned-comparison count, see docs/analysis-protocol.md Sec 3), n=5/group, both groups drawn from the SAME N(0,1) in every trial (a true null -- any p<=alpha is by construction a false positive), 3000 independent trials, alpha=0.05.

| Quantity | Value | Expected |
|---|---|---|
| Uncorrected per-comparison false-positive rate | 0.0537 | ~= alpha (0.05) -- each individual test is calibrated |
| Uncorrected family-wise error rate (>=1 false positive in 10) | 0.4257 | >> alpha (analytic, independent tests: 1-(1-0.05)^10 = 0.4013) |
| Holm-corrected family-wise error rate | 0.0000 | <= alpha (0.05) -- this is exactly the guarantee Holm-Bonferroni provides |

**Result: PASS** -- the uncorrected family-wise error rate (0.4257) is far above alpha while the Holm-corrected rate (0.0000) tracks/undercuts it, exactly as docs/analysis-protocol.md Sec 3 asserts.

## 2. Dry run: real pipeline on committed surrogate data

`results/runs/strategy_comparison/` (surrogate, NOT real PPO fitness) loaded via `evodevo.analysis.aggregate.load_all_runs` and run through `stats_protocol.run_protocol` end-to-end, exactly as it will be run on real results.

- Arms: ['direct', 'nca', 'rule_based']
- Strategies: ['ea', 'random_search']
- Tasks: ['Walker-v0']
- Planned comparisons enumerated: **6** (this data's matrix: 3 arms x 2 strategies x 1 task -> C(3,2)=3 cross-arm + 3 arm-vs-own-random_search, per task)

| Comparison | Kind | n_a | n_b | mean_a | mean_b | A12 | U | p_raw | p_holm | significant (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| direct/ea vs nca/ea (Walker-v0) | cross_arm | 3 | 3 | 14.1333 | 26.0000 | 0.0000 | 0.00 | 0.0495 | 0.2972 | no |
| direct/ea vs rule_based/ea (Walker-v0) | cross_arm | 3 | 3 | 14.1333 | 26.0000 | 0.0000 | 0.00 | 0.0495 | 0.2972 | no |
| nca/ea vs rule_based/ea (Walker-v0) | cross_arm | 3 | 3 | 26.0000 | 26.0000 | 0.5000 | 4.50 | 1.0000 | 1.0000 | no |
| direct/ea vs direct/random_search (Walker-v0) | arm_vs_own_random_search | 3 | 3 | 14.1333 | 10.0800 | 1.0000 | 9.00 | 0.0495 | 0.2972 | no |
| nca/ea vs nca/random_search (Walker-v0) | arm_vs_own_random_search | 3 | 3 | 26.0000 | 26.0000 | 0.5000 | 4.50 | 1.0000 | 1.0000 | no |
| rule_based/ea vs rule_based/random_search (Walker-v0) | arm_vs_own_random_search | 3 | 3 | 26.0000 | 26.0000 | 0.5000 | 4.50 | 1.0000 | 1.0000 | no |

**Result: PASS** -- the protocol ran end-to-end against real (surrogate) project data with no errors, producing correctly-labelled per-comparison rows and a Holm correction sized to the data's own matrix shape. This is the rehearsal for the day real PPO results arrive: the same call, `stats_protocol.run_protocol(fitness_by_cell, arms, strategies, tasks)`, against real per_seed_finals-derived data, produces the final report's significance table.
