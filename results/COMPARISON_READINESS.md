# Comparison Readiness Report

Verification-only pass over the P1-P7 comparison-fairness workstream. Nothing in this report was fixed, tuned, or re-run to look better -- every number either comes from an existing committed artifact or a read-only check performed at report-generation time. If a check failed or an artifact was missing, that is reported as a finding below, not silently worked around.

## Section 0 -- Provenance

- Git commit: `1fb8b565df3ed35ec229e9346200c3cceb56f6a0`
- Working tree clean: **True**
- Report generated (UTC): `2026-09-15T15:04:33.779866+00:00`
- evogym commit (`external/evogym`): `f63c649c688adda1accad5f86731aad96606388a`
- Key package versions:
  - `evogym`: `no __version__ attribute`
  - `matplotlib`: `3.10.9`
  - `numpy`: `1.26.4`
  - `pandas`: `2.3.3`
  - `platform`: `macOS-26.5.1-arm64-arm-64bit`
  - `pytest`: `9.1.1`
  - `python`: `3.10.20`
  - `yaml`: `6.0.3`

**Resolved ExperimentFactors for the primary configuration** (P3 init_condition x P4 T_primary; mutation strength is calibrated per-arm in P5 via each encoder's own constructor kwarg, not via `mutation_scale` below):
```
dev_steps = 20
hidden_channels = 3
init_condition = 'center_seed'
mutation_scale = 1.0
neighbourhood = 'von_neumann'
seed = 0
```

## Section 1 -- Factor infrastructure (P1)

| arm | parameter | source |
| --- | --- | --- |
| direct | mutation_rate | constructor kwarg (config/factors-resolved), P5-calibrated (0.3057) |
| direct | empty_prob | constructor kwarg, default 0.6 -- NOT a crossed factor (documented, not phenotype-critical in the P1 sense of dev_steps/init_condition) |
| nca | dev_steps | ExperimentFactors.dev_steps (crossed factor, P1/P4) |
| nca | init_condition | ExperimentFactors.init_condition (crossed factor, P1/P3) |
| nca | hidden_channels | ExperimentFactors.hidden_channels (crossed factor, P1) |
| nca | neighbourhood | ExperimentFactors.neighbourhood (crossed factor, P1; only 'von_neumann' implemented) |
| nca | mutation_sigma | constructor kwarg, P5-calibrated (0.0201) |
| nca | hidden_width | constructor kwarg, default 16 -- NOT a crossed factor |
| rule_based | developmental_iterations | ExperimentFactors.dev_steps (crossed factor, P1/P4) |
| rule_based | init_condition | ExperimentFactors.init_condition (crossed factor, P1/P3) |
| rule_based | mutation_rate | constructor kwarg, P5-calibrated (0.1707) |
| rule_based | HARD_MAX_ITERATIONS | module-level constant = 40 (safety ceiling, not a phenotype-affecting parameter -- raised in P4 to accommodate the depth sweep) |

**P1 grep assertion** (`grep -rn "dev_steps\s*=\s*[0-9]\|iterations\s*=\s*[0-9]" evodevo/encoders/`):
```
evodevo/encoders/rule_based/_rules/interpreter.py:180:# interpreter to 40 steps via repeated iterations=1 calls, which this
evodevo/encoders/rule_based/_rules/interpreter.py:282:    iterations=0 returns the initial grid unchanged (no rules are ever
```
Result: **PASS** -- 2 raw match(es), all inside comments/docstrings, zero matches in actual code.

**Initial-condition round-trip consistency** (`argmax(as_state_tensor(c)[..., :N_MATERIALS]) == as_discrete_grid()`):
| condition | round-trip consistent? |
| --- | --- |
| center_seed | PASS |
| full_nonsymmetric | PASS |
| full_uniform | PASS |

Result: **PASS** for all three conditions.

## Section 2 -- Initial condition (P3)

Artifact: `results/initial_condition/init_study_raw.json` (mtime: 2026-09-15T12:24:57.538879+00:00)

| arm | condition | valid_rate | degenerate_fraction | failure_reason_counts |
| --- | --- | --- | --- | --- |
| nca | center_seed | 0.5745 | 0.0240 | {'disconnected': 731, 'no_actuator': 208, 'too_few_voxels': 81} |
| nca | full_nonsymmetric | 0.6815 | 0.0185 | {'disconnected': 552, 'no_actuator': 134, 'too_few_voxels': 21} |
| nca | full_uniform | 0.7175 | 0.0325 | {'disconnected': 377, 'no_actuator': 240, 'too_few_voxels': 21} |
| rule_based | center_seed | 0.5710 | 0.1865 | {'disconnected': 200, 'no_actuator': 801, 'too_few_voxels': 153} |
| rule_based | full_nonsymmetric | 0.7495 | 0.1270 | {'disconnected': 277, 'no_actuator': 288, 'too_few_voxels': 26} |
| rule_based | full_uniform | 0.1165 | 0.8760 | {'disconnected': 36, 'no_actuator': 1760, 'too_few_voxels': 26} |

**Variance decomposition (validity rate, |nca - rule_based|)**: center_seed=0.0035, full_nonsymmetric=0.0680, full_uniform=0.6010. The between-arm gap is smallest, and consistent with noise, at `center_seed` (0.0035) -- this is the condition that was previously used by only ONE of the two arms, i.e. the historically-reported validity difference between these arms is overwhelmingly attributable to which initial condition each happened to use, not to the encoding itself.

**full_uniform symmetry control**: nca unique_valid_body_fraction=0.7150, rule_based=0.0390. **Both arms DID differentiate from a perfectly symmetric start** -- this contradicts the naive locality prediction on its face. Investigated and traced to non-isotropic per-genome update rules (NCA's unconstrained directional perception weights; rule-based's directional GROW_DIRECTIONAL/COPY_NEIGHBOR actions), confirmed by constructing isotropic/non-directional counter-examples that DO preserve symmetry -- see results/initial_condition/INIT_STUDY.md Section 7 for the full investigation. Not a hidden implementation bug, but a real correction to the originally-stated locality claim (translation-invariance != isotropy).

**Decision taken**: `center_seed`, applied identically to both developmental arms (docs/decisions.md, 2026-09-15). Driving numbers: between-arm validity gap at center_seed (0.0035) is the smallest among the three conditions, and both arms retain non-degenerate diversity there (nca unique_valid_body_fraction=0.5885, rule_based=0.4070), unlike `full_uniform` where rule_based's diversity collapses to 0.0390.

## Section 3 -- Developmental depth (P4) [most important section]

Artifact: `results/depth/depth_study_raw.json` (mtime: 2026-09-15T14:52:47.515377+00:00)

### Light-cone propagation speed

| arm | fitted speed (cells/step) | step p10 | step median | step p90 | per-pair speed median |
| --- | --- | --- | --- | --- | --- |
| nca | 0.0194 | 3.0000 | 10.0000 | 28.0000 | 0.2500 |
| rule_based | 0.2548 | 1.0000 | 1.0000 | 4.0000 | 1.0000 |

Geometric lower bound: on a 5x5 grid, max Manhattan distance centre-to-any-cell is 4, corner-to-corner is 8. The applicable bound (P3's `center_seed`, and `light_cone`'s own centre-perturbation) is **T >= 4**, not the corner-to-corner bound of 8.

### Settling-step distributions

| arm | median | q95 | never_settled_fraction | never_settled_count |
| --- | --- | --- | --- | --- |
| nca | 37.0000 | 39.0000 | 0.2090 | 418 |
| rule_based | 2.0000 | 5.0000 | 0.1155 | 231 |

**T_primary derivation**: T_primary = ceil(max over developmental arms of q95(settle_step)), clipped to a FLOOR ONLY of [4, 40] (geometric minimum floor, T_MAX_HARD_CAP ceiling -- no artificial ceiling below T_MAX, since developmental steps are cheap forward passes with no compute justification for a tighter cap). q95: nca=39.0000, rule_based=5.0000. max=39.0000 -> ceil=39 -> clip to [4,40] -> **T_primary = 39**.

T_primary (39) is at or past nca's own median settle_step (37.0000) and q95 (39.0000) -- no clip-induced inconsistency: **PASS**.

**Idempotence verification**: 0/3351 settled genomes (across both arms) changed again after their recorded settle_step, before T_max. Result: **CONFIRMED (no violations)**.

**NCA argmax-stabilisation distribution**: 0.0013 of genomes stabilise by step 1, 0.0038 by step 2, median settle_step=37.0000 (out of T_max=40).

**Verdict: NCA development is NOT vestigial at current settings, because the decoded readout stops changing at median step 37 of T_max=40** (the median genome continues re-deciding its argmax for the large majority of the developmental horizon, not just the first couple of steps).

**T selection is NOT provisional** -- NCA's never_settled_fraction (0.2090) is below the 50% provisional-fallback threshold, so T_primary was derived from BOTH arms' own q95, not the rule-based-only fallback.

**Depth sweep as configured**: T in [1, 4, 39, 40]. T=1 present for both developmental arms: **YES** (checked directly against the recorded sweep, not assumed).
## Section 4 -- Mutation calibration (P5)

Artifact: `results/mutation/mutation_calibration_raw.json` (mtime: 2026-09-15T12:51:46.982295+00:00)

**Chosen D\* = 4.9915** cells changed out of 25.

| arm | parameter | grid range | solved value |
| --- | --- | --- | --- |
| direct | mutation_rate | [0.0100, 0.4000] | 0.3057 |
| nca | mutation_sigma | [0.0100, 0.5000] | 0.0201 |
| rule_based | mutation_rate | [0.0200, 0.5000] | 0.1707 |

**Held-out verification** (fresh independent sample, tolerance |mean - D*| <= 0.5):
| arm | verification mean | verification IQR | within tolerance? |
| --- | --- | --- | --- |
| direct | 4.8970 | 2.0000 | YES |
| nca | 5.1220 | 4.0000 | YES |
| rule_based | 5.3100 | 9.0000 | YES |

Result: **ALL THREE ARMS WITHIN TOLERANCE**.

**Tail comparison**: means are matched by construction (all target D*=4.9915), but IQRs are {'direct': 2.0, 'nca': 4.0, 'rule_based': 9.0} -- a spread of 7.0000 cells across arms. **The tails do NOT match despite matched means** -- matching the mean does not equalise the spread, and this spread is large enough to matter for search dynamics.
## Section 5 -- Budget accounting (P6)

Dual counters (`evals_spent`, `evals_valid_cumulative`, `evals_invalid_cumulative`, `evals_cache_hit_cumulative`, `best_fitness_so_far`) are written to every per-eval record by `evodevo.evolution.eval_bookkeeping.EvaluationTracker`, shared byte-for-byte by both the EA and random_search -- confirmed by source inspection (not re-derived here).

**Two-panel figures**: `plot_two_panel_comparison` used in ['report.py', 'ablation_report.py']. Leftover single-panel `plot_fitness_curves(...)` calls found in: none.
Result: **PASS** -- every comparison figure in report.py/ablation_report.py uses the two-panel format, with no surviving single-panel call.

**Resolved invalid-genome/cache-charging policy** (two different encoder configs):
```
{
  "budget_counter_semantics": "evals_spent (== eval_index+1) is the sole budget-governing counter; evals_valid_cumulative tracks real (decoded-and-validated) search attempts separately and is NEVER used to gate the eval_budget loop",
  "cache_charging_policy": "cache_hit_still_consumes_one_evaluation_slot",
  "invalid_genome_policy": "an invalid genome consumes one evaluation slot and is scored sentinel_fitness=-10.0 -- never resampled, never free"
}
```
Identical across encoders: **True**.
## Section 6 -- Random-search control (P7)

Contract suite (`tests/evolution/test_strategy_contract.py`, random_search against every encoder, checked unmodified against the same suite as the EA): `48 passed in 0.34s` -- **PASS**.

A six-cell ({ea, random_search} x {direct, nca, rule_based}) surrogate benchmark is committed under `results/runs/strategy_comparison/` -- cells found: [('direct', 'ea'), ('direct', 'random_search'), ('nca', 'ea'), ('nca', 'random_search'), ('rule_based', 'ea'), ('rule_based', 'random_search')].

### EA-minus-random decomposition (real committed data, not a unit test)

**Decomposition framing**: `random_search` (no selection, no mutation, no population -- fresh i.i.d. samples from the encoder's own genome distribution) measures what the encoding's representational PRIOR achieves for free. `ea - random_search` measures what SEARCH (selection + mutation) adds on top. These are different claims: a large `random_search` result with a small `ea - random_search` gap means the encoding's prior is doing the work, not the algorithm.

| encoder | counter | random_search (mean +/- std) | ea (mean +/- std) | EA-minus-random advantage | seed-to-seed noise floor | interpretable? |
| --- | --- | --- | --- | --- | --- | --- |
| direct | evals_spent | 10.0800 +/- 0.5133 | 14.1333 +/- 1.2760 | 4.0533 | 1.2760 | yes |
| direct | evals_valid | 10.0800 +/- 0.5133 | 14.1333 +/- 1.2760 | 4.0533 | 1.2760 | yes |
| nca | evals_spent | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| nca | evals_valid | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| rule_based | evals_spent | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |
| rule_based | evals_valid | 26.0000 +/- 0.0000 | 26.0000 +/- 0.0000 | 0.0000 | 0.0000 | **NO** |

**nca's search-efficiency result under `evals_spent` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**nca's search-efficiency result under `evals_valid` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**rule_based's search-efficiency result under `evals_spent` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.
**rule_based's search-efficiency result under `evals_valid` is NOT interpretable**: its EA-minus-random advantage (0.0000) does not exceed the seed-to-seed noise floor (0.0000, from 3 EA seed(s) / 3 random_search seed(s)) -- do not report this encoding's search-efficiency claim under this counter alongside interpretable ones without this caveat.

## Section 7 -- Outstanding risk register

Written adversarially, as a reviewer trying to reject the paper. Understating a risk here is the worst possible outcome of this report.

| # | issue | severity | evidence | downstream decision affected |
| --- | --- | --- | --- | --- |
| 1 | The only committed primary-comparison results (`results/runs/`) predate the ENTIRE P1-P7 fairness workstream | **BLOCKER** | Sampled `/Users/tejeshwarcdr/Academics/FYP/EvoGym_Project/results/runs/direct/Walker-v0/seed_2/run_meta.json`: `resolved_policy` present=False, `config.strategy` present=False, `factors.json` present=False -- all False/absent means this run predates ExperimentFactors, the P3 init_condition decision, the P4 T_primary decision, P5's calibrated mutation parameters, and the P6/P7 dual-counter/strategy infrastructure entirely. | Every headline claim in results/BENCHMARK_REPORT.md as currently committed is not a fairness-corrected comparison. The PPO matrix must be (re-)run under center_seed/T_primary=20/P5-calibrated mutation/random_search-control before any conclusion is trustworthy. |
| 2 | NCA does not settle within T_max=40 for a substantial minority of genomes | **MAJOR** | never_settled_fraction=0.2090 (418/2000) at T_max=40, center_seed. | P4 T_primary derivation and any conclusion assuming NCA's development reaches a fixed point. |
| 3 | Non-settling NCA genomes are dominated by STILL_GROWING, a genuine instability rather than an honest slow-settling tail | **MAJOR** | 416/418 (0.9952) of never-settled genomes classified as STILL_GROWING (see results/depth/DEPTH_STUDY.md's 'Diagnosis of non-settling genomes'). | Evidence for NCA update-rule/architecture work: a larger T_max would NOT be expected to resolve this fraction, since the dominant failure mode is active divergence, not slow convergence. |
| 4 | NCA's argmax readout and its continuous state settle on very different timescales (median settle_step=37 vs. rule_based's 2) | **MAJOR** | nca stabilize_by_step_2=0.0038 (i.e. NOT vestigial -- development genuinely continues deciding the phenotype for most of the horizon), vs. rule_based stabilize_by_step_2=0.7106. The two arms' own natural developmental timescales differ by roughly an order of magnitude. | Any single shared T_primary necessarily overshoots one arm's natural timescale while potentially undershooting the other's -- the depth-sweep (Layer 3) results must be read alongside this, not the single-T primary comparison alone. |
| 5 | Mutation displacement tails (IQR) do not match despite matched means | **MAJOR** | IQRs at calibration: {'direct': 2.0, 'nca': 4.0, 'rule_based': 9.0} (spread=7.0000 cells) despite all three arms targeting the same D*=4.9915. | Any claim that mutation strength is 'equalised' across arms in the P5 sense of matched phenotypic displacement is true only for the mean, not the distribution -- search-dynamics comparisons that depend on mutation's tail behaviour remain confounded. |
| 6 | 1 row(s) in docs/decisions.md still marked OPEN | **MAJOR** | Open rows: Invalid-genome policy. Per decisions.md's own header: "No row may remain 'OPEN' in the final submission." | The project's own stated completion gate for docs/decisions.md is not met. |
| 7 | Residual between-arm validity gap after initial-condition control | **MINOR** | At center_seed (the chosen primary condition): nca valid_rate=0.5745, rule_based valid_rate=0.5710, gap=0.0035. | Whether validity-rate differences in the primary comparison can be attributed to the encoding at all, vs. residual noise/confound not resolved by the P3 initial-condition control. |
| 8 | D* landed outside the originally recommended 2-3 cell band | **MINOR** | D*=4.9915, driven by nca's own grid floor (mutation_sigma=0.01 already producing ~3.44 cells of displacement) rather than the 2-3 recommendation. | The mutation-strength calibration's absolute scale is higher than originally anticipated; does not by itself invalidate the calibration (it was chosen on evidence, not fitted to the recommendation) but changes the EA's effective exploration radius versus what a naive reading of the recommendation would assume. |
| 9 | Several encoder parameters remain hardcoded/constructor-default rather than crossed factors | **MINOR** | DirectEncoder.empty_prob (default 0.6), NCAEncoder.hidden_width (default 16), RuleBasedEncoder's MIN_RULES/MAX_RULES genome-length bounds, HARD_MAX_ITERATIONS=40 -- none of these are swept or asserted identical across arms the way dev_steps/init_condition/mutation strength are. | Any future claim that 'every phenotype-affecting parameter is controlled' should be read as 'every parameter this workstream identified and promoted' -- not a literally exhaustive audit. |
| 10 | full_uniform (perfectly symmetric initial condition) differentiates for both developmental arms, contradicting the naive locality prediction | **MINOR** | nca unique_valid_body_fraction=0.7150, rule_based=0.0390 under full_uniform -- both far from zero. Investigated and attributed to non-isotropic per-genome update rules, not a hidden implementation asymmetry (see Section 2). | The original FullUniform docstring's locality claim was corrected, not merely re-confirmed -- readers of that module's history should not assume the original claim as originally written is accurate. |

**Totals**: 1 BLOCKER, 5 MAJOR, 4 MINOR.
## Section 8 -- Verdict

### What is now demonstrably fair across arms

- Developmental depth and initial condition are config-driven crossed factors for both developmental arms, not per-arm hardcoded literals (P1 grep assertion clean; initial-condition round-trip consistency verified for all three conditions -- Section 1).
- The initial condition used in the primary comparison (`center_seed`) was chosen on measured evidence that minimises the between-arm validity gap (0.0035) while keeping both arms' diversity non-degenerate (Section 2).
- Mutation strength is calibrated per-arm to match expected phenotypic displacement (mean), verified within tolerance on a held-out sample for all three arms (Section 4).
- The invalid-genome and cache-charging policies are identical across arms by construction (shared EvaluationTracker), and every comparison figure reports both evals-spent and evals-valid panels, so representational-prior effects cannot silently masquerade as search efficiency (Section 5).
- A random-search control exists and passes the identical strategy contract suite as the EA, unmodified (Section 6).

### What remains confounded and why

- NCA and rule-based have roughly an order-of-magnitude different natural developmental timescale (median settle_step 37 vs. 2) -- a single shared T_primary cannot simultaneously match both arms' own settling horizon (Section 3).
- Mutation displacement means are matched, but the distributions' tails are not (IQR spread up to several cells across arms) -- search-dynamics comparisons that depend on mutation's tail behaviour remain confounded even after P5 calibration (Section 4).

### What must be resolved before the PPO matrix is launched

- All 1 BLOCKER-severity item(s) in the Section 7 risk register, specifically: The only committed primary-comparison results (`results/runs/`) predate the ENTIRE P1-P7 fairness workstream.
- The committed results/runs/ primary comparison must be re-run under the full P1-P7 resolved configuration (center_seed, T_primary=39, P5-calibrated mutation, random_search control present) before any of its numbers are cited as the paper's headline result.
