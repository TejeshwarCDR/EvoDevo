> **SUPERSEDED as of 2026-09-19.** This is the lay-audience explainer for
> `results/BENCHMARK_REPORT.md`'s run — see that file's own superseded banner
> for the full detail. In short: this run predates the entire NCA/rule_based
> architecture-redesign workstream (committed 2026-09-17). Its NCA numbers
> (e.g. Section 6's "68.35% of NCA genomes are still changing state," "~40x
> the initial-state scale") describe the old `update_rule` (unbounded
> residual, no `bounded_leaky`/`decay`/`state_bound`), `dev_steps=10` (current
> shared `T_primary=100`), `hidden_channels=3` (current PRIMARY `0`). See
> `evodevo/encoders/nca/resolved_config.py::PRIMARY` and
> `evodevo/encoders/rule_based/resolved_config.py::PRIMARY` for the current
> resolved configuration, and `docs/decisions.md`'s NCA/rule_based redesign
> rows for what changed and why. **The real, final results explainer does not
> yet exist** — it will need to be rewritten against the actual primary-matrix
> run once that is executed (see `docs/PRE_PARALLELIZATION_READINESS.md`).

# Results Explained

**Headline: no detectable difference between the three encodings.** Across every pairwise comparison (direct vs. NCA, direct vs. rule-based, NCA vs. rule-based), the 95% confidence interval on the difference in final best fitness includes zero. This is a real, expected-to-be-reported outcome under this project's falsifiability commitment, not a failed experiment — see below for what it does and doesn't mean given the (small, early-training) scale of this run.

This document is written for a reader with no prior context on the implementation. For the underlying numbers and figures, see `results/BENCHMARK_REPORT.md` and `results/media/`.

## 1. The research question

Soft robots in [EvoGym](https://evolutiongym.github.io/) are built from a grid of voxels (rigid, soft, and horizontally/vertically actuating materials). There are many ways to encode "which voxel goes where" as a genome that an evolutionary algorithm can mutate and select on. This project asks: **does the choice of genome-to-body encoding change how efficiently evolution finds good locomoting bodies, or how good the bodies it finds are** — when every other part of the pipeline is held identical?

Three encodings were compared, sharing one evolutionary loop, one fitness function (EvoGym PPO-trained locomotion reward), and one evaluation budget:

- **Direct**: the genome *is* the voxel grid — one categorical value per cell, mutated cell-by-cell.
- **NCA (Neural Cellular Automaton)**: the genome is a small neural network's weights; the network runs for a fixed number of steps, growing the body from a seed cell, the way a Neural Cellular Automaton grows a pattern.
- **Rule-based**: the genome is a variable-length list of symbolic if-then developmental rules ("if neighbor count > 2, set actuator"), applied iteratively to grow the body from a seed.

All three plug into the same outer evolutionary loop (population/offspring sizes, selection, mutation-only reproduction, invalid-genome handling) and the same PPO-based fitness evaluation — only the genome representation and its decode-to-body step differ.

## 2. What was actually run

- **3 encoders × 5 seeds × 1 task (Walker-v0)** = 15 independent evolutionary runs.
- **eval_budget = 300** genome evaluations per run (population 15, offspring 15 per generation → 20 generations).
- **ppo_timesteps = 5000** per evaluation — each candidate body is trained with PPO for 5,000 timesteps before its fitness is recorded.
- **Fairness controls in force and mechanically checked before the run started**: identical eval budget, identical PPO budget, mutation-only reproduction (no crossover) for all three arms, identical invalid-genome accounting (an invalid body gets a fixed sentinel fitness and is never evaluated with PPO), and a config-hash lock (`56b3ed67...`) confirming the three arms' shared settings were byte-identical at launch and stayed that way for all 15 runs.

## 3. The headline finding, with uncertainty

| Encoder | Final best fitness (mean ± std, 5 seeds) |
|---|---|
| Direct | −1.74 ± 4.13 |
| NCA | 0.28 ± 0.09 |
| Rule-based | 0.18 ± 0.07 |

These point estimates alone look different, but the sample is tiny (5 seeds per arm) and direct's variance is large. The actual statistical comparisons:

| Comparison | Mean difference | 95% CI | Verdict |
|---|---|---|---|
| Direct − NCA | −2.01 | [−6.15, 0.17] | no detectable difference |
| Direct − Rule-based | −1.92 | [−6.06, 0.27] | no detectable difference |
| NCA − Rule-based | +0.10 | [−0.001, 0.207] | no detectable difference |

Every interval spans zero (the NCA vs. rule-based interval comes closest, but still crosses it). With this few seeds, the correct, honest statement is that this run does not distinguish the three encodings on final fitness — not that they are "equally good" (absence of evidence is not evidence of equivalence at n=5).

## 4. Walking through the table and figures

- **`results/BENCHMARK_REPORT.md`'s three-way table** also reports: evaluations needed to reach 50%/90% of the best arm's final fitness (a search-efficiency proxy — lower is faster to a given quality level), validity rate (fraction of genomes that decoded to a valid, PPO-evaluable body), unique bodies explored, and mean wall-clock per evaluation. Read the evaluations-to-threshold columns as noisy — same small-sample caveat applies.
- **`results/figures/fitness_curves.png`**: best-fitness-so-far vs. evaluation count, one line per arm, with per-seed variance bands. Look at whether the bands overlap (they do, substantially, for most of the run) rather than which line is nominally on top at evaluation 300.
- **`results/figures/morphology_comparison.png`**: box plots of actuator fraction, occupied voxel count, and symmetry for each arm's best body across seeds. This shows what kind of bodies each encoding tends to converge on, independent of fitness.
- **`results/figures/validity_efficiency.png`**: three panels — each encoder's true pre-run random-genome validity prior (from `results/inspection/{encoder}/*_sanity.json`, PPO-free), the realized in-run validity rate during evolution (from the three-way table's `validity_rate` column), and search efficiency. The first two are not the same quantity: the realized rate is typically much higher, since evolution concentrates search once it finds valid regions of genome space (see section 5 below).
- **`results/media/`**: a rollout gif/mp4 of each arm's best body actually walking, development-trajectory figures for NCA and rule-based (showing how the body grows from its seed step by step), and `comparison_Walker-v0.png`, a side-by-side of all three arms' best morphologies.

None of the figures should be read as showing one encoding "winning" — they're included to make the comparison inspectable, not to substitute for the statistical test in Section 3.

## 5. Run integrity: timing investigation (clean)

The run finished in ~7.5 hours against an ~11-hour pre-run estimate, a 32% speedup that was investigated before any results were reported, per this project's standard of treating unexplained discrepancies as suspect until checked. The investigation (documented in full in the session's Part 0 checks) confirmed:

- All 15 (encoder, seed) runs completed with exactly the planned 300 evaluations each — nothing terminated early or was truncated.
- The resolved configuration embedded in every run's metadata rehashes to the exact locked hash from before launch — no configuration drift on any run.
- The evaluation cache was confirmed off (`cache_enabled=false`) in all 15 runs, with zero cache hits recorded — the speedup is not an artifact of caching turning on and biasing the arms unevenly.
- Zero simulation-level exceptions were silently swallowed across all 4,500 logged evaluations.
- The actual explanation for the speedup: all three arms' *real* validity rates, measured across the full evolutionary run, came in well above the validity priors measured beforehand on random (non-evolved) genomes — because evolution concentrates search on the region of genome space that decodes to valid bodies once it finds one, which the original 11-hour estimate did not model (it assumed close to full-PPO cost for every evaluation). Invalid genomes are rejected in microseconds without ever running PPO, so a higher realized validity rate speeds up the run rather than slowing it down.

This is reported here as a **verified, clean run** — the timing gap has a specific, checked, benign explanation rather than being an open question.

## 6. Threats to validity

- **Seed count.** 5 seeds per arm is a small sample. The confidence intervals in Section 3 are correspondingly wide; this run can rule out large effects but not moderate ones.
- **PPO budget.** `ppo_timesteps=5000` per evaluation is far below EvoGym's own recommended >1,000,000 timesteps for a "serious" locomotion-training run. Every fitness value in this report should be read as **early-training signal** — a rough proxy for whether a body *can* locomote at all under minimal training — not as a converged-performance measurement. A body that scores low here might still be a strong walker under a realistic PPO budget, and vice versa.
- **Single task.** Only Walker-v0 was tested. No claim is made about generalization to other EvoGym tasks (climbers, carriers, etc.).
- **NCA's non-settling behavior.** Pre-run inspection (2,000 random genomes) found that 68.35% of NCA genomes are still changing state at the final developmental step — the growth process has not converged within the fixed number of developmental steps used, and residual state magnitude grows to roughly 40x the initial-state scale, i.e., unbounded rather than self-stabilizing. This is a known, disclosed limitation of the NCA arm as configured here, not something fixed for this run.
- **Mutation-strength tuning asymmetry.** Direct's genome is a categorical grid, NCA's is a continuous weight vector, and rule-based's is a variable-length symbolic list — there is no single shared knob for "mutation strength" across all three, so each arm's mutation rate/sigma was set independently rather than through a common tuning procedure. Any efficiency difference between arms partly reflects this per-arm tuning choice, not only the representation itself.
- **Out-of-scope by design.** Ablations (e.g., varying grid size, developmental step count, mutation strength systematically) and a second-task comparison were explicitly deferred from this run's scope and remain future work — they were not silently dropped, just never in plan for this run.

## 7. What is and isn't novel here

Neither Neural Cellular Automata for growing soft robot bodies nor symbolic/rule-based developmental encodings are new ideas — both predate this project. The contribution of this work is the **controlled three-way comparison**: running direct, NCA, and rule-based encodings through one identical evolutionary pipeline, with mechanically enforced (config-hash-locked, pre-launch-audited) fairness in evaluation budget, PPO budget, reproduction operator, and invalid-genome accounting, so that any observed difference (or, as found here, lack of one) can be attributed to the representation rather than to incidental differences in how each arm was run.

## 8. What a larger follow-up study would need

- **More seeds** (10–20+ per arm) to narrow the confidence intervals enough to detect moderate effect sizes, if any exist.
- **A realistic PPO budget** (EvoGym's own >1e6-timestep guidance) so fitness reflects converged locomotion performance rather than early-training signal.
- **The deferred ablations**: grid size, developmental step count, and a shared/normalized mutation-strength sweep across all three representations.
- **A second task family** (e.g., a climbing or carrying task) to test whether any encoding-dependent effect is task-specific or general.
- **A settling-time fix or explicit disclosure protocol for NCA** so its non-settling behavior doesn't confound comparisons in a follow-up with a larger developmental-step budget.
