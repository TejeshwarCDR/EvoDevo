# Pre-registration: campaign extension to three further EvoGym tasks

**Date: 2026-10-15. Status: ACCEPTED — scope decided by Tejeshwar; rules specified before any extension measurement existed.**
This document is committed before any preflight training on the extension tasks has run, and before any extension campaign run. It is not edited afterwards: any later change is a dated "Deviation from pre-registration (extension)" row in `docs/decisions.md`. The executable form of every rule below is `evodevo/extension/rules.py` (constants and functions, unit-tested in `tests/extension/test_rules.py`); where this text and that module disagree, the disagreement is itself a defect to be recorded, not silently resolved.

The original campaign (Walker-v0, UpStepper-v0; four arms) and its pre-registered analysis plan (`docs/decisions.md`, 2026-09-27) are unchanged by this extension. Its plan, config hash and resume behaviour are pinned by `tests/campaign/test_extension_invariance.py`.

---

## 1. Purpose and task selection

**Goal.** Cover every EvoGym difficulty tier with at least one task. EvoGym classifies its tasks as easy, medium or hard from the performance of the benchmark's baseline co-design algorithms (Bhatia et al., NeurIPS 2021, §3.5; the per-task ratings are on the official task pages, `evolutiongym.github.io/all-tasks`). The original campaign covers easy (Walker-v0) and medium (UpStepper-v0).

| Task | Official tier | Terrain / objective | Interface relative to the original tasks |
|---|---|---|---|
| **ObstacleTraverser-v1** | hard | very bumpy terrain; reward = forward displacement of the centre of mass | identical to UpStepper-v0 (velocity, orientation, local terrain observation) |
| **Climber-v0** | medium | flat vertical channel; reward = **upward** displacement | identical to Walker-v0 |
| **CaveCrawler-v0** | medium | caves and low ceilings; reward = forward displacement | Walker-v0 plus ceiling observations |

All three are locomotion tasks with the same voxel action space, so D2's argument against manipulation tasks (a changed controller problem confounding the morphology comparison) still holds. They differ from the original tasks in terrain and, for Climber-v0, in the reward axis. That is why tasks are analysed separately (§7).

**Why ObstacleTraverser-v1 for the hard tier.** D2 rejected hard tasks for the *second* task because EvoGym's authors report that no tested algorithm succeeds on the hardest environments, which risks a floor effect. The extension adds exactly one hard task *on purpose*, because the hard tier is the one the paper is missing, and pre-registers how a floor effect is reported (§7.4). Among the hard tasks, ObstacleTraverser-v1 keeps the reward (forward displacement) and observation interface of UpStepper-v0, so it is the hard task closest to the existing ladder. Traverser-v0 was rejected because its pit of rigid blocks makes simulation much more expensive and makes a floor effect most likely. ObstacleTraverser-v0 was rejected because it is rated medium.

**Static construction check.** For every extension task, a 5×5 body at the environment's spawn point overlaps no terrain (checked against the EvoGym world files at commit `f63c649`). Climber-v0's channel is 7 voxels wide, and a 5×5 body fits between its walls.

## 2. Fixed design

Everything not decided by a rule in §4–§6 is fixed here, identical to the original campaign unless stated.

- **Arms:** `direct`, `nca`, `rule_based`, the three core arms with the same calibrated mutation strengths and resolved encoder settings as the original campaign. `rule_based_monotone` is **not** run on the extension tasks. The mechanistic `rule_based` vs `rule_based_monotone` contrast therefore stays a two-task result (Walker-v0, UpStepper-v0).
- **Replicates:** seeds 0–4 (n = 5 per arm per task).
- **EA:** population 15, offspring 15, `eval_budget` 300 genomes (20 generations), mutation only, invalid genome = sentinel −10 consuming one budget slot.
- **Development:** `dev_steps` 100, `init_condition` center_seed, 5×5 grid.
- **Controller:** PPO with stable-baselines3 defaults, one deterministic evaluation episode, genome fitness = max over K PPO seeds.
- **D7 order:** ObstacleTraverser-v1 → Climber-v0 → CaveCrawler-v0. The hard task goes first because it is the only one filling a missing tier. If the campaign is interrupted, the most valuable result exists first.
- **Execution:** a new campaign (new campaign id) on the lab machine's validated production profile (18 workers, P8+E10, power controls). It is launched only after the original campaign is complete.

## 3. Extension preflight (B4x, B5x)

Run on the lab machine after the original campaign completes, under the production placement and power controls (`docs/campaign_prep/ext/RUNBOOK_EXT.md`).

**Representative bodies (6).** For each core arm: (i) the original campaign's Walker-v0 final best body (`milestones/Walker-v0/best_bodies/<arm>.npz`), an evolved locomotor, and (ii) the first structurally valid body decoded from a random genome of that arm's campaign encoder (seed `[20261015, arm index, attempt]`). Rationale: B4 in the original preflight used three random bodies, and one Walker-v0 body never learned, which left the Walker rule resting on two bodies. Evolved bodies are what the EA evaluates in late generations, where the budget matters most. Random bodies are what it evaluates first.

**B4x.** Each body × each extension task: one PPO training (seed 20261015) to 2,000,000 timesteps, with fitness recorded every 100,000 steps by the same deterministic rollout used for campaign fitness.

**B5x.** Each body × each extension task: 3 independently seeded trainings (seeds `ppo_seed(20261016, r)`, r = 0, 1, 2) at that task's D6-ext budget. In the same batch, as cost references, each body is trained once on Walker-v0 at 1,000,000 and once on UpStepper-v0 at 1,200,000.

## 4. D6-ext — PPO timestep budget per task

Applied to the B4x curves; bodies whose training failed or stopped early are excluded first.

1. **Qualifying bodies:** final (2M) fitness ≥ 10% of the best body's final fitness, with the best final fitness > 0.
2. **Non-learning task:** fewer than 2 qualifying bodies → budget 1,000,000 (the Walker-v0 budget), disclosed as a floor.
3. **Registered rule:** D6's pre-registered knee rule (`select_knee`, unchanged: the smallest budget at which median marginal improvement per 100k falls below 1% of fitness so far, sustained over 3 windows), applied to the per-checkpoint median of the qualifying bodies. A knee is rounded to its 100k checkpoint and clipped to [500,000, 1,500,000]. The lower clip is D6's own stated lower expectation, and it guards against a spurious early plateau. The upper clip is a compute cap declared here, and a capped value is disclosed as a floor.
4. **No knee:** the Walker-v0 fallback rule, verbatim, except that the ordering condition is evaluated over the qualifying bodies only (with six bodies, the order of near-zero non-learners is noise). Candidates are T ∈ {1.0M, 1.2M, 1.5M, 2.0M}. Choose the smallest T at which every qualifying body has ≥ 85% of its own 2M fitness and the qualifying bodies' order at T equals their order at 2M. If none passes, T = 2.0M as a floor. The result is then capped at 1,500,000 and disclosed.

The rule uses only controller learning curves of fixed bodies, never an arm-versus-arm comparison of evolutionary outcomes. The budget is identical for every arm on a task.

## 5. D16-ext — PPO seeds (K) per task

Applied to the B5x repeats at the chosen budget; bodies with fewer than 2 completed repeats are excluded.

- between-body SD = SD of the per-body mean fitness (≥ 3 bodies required, otherwise K = 1);
- r₁ = pooled within-body SD of single trainings ÷ between-body SD;
- r_max2 = pooled within-body SD of the max over all pairs of repeats ÷ between-body SD.

**K = 2 (max aggregation) if r₁ > 0.33 and r_max2 ≤ 0.5·r₁; otherwise K = 1.** If the between-body SD is ~0 (bodies indistinguishable at this budget), K = 1. The 0.33 threshold lies between the two ratios that drove the original D16 decisions (Walker-v0 0.165 → K = 1; UpStepper-v0 0.666 → K = 2), at roughly their geometric mean. The second condition requires max-of-2 to actually suppress the noise, as it did on UpStepper-v0 (collapse-type noise). The between-body spread is defined on the six B5x bodies rather than on B3 random genomes, and this definition is disclosed. Mean-of-K is the pre-registered sensitivity analysis for any task with K = 2.

## 6. Cost projection and GUARD-ext

**Projection.** The phase duration for a task is the original campaign's realised Walker-v0 phase makespan, scaled by three factors:

- the share of that phase's worker-busy time spent on the three core arms (from its journal);
- the measured cost ratio of one training on the task at its budget to one training on Walker-v0 at 1M, using the same six bodies in the same batch, with E-core samples divided by the measured 2.3172 E/P ratio;
- K.

**Backtest.** The same method predicts the original UpStepper-v0 phase from the Walker-v0 phase, and that prediction is compared with UpStepper-v0's realised makespan. Each band is ± max(10%, |backtest error|). The per-task PPO-task timeout is 4 × (mean + 1.645 SD) × 2.3172 of the B5x P-core-equivalent training time, the same formula as the original campaign.

**GUARD-ext (threshold: p95 total ≤ 360 h = 15 days, launch ≈ 2026-10-22 → ≈ 2026-11-06).** It is applied mechanically to the p95 projected total:

- (a) while over, set K = 1 on the latest task (in D7 order) that has K = 2;
- (b) while still over, lower the budget to 1,200,000 (the UpStepper-v0 budget) on the latest task whose budget exceeds it, disclosed as a floor;
- (c) if still over, **nothing else is cut**: no arm, seed, task or generation. The campaign proceeds and the projected completion date is reported.

Generations are deliberately not reduced, because the primary endpoint is fixed at evaluation 300.

## 7. Analysis plan (extension tasks)

Unit of analysis: one EA run (arm × task × seed). Tasks are analysed separately, with no pooling across tasks.

1. **Primary endpoint:** best-so-far fitness at evaluation 300, analysed per task.
2. **Primary test (per task):** Kruskal–Wallis across the 3 arms. Then pairwise two-sided Mann–Whitney U for the 3 arm pairs, Holm-corrected **within the task** (family of 3), with Vargha–Delaney A12 and a bootstrap 95% CI for each pair. Arms are summarised by median and IQR. This per-task Holm family is the primary analysis, as the original pre-registration states ("Per task: … Holm correction"). The cross-task family of `evodevo/analysis/stats_protocol.py` (one Holm correction over all tasks and comparison kinds) is reported only as a secondary, more conservative view, for the original campaign too.
3. **Secondary endpoints:** as in the original plan: validity rate by generation; final-best morphology descriptors; evaluations to 90% of the arm's final median. Mean-of-K sensitivity for any task with K = 2.
4. **Floor-effect rule (all extension tasks, decisive for ObstacleTraverser-v1).** Let Q75 be the 75th percentile of all valid generation-0 fitness values on the task, pooled over arms and seeds. These are random-genome bodies, the EA's starting point. If **every** arm's median final best-so-far fitness is ≤ Q75, the task is reported as **floor-limited at this budget**: its tests are still reported, but no inference for or against H1 is drawn from it. The milestone package writes this check to `stats/floor_effect_check.csv`.
5. **Cross-task view (descriptive only):** for the three core arms across all five tasks (Walker-v0, UpStepper-v0 and the three extension tasks), the per-task median ranks and A12 against `direct`. There is no pooled test.
6. **Exploratory prediction P4 (stated in advance, not confirmatory):** consistent with H1 and P2, the developmental arms' A12 against `direct` is at least as large on the medium and hard extension tasks as on Walker-v0.

## 8. Disclosed limitations carried forward

- All original limitations: NCA development does not settle; the validity proxy assumes random fitness; n = 5 seeds.
- Three of the six B4x/B5x bodies evolved on Walker-v0, so they may favour walking morphologies. This affects only budget and K calibration, never the arm comparison.
- The cost projection assumes the extension tasks' validity trajectories resemble Walker-v0's. Its error on the original campaign is measured by the backtest and propagated into the band.
- Climber-v0 rewards upward displacement and CaveCrawler-v0 adds ceiling observations. Cross-task differences confound terrain with these interface differences, which is why there is no pooling.

## Amendment 1 (2026-10-15, before any extension preflight data)

No extension preflight measurement exists at the time of this amendment. Constants not named here are unchanged.

1. **B5x uses 4 repeats instead of 3** per (body, task): seeds `ppo_seed(20261016, r)`, r = 0..3. Noise trainings become 6 bodies × 3 tasks × 4 repeats = 72 (plus the 12 cost references).
2. **D16-ext max-of-2 within-body variance is the unbiased U-statistic.** For a body with repeats x₁..xₙ (n ≥ 4) it is E[max²] − E[max]², where E[max²] is the mean of max(a,b)² over all distinct pairs and E[max]² is the mean of max(a,b)·max(c,d) over all pairs of pairs with four distinct indices. The earlier overlapping-pairs sample variance is biased low (about 0.57× the true variance at 3 repeats under Gaussian noise), which would favour K = 2. The max-of-2 within variance is the mean over bodies of this quantity, clipped at 0; the single-training within variance is the mean over bodies of the ddof=1 sample variance; the SDs are their square roots. The 0.33 and 0.5 thresholds and all other D16-ext branches are unchanged.
3. **"Bodies with ≥ 4 repeats" replaces "≥ 2"** for the max-of-2 term (the single-training term still uses bodies with ≥ 2 repeats). Fewer than 3 bodies with ≥ 4 repeats gives K = 1 (branch `insufficient_repeats`).
4. **`derive` refuses unfinished, dirty or multi-commit preflight runs.** It requires a PRODUCTION label (production parameters, never TEST), no stage run on a dirty tree, a single git commit across all stages, and exactly the expected number of recorded trainings (failed trainings count as recorded and are listed in the decision report).
5. **The floor-effect check applies to the extension tasks only.** The original campaign's milestone packages are unchanged.

## Amendment 2 (2026-10-25, after the extension preflight ext-20261023 and before any campaign data for ObstacleTraverser-v1 or for seeds 5-9)

The extension preflight `ext-20261023` has been run and its records are committed unchanged
(`EXT_DECISIONS.json`, `EXT_DECISIONS.md`, `final_estimate_ext.json`, `campaign_ext.yaml`). No campaign data exist for
ObstacleTraverser-v1, and none for Walker-v0 or UpStepper-v0 at seeds 5-9. The 3-task design of this document is retired:
`SUPERSEDED_BY_AMENDMENT_2.json` holds the sha256 of `EXT_DECISIONS.json`, and the plan builder and the launcher refuse a
campaign whose `decisions_path` is that record. Constants and rules not named here are unchanged. The generated record of
this amendment is `A2_DECISIONS.json` (with `A2_DECISIONS.md`, `final_estimate_a2.json` and
`campaign_a2.yaml`, all produced by `python -m evodevo.extension amend2 derive` and verified byte for byte by
`amend2 check`). The gate evidence is [A2_GATES_REPORT.md](A2_GATES_REPORT.md).

**(a) Tasks.** Walker-v0, UpStepper-v0 and ObstacleTraverser-v1.

- **Climber-v0 is excluded for a floor effect.** The preflight found no learning signal: the median fitness over the six
  bodies lies between -0.003 and 0.134 at every one of the 20 checkpoints from 100k to 2M steps (0.042 at 2M), and the
  six bodies' fitness at 2M lies between -0.011 (nca:random) and 0.189 (direct:random). D6-ext found no knee and fell back
  to its registered fallback (disclosed floor); no budget makes the arms distinguishable. `EXT_DECISIONS.md` has the
  table and the curves.
- **CaveCrawler-v0 is excluded for saturation.** The median fitness over the six bodies is 1.66 at 100k and 1.87 at 200k,
  and 1.85-1.91 at every later checkpoint to 2M; D6-ext found its knee at 200k (clipped to the 500k floor). At 2M five of
  six bodies lie between 1.72 and 1.94 (direct:evolved 1.89, direct:random 1.92, nca:random 1.72, rule_based:evolved
  1.88, rule_based:random 1.94; nca:evolved 0.72 is the exception), so the task leaves little room for an arm
  difference.
- All three preflight results (ObstacleTraverser-v1, Climber-v0, CaveCrawler-v0) are reported in the paper's appendix,
  with their numbers and curves.

**(b) Seeds and arms.** Seeds 0-9 per task for the arms direct, nca and rule_based. The monotone variant
(rule_based_monotone) stays at its original seeds 0-4 on Walker-v0 and UpStepper-v0 and is a secondary contrast only.

**(c) ObstacleTraverser-v1 PPO budget = 1,500,000 steps, K = 1.** Gate T applied the registered D6-ext fallback test (every
qualifying body reaches at least 85% of its 2M fitness, and the rank order of the bodies is kept) at 1.0M and 1.2M. Both
failed, so the rule fell through to the registered cap, 1.5M. We disclose plainly:

- 1.5M fails the same test as well (direct:evolved reaches 58% of its 2M fitness, direct:random 22%, rule_based:evolved
  38%, and the rank order differs). Only 2M passes, trivially, because it is compared with itself. 1.5M is the cap, not a
  budget at which training has converged.
- The test compares each body with a single noisy 2M value (one training per body, six bodies), and single-checkpoint
  fitness is itself noisy: direct:random is 0.26 at 1.5M against 0.69 at 1.4M and 1.14 at 1.8M, and nca:random is 0.24 at
  1.8M against 1.07-1.13 at its neighbouring checkpoints.
- The preflight shows genuine slow learners. direct:evolved reaches 0.21 at 1.0M, 0.63 at 1.2M and 1.09 at 2M.
  rule_based:evolved sits on a plateau near 1.15-1.2 with jumps (1.68 at 1.2M, 1.74 at 1.4M and 1.8M) and ends at 3.10 at
  2M, a single observation that one training per body cannot separate from seed noise.
- Not every body has converged at 1.5M. The Spearman correlation between the bodies' fitness at 1.5M and at 2M is 0.54
  (Kendall tau-b 0.33), with n = 6.
- The D6-ext knee (600k steps, where the median marginal improvement falls below 1% for three windows) is reported. It is
  not used.
- K = 1 per D16-ext: the single-training noise ratio is 3.01 and max-of-2 reduces it to 1.87, a 38% reduction, short of
  the required 50%.

**(d) Walker-v0 and UpStepper-v0 seeds 5-9** use the original per-run configurations unchanged (Walker-v0 1.0M steps with
K = 1; UpStepper-v0 1.2M steps with K = 2) and pool with seeds 0-4. Gate R evidence (A2_GATES_REPORT.md): the config
differs only in the seed; the library versions are identical, including evogym f63c649c; no behavioural code hunk exists
between the original campaign's commit and the current code; and 12 of 12 replayed original evaluations are bit-identical,
including both K = 2 trainings. The launch refuses if any config field other than the seed differs from the original
campaign's, or if any recorded library version (python, torch, stable-baselines3, gymnasium, numpy, cloudpickle, evogym
and its commit) differs from the installed one.

**(e) One campaign, strict phase order.** P1 ObstacleTraverser-v1 seeds 0-4, P2 ObstacleTraverser-v1 seeds 5-9, P3
UpStepper-v0 seeds 5-9, P4 Walker-v0 seeds 5-9 (phase ids `ObstacleTraverser-v1:s0-4`, `ObstacleTraverser-v1:s5-9`,
`UpStepper-v0:s5-9`, `Walker-v0:s5-9`; 15 runs each, 60 in all). Every phase is committed unconditionally. The P1 package
is INTERIM (five seeds, with its re-evaluation) and cannot stop or change later phases.

**(f) Primary endpoint unchanged:** each run's recorded best fitness (best-so-far at evaluation 300).

**(g) Secondary endpoint: champion re-evaluation.** After each phase, each run's champion (the milestone definition: the
evaluation with the maximum recorded fitness, ties to the lowest evaluation index) is re-trained 10 times, each a single
training at the task's budget, with PPO seeds `ppo_seed(20261101, k)`, k = 0..9. The mean and spread are reported. There
are 30 champions per task; the Walker-v0 and UpStepper-v0 champions of seeds 0-4 are read from the original campaign,
read-only. The stages are, in order: after P1 (150 trainings), after P2 (150), after P3 for UpStepper-v0 seeds 0-9
(300), after P4 for Walker-v0 seeds 0-9 (300).

**(h) Robustness at 2x each task's budget.** Every champion (90) is re-trained 3 times with `ppo_seed(20261201, k)`,
k = 0..2, at 3,000,000 (ObstacleTraverser-v1), 2,400,000 (UpStepper-v0) and 2,000,000 (Walker-v0) steps,
ObstacleTraverser-v1 first (270 trainings). Arm rankings and A12 are reported at the task budget and at 2x.

**(i) PPO state.** The worker already saves each trained policy with its optimizer state after the fitness rollout, which
draws from no random stream. No observation normaliser exists, and the learning rate is constant (3e-4). Re-trainings are
saved the same way, each under a unique path per (task, arm, seed, champion evaluation index, stage, k).

**(j) Analysis per task, no pooling.** Kruskal-Wallis across the three arms, pairwise Mann-Whitney U Holm-corrected over
the 3 pairs, Vargha-Delaney A12 with bootstrap 95% intervals, and the floor-effect check on ObstacleTraverser-v1.
Walker-v0 and UpStepper-v0 merge seeds 0-4 and 5-9 with a provenance column; the monotone contrast uses the original
seeds 0-4 only. Cross-task results are descriptive only.

**(k) No cost guard.** The extension's 360 h guard does not apply to this campaign. The projected timeline (about 1,160 h
mean, p5-p95 1,044-1,276 h, from the planned launch on 2026-10-26 12:00 IST) is informational.

**(l) Any fourth task** gets its own pre-registration and preflight, outside this campaign.
