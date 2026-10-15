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
