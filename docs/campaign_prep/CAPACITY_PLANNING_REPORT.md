> Version-controlled copy, made 2026-09-28, of `results/preflight/20260922T122730Z_4be8402b/report/CAPACITY_PLANNING_REPORT.md` (a gitignored output directory). Relative paths below are as in the original.

# Capacity Planning Report — campaign wall-clock, simulated and cross-checked against classical scaling laws

Run: `20260922T122730Z_4be8402b`. Written 2026-09-24. This is law-based simulation on already-collected data plus one cheap new measurement (Step 2). It is **not** the empirical cost projection (`projection.md`). No pre-flight stage and no campaign run was executed, and no existing stage JSON was modified.

## Final campaign estimate (2026-09-27): 4 arms, per-task budgets, D16 K=2; supersedes every earlier total below

**Scope caveat RESOLVED.** `rule_based_monotone` is now **measured and included** as the fourth co-primary arm. The measurements come from `results/campaign_prep/CAMPAIGN_PREP_P1.md` Step 4, and the co-primary status is confirmed in `docs/decisions.md` (MONO, 2026-09-27). Every total in the rounds below (165.6 h and so on) is a 3-arm, uniform-1M, K=1 figure, kept for history only.

**Design simulated** (`scripts/capacity_planning/final_estimate.py`, commit `77bc842`; output `results/campaign_prep/final_estimate/final_estimate.json`; 500 Monte Carlo trials):
- 4 arms × 5 seeds, population 15, offspring 15, 300 genomes (20 generations).
- D7 strict phases (Walker-v0, then UpStepper-v0).
- Walker-v0 at 1,000,000 timesteps with K=1; UpStepper-v0 at 1,200,000 with K=2, each PPO seed a separate pool task.
- Validity resolved in the scheduler: invalid genomes occupy no worker slot.
- 20 workers (8 P-core primaries plus 12 E-cores at the production-scale E/P ratio 2.3172×), lowest-generation-first dispatch.

**t_PPO inputs**, all at B3's measurement footing (3 concurrent P-core trainings):

| Arm | Walker-v0 (1.0M) | UpStepper-v0 (1.2M) |
|---|---|---|
| direct | 1298.0 ± 16.6 s (B3) | 1485.3 ± 20.4 s (B3 × 1.182) |
| nca | 1378.3 ± 6.0 s (B3) | 1559.4 ± 22.5 s (B3 × 1.182) |
| rule_based | 1283.6 ± 73.2 s (B3) | 1463.8 ± 101.8 s (B3 × 1.182) |
| rule_based_monotone | 1330.4 ± 19.2 s (P1 measured ÷ c) | 1532.5 ± 26.4 s (P1 measured ÷ c) |

Where the factors come from:
- **1.182** is the **measured** 1M → 1.2M scaling (a same-genome rule_based anchor; P1 Step 4f). Pure linear scaling would be 1.200.
- **c = 1.0596** is the measured 8-way vs. 3-way concurrency condition factor.
- Validity: current-calibration round-3 curves for all 4 arms, monotone measured with the same proxy (offspring validity 0.933, generations 1–19).
- t_sentinel and t_develop: B3, plus P1 Step 4c for monotone.

**Result:**

| | Mean | p5 | p95 |
|---|---|---|---|
| Walker-v0 phase | **123.0 h** | 121.9 h | 124.1 h |
| UpStepper-v0 phase | **280.2 h** | 277.8 h | 282.9 h |
| **Total** | **403.2 h (16.8 days)** | **400.6 h** | **406.0 h** |

**Walker-v0 milestone:** complete after **30.5%** of total runtime (about 123 h, 5.1 days).

**Budget guard** (`docs/decisions.md`, 2026-09-27): p95 406.0 h ≤ 432 h, so **no descoping step applied**. The configuration after the guard is the full design: 4 arms on both tasks, UpStepper-v0 K=2.

**Sensitivities** (final configuration):

| Variant | Total mean | p95 |
|---|---|---|
| Pure linear scaling ×1.2 | 406.0 h | 408.8 h |
| No condition correction (c = 1, i.e. production P-cores as slow as the 8-way batch) | 422.2 h | 425.2 h |
| Resampled per-seed validity trajectories | 403.1 h | 414.6 h |

All stay under 432 h.

**Read the p5–p95 band carefully.** It reflects only per-evaluation randomness around *fixed* mean costs. It does not include uncertainty in those means, which rest on n = 3 trainings per arm and task. The sensitivity rows are the better guide to that uncertainty. Headroom to the guard is 26 h (6%) at p95.

---

## Round 4 update (2026-09-27): D16 noise policy costed; D15 monotone-arm scoping BLOCKED (contradicts current code and an existing decision)

### Step 1 -- rule_based_monotone scoping: NOT DONE. The requested premise is factually contradicted by the current code and by an unsuperseded prior decision.

The task asked me to formalize `rule_based_monotone` as a labelled ablation, out of the D15 primary grid, on the stated precedent of `rule_based_xover`. I checked that precedent against the actual code and docs before writing anything, per this project's own established discipline (`results/STALENESS_AUDIT.md`'s whole method), and it does not hold:

- **`evodevo/experiments/configs/benchmark.yaml:78`** (the live config the real benchmark launcher reads): `encoders: [direct, nca, rule_based, {encoder: rule_based, variant: monotone}]` -- **four** primary arms, not three.
- **`evodevo/encoders/rule_based/resolved_config.py:119`**: `PPO_MATRIX_CANDIDATES = (PRIMARY, MONOTONE_ABLATION)`, consumed by `benchmark.py`'s `rule_based_monotone` arm wiring (`_BUILTIN_ARM_DEFAULT_KWARGS`).
- **`docs/decisions.md`'s "Monotone vs unrestricted: an explicit decision" row (2026-09-16)** considered exactly the three options the current task lists (adopt monotone as primary / keep it as ablation / defer as co-primary) and explicitly **rejected "keep as ablation"** as under-argued ("found the prior monotone-ablation task treated 'kept as ablation' as self-evidently correct without weighing the trade-off explicitly"), choosing **DEFER: run both as co-primary, first-class configurations** instead.
- `results/rule_based/MONOTONE_ABLATION.md` itself carries a banner: "the actual decision is DEFER -- monotone now runs as a co-primary ... candidate," and `results/STALENESS_AUDIT.md` lists that file's own former "Status: ABLATION, not primary" line as **superseded** by the DEFER decision.

So `rule_based_xover` and `rule_based_monotone` are not the same pattern: `rule_based_xover` was scoped as an ablation and stayed one; `rule_based_monotone` was scoped as an ablation, then **explicitly promoted** to co-primary by a reasoned decision that is still in force and enforced in the live config. Writing a new decisions.md entry or a projection.md note that calls it "out of primary D15 scope" would silently reverse the 2026-09-16 DEFER decision without saying so, and would contradict `benchmark.yaml` as it stands today.

**I did not make either edit requested in Step 1** (no `docs/decisions.md` entry re-scoping monotone; `projection.md`'s blank `rule_based_monotone` row is unchanged). This needs a decision from Tejeshwar first: either (a) monotone stays co-primary as already decided, and its cost belongs in the primary campaign projection (not costed here, since Step 1 asked me not to measure it), or (b) the 2026-09-16 DEFER decision is being deliberately reversed now, in which case that reversal -- not a claim that it was always an ablation -- is what belongs in `docs/decisions.md`, with its own rationale (e.g. the cost/scope pressure this whole capacity-planning exercise has been surfacing). I have not guessed which; see the message accompanying this update.

### Step 2 -- D16 asymmetric evaluation-noise policy: formalized as PROPOSED

Added to `docs/decisions.md` (2026-09-27 row), flagged **PROPOSED, pending Tejeshwar's sign-off**, exactly as `docs/adr/launcher-scheduling.md` is flagged. Summary: single PPO evaluation per genome on Walker-v0 (B5: within-body noise is 16.5% of B3's between-body spread); multi-seed PPO evaluation per genome on UpStepper-v0 (B5: 66.6%, `single_eval_defensible: false`), exact seed count left to the costing below. This is stated explicitly as an asymmetric, evidence-driven default, not a fairness-uniform one, and the entry notes that the D15 grid's uniform `seeds=5` is a different axis (EA-replicate seeds) and therefore not a precedent either way for this PPO-training-noise axis.

### Step 3 -- cost of the proposed D16 policy (`d16_ppo_seed_cost.py`, 500 Monte Carlo trials, interleaved scheduling, current-calibration validity curves)

A "PPO seed" here means the genome's PPO training+eval is repeated N times and the wall-clock cost is the **sum**, not the mean, of the N repeats (`evaluation_noise.json`'s own `cost_multiplier_2_seeds: 2` / `cost_multiplier_3_seeds: 3`, confirming cost multiplies). Invalid genomes are unaffected -- decode+validate is deterministic given the genome, so re-running it for a structurally invalid body would just repeat the same sentinel result once already produced. Baseline (single eval, both tasks, current-calibration validity curves): **165.6 h** (164.3–167.0), matching Round 3's 165.6 h within Monte Carlo noise (different trial draws).

| option | new total (mean, p5–p95) | delta vs. 165.6 h baseline | delta % |
|---|---|---|---|
| 2-seed, **asymmetric** (UpStepper-v0 only) | 246.7 h (244.6–248.8) | +81.1 h | +48.9% |
| 3-seed, **asymmetric** (UpStepper-v0 only) | 327.9 h (324.9–331.2) | +162.2 h | +97.9% |
| 2-seed, **uniform** (both tasks) | 331.1 h (328.5–333.7) | +165.5 h | +99.9% |
| 3-seed, **uniform** (both tasks) | 496.8 h (492.7–501.2) | +331.2 h | +199.9% |

Sanity check: the uniform options cost almost exactly double their asymmetric counterparts (both task phases pay the multiplier instead of one), which is the expected relationship given the two tasks' near-identical per-arm evaluation costs and identical grid shape.

### Step 4 -- summary for the scoping gate

| option | total campaign wall-clock | vs. 165.6 h baseline |
|---|---|---|
| baseline: single eval, both tasks | 165.6 h (164.3–167.0) | -- |
| 2-seed asymmetric (UpStepper-v0 only) | 246.7 h (244.6–248.8) | +81.1 h (+48.9%) |
| 3-seed asymmetric (UpStepper-v0 only) | 327.9 h (324.9–331.2) | +162.2 h (+97.9%) |
| 2-seed uniform (both tasks) | 331.1 h (328.5–333.7) | +165.5 h (+99.9%) |
| 3-seed uniform (both tasks) | 496.8 h (492.7–501.2) | +331.2 h (+199.9%) |

**This is Tejeshwar's call.** Nothing here picks an option. The asymmetric variants buy exactly the noise-robustness B5 flagged as missing (UpStepper-v0 only) at roughly half the cost of the uniform variants, which also pay the same multiplier on Walker-v0 where B5 found the noise was already small relative to signal.

No PPO was run, `ea.py` was not modified, no launcher was built, and the real pre-flight/campaign were not run. `rule_based_monotone`'s cost was not measured, per Step 1's own instruction not to. No sign-off checkbox was checked.

---

## Round 3 update (2026-09-24): validity is pre-PPO, current-calibration curves measured, ambiguity and inline-sentinel idea resolved

This section supersedes Round 2 wherever they differ.

**Bottom line.** Measured with the real `run_ea` loop under the *current* calibrated configuration, offspring validity is higher than in the stale historical run for nca (+7.8 points over gens 1–19, p = 2.4×10⁻¹⁴) and rule_based (+13.1 points, p = 1.6×10⁻⁴⁰), and lower for direct (−3.5 points, p = 0.0065); all three differences are statistically significant. The interleaved campaign estimate moves from 151.9 h to **165.6 h (p5–p95 164.2–167.0)**, about 43% of the way from the 109.4 h flat floor to the 239.9 h ceiling. Serial-per-run is **480.0 h (474.5–485.9)**, so the scheduling gap is +314.5 h. The inline-sentinel idea does not change the campaign total at all (165.58 h either way).

### 1. Is validity pre-PPO and task-independent? (read-only code check)

**(a) Structural validity is decided before PPO, and (b) does not depend on the task. Both hold, with one carve-out.** From `evodevo/evaluation/ppo_runner.py:72-83`, `evaluate_body` runs validation first and returns before EvoGym or PPO are imported:

```python
h = compute_body_hash(body)

validation = validate_body(body)
if not validation.valid:
    return EvalResult(fitness=sentinel_fitness, valid=False, reason="; ".join(validation.reasons), ...)

import evogym.envs  # noqa: F401  registers evogym gym environments
```

`validate_body(body, grid_shape=(5, 5), min_voxels=3, min_actuators=1)` (`evodevo/morphology/validate.py:72`) checks shape, integer dtype, material ids, voxel count, single connected component, and actuator count. It takes no task argument and contains no reference to a task. `Encoder.decode(genome)` (`evodevo/encoders/base.py:53`) also takes only a genome. So the structural validity of a given genome depends on encoder, genome and grid only.

**The carve-out.** The `valid` flag recorded in `run_log.jsonl` can also be set False *after* validation, in the `except` branch that wraps `make_vec_env`/`PPO.learn`/rollout (`reason="simulation_instability: ..."`, `ppo_runner.py:121-130`). That path is PPO- and task-dependent. In the historical logs it never fired: of 4,500 records, 1,749 are invalid and **all 1,749 carry structural reasons; none carries `simulation_instability`**. That is Walker-v0 at 5,000 timesteps only, so an instability rate at 1M timesteps or on UpStepper-v0 is unmeasured. I treat it as negligible, but it is an assumption.

**What this retracts and what it does not.**
- **Retracted:** the caveat that the 5,000-versus-1,000,000 `ppo_timesteps` mismatch limits transfer of the validity curves. Validity is decided before any PPO step, so the mismatch does not apply to the validity of any genome. (It would still matter for reusing fitness magnitudes or timings, which this analysis does not do.)
- **Justified, not merely convenient:** applying the Walker-v0 curves to UpStepper-v0 (validation has no task input).
- **Not retracted, narrowed:** the validity *trajectory* over generations still depends weakly on which valid parents are selected, and among valid genomes that ordering comes from task- and PPO-dependent fitness. The Step 2 proxy below tests exactly this, and it held (§2). The other Round-2 caveat, that the historical run used a stale configuration, is real and is what Step 2 measures.

### 2. Current-calibration validity-by-generation, measured (`measure_validity_current_calibration.py`, `analyze_validity_round3.py`)

- **Method.** The real `evodevo.evolution.ea.run_ea` loop (unmodified) with population 15, offspring 15, 20 generations, and **20 independent seeds per arm** (n = 300 offspring per arm per generation). Encoders are built through the live benchmark path (`benchmark._resolved_arm_kwargs` + `extract_factors` + `get_encoder`), so the calibrated mutation strengths, T_primary = 100 and the NCA/rule_based resolved dynamics come from the live config: direct `mutation_rate` 0.1535, nca `mutation_sigma` 0.0305, rule_based `mutation_rate` 0.0779. No PPO or EvoGym was run; 3.0 minutes on 24 cores.
- **Fitness proxy.** The evaluator calls the same `validate_body` that `evaluate_body` calls first. Invalid genomes get the sentinel (−10, always ranked last, as in reality). Valid genomes get random U[0,1) fitness. A constant-fitness variant is in the JSON as a sensitivity check. Justification: validity is decided before fitness exists, and in a (μ+λ) loop the dominant selection effect on validity is that valid outranks invalid. **Limitation:** among valid parents, real fitness would favour some over others, and if fitness correlated with mutational robustness the real curves could differ.
- **Proxy check against real fitness.** Direct with the *historical* mutation setting (0.1) under the random proxy, 20 seeds: mean validity over gens 1–19 is 26.9% against 26.3% for the historical run (real PPO fitness), and last-5-generation validity is 43.5% against 48.3%. **No generation has non-overlapping 95% Wilson intervals** (0 of 20). This is direct only; nca and rule_based cannot be checked this way because their historical encoders differ from the current code. The constant-fitness variant gives the same curves as the random one within noise.
- **Why generation 1 is still near the prior.** With population 15 and offspring 15, generation 1 selects `sorted(population)[:15]` from a 15-member population, which is everyone, so there is no selection until generation 2 (nca: 62.7% at gen 1, then 84.3% at gen 2; rule_based: 56.7% then 88.7%). This follows from `ea.py:95`.

**Current-calibration offspring validity, pooled over 20 seeds (Wilson 95% in brackets; n = 300 per generation):**

| gen | direct | nca | rule_based |
|---|---|---|---|
| 0 | 2.0% (0.9–4.3) | 65.0% (59.4–70.2) | 54.7% (49.0–60.2) |
| 1 | 1.7% (0.7–3.8) | 62.7% (57.1–67.9) | 56.7% (51.0–62.2) |
| 2 | 2.3% (1.1–4.7) | 84.3% (79.8–88.0) | 88.7% (84.6–91.8) |
| 5 | 9.0% (6.3–12.8) | 87.7% (83.5–90.9) | 94.3% (91.1–96.4) |
| 10 | 25.3% (20.7–30.5) | 92.3% (88.8–94.8) | 95.7% (92.7–97.5) |
| 15 | 34.0% (28.9–39.5) | 93.0% (89.5–95.4) | 95.0% (91.9–96.9) |
| 19 | 41.7% (36.2–47.3) | 93.0% (89.5–95.4) | 93.3% (89.9–95.6) |

Full 20-generation rates with intervals: `capacity_planning/validity_round3_analysis.json`; plot: `capacity_planning/validity_current_vs_historical.png`.

**Comparison against the historical curves (historical n = 75 per generation, 5 seeds):**

| arm | generations with non-overlapping Wilson intervals | pooled gens 1–19: current vs historical (difference; Fisher exact p) | last 5 gens: current vs historical (difference; Fisher exact p) |
|---|---|---|---|
| direct | 0 of 20 | 22.9% (n=5,700) vs 26.3% (n=1,425); −3.5 points; p = 0.0065 | 38.1% vs 48.3% (n=1,500 vs 375); −10.1 points; p = 4.0×10⁻⁴ |
| nca | 9 of 20 (gens 1, 9, 13–19) | 89.2% vs 81.5%; +7.8 points; p = 2.4×10⁻¹⁴ | 93.7% vs 78.4%; +15.3 points; p = 1.5×10⁻¹⁶ |
| rule_based | 16 of 20 (gens 3, 5–19) | 92.1% vs 78.9%; +13.1 points; p = 1.6×10⁻⁴⁰ | 93.7% vs 75.2%; +18.5 points; p = 3.1×10⁻²² |

All six comparisons use the same test (two-sided Fisher exact on pooled valid/invalid counts) and all are significant at 0.01. The evidence is strongest for rule_based, then nca, then direct, and the size of the difference in points follows the same order, so the language is the same for all three arms: current-calibration validity is higher for nca and rule_based and lower for direct. **Caution:** pooling treats every evaluation as independent, but evaluations within a seed's trajectory are correlated, so these p-values are optimistic; they order the arms correctly but should not be read as exact. Per-generation intervals overlap for direct in all 20 generations mainly because the historical sample is only n = 75 per generation.

The two arms that dominate cost (nca and rule_based, ~85% of the work) are higher under the current calibration, which is the direction predicted in Round 2 from their smaller current mutations. Both match the historical run in shape (a jump at gen 2) and differ in level.

### 3. Re-run Monte Carlo with the current-calibration curves (500 trials, same seed)

Curves: direct = logistic fit to the current data (slope +0.159/gen, SE 0.007; note the fit gives 4.7% at gen 0 against 2.0% observed, a small overestimate); nca and rule_based = empirical pooled per-generation rates (a step-then-plateau shape, not logistic).

| variant | interleaved mean (p5–p95) | serial-per-run mean (p5–p95) | serial − interleaved |
|---|---|---|---|
| flat random-genome validity (floor) | 109.4 h (107.5–111.0) | 381.2 h (374.3–388.0) | +271.8 h |
| historical curves | 151.9 h (150.3–153.6) | 480.9 h (475.3–486.5) | +328.9 h |
| **current-calibration curves** | **165.6 h (164.2–167.0)** | **480.0 h (474.5–485.9)** | **+314.5 h** |
| current, each run resamples one measured run's trajectory | 165.7 h (159.4–172.1) | 443.4 h (420.2–465.2) | +277.7 h |
| ceiling: all offspring after gen 0 valid | 239.9 h (239.4–240.3) | 513.0 h (511.3–514.9) | +273.1 h |

**Where the number falls.** The interleaved estimate of about 166 h lies **43% of the way from the 109.4 h floor to the 239.9 h ceiling**, 13.6 h (9%) above the historical-curve estimate. It cannot exceed 240 h. The remaining uncertainty is the fitness-selection effect that the proxy cannot capture (§2) and the round-1 caveats (n = 3 timing observations, the P/E ratio, no contention beyond it). The scheduling gap stays large in every scenario (+272 to +329 h).

### 4. Statistical language: generation-0 validity against the random-genome rate, all three arms

Prior = `results/inspection/{arm}/{arm}_sanity.json`, n = 2,000. "Inside/outside" refers to whether the gen-0 rate falls inside the Wilson 95% interval around the *prior* rate; the exact binomial p-value tests the gen-0 count against the prior rate.

| arm | prior (Wilson 95%, n=2,000) | historical gen 0 (k/n) | inside prior CI? | binomial p | current gen 0 (k/n) | inside prior CI? | binomial p |
|---|---|---|---|---|---|---|---|
| direct | 2.15% (1.60–2.88%) | 1/75 = 1.3% | outside (below) | 1.00 | 6/300 = 2.0% | inside | 1.00 |
| nca | 69.15% (67.09–71.14%) | 50/75 = 66.7% | outside (below) | 0.62 | 195/300 = 65.0% | outside (below) | 0.13 |
| rule_based | 59.75% (57.58–61.88%) | 39/75 = 52.0% | outside (below) | 0.195 | 164/300 = 54.7% | outside (below) | 0.077 |

Reading this: the prior's own interval is narrow (about ±2 points at n = 2,000), while a gen-0 sample of 75 or 300 has a far wider one, so falling outside the prior's interval is expected for a modest sample and is not the right test; the prior is inside every gen-0 Wilson interval, and no single-sample binomial test rejects at the 5% level. **One caution:** pooling the historical and current gen-0 samples for rule_based gives 203/375 = 54.1%, which is distinguishable from the 59.75% prior (p = 0.027). So rule_based gen 0 in the EA loop appears slightly (about 5–6 points) lower than the inspection rate. I have not investigated why. It moves the cost estimate by well under 1%, since only one of 20 generations is affected. nca pooled: 65.3% (p = 0.12); direct pooled: 1.9% (p = 0.86).

### 5. The "~5 h alone" ambiguity: resolved, and the ADR was misleading

The "about 5 h alone" in the ADR was the **5-seed direct row** under **flat validity** (Round 1: 5.07 h), compared against a **single direct run's** finish time in the pool. Those are different units. Solo means one run (or one row) with the whole 20-worker pool to itself, no contention (500 trials each):

| validity scenario | solo single direct run (mean, p5–p95) | solo 5-seed direct row | same single run pooled with all 15 Walker runs (FIFO) | slowdown from pooling |
|---|---|---|---|---|
| flat | **3.34 h** (1.21–5.91) | 5.06 h (3.58–6.83) | 38.8 h | 11.6× |
| historical curves | **13.16 h** (11.37–15.02) | 15.15 h (14.04–16.33) | 69.3 h | 5.3× |
| current curves | **12.95 h** (10.99–14.97) | 14.46 h (13.29–15.75) | 74.6 h | 5.8× |

The difference is large: a solo direct run is 3.3 h with flat validity but about 13 h once validity rises across generations (validity makes valid evaluations, and therefore ~1,300 s PPO tasks, more frequent). The pooling delay is real but is 5–6×, not the ~8–14× the old sentence implied. The ADR is corrected.

### 6. The "resolve sentinels inline" idea: quantified (`scheduling_trace.py`, `round3_solo_and_inline.json`)

Third policy "inline": invalid genomes (decode + validate only) never take a pool slot; their measured sentinel time is charged serially to the run. Same random draws for both policies in each trial.

| validity scenario | direct-run finish in the pooled Walker phase: FIFO → inline | solo single direct run: FIFO → inline | **full campaign, interleaved: FIFO → inline** |
|---|---|---|---|
| flat | 38.8 h → 14.6 h | 3.34 h → 1.99 h | 109.39 h → 109.36 h |
| historical curves | 69.3 h → 64.4 h | 13.16 h → 7.21 h | 151.99 h → 152.00 h |
| current curves | 74.6 h → 68.5 h | 12.95 h → 6.71 h | 165.58 h → 165.58 h |

- **The campaign total is unchanged in every scenario**, because the pool is work-bound and sentinels use almost none of its capacity. The Walker-phase makespan is identical (84.46 h, current).
- **Only direct's own latency improves.** Of the pool-induced queueing delay for direct (pooled finish minus solo finish), inline recovers 68% under flat validity but only **about 10% under current validity** (74.6 → 68.5 h, against a delay of 61.6 h), and 9% under the historical curves. At realistic validity, direct's valid PPO evaluations still queue behind the others.
- **Not a substitute for the propose/evaluate/tell refactor.** Inline resolution still needs batched dispatch, so it is an increment on top of the refactor. At a high level it would touch the (not yet existing) launcher's dispatch layer, `evodevo/evaluation/ppo_runner.py` (expose the validate stage so the launcher can call it; `evaluate_body` already validates first), and `evodevo/evolution/eval_bookkeeping.py` (accept a pre-resolved result while keeping the sentinel record byte-identical: reason string, `wall_clock_s`, counters). That is a small addition once the refactor exists. Not implemented.
- **Conclusion.** Not worth prioritising for wall-clock. It could matter only if direct's early completion mattered for partial results, and lowest-generation-first scheduling already handles balance at no cost to the makespan.

### 7. Accepted risks, not further investigated

- **(a) Random fitness among valid genomes (largest remaining uncertainty).** The current-calibration validity-by-generation proxy assigned random fitness to valid genomes. It cannot capture whether real PPO-based selection among valid parents moves the validity trajectory differently from random selection. The one check available (direct with the historical mutation setting, 26.9% against 26.3% over gens 1–19) supports the proxy for direct only. This is the largest remaining source of uncertainty in the wall-clock estimate.
- **(b) Post-PPO `simulation_instability` invalidation.** It was never observed in the historical logs (0 of 1,749 invalid records), but that data is Walker-v0 at 5,000 timesteps only. Its rate at 1,000,000 timesteps, or on UpStepper-v0, is unmeasured.
- **(c) Neither risk changes the scheduling recommendation.** The interleaved-versus-serial comparison holds in every validity scenario tested: serial-per-run is **3.49× slower (flat), 3.17× (historical), 2.90× (current) and 2.14× (ceiling)** than interleaved, that is 2.1–3.5×, with an absolute gap of +272 to +329 h. (An earlier draft of this item said "2.9–3.5×"; the ceiling scenario is 2.14×, so the range is corrected here.) Only the *absolute* wall-clock estimate (best current figure: **165.6 h**, hard ceiling 239.9 h) carries this residual uncertainty.
- **(d) Closing (a) is not recommended before the campaign.** It would need real PPO time. The two independently measured calibration estimates (historical curves 151.9 h, current curves 165.6 h) already agree within about 9%, and the real launcher's first real generations will reveal the true validity trajectory as a natural by-product of running it.

### 8. Documents

The ADR's validity-sensitivity, fairness/solo-run, and inline-sentinel bullets are updated; its sign-off section and core recommendation (interleaved cross-run scheduling per D7 phase) are unchanged, since nothing here changes which design is recommended. No `ea.py`, stage JSON, historical log or PPO was touched or run.

Reproduce: `python3 scripts/capacity_planning/measure_validity_current_calibration.py && python3 scripts/capacity_planning/analyze_validity_round3.py && python3 scripts/capacity_planning/simulate_campaign.py 500 && python3 scripts/capacity_planning/scheduling_trace.py`.

---

## Round 2 update (2026-09-24): real validity-by-generation data, corrected claims, reconciled documents

This section supersedes the flat-validity headline below where they differ. Everything under "Headline" and Sections 1–6 is the round-1 analysis, left as written apart from the one marked correction.

### 1. The historical run, and what it is

- **Location:** `results/runs/{direct,nca,rule_based}/Walker-v0/seed_{0..4}/{run_log.jsonl,run_meta.json}`: 15 runs (3 arms × 5 seeds), Walker-v0 only. Each has population 15, offspring 15, eval_budget 300 (= 20 generations of 15), cache off. This is the D15 shape exactly.
- **The names `project-history.md` and `overview.md` do not exist in the repository.** The run is documented in `docs/RESULTS_EXPLAINED.md` §5, `docs/READY_TO_RUN.md` and `docs/decisions.md`: it was estimated at ~11 h and took ~7.5 h, and the documented explanation is that realised validity was far above the random-genome priors.
- **Two limits on how far it transfers:**
  - **`ppo_timesteps` was 5,000, not the campaign's 1,000,000.** Only the validity trend is used, never its timings. Selection prefers valid bodies whatever the PPO budget (sentinel fitness −10), so the trend should transfer, but that is an assumption.
  - **It is a pre-calibration run, committed as "stale, for provenance only"** (`results/COMPARISON_READINESS.md` §7 flags it as a blocker for headline use). It used uncalibrated mutation (direct 0.10, nca σ 0.10, rule_based 0.20) and older NCA/rule-based dynamics. The current calibrated values are direct 0.1535, nca σ 0.0305, rule_based 0.0779.
- The original logs were only read, never written (`git status` shows them unchanged).

### 2. Offspring validity by generation (`capacity_planning/validity_by_generation.{png,json}`, script `validity_by_generation.py`)

Every `run_log.jsonl` record is one evaluated offspring, so grouping by generation gives per-generation offspring validity (75 evaluations per arm per generation, pooled over 5 seeds). Generation 0 is the random initial population.

| arm | gen 0, historical (k/n, Wilson 95%) | random-genome prior (`inspection`, n=2000, Wilson 95%) | consistent? (see Round 3 §4 for the full statistics) | gens 1–19 mean | last 5 gens mean | trend |
|---|---|---|---|---|---|---|
| direct | 1/75 = 1.3% (0.2–7.2%) | 2.15% (1.60–2.88%) | prior inside the historical CI; exact binomial p = 1.00 | 26.3% | 48.3% | **strongly rising**, logit slope +0.183/gen (SE 0.013, z = 14); fitted 4.3% → 59.5% |
| nca | 50/75 = 66.7% (55.4–76.3%) | 69.15% (67.09–71.14%) | prior inside the historical CI; p = 0.62 | 81.5% | 78.4% | jumps to ~0.80 by gen 1 then flat; slope −0.017 (SE 0.011) over gens 0–19, −0.037 (SE 0.013) over gens 1–19 (slight decline) |
| rule_based | 39/75 = 52.0% (40.9–62.9%) | 59.75% (57.58–61.88%) | prior inside the historical CI; p = 0.195 | 78.9% | 75.2% | jumps to ~0.87 by gen 2, then drifts to ~0.77; slope +0.014 (SE 0.011) |

- **Generation 0 is statistically consistent with the random-genome rates for all three arms** (exact binomial p = 1.00, 0.62, 0.195; the prior lies inside each historical Wilson interval). The historical point estimates do fall slightly below the prior's own narrow n = 2000 interval, because n = 75 is far noisier; "agree" in the first draft of this section was qualitative and is replaced here by these numbers (Round 3 §4 adds the current-calibration gen-0 data).
- **direct** is monotone and well described by a logistic in generation. Its seeds are very heterogeneous: four seeds reach 57–68% over the final 5 generations, while **seed 4 never found a valid body in 300 evaluations**.
- **nca and rule_based are not monotone**. They jump after generation 0 and then plateau, so a logistic trend is the wrong model. For them I used the empirical pooled per-generation rates.
- **All three arms are characterised**, but only for Walker-v0 and only under the pre-calibration configuration. UpStepper-v0 and the calibrated encoders are **uncharacterised**; the simulation assumes the Walker curves hold for both tasks.

### 3. Re-run Monte Carlo with the real curves (500 trials, same seed and draws as round 1)

| variant | interleaved mean (p5–p95) | serial-per-run mean (p5–p95) | serial − interleaved |
|---|---|---|---|
| round 1, flat random-genome validity | 109.4 h (107.5–111.0) | 381.2 h (374.3–388.0) | +271.8 h |
| **round 2: fitted/empirical curves** | **151.9 h (150.3–153.6)** | **480.9 h (475.3–486.5)** | **+328.9 h** |
| round 2: each run resamples one historical run's own trajectory (keeps direct's dead seed) | 152.1 h (144.0–160.0) | 435.3 h (409.0–461.2) | +283.2 h |
| round 1 ceiling: all offspring after gen 0 valid | 239.9 h (239.4–240.3) | 513.0 h (511.3–514.9) | +273.1 h |

**Where the real number falls.** The interleaved estimate is about **152 h, roughly a third (33%) of the way from the 109 h flat floor to the 240 h ceiling**. Serial-per-run is 75% of the way from 381 h to 513 h (435–481 h). The fitted curve slightly overestimates direct's early generations (4.3% fitted against 1.3% observed at generation 0), which is negligible for cost.

**Which way the real campaign probably deviates (not simulated).** Under the current calibration, nca and rule_based use much smaller mutations (σ 0.0305 against 0.10; rate 0.0779 against 0.20). Smaller mutations keep offspring closer to a valid parent, which would raise their validity above the historical ~0.8 and push cost toward the ceiling. Direct's mutation is larger (0.1535 against 0.10), which may slow its rise. nca and rule_based account for about 85% of the work, so the risk is skewed upward, bounded by 240 h. The historical run is therefore a good central estimate, not a guarantee, and a short calibrated-configuration check of validity-by-generation would tighten it.

### 4. The round-1 claim "validity uncertainty is larger than any scheduling effect"

**Quoted** from the bullet at the top of the round-1 headline: *"…interleaved rises to 239.9 h … and serial to 513.0 h. The serial penalty stays about +273 h. This uncertainty is larger than any scheduling effect and should be resolved before the scoping gate."*

It compared the validity swing on the interleaved total (240 − 109 = 130.5 h) with "any scheduling effect". By this report's own numbers that is **false**: the serial-versus-interleaved gap is +272 h (+329 h with the real curves), about twice the whole validity swing. The claim is corrected in place. What the numbers do support:

- The scheduling gap is large in every validity scenario (+272 to +329 h), so the interleaving recommendation does **not** depend on resolving validity.
- The validity assumption moves the absolute budget by up to 131 h (a factor of 2.2 on the interleaved total). It is larger only than the *secondary* scheduling effects: the strict phase barrier (~1%), timing spread (0.3 h for interleaved) and FIFO against lowest-generation-first (identical makespan).

### 5. Fair-scheduling claim (ADR): substantiated, but the original wording was wrong and is corrected

The ADR claimed FIFO "leaves direct runs finishing late". The trace shows the opposite in ordering: **direct finishes first** (Walker phase, 60 trials, `scheduling_trace.json`):

| policy, validity | phase makespan | direct / nca / rule_based finish (h) | generations done at 25% of the phase (direct / nca / rule_based) |
|---|---|---|---|
| FIFO, flat | 55.8 h | 38.7 / 54.1 / 55.1 | 7.2 / 5.0 / 4.6 |
| lowest-generation-first, flat | 55.8 h | 52.4 / 54.1 / 55.1 | 5.8 / 5.0 / 4.7 |
| FIFO, historical curves | 77.5 h | 69.4 / 75.4 / 76.8 | 6.8 / 5.7 / 5.0 |
| lowest-generation-first, historical curves | 77.5 h | 73.8 / 75.4 / 76.8 | 6.0 / 5.8 / 5.1 |

**Mechanism (worked example, first direct run, historical curves, FIFO):** in generation 3 all 15 evaluations were invalid sentinels (longest task 0.0 s), yet the generation took **10,276 s** in the pool. Generation 5 was the same (0.0 s of work, 12,021 s in the pool). Even a generation with a 1,287 s valid task took 12,290 s. The reason is that a sentinel evaluation still needs a worker slot, and all 20 workers are busy with about 1,300 s PPO tasks from the nca and rule_based runs. Each direct generation therefore waits about one queue turnaround (roughly 3–4 h) regardless of how cheap it is. Twenty generations then take about 69 h in the pool against about 5 h alone. Direct is only modestly ahead of the other arms because cheapness buys almost nothing when everything waits in the same FIFO queue.

So the accurate statements are: (a) FIFO does not starve direct; (b) FIFO leaves arms unevenly advanced at any early stop, with direct about 2 generations ahead at 25% of the phase; (c) lowest-generation-first equalises progress at the same makespan. The ADR has been rewritten with this. The idea of resolving sentinel evaluations inline without a pool slot is recorded there as untested.

### 6. Document reconciliation

- **`projection.md`** now carries a prominent note at the top: it uses a simplified throughput-division model that ignores generation dependencies and the stale 2.1440× P/E ratio, and `CAPACITY_PLANNING_REPORT.md` is the source for any wall-clock decision. This is intentional. The methodology and tables are unchanged, its per-arm cost breakdown remains valid, and only its wall-clock totals are superseded. The generator `b6_projection.py` was **not** modified, so regenerating would drop the note.
- **`docs/adr/launcher-scheduling.md`** updated: new interleaved/serial figures, the corrected fairness bullet, and the sensitivity bullet. It remains a proposal awaiting sign-off.
- Not touched: `ea.py`, any stage JSON, `b6_projection.py`, the historical logs. No launcher was built and no pre-flight or campaign run.

### 7. What is still open

- The calibrated-configuration validity curve, and UpStepper-v0 (Walker curves assumed).
- nca and rule_based cost is the main driver of the 152 h figure, and its main uncertainty is the calibrated-mutation direction discussed above.
- Round-1 caveats on n = 3 timing observations and the P/E ratio still apply.

Reproduce: `python3 scripts/capacity_planning/validity_by_generation.py && python3 scripts/capacity_planning/simulate_campaign.py 500 && python3 scripts/capacity_planning/scheduling_trace.py`.

---

## Round 1 (flat random-genome validity assumption)

> **✅ RESOLVED 2026-09-27 (final estimate, top of this report): `rule_based_monotone` is now measured and included; the 4-arm total is 403.2 h mean / 406.0 h p95.** The original caveat follows, kept for history.
>
> **⚠ SCOPE CAVEAT, added 2026-09-27 (Round 4): every wall-clock figure in this report may be missing a fourth arm.**
> Every total in this document -- the 165.6 h current-calibration figure, the 109.4/151.9/165.6/239.9 h floor/historical/current/ceiling range, the D16 costing table below, and every earlier round's numbers -- assumes the D15 grid's original 3-arm scope (`direct`, `nca`, `rule_based`) and excludes `rule_based_monotone`. But `evodevo/experiments/configs/benchmark.yaml`, `evodevo/encoders/rule_based/resolved_config.py`'s `PPO_MATRIX_CANDIDATES`, and a 2026-09-16 `docs/decisions.md` entry ("Monotone vs unrestricted: an explicit decision") currently treat `rule_based_monotone` as a **co-primary 4th arm**, not an excluded ablation (see Round 4 §Step 1 below for the full citation trail). This was surfaced, not resolved, in Round 4: no `docs/decisions.md` or `projection.md` edit was made either way, per instruction, pending Tejeshwar's confirmation of which scoping is actually intended. **If the 4-arm scoping is confirmed current, every total in this report understates the campaign by roughly one arm's worth of cost** (`rule_based_monotone` was not measured or costed here, so no corrected number is offered) **and must be resolved as its own dedicated task before any figure in this report is treated as final for the scoping gate.**

## Headline

| D15 grid (pop 15 × 20 gens × 5 seeds × 3 arms × 2 tasks = 9,000 evaluations) | mean | median | p5 – p95 | 95% CI of mean |
|---|---|---|---|---|
| **INTERLEAVED** (all replicates of a D7 task phase pooled on 20 workers) | **109.35 h** | 109.38 | **107.47 – 110.95 h** | 109.26 – 109.44 |
| SERIAL_PER_RUN (one replicate at a time, width 15) | 381.15 h | 381.09 | 374.28 – 388.00 h | 380.79 – 381.52 |
| **Cost of not building cross-run interleaving** | **+271.8 h** (11.3 days) | 271.8 | 265.2 – 278.5 h | 271.5 – 272.2 |

Serial-per-run is **3.49×** slower (p5–p95 of the ratio: 3.41–3.56). It leaves 25% of the workers idle by construction, and each generation waits for its slowest worker, which is an E-core 2.3172× slower.

Read the following before using the 109 h:

- **The interleaved figure is a throughput-bound floor under the B6 validity assumption.** That assumption is that every generation's evaluations are valid at the random-genome rate (direct 2.15%, nca 69.2%, rule_based 59.8%). After generation 0, offspring come from valid, selected parents, so their real validity is probably higher. If every post-generation-0 offspring is valid, which is an upper bound and not a prediction, interleaved rises to **239.9 h** (239.4–240.3) and serial to 513.0 h. The serial penalty stays about +273 h. *(Correction, round 2: the original closing sentence here said this uncertainty was "larger than any scheduling effect". That was wrong against these very numbers — see the top section, item 4. Round 2 also resolved much of the validity uncertainty with real historical data.)*
- **`projection.md` is not directly comparable.** It says 105.0 h using a 2.1440× P/E ratio from a 20,000-timestep probe. This report uses the production-scale 2.3172× (`pe_core_ratio_measurement_production_scale.json`), which explains most of the +4.1% gap. `projection.md` and `b6_projection.py` still carry 2.1440× and were not touched here.
- **Per-evaluation variance is under-measured.** The timing distribution rests on n = 3 raw observations per (arm, task) (see Step 3). Measured CVs are 0.5–7%. With an assumed CV of 10% or 25%, serial rises to 405.8 h or 463.9 h, while interleaved stays at 109.4 h or 109.7 h. Interleaving is robust to timing spread and the barrier-bound serial mode is not.
- The strict D7 phase barrier (Walker fully completes first) costs at most about 1%: the interleaved pool realises 99.1% of its capacity.

---

## 1. Step 1 — Verification of the scheduling-model hypothesis (read-only)

**(a) The generation barrier is algorithmic. Confirmed.** In `evodevo/evolution/ea.py:95` (`survivors = sorted(population, key=fitness, reverse=True)[:mu]`), the population includes all λ offspring of the previous generation (`population = survivors + offspring`). Generation g+1's parents cannot be chosen until every generation-g fitness exists. This is a property of (μ+λ) generational selection. An asynchronous or steady-state EA would not need it, but that is a different algorithm.

Two nuances:

- As coded, the barrier is trivially satisfied, because `run_ea` evaluates one genome at a time (`tracker.evaluate`, `eval_bookkeeping.py:120`). There is no intra-generation concurrency in `ea.py` today.
- The λ offspring genomes depend only on the survivors and the RNG, not on each other's fitness. A launcher can therefore generate a batch and evaluate it concurrently, but that needs a propose/evaluate/tell split of the loop (a code change, not made here).

**(b) "Seeds" are independent replicate runs. Confirmed.** `benchmark.py` builds a `(arm_id, task, seed)` run matrix. Each run gets `dataclasses.replace(config, seed=seed, ...)`, its own output directory (`_output_dir_for`) and its own `np.random.default_rng(config.seed)` inside `run_ea`. B6 multiplies `evaluations_per_seed * seeds`. The fitness cache is off for the primary comparison (`cache_enabled` defaults to False) and is per-run when enabled (`resolve_cache`). No algorithmic state is shared. The only shared artifacts are a monitoring heartbeat file and a read-only mutation-calibration file. Replicates can be scheduled concurrently, and D7 already states that results do not depend on concurrency. The existing launcher (`benchmark.py:695`) runs them in a serial `for` loop, so nothing interleaved exists today.

Neither (a) nor (b) is false, so the rest of the analysis stands.

**(c) D7, quoted from `docs/final_decisions.md`:**

> **Decision (one machine): enqueue all Walker-v0 cells first, then all UpStepper-v0 cells.** The launcher keeps every core busy throughout — this is purely a queue ordering, not a throughput choice.

The stated reason is "you have **one complete, publishable task** rather than two half-finished ones". D7's text is an enqueue ordering. The strict "Walker phase fully completes before UpStepper starts" barrier used in the simulator is the more conservative reading. Its cost is bounded at about 1% (above).

**(d) λ against the corrected optimal concurrency (20).** The `projection.md` grid has six rows, all with population 15. λ (offspring_size) is taken as 15. B6 models one population evaluated per generation, `FAST_DEV` sets population 15 and offspring 15, and D15 formally leaves it open (`OpenProfileDecisionError`).

| # | population (λ) | generations | seeds | arm | task | λ vs 20 | λ-limited in isolation? |
|---|---|---|---|---|---|---|---|
| 1 | 15 | 20 | 5 | direct | Walker-v0 | 15 < 20 | **yes** |
| 2 | 15 | 20 | 5 | direct | UpStepper-v0 | 15 < 20 | **yes** |
| 3 | 15 | 20 | 5 | nca | Walker-v0 | 15 < 20 | **yes** |
| 4 | 15 | 20 | 5 | nca | UpStepper-v0 | 15 < 20 | **yes** |
| 5 | 15 | 20 | 5 | rule_based | Walker-v0 | 15 < 20 | **yes** |
| 6 | 15 | 20 | 5 | rule_based | UpStepper-v0 | 15 < 20 | **yes** |

**All six rows are λ-limited.** A single run can occupy at most 15 of 20 workers (75%), and `cpu_slots(15)` places them on 8 P-cores plus 7 E-cores.

## 2. Step 2 — Measured orchestration overhead (real `run_ea`, mocked instant evaluator)

Script: `scripts/capacity_planning/measure_orchestration_overhead.py`. It runs the real `run_ea` with the real encoders, the real `RunLogger` (per-record JSONL flush) and a mock evaluator that returns instantly. There is no PPO and no EvoGym. Setup: population 15, offspring 15, 20 generations, 20 repetitions per arm (400 generation samples per arm). The whole measurement took 16.0 s. Per-generation orchestration is the generation's window minus the time inside the mock evaluator, so it covers selection, mutation, per-record bookkeeping and logging.

| arm | mean (ms) | sd (ms) | min (ms) | max (ms) |
|---|---|---|---|---|
| direct | 0.805 | 0.080 | 0.730 | 2.072 |
| nca | 1.155 | 2.149 | −0.829 | 19.916 |
| rule_based | 1.106 | 0.079 | 0.875 | 1.496 |

Caveats:

- `decode()` runs in-process in `EvaluationTracker.evaluate`. It was timed separately and subtracted. For nca the subtraction is spread evenly across generations, so a few nca samples came out slightly negative. That is a measurement artifact, and it explains the nca sd.
- Real `decode()` costs 0.030 s per generation for nca. Rule_based decode costs about 0.24 s per genome (`evaluation_cost.json` `t_develop`), which is about 3.6 s per generation. It was stubbed with a cached body so the run stays under a minute, so the rule_based row is orchestration only. The simulator's BASE case charges the launcher for decode as a conservative upper bound (mean 1.21 s per generation); the "orch only" variant charges 1.0 ms.
- Either way the overhead is about 10⁻⁶ – 10⁻³ of a generation's roughly 1,300 s evaluation time. It is negligible for scheduling.

## 3. Step 3 — Monte Carlo discrete-event simulation

Script: `scripts/capacity_planning/simulate_campaign.py` (500 trials per mode, seed 20260924, both modes fed the same random draws in each trial). Output: `capacity_planning/simulation_results.json`.

**Model**

- **Evaluation-time distribution.** `evaluation_cost.json` stores only n = 3 raw t_PPO observations per (arm, task). That is too few to resample, so the **summary-stats case applied**: a lognormal is fitted to each measured mean and sd. The same was done for t_sentinel (n = 100 summaries).
- **Validity.** Each evaluation is a Bernoulli mixture using the B6 formula: t_PPO with probability validity_rate, otherwise t_sentinel. Rates come from `results/inspection/{arm}/{arm}_sanity.json`.
- **Workers.** They follow `cpu_slots()` and `allocation_order()`. Width 20 is 8 P + 12 E. Width 15 is 8 P + 7 E. An E-core draw is multiplied by **2.3172×** (`pe_core_ratio_measurement_production_scale.json`, 3 P-core against 3 E-core evaluations at 1M timesteps).
- **Runs.** Each run is 20 batches of 15, that is 300 evaluations, matching `FAST_DEV` `eval_budget = 300`. Overhead is added between batches.
- **SERIAL_PER_RUN.** One replicate at a time, batch width 15, arms in order, Walker phase then UpStepper phase.
- **INTERLEAVED.** All 15 runs of a task phase (3 arms × 5 seeds) are pooled on 20 workers under greedy list scheduling: fastest idle worker first, ready tasks in FIFO order. A run's generation g+1 becomes ready only after its own generation g completes, plus overhead. The Walker phase completes before the UpStepper phase.

**Full campaign, per phase (interleaved):** Walker-v0 55.76 h (54.55–56.98), UpStepper-v0 53.58 h (52.48–54.79).

**Per D15 grid row (hours, mean with p5–p95).** A row cannot have its own wall-clock in a pooled phase, so it is shown two ways: run alone with its 5 seeds interleaved on 20 workers (comparable with serial), and the time its last run finished inside the pooled phase.

| arm | task | SERIAL row | INTERLEAVED, row alone | difference | finish inside pooled phase |
|---|---|---|---|---|---|
| direct | Walker-v0 | 16.70 (11.78–22.16) | 5.07 (3.57–6.93) | 11.63 | 40.3 |
| direct | UpStepper-v0 | 16.22 (11.92–20.81) | 4.96 (3.58–6.60) | 11.27 | 38.7 |
| nca | Walker-v0 | 89.14 (89.10–89.20) | 30.59 (29.49–31.38) | 58.55 | 55.0 |
| nca | UpStepper-v0 | 86.24 (86.09–86.39) | 29.29 (28.20–29.87) | 56.95 | 52.7 |
| rule_based | Walker-v0 | 87.48 (86.74–88.12) | 24.88 (24.08–25.72) | 62.59 | 55.7 |
| rule_based | UpStepper-v0 | 85.37 (84.51–86.17) | 24.03 (23.27–24.84) | 61.34 | 53.6 |

Reading the table:

- The direct row alone takes 5 h, not the 0.86 h of `projection.md`. With 2% validity a generation's cost is set by the rare valid evaluation, so the row is bounded by its 20-generation dependency chain and not by throughput. Only about 17% of the worker-time is busy.
- Pooling the direct runs with the other arms hides that idle time, which is why the total is throughput-bound.
- The late finish of the direct runs in the pooled phase is an artifact of FIFO priority (see the ADR).

**Sensitivity of the campaign totals (hours, mean with p5–p95):**

| variant | serial | interleaved | difference |
|---|---|---|---|
| BASE (measured CV, B6 validity, decode counted in the launcher) | 381.15 (374.28–388.00) | 109.35 (107.47–110.95) | 271.8 |
| orchestration only, no decode | 380.95 | 109.35 (107.55–110.96) | 271.6 |
| assumed CV 10% (not measured) | 405.79 | 109.38 | 296.4 |
| assumed CV 25% (not measured) | 463.94 | 109.67 | 354.3 |
| upper bound: all offspring after generation 0 valid (not a prediction) | 512.95 | 239.88 (239.40–240.30) | 273.1 |

## 4. Step 4 — Cross-validation against classical laws

Script: `scripts/capacity_planning/laws_crosscheck.py`. Output: `capacity_planning/laws_results.json`.

### 4a. Amdahl's law

The serial fraction is (1−P) = o / (o + G(N)), with o = 1.21 s mean overhead per generation (decode included, conservative) and G(N) the mean generation evaluation time at width N. For serial-per-run, G(15) is the simulated mean batch makespan, 2,286 s. For interleaved, G(20) is the pool wall-clock per generation-batch of work (simulated total ÷ 600 batches), 656 s. T₁ is one P-core doing everything: 1,428 h (59.5 days).

| mode | N | (1−P) | S(N) | Amdahl T (h) | sim mean (h) | Amdahl vs sim |
|---|---|---|---|---|---|---|
| serial-per-run | 15 | 5.3×10⁻⁴ | 14.89 | 95.9 | 381.2 | **−74.8%** |
| interleaved | 20 | 1.85×10⁻³ | 19.32 | 73.9 | 109.3 | **−32.4%** |

**Why they disagree.** Amdahl's law assumes N identical processors. Here the cores are not identical (2.3172× P/E ratio), so 20 workers are worth only 13.18 P-core-equivalents. Amdahl therefore under-predicts wall-clock. Replacing N with the P-equivalent capacity gives a heterogeneity-adjusted variant, a check I added and did not ask for:

- **Interleaved, N_eff = 13.18:** 110.8 h, which is **+1.3%** against the sim. The heterogeneity fully explains the interleaved gap.
- **Serial, N_eff = 11.02:** 130.3 h, still **−65.8%**. The remaining gap is not a serial fraction. It is load imbalance: each batch waits for the slowest of 15 draws, an E-core running a valid evaluation, and the mean-based law cannot see that. It is also the 5 idle workers. The simulator realises 34% of the width-15 capacity in serial mode and 99% in interleaved mode.

The measured orchestration overhead is too small to matter in either mode. The barrier hurts because of stragglers and idle workers, not because of the serial fraction.

### 4b. Universal Scalability Law (Gunther)

I fitted X(N) = λN / (1 + α(N−1) + βN(N−1)) by non-linear least squares to the corrected `concurrency_sweep.json` (6 points, 3 parameters, so 3 degrees of freedom). The sweep was run at 20,000 timesteps, not the campaign budget.

| N | measured (genomes/min) | USL fit |
|---|---|---|
| 8 | 30.42 | 28.33 |
| 12 | 34.02 | 34.73 |
| 16 | 33.83 | 39.16 |
| 20 | 46.27 | 42.41 |
| 24 | 45.86 | 44.89 |
| 28 | 46.15 | 46.84 |

Fitted: **λ = 5.81** (SE 3.9), **α = 0.0916** (SE 0.153), **β = 6×10⁻¹⁸** (about 0; SE 0.0027), R² = 0.82.

- **USL predicted throughput: 38.19 genomes/min at N = 15 and 42.41 at N = 20.** The measured value at 20 is 46.27, so the fit is 8.3% low there.
- **Interpretation.** β is about 0, so the fit is contention-only (Amdahl-like, no retrograde throughput), and the plateau at 20–28 is the core count. The parameters are poorly identified: the standard errors are as large as the estimates, and the data are stepwise rather than smooth (the P-core boundary at 8, the physical-core boundary at 20).

**Comparison against the simulator, as throughput relative to N = 8.** Absolute genomes/min are not comparable, because the sweep is a 20k-timestep probe with about 49% invalid genomes.

| N | measured | USL | simulator (P-equivalent capacity) |
|---|---|---|---|
| 12 | 1.118 | 1.226 | 1.216 |
| 15 | not measured | 1.348 | 1.378 |
| 16 | 1.112 | 1.383 | 1.432 |
| 20 | 1.521 | 1.497 | 1.647 |

At N = 15 the USL and the simulator agree within 2.2%. At N = 20 the simulator exceeds the measurement by 8.3% and the USL by 10.0%. The simulator has no contention, thermal, memory-bandwidth or 120-genome tail effects, so it is expected to be optimistic. This matches the ~0.92 realised-versus-ideal factor implied by the sweep.

The measured N = 12 and N = 16 points fall well below the simulator (N = 16 by 22%). I have not verified why. The finite 120-genome sweep batches on mixed-speed workers (quantisation) is a hypothesis only.

The USL describes an unbarriered independent stream. It says nothing about the generation barrier, which the simulator shows is the dominant loss in serial mode (34% of capacity realised at N = 15).

### 4c. Gustafson's law (a different question: "how much more could we afford", not "how long")

Fixed time budget T: the interleaved wall-clock of the reference cell (pop 15, gen 20, 5 seeds, direct, Walker-v0), **T = 5.13 h** (p5–p95 3.62–6.93), 400 trials.

- **Classical Gustafson:** S = N − α(N−1), with α = 1.85×10⁻³ from 4a. This gives **S = 19.96 homogeneous** and **13.16 P-equivalent** (N_eff = 13.18). The second number is the honest one for this machine.
- **Empirical scaled workload for the reference cell.** N = 1 (one P-core) finishes only **2.2 seeds** in T; five seeds would need 11.6 h. N = 20 finishes 5 by definition, and adding seeds is almost free:

| seeds | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| N = 20 wall-clock (h) | 5.13 | 5.52 | 6.21 | 8.58 | 11.79 |

  Within +25% of T, N = 20 absorbs about **20 seeds**, roughly 9× what N = 1 absorbs.
- **The catch.** That headroom exists only because the direct row is latency-bound at 2% validity. Under the all-valid upper bound the reference cell takes 39.4 h and 10 seeds take 78.6 h, exactly proportional, so the gain falls back to the saturated ~13× (13.16 P-equivalent).

**Tied to the D14 option table** (N = 20 interleaved simulation against N = 1 total work; "fits in T" means the delta ≤ 5.13 h):

| D14 option | N = 20 delta | N = 1 delta | fits in T at N = 20? | `projection.md` delta (h) |
|---|---|---|---|---|
| More seeds 5 → 10, reference cell only | +0.39 h | +11.6 h | yes | +0.86 |
| More seeds 5 → 10, whole campaign | +108.5 h | +1,428 h | no | not tabulated |
| Second grid size (2× cross product, B6's model) | +108.4 h | +1,428 h | no | +105.01 |
| Extra arm `direct_repaired` (direct as proxy) | +1.9 h | +22.9 h | yes | +1.68 |
| Extra arm `nca_compact` (nca as proxy) | +59.2 h | +777 h | no | +57.15 |

Notes:

- The single-cell more-seeds row is not representative of the whole campaign. The whole-campaign row is work-bound and roughly doubles the time.
- The second-grid figure uses B6's 2.0× model. `final_decisions.md` D14 quotes about 2.5× for a 9×9 grid, and neither has been measured.
- `direct_repaired` and `nca_compact` have never been measured. They are costed with their nearest measured arm, as B6 does.
- `larger_timestep_budget` was not asked for. It scales linearly.

## 5. Step 5 — Architecture decision record

`docs/adr/launcher-scheduling.md` (**status: PROPOSED, awaiting sign-off, not a decision**). It records the λ-versus-concurrency finding, the +271.8 h serial-per-run cost, and the recommendation to interleave replicate runs within each D7 task phase. It states that this changes only scheduling and not the (μ+λ) algorithm, and it lists the required `ea.py` propose/evaluate/tell split and the fair-priority requirement. Nothing in it is implemented.

## 6. Limitations

- Validity is at the random-genome rate in every generation (the B6 assumption). This is the largest uncertainty: 109 h against 240 h.
- t_PPO comes from n = 3 observations per (arm, task), measured on quiet P-cores at concurrency 8. The simulator has no contention beyond the P/E ratio, so it is optimistic against the measured sweep by about 8% at N = 20.
- The P/E ratio rests on 3 + 3 evaluations. The 20k-timestep probe gave 2.1440× and the 1M-timestep run 2.3172×.
- The rule_based orchestration overhead excludes its real decode, which is charged from `t_develop` instead.
- The serial mode uses simple greedy placement on 15 workers. A smarter placement, such as running 15 tasks in two waves on the 8 P-cores, was not modelled.
- Nothing here depends on the uncommitted working-tree edits in `evodevo/preflight/`; they were not modified.

## Reproduce

```
python3 scripts/capacity_planning/measure_orchestration_overhead.py results/preflight/20260922T122730Z_4be8402b/report/capacity_planning/orchestration_overhead.json
python3 scripts/capacity_planning/simulate_campaign.py 500
python3 scripts/capacity_planning/laws_crosscheck.py
```

Outputs: `results/preflight/20260922T122730Z_4be8402b/report/capacity_planning/{orchestration_overhead,simulation_results,laws_results}.json`.
