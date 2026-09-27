> Version-controlled copy, made 2026-09-28, of `results/campaign_prep/CAMPAIGN_PREP_P1.md` (a gitignored output directory). Relative paths below are as in the original.

# Campaign prep P1

Date: 2026-09-27. First pass: stopped at Step 2 (Walker-v0 had no D6 knee). Second pass (the "Unblock P1" prompt): D6 resolved by a pre-specified fallback rule (commit `1e79a56`), and Steps 3–5 run.

**Overall status: COMPLETE. D6 is CLEAR, and Steps 3, 4 and 5 are done.**

The t_PPO batch (4d–4f) ran in a third pass, after Tejeshwar explicitly approved it with a 75-minute limit. It took 27.5 min.

No project code or configuration was modified by P1. One exception: `.gitignore` gained `results/campaign_prep/`, which P1 Step 4 explicitly permits.

Commits from this prompt:

| Commit | Contents |
|---|---|
| `294ee55` | ADR approved (first pass) |
| `f2689b1` | Pre-registered analysis plan (alone) |
| `1e79a56` | D6 application plus the Deviation-from-pre-registration entry |

---

## Step 1 — ADR sign-off: DONE

- Before this step, `git log --oneline -- docs/adr/launcher-scheduling.md` was **empty**. The file had never been committed (it was untracked under `docs/adr/`), its Status was `PROPOSED`, and every sign-off box was unticked.
- Change made: ticked `[x] Approved as the design for Step 4 (full launcher)`. The Status line now reads `ACCEPTED — approved by Tejeshwar, 2026-09-27. Nothing described here is implemented anywhere in the repository yet.` The old sentence "a recommendation awaiting sign-off. Not a decision." was dropped because it no longer applies. The factual "nothing implemented" clause stays.
- Commit: **`294ee55` "ADR: interleaved launcher scheduling approved"**, with only this file in it (1 file, +66).
- `git log --oneline -- docs/adr/launcher-scheduling.md` now gives `294ee55 ADR: interleaved launcher scheduling approved`.

## Step 2 — D6 timestep budget from B4: BLOCKING

Sources:
- `results/preflight/20260922T040932Z_4be8402b/stages/learning_curves.json` (evaluator_mode `real`, partial `false`, curve_max_timesteps 2,000,000, eval interval 100,000, 3 bodies per task, concurrency 8)
- `docs/final_decisions.md` §D6
- `evodevo/preflight/stages/b4_learning_curves.py`

### 2a. The rule and its implementation

D6, verbatim (`docs/final_decisions.md` §D6):

> **Selection rule (stated in advance):** the smallest budget at which median fitness improvement per additional 100k steps falls below 1% of the fitness achieved so far, sustained over three consecutive windows.

`select_knee()` (`evodevo/preflight/stages/b4_learning_curves.py:61`), core logic:

```python
sorted_series = sorted(series, key=lambda p: p[0])
windows = []
for (t0, f0), (t1, f1) in zip(sorted_series, sorted_series[1:]):
    step = t1 - t0
    if step <= 0:
        continue
    raw_delta = f1 - f0
    normalized_per_100k = raw_delta / step * 100_000.0
    denom = abs(f0)
    if denom > 1e-12:
        relative = normalized_per_100k / denom
    else:
        relative = 0.0 if abs(normalized_per_100k) < 1e-12 else float("inf")
    windows.append({"window_start": t0, "window_end": t1, "relative_improvement": relative})

knee_timesteps = None
for i in range(n - sustained_windows + 1):          # sustained_windows = 3
    candidate = windows[i : i + sustained_windows]
    if all(w["relative_improvement"] < threshold for w in candidate):   # threshold = 0.01
        knee_timesteps = candidate[0]["window_start"]
        break
```

The input `series` is `_median_series(curves)`: the per-checkpoint median across the task's 3 bodies, aligned by checkpoint index. Two implementation facts matter when reading the results:
- The comparison is **signed**, so a fitness *drop* counts as "below 1%".
- The knee is reported as the **start** of the first qualifying 3-window run.

As a check, I re-ran `select_knee(_median_series(...))` on the stored curves. It reproduced the stored median series and knees exactly for both tasks.

### 2b. Per task

| Task | Knee (rule output) | `selected_budget` | `no_plateau_observed` | Warning |
|---|---|---|---|---|
| UpStepper-v0 | **1,200,000** (knee_found = true, window index 11) | 1,200,000 | false | none |
| Walker-v0 | **none** (knee_found = false) | 2,000,000 (a floor, not a knee) | **true** | *"On Walker-v0, no sustained plateau was observed within curve_max_timesteps=2,000,000 -- the reported budget (2,000,000) is a FLOOR, not a knee: fitness may still be improving past this point. Stated plainly here, not averaged away."* |

**UpStepper-v0: 3 bodies, arms rule_based / rule_based / nca, all 20 checkpoints present, none failed**

| Label | Arm | B3 fitness | Curve summary |
|---|---|---|---|
| low | rule_based | 0.0020 | Flat at about 0 (−0.002 to 0.004) for the whole 2M. It never learns. |
| middle | rule_based | 1.426 | Jumps from 0.14 to 1.39 by 200k, then sits flat at 1.31–1.46 up to 2M (final 1.378). Effectively plateaued from 200k. |
| high | nca | 1.637 | Jumps from 0.01 to 1.44 by 200k, then rises slowly to about 1.67 by 1.3M. After that it is unstable: 2.56 at 1.6M, **−1.72 at 1.7M**, 1.67 at 1.8M, 2.37 at 2M. |

Median series (fitness at 100k … 2M): 0.010, 1.393, 1.391, 1.434, 1.308, 1.229, 1.406, 1.407, 1.421, 1.406, 1.444, 1.459, 1.409, 1.411, 1.346, 1.427, **0.004**, 1.444, 1.432, 1.378. The drop to 0.004 at 1.7M comes from the nca body's −1.72 checkpoint.

Windows below 1% (Y) by window start 100k…1.9M: `.Y.YY.Y.Y..YYY.Y.YY`. The first run of three Ys starts at 1.2M, with values −3.44%, +0.12% and −4.59%.

Caveats a human should see before accepting 1.2M:
- **The knee is fragile.** The window starting at 800k is +1.007%, just 0.007 percentage points over the threshold. At 1.00% or below, windows 700k/800k/900k (+0.07%, ~1%, −1.10%) would have qualified and the knee would have been **700,000**.
- After 200k the median curve is flat to within its own checkpoint noise (1.23–1.46). So the rule is effectively picking the first point where three consecutive noise draws are all non-positive-ish, not a genuine plateau onset.
- The low body is about 0 throughout, so the median of 3 is mostly the middle rule_based body's curve.
- The nca body is still gaining on average: +4.46% per 100k over 1M→2M, but noisy.

**Walker-v0: 3 bodies, arms rule_based / direct / nca, all 20 checkpoints present, none failed**

| Label | Arm | B3 fitness | Curve summary |
|---|---|---|---|
| low | rule_based | −9.2e-14 | Exactly 0 (magnitude below 1e-10) for the whole 2M. The body does not move. |
| middle | direct | 3.162 | 1.00 at 100k, 2.77 at 200k, then a slow, steady climb: 3.01 at 1M, 3.55 at 1.9M, 3.44 at 2M. Mean +1.40% per 100k over 1M→2M. |
| high | nca | 9.107 | About 0 until 200k, 6.28 at 300k, then a steady climb: 9.14 at 1M, 10.57 at 2M. Mean +1.56% per 100k over 1M→2M. |

Because the low body is about 0 and the high body is always the maximum, the median series equals the **direct body's curve** from 300k onward. Windows below 1% by start 100k…1.9M: `...Y..Y.Y.Y..Y.Y..Y`. There are never even two consecutive Ys, so no qualifying three-window run exists. Both bodies that do learn are still improving at more than 1% per 100k on average across the second million steps, so this is a real no-plateau result, not a threshold artefact.

### 2c. Comparison with 1,000,000

Every B3 cost measurement and every capacity estimate assumed 1,000,000 timesteps.

| Task | D6 output | vs 1M | Implied t_PPO scale factor |
|---|---|---|---|
| UpStepper-v0 | 1,200,000 | **higher** | **×1.2** (would be ×0.7 if the 800k window had fallen 0.007 pp lower) |
| Walker-v0 | no knee; floor 2,000,000 | **higher, by at least** | **≥ ×2.0** (a lower bound, since the rule says fitness is still rising at 2M) |

**Assumption, stated explicitly:** these factors assume t_PPO scales roughly linearly with training timesteps, with a fixed per-evaluation overhead (env setup, eval_episodes rollouts) that is negligible or also scales. Nobody measured this here. The fixed overhead would make the true factor slightly *below* the linear figure.

Implication, not a decision: at the Walker floor of 2M, PPO cost on Walker-v0 at least doubles. Every Walker capacity/projection number in B6 and the capacity-planning report is then under-stated by about 2× in its PPO-dominated part.

### 2d. Verdict

**First pass: BLOCKING.** Walker-v0 had `knee_found: false`, `no_plateau_observed: true` and the "FLOOR, not a knee" warning.

**Now: CLEAR.** D6 was resolved per task in `docs/decisions.md` (commit `1e79a56`):

| Task | `ppo_timesteps` | Basis |
|---|---|---|
| UpStepper-v0 | **1,200,000** | The registered rule's knee, accepted as produced. The 800k window missed by 0.007 points; the knee is not overridden. |
| Walker-v0 | **1,000,000** | A fallback rule fixed in writing before its inputs were computed (Step 2e). Recorded as a "Deviation from pre-registration". |

The implied t_PPO factors from 2c become ×1.2 for UpStepper-v0 and ×1.0 for Walker-v0 under the linear assumption. The measured factor for UpStepper-v0 is 1.182 (Step 4f).

### 2e. Walker-v0 fallback rule: full table

The rule, verbatim as given:
- Candidates: T in {1.0M, 1.2M, 1.5M, 2.0M}.
- Qualifying bodies: fitness at 2M is at least 10% of the best body's 2M fitness.
- Choose the smallest T such that (i) every qualifying body reaches at least 85% of its own 2M fitness at T, and (ii) the fitness ordering of ALL bodies at T equals their ordering at 2M.
- Otherwise use 2.0M, disclosed as a floor.

Source: `learning_curves.json`, `tasks.Walker-v0.bodies[*].curve`.

Qualifying-body threshold: 10% of the best 2M fitness (nca, 10.5683) is 1.0568.
- direct (3.4364): qualifies.
- nca (10.5683): qualifies.
- rule_based (−8.5e-13): does not qualify.

Ordering at 2M: nca > direct > rule_based.

| T | rule_based (fitness; fraction of 2M) | direct | nca | Ordering at T | (i) | (ii) |
|---|---|---|---|---|---|---|
| 1.0M | −1.33e-12; n/a (numerical zero) | 3.0137; **0.8770** | 9.1405; **0.8649** | nca > direct > rule_based | ✅ | ✅ |
| 1.2M | 1.00e-12; n/a | 3.0827; 0.8971 | 8.8206; **0.8346** | nca > direct > rule_based | ❌ | ✅ |
| 1.5M | −5.24e-14; n/a | 3.3178; 0.9655 | 9.0821; 0.8594 | nca > direct > rule_based | ✅ | ✅ |
| 2.0M | −8.53e-13; 1.0 | 3.4364; 1.0 | 10.5683; 1.0 | nca > direct > rule_based | ✅ | ✅ |

**Chosen T = 1,000,000.** This is not monotone: 1.2M fails (i), but the rule takes the smallest passing T. No threshold was changed.

Condition (ii) is trivially stable here. The rule_based body sits at about 1e-12 and the other two are far apart, so the ordering never changes.

Disclosure, also recorded in `decisions.md`: the raw 1M and 2M values had already appeared in this report's Step 2b before the rule was written.

---


## Step 3 — rule_based_monotone scope facts (read-only)

### 3a. The 2026-09-16 decision

The entry is the `docs/decisions.md` row "Monotone vs unrestricted: an explicit decision". It is quoted in full, as its Chosen and Rationale cells, in Appendix A at the end of this report.

The decision itself is **DEFER**: run both `unrestricted` (PRIMARY) and `monotone` as co-primary configurations in the PPO matrix.

Its reason for rejecting "keep as ablation" (the status quo) and "adopt outright", quoted:

> "the ONE genuine cost (reduced mutation-neutrality/search-efficiency) is specifically a claim about EA SEARCH DYNAMICS OVER GENERATIONS -- exactly the kind of quantity this project's entire fitness-blind structural workstream has consistently declined to assert without measuring [...] Adopting monotone outright would risk exactly the failure mode this project's whole discipline exists to prevent: inferring a performance-relevant conclusion from a structural snapshot. Deferring costs only 2x compute for this one arm in the primary matrix and preserves both candidates' evidence for the actual PPO comparison to adjudicate."

The weighing it records: monotone beats unrestricted on 4 of 5 core structural metrics and adds a hard termination guarantee. Against that, the only cost is more neutral mutations.

### 3b. What "monotone" restricts, in plain words

Source: `results/rule_based/MONOTONE_ABLATION.md`.

**The restriction.** The five materials get a fixed rank: EMPTY 0 < SOFT 1 < RIGID 2 < H_ACT 3 < V_ACT 4. In `rule_based`, a rule may rewrite a cell to any material. In `rule_based_monotone`, a rule that rewrites a cell *itself* (SET_* or COPY_NEIGHBOR) is silently skipped if it would lower that cell's rank. GROW_DIRECTIONAL is unaffected, because it only ever fills an EMPTY cell.

In effect, development can only add or "upgrade" material, never remove or downgrade it. Everything else is identical: genome, rule language, interpreter, conflict resolution and readout.

**The consequence.** The rank sum over 25 cells can only rise and is capped at 100. So development provably terminates and cannot cycle (Proposition R4):
- 0 of 5,710 rank-sum violations.
- 0 of 2,000 genomes with period > 1.
- Maximum termination step 7.

**Structural cost table** (N = 2000, T_primary = 100):

| Metric | unrestricted | monotone | Delta |
|---|---|---|---|
| valid_rate | 0.5470 | 0.6530 | +0.1060 |
| degenerate_fraction | 0.2380 | 0.1165 | −0.1215 |
| unique_valid_body_fraction | 0.3315 | 0.3865 | +0.0550 |
| material_entropy_bits | 2.3043 | 2.3155 | +0.0112 |
| uniform_body_fraction | 0.2380 | 0.1165 | −0.1215 |
| never_settled_fraction | 0.1080 | 0.0000 | −0.1080 |
| mean_settle_step | 2.1754 | 1.8550 | −0.3204 |
| mean_hamming_displacement | 5.8280 | 5.1735 | −0.6545 |
| fraction_zero_displacement | 0.5230 | 0.5480 | +0.0250 |

Monotone is better on validity, degeneracy, diversity and settling. Its one cost is more mutations that change nothing (zero displacement rises from 52.3% to 54.8%).

### 3c. Config agreement

- `evodevo/experiments/configs/benchmark.yaml` lists `encoders: [direct, nca, rule_based, {encoder: rule_based, variant: monotone}]`.
- `resolved_config.py` defines `PPO_MATRIX_CANDIDATES = (PRIMARY, MONOTONE_ABLATION)`.
- **Programmatic check:** the live benchmark path (`benchmark._resolved_arm_kwargs`, then `get_encoder`) builds `rule_based` and `rule_based_monotone` encoders. Their `describe()` output matches every one of the 7 resolved fields of `PRIMARY` and `MONOTONE_ABLATION` respectively: **0 mismatches**.
- `git log -1` for each:
  - `benchmark.yaml`: `c4d3f5f` (2026-09-17), "Introduce fast_dev/primary_matrix config profiles; fix NCA hidden_channels wiring gap"
  - `resolved_config.py`: `217326e` (2026-09-17), "Commit rule_based order-independence fixes, rule-language spec, resolved config"
- Note for Prompt 2: `benchmark.yaml` currently has `profile: fast_dev` and `tasks: [Walker-v0]` only.

## Step 4 — rule_based_monotone measurements

Outputs are in `results/campaign_prep/monotone_cost/`:

| File | Contents |
|---|---|
| `cpu_measurements.json` | Raw 4a–4c data |
| `validity_analysis.json` | 4a/4b analysis |
| `sanity/` | 4a output files |
| `scripts/campaign_prep/measure_cpu.py`, `analyze_validity.py` | The scripts (moved from this directory on 2026-09-28; committed) |
| `scripts/campaign_prep/measure_tppo.py` | The 4d/4e script (moved and committed 2026-09-28). Raw results: `tppo_measurements.json`, `tppo_results.jsonl`. |

Expected duration, stated before starting: about 45 min (about 10 min CPU work plus about 30 min of PPO).

**What actually happened:** the CPU-only part took **7,299 s (about 2.0 h)**. The cause was a bug in my measurement script. It pinned the parent process to CPU 0 before creating the multiprocessing Pool, and the Pool's 26 workers inherited that affinity, so all the work ran on one core. At the correct parallelism it would have taken about 10 min. **The measured values are unaffected:** validity counts don't depend on speed, and 4c ran single-process before the Pool existed. But Step 4's wall time exceeded the 90-minute limit, so the PPO batch was held until Tejeshwar approved it (75-minute limit). It then ran in 27.5 min.

### 4a. Random-genome validity (n = 2000)

Method: `inspect_encoder()`'s procedure (seed 0, n = 2000) with the campaign-resolved kwargs from the live benchmark path. These are:
- dev_steps 100
- center_seed
- mode_of_window, W = 6
- calibrated mutation_rate 0.0779
- plus `action_monotonicity=monotone` for monotone

rule_based was re-run under identical conditions for comparison.

Why not compare with the stored sanity file: `results/inspection/rule_based/rule_based_sanity.json` is superseded. It used pre-redesign defaults and the CLI's dev_steps = 10 / full_nonsymmetric factors, so it isn't comparable.

| Arm | Valid / n | Rate | Wilson 95% CI | Degenerate fraction | Invalid reasons |
|---|---|---|---|---|---|
| **rule_based_monotone** | 1306 / 2000 | **0.6530** | [0.6319, 0.6736] | 0.1165 | no_actuator 678, too_few_voxels 47, disconnected 15 |
| rule_based (same conditions) | 1150 / 2000 | 0.5750 | [0.5532, 0.5965] | 0.2320 | no_actuator 831, disconnected 174, too_few_voxels 160 |

Monotone's 0.6530 matches `MONOTONE_ABLATION.md` exactly. rule_based's 0.5750 differs from that report's 0.5470. The likely cause is the readout: this run uses the campaign's mode_of_window (W = 6), which matters only for genomes that cycle, and monotone genomes never cycle. This explanation was not verified further.

### 4b. Validity by generation (round-3 no-PPO proxy)

Method: `scripts/capacity_planning/measure_validity_current_calibration.run_one`, imported unchanged:
- real `run_ea`, population 15, 20 generations, 20 seeds
- random U[0,1) proxy fitness for valid genomes, sentinel for invalid ones

Consistency check: my rule_based re-run reproduced the stored round-3 counts **exactly** (all 20 seeds, both proxies), so the monotone curve is directly comparable. n = 300 genomes per generation (15 × 20 seeds). The other arms' curves come from `results/preflight/20260922T122730Z_4be8402b/report/capacity_planning/validity_round3_analysis.json`.

| Generation | monotone | Wilson 95% | rule_based | nca | direct |
|---|---|---|---|---|---|
| 0 | 0.620 | [0.564, 0.673] | 0.547 | 0.650 | 0.020 |
| 1 | 0.600 | [0.547, 0.657] | 0.570 | 0.630 | 0.020 |
| 2 | 0.940 | [0.903, 0.959] | 0.890 | 0.840 | 0.020 |
| 5 | 0.950 | [0.915, 0.967] | 0.940 | 0.880 | 0.090 |
| 10 | 0.960 | [0.936, 0.979] | 0.960 | 0.920 | 0.250 |
| 15 | 0.970 | [0.944, 0.984] | 0.950 | 0.930 | 0.340 |
| 19 | 0.960 | [0.931, 0.977] | 0.930 | 0.930 | 0.420 |
| **Offspring, generations 1–19 pooled** | **0.933** (5316 / 5700) | [0.926, 0.939] | 0.921 | 0.892 | 0.229 |
| Last 5 generations | 0.955 | | 0.937 | 0.937 | 0.381 |

All 20 generations with CIs are in `validity_analysis.json`. The fixed-proxy curve is almost identical (0.59–0.97).

Monotone's curve sits at or slightly above rule_based's. For capacity planning the two are effectively the same. Both approach about 95% validity from generation 2 on, so almost every monotone offspring costs a full PPO training.

### 4c. t_develop and t_sentinel (n = 100 each)

Method: B3's own `_measure_develop` and `_measure_sentinel`, seeded as B3 would seed arm index 3. A single process pinned to P-core 0, with the machine otherwise idle. rule_based was re-measured on B3's own seeds for comparison.

| Arm | Timing | Mean | Median | SD | p99 | Min | Max | n |
|---|---|---|---|---|---|---|---|---|
| monotone | t_develop | 0.2483 s | 0.2326 | 0.1031 | 0.5254 | 0.0961 | 0.5637 | 100 |
| monotone | t_sentinel | 0.2248 s | 0.2159 | 0.1016 | 0.4897 | 0.0961 | 0.5073 | 100 (318 draws) |
| rule_based (re-measured) | t_develop | 0.2438 s | 0.2149 | 0.1083 | 0.4998 | 0.0852 | 0.5324 | 100 |
| rule_based (re-measured) | t_sentinel | 0.2337 s | 0.2136 | 0.0886 | 0.4180 | 0.0856 | 0.4232 | 100 (253 draws) |
| rule_based (B3, for reference) | t_develop | 0.2407 s | | | | | | 100 |
| rule_based (B3, for reference) | t_sentinel | 0.2297 s | | | 0.4115 | | | 100 |

The rule_based re-measurement reproduces B3 to within 2%. Monotone's development cost equals rule_based's, since the interpreter always runs all 100 steps. Both are negligible next to t_PPO (about 1,300 s).

### Pinning bug: root cause, fix, and effect on recorded values

**Root cause.** Linux CPU affinity is inherited by child processes. `measure_cpu.py` pinned its *parent* process to CPU 0 (intended, for the 4c timings) and then created a `multiprocessing.Pool` in that same process. All 26 workers inherited the single-CPU mask, so 4a and 4b ran time-sliced on one core.

**Fix.**
- `measure_cpu.py`: the pinned timing measurement now runs in its own child process.
- `measure_tppo.py`: no parent pinning at all. The pool's workers pin themselves in `evodevo.parallel.worker_pool._worker_initializer`, which applies `sched_setaffinity` to each worker's own slot. That was already true, so the t_PPO batch was never exposed to this bug.

**A second launch bug, found and fixed before any training ran.** The pool uses the `spawn` start method, which re-imports the main script in every worker. My script imported numpy at module level, and the pool's strict thread-count guard raised `ThreadingConfiguredTooLateError`. All heavy imports moved inside functions. That first launch trained nothing and produced no data.

**Effect on recorded values: none.**
- 4c's t_develop and t_sentinel ran in the parent *before* the Pool was created. The code order and the log line `timings done 196.7` confirm it. They ran single-process on the intended CPU 0 with no contention, so they were not re-measured. They also reproduce B3's rule_based values to within 2%.
- 4a and 4b are deterministic counts. The rule_based rerun reproduced the stored round-3 counts exactly.

### 4d / 4e. t_PPO: monotone, plus the rule_based anchor

Setup:
- One `EvaluationPool` of 8 workers, mode "real", eval_episodes = 1 (`CAMPAIGN_EVAL_EPISODES`).
- Walker-v0 at **1,000,000** timesteps (the Step B budget, reported before starting); UpStepper-v0 at **1,200,000**.
- **Pinning verified live**, about 76 s into the workers' life, with `taskset -cp` and `ps -o psr`. The 8 workers had masks {0}, {2}, {4}, {6}, {8}, {10}, {12}, {14}: 8 distinct P-core primaries, no E-core, no hyperthread sibling, each running at about 100% CPU. The ninth child (mask 0–27, 0% CPU) is multiprocessing's resource tracker. Every worker's returned `AffinityResult` also reports `applied=True` on its own P-core.
- Monotone genomes: the first 3 valid draws from B3's valid-draw range for arm index 3 (seeds 303000000–303000002; 3 draws needed).
- Anchor: B3's own rule_based genome 203000001. Its body hash was asserted equal to B3's record.

Outcome:
- Batch wall time **1,651.6 s**. 8 of 8 succeeded, with 0 failures, timeouts or retries.
- Peak RSS about 400 MB per worker. P-core clocks averaged about 4.9 GHz (111 samples).
- Raw data: `monotone_cost/tppo_measurements.json` and `tppo_results.jsonl`.

| Task (budget) | Arm | Genome | t_PPO (s) | Fitness | CPU |
|---|---|---|---|---|---|
| Walker-v0 (1.0M) | monotone | 303000000 | 1387.6 | 5.559 | 8 |
| Walker-v0 (1.0M) | monotone | 303000001 | 1427.7 | 8.843 | 6 |
| Walker-v0 (1.0M) | monotone | 303000002 | 1414.1 | 10.561 | 4 |
| Walker-v0 (1.0M) | rule_based anchor | 203000001 | 1402.6 | **2.632** (B3: 2.632, reproduced exactly) | 10 |
| UpStepper-v0 (1.2M) | monotone | 303000000 | 1594.7 | 1.428 | 2 |
| UpStepper-v0 (1.2M) | monotone | 303000001 | 1650.4 | 1.426 | 0 |
| UpStepper-v0 (1.2M) | monotone | 303000002 | 1626.5 | 1.431 | 14 |
| UpStepper-v0 (1.2M) | rule_based anchor | 203000001 | 1614.1 | 1.173 | 12 |

**Monotone t_PPO (this batch's conditions):**

| Task (budget) | Mean | SD | Min | Max | n |
|---|---|---|---|---|---|
| Walker-v0 (1.0M) | **1409.8 s** | 20.4 | 1387.6 | 1427.7 | 3 |
| UpStepper-v0 (1.2M) | **1623.9 s** | 28.1 | 1594.7 | 1650.4 | 3 |

**Anchor ratio, monotone ÷ rule_based under the same conditions: 1.005 on Walker-v0 and 1.006 on UpStepper-v0.** Monotone costs the same per PPO training as rule_based.

### 4f. t_PPO timestep-scaling check (rule_based anchor vs. B3)

B3 ran 3 trainings concurrently; this batch ran 8. The same genome, task and budget on Walker-v0 therefore isolates the effect of measurement conditions.

| Quantity | Value |
|---|---|
| Walker-v0 at 1M: anchor 1402.6 s vs. B3 same genome 1323.7 s | **condition factor c = 1.0596** (about 6% slower at 8-way concurrency) |
| B3 rule_based arm mean at 1M, as linear extrapolation to Walker's chosen budget (×1.0) | 1283.6 s. Measured 1402.6 s = 1.093×, of which 1.060 is conditions and the rest genome-to-genome spread (this genome is B3's slower one). |
| UpStepper-v0 at 1.2M: anchor 1614.1 s vs. B3 same genome at 1M 1288.3 s | raw ratio 1.2529 |
| **Measured scaling, 1M → 1.2M** (raw ratio ÷ c) | **1.182** (pure linear assumption: 1.200) |
| Linear extrapolation from B3's same-genome 1M value (×1.2) | 1546.0 s at B3 conditions. Measured, condition-corrected: 1614.1 ÷ 1.0596 = 1523.3 s (−1.5%). |
| Linear extrapolation from B3's rule_based arm mean (×1.2) | 1485.6 s. Measured, condition-corrected, same genome: 1523.3 s. The difference is genome spread, not scaling. |

**Conclusion.** t_PPO scales very slightly sublinearly from 1M to 1.2M timesteps (1.182× against a linear 1.200×). The linear assumption is 1.5% conservative.

Caveat: this rests on **one genome** per task, so it is a consistency check, not a precise estimate. The −1.5% is within the anchor's own run-to-run variability.

Prompt 2 uses the measured 1.182 for UpStepper-v0 and ×1.0 for Walker-v0, whose budget equals B3's 1M.

## Step 5 — D16 aggregation check from B5 raw data

Source: `results/preflight/20260922T040932Z_4be8402b/stages/evaluation_noise.json`. The `20260922T122730Z` copy is byte-identical. Script: `scripts/campaign_prep/d16_aggregation.py`. Output: `results/campaign_prep/d16/d16_aggregation.json`.

Definition, exactly B5's:
- Take each body's within-body sample SD of the estimator.
- Pool across bodies with B5's own `_pooled_sd` ((n−1)-weighted).
- Divide by B3's between-body SD: the SD of B3's 9 pooled per-task fitness values, 0.5772 on UpStepper-v0 and 3.1583 on Walker-v0.

For K = 2, the estimator's values are the 10 unordered pairs of each body's 5 values.

**The K = 1 row reproduces B5's reported 0.6658 and 0.1651 exactly.**

| Task | Single (K = 1) | Mean of 2 | Max of 2 |
|---|---|---|---|
| UpStepper-v0 | 0.6658 | 0.3844 | **0.0611** |
| Walker-v0 | 0.1651 | **0.0953** | 0.1307 |

Per body, within-body SD for single / mean-of-2 / max-of-2, and collapse events (a value below 50% of that body's median):

| Task | Body | Median | SD: single / mean-2 / max-2 | Collapses |
|---|---|---|---|---|
| UpStepper-v0 | low, rule_based | 0.0273 | 0.0133 / 0.0077 / 0.0075 | 2 (0.0027, 0.0038). Near-zero noise, not meaningful collapses. |
| UpStepper-v0 | middle, rule_based | 1.408 | 0.0283 / 0.0164 / 0.0184 | 0 |
| UpStepper-v0 | high, nca | 1.627 | 0.6649 / 0.3839 / 0.0577 | **1 (0.1396)** |
| Walker-v0 | low, rule_based | about 0 (−1e-16) | 0 / 0 / 0 | n/a, median about 0 |
| Walker-v0 | middle, direct | 3.094 | 0.7320 / 0.4226 / 0.6161 | 0 |
| Walker-v0 | high, nca | 8.243 | 0.5288 / 0.3053 / 0.3631 | 0 |

**On UpStepper-v0, max-of-2 gives the lowest noise:** 0.061 against 0.666 for a single evaluation (about 10.9× lower) and 0.384 for mean-of-2 (about 6.3× lower).

This comes almost entirely from **one** genuine collapse event: the nca body's 0.14 among values of about 1.6. Max-of-2 discards a lone collapse whenever the other seed didn't collapse. The middle body shows no collapse, and there max-of-2 is no better than mean-of-2 (0.018 vs 0.016).

On Walker-v0, which has no collapses, mean-of-2 is lowest and max-of-2 is in between.

**Sample-size limitation:**
- Only 3 bodies × 5 repeats per task.
- The UpStepper conclusion rests on a single collapse in 15 runs.
- The 10 pairs per body are not independent (each value appears in 4 pairs).
- Max-of-2 also estimates a different, upward-biased quantity from the single value, so the fractions compare noise, not accuracy.

## Acceptance status (P1, after the unblock pass)

1. ADR approved and committed: **yes, `294ee55`**.
2. D6 per task with explicit verdict: **CLEAR**. UpStepper-v0 1,200,000 (registered rule); Walker-v0 1,000,000 (fallback rule, commit `1e79a56`).
3. monotone restriction explained and its 2026-09-16 rationale quoted: **yes**.
4. monotone measurements:
   - Validity rate, validity-by-generation, t_sentinel and t_develop: **yes**.
   - t_PPO on both tasks (n = 3 each: Walker-v0 1409.8 ± 20.4 s, UpStepper-v0 1623.9 ± 28.1 s): **yes**.
   - Anchor ratio (1.005 / 1.006) and the measured scaling (1.182 vs. linear 1.200): **yes**.
5. D16 aggregation table for both tasks: **yes**.

## Appendix A — 2026-09-16 "Monotone vs unrestricted" entry (Chosen and Rationale cells, verbatim)

> **Options:** **ADOPT monotone as PRIMARY** vs. **KEEP unrestricted as PRIMARY, monotone as ablation** (the status quo) vs. **DEFER to the PPO matrix: run BOTH as co-primary, first-class configurations, let task performance decide**. Argued entirely from `results/rule_based/MONOTONE_ABLATION.md`'s already-collected structural evidence -- no PPO/fitness evaluation was run to make this call, per this task's own explicit prohibition.
>
> **Chosen + rationale:** **What "reduced mutation displacement sensitivity" actually predicts, checked rather than assumed**: `fraction_zero_displacement` RISES under monotone (0.5230 -> 0.5480, +0.0250, a ~4.8% relative increase) -- this is evidence for "more neutral mutations" specifically (a materially larger share of (parent, mutated-child) pairs decode to a LITERALLY IDENTICAL body), not merely "smaller but still non-zero displacement"; `mean_hamming_displacement` also drops (5.8280 -> 5.1735, -0.6545, ~11.2% relative), but part of that drop is mechanically driven by the extra zero-displacement mutations pulling the mean down, and this report's own available data (mean and zero-fraction only, no median-among-nonzero) cannot fully separate "more zeros" from "individually smaller nonzero jumps" -- both plausible, not claiming more than the data supports. A genuine, non-cosmetic concern for EA search efficiency either way: this project's own mutation-rate calibration (`mutation_calibration.py`, P5-style) targets a fixed displacement band per arm, and a shift in the neutral-mutation fraction is exactly the kind of change that calibration is sensitive to. **Weighed against monotone's structural dominance**: monotone beats unrestricted on 4/5 core structural metrics (valid_rate +0.1060, degenerate_fraction -0.1215, unique_valid_body_fraction +0.0550, uniform_body_fraction -0.1215, all N=2000/T_primary=100) PLUS a hard termination guarantee (Proposition R4: 0 Phi-decrease violations, 0 period>1 cycles, max termination step 7 <= bound 100) that categorically eliminates the cycling-population/readout-window complexity this project has spent five separate rounds (N14 SETTLING_AUDIT.md, N15/N16 readout-policy and tie-break work, R1's exact-cycle characterisation, R5/this-task's readiness audit) diagnosing and patching for the unrestricted arm. **DECISION: DEFER** -- run BOTH `unrestricted` (unchanged PRIMARY) and `monotone` as co-primary configurations in the PPO matrix, rather than picking one now. **Why defer rather than adopt outright, given the lopsided structural evidence**: the ONE genuine cost (reduced mutation-neutrality/search-efficiency) is specifically a claim about EA SEARCH DYNAMICS OVER GENERATIONS -- exactly the kind of quantity this project's entire fitness-blind structural workstream has consistently declined to assert without measuring (no structural proxy anywhere in this project has been treated as a substitute for an actual task-performance result). Adopting monotone outright would risk exactly the failure mode this project's whole discipline exists to prevent: inferring a performance-relevant conclusion from a structural snapshot. Deferring costs only 2x compute for this one arm in the primary matrix and preserves both candidates' evidence for the actual PPO comparison to adjudicate. **Mechanism**: `evodevo/encoders/rule_based/resolved_config.py::PRIMARY.action_monotonicity` stays `"unrestricted"` (RuleBasedEncoder's own class default is UNCHANGED, minimising blast radius); `MONOTONE_ABLATION` is elevated in status (not renamed, for backward compatibility with existing citations) from diagnostic-only to a co-primary PPO-matrix candidate; a new `PPO_MATRIX_CANDIDATES = (PRIMARY, MONOTONE_ABLATION)` tuple in `resolved_config.py` is the single source of truth for which rule_based dynamics configurations the primary comparison includes, consumed by `evodevo/experiments/benchmark.py`'s new `rule_based_monotone` arm (see the "Wire the resolved readout policy into the primary benchmark path" row below for the mechanics of how per-arm encoder_kwargs now reach real runs). No PPO/fitness evaluation was run to reach this decision; R2/R3's own conflict-resolution/position-predicate decisions were not reopened; NCA's encoder/configuration was not touched.
