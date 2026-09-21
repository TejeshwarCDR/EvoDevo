# final_decisions.md

**Single source of truth for the EvoDevo experimental campaign.** Date: 2026-09-19 · Status: FINALISED pending pre-flight measurements (marked ⏳ where so) Referenced by: implementation, experiment execution, analysis, and paper writing.

Each decision is written to be read aloud to a supervisor or lifted into a paper's methods section. Every one states: what we decided, what the alternatives were, why we chose as we did, and what the risk is.

---

# PART A — SCIENTIFIC FRAMING

## D1. The hypothesis (CORRECTED — this replaces the earlier informal framing)

**The earlier framing was:** *"Developmental encoding performs better on complex tasks because it can develop properly and adapt to the task."*

**This is mechanistically false for our architecture and must not be used.** In our pipeline the order is: genome → development (NCA or rule interpreter) → body → PPO trains a controller → fitness. **Development completes before the task is ever encountered.** The NCA never observes the terrain; the rule interpreter never knows which task is being run. Development therefore cannot adapt to the task.

**What actually happens:** development imposes a *structural prior* on which morphologies the evolutionary search can reach. Growth from a seed with local rules makes connected, regular bodies common and disconnected fragments rare. Direct encoding, sampling each voxel independently, produces a valid body only \~2.25% of the time. Adaptation to the task is performed by *evolution* (selection across generations), not by development.

**FINAL HYPOTHESIS (pre-registered):**

> **H1.** Developmental encodings (NCA, rule-based) impose a structural prior over the morphology space. Where task performance depends on structured morphology, this prior improves search efficiency under a fixed evaluation budget relative to direct encoding. Where nearly any locomoting body suffices, the prior confers little or no advantage.

**Pre-registered predictions:**

- P1: On Walker-v0 (flat terrain, easy), the advantage of developmental over direct encoding is small or absent.
- P2: On UpStepper-v0 (stairs, medium), the advantage is larger than on Walker-v0.
- P3: The advantage shrinks or vanishes when measured against *valid* evaluations rather than *total* evaluations — because much of it is attributable to the validity prior rather than to search dynamics. (This is why P6's dual-counter reporting exists.)

**What would disconfirm H1:** no difference between arms on either task; or a difference on Walker but not UpStepper; or an advantage that disappears entirely under valid-evaluation normalisation and under the repaired-direct baseline.

**Why pre-registering disconfirmation matters:** a paper that states in advance what would refute its hypothesis, and then reports honestly, is far more credible than one that appears to have searched for a supporting result. A null result, reported well alongside our capacity analysis, is publishable. A result that looks fitted to a desired conclusion is not.

**Risk:** H1 may be false at 5×5. See D3.

---

## D2. Second task: **UpStepper-v0**

**Decision: UpStepper-v0.** Walker-v0 remains the easy/control task.

**Why not a "hard" task (GapJumper-v0, PlatformJumper-v0, Traverser-v0, ObstacleTraverser-v1):** EvoGym's own benchmark paper reports that all tested algorithms fail to find robots that succeed in its hardest environments. If every arm scores near zero, we cannot distinguish encodings at all — an expensive, uninterpretable **floor effect**. This is the single biggest task-selection risk and it is why "pick the hardest task" is the wrong instinct.

**Why not a manipulation task (Carrier, Thrower, Catcher, Lifter):** these change the observation and action interface and introduce object dynamics. Our PPO configuration, episode length, and fitness scale were validated for locomotion. Switching task *family* adds a confound: any difference between Walker and task 2 becomes partly attributable to the controller problem changing, not to morphology. We want the morphology comparison isolated.

**Why UpStepper-v0 specifically:**

1. **Officially rated medium** by EvoGym — hard enough to discriminate between encodings, not so hard that everything fails.
2. **Requires structured morphology.** Climbing a step demands a body that can lift and advance part of itself — you cannot climb by wiggling a symmetric blob. This is precisely the condition H1 predicts developmental priors should help with.
3. **Same interface as Walker-v0.** Locomotion observation/action space, same episode structure. PPO hyperparameters transfer. The *only* thing changing between our two tasks is terrain structure, which is exactly the variable we want.
4. **Official benchmark task.** Our numbers are directly comparable to published EvoGym baselines, which strengthens the paper.

**Fallback if UpStepper proves degenerate in pre-flight** (e.g. all arms at floor or all at ceiling): ObstacleTraverser-v0, also officially medium, "walks across terrain that gets increasingly more bumpy." Decide this from pre-flight data, not mid-run.

**Sentence for the paper:** *"We select Walker-v0 (flat terrain, rated easy) and UpStepper-v0 (stairs, rated medium) from the EvoGym benchmark. Both are locomotion tasks sharing an observation and action interface, isolating terrain structure as the manipulated variable. We deliberately avoid tasks rated hard, since the benchmark's authors report that no tested algorithm succeeds on them, which would produce a floor effect incapable of discriminating between encodings."*

---

## D3. Grid size: run 5×5, and report the capacity analysis regardless ⏳

**The problem, stated plainly:** the theoretical case for developmental encoding is compression — a small genome specifying a large structure. Our measured genome sizes are: direct = 58 bits, rule-based = 21–111 bits, **NCA = 16,032 bits**. On a 25-voxel grid the NCA genome is \~276× *larger* than the direct genome it is supposed to compress. At this scale there is almost nothing to compress, and the mechanism under test has little room to operate.

**Decision:**

- **Primary matrix runs at 5×5** (our validated configuration; changing it now would invalidate the entire calibration chain).
- **The capacity-vs-grid-size analysis is produced regardless**, fitness-blind and free: a table and figure showing direct's genome growing as O(n²) while NCA's stays constant and rule-based's stays near-constant across 5×5 / 7×7 / 9×9. This makes the scaling argument concrete without running a single extra PPO step.
- **A PPO run at a second grid size is the highest-value optional addition** if pre-flight shows headroom. ⏳ Decide at the scoping gate (D14).

**Why this matters for the paper:** it converts a potential weakness ("you tested at a scale where your mechanism can't work") into a stated, quantified limitation with a concrete future direction — which is how limitations are supposed to be handled.

---

# PART B — EXECUTION ORDER AND COMPUTE

## D4. Order of work: partial parallelisation → pre-flight → full launcher

**Your question:** pre-flight first, or parallelise first?

**Decision: neither, exactly. Do this in three steps.**

1. **Build the thread-pinning foundation and inner worker pool first** (\~1 day).
2. **Run pre-flight under realistic concurrent load.**
3. **Finish the full launcher** using pre-flight's numbers.

**Why not pre-flight first:** timing one evaluation in isolation and multiplying by core count **will give you a wrong number**. PyTorch and NumPy default to using every available core for internal threading. Run 20 workers on a 20-core machine, each also trying to use 20 cores, and you get severe contention — potentially 5–10× slower than predicted — and it fails *silently*: nothing crashes, everything is just slow. A serial measurement extrapolated to full concurrency could underestimate real cost by a large factor, which is exactly the number you don't want to hand your professor.

**Why not full parallelisation first:** the launcher's core sizing (`concurrent_cells × workers_per_cell`) depends on measurements pre-flight hasn't produced yet. Building it on guessed numbers means rebuilding it.

**Why this order is defensible to a supervisor:** *"We built the minimum infrastructure needed to measure honestly, measured under the conditions the real run will actually experience, then sized the full system from real data."*

---

## D5. What pre-flight must measure — the complete list

This is the deliverable you show your professor. Organised by what each number is for.

### B1. Machine profile

| **MeasurementWhy**                                    |                                                         |
| ----------------------------------------------------- | ------------------------------------------------------- |
| Physical vs logical cores (20 / 28 on this i7-14700K) | Worker sizing; hyperthreads ≠ 2× throughput             |
| P-core vs E-core split (8 P + 12 E)                   | Heterogeneous cores → optimum is not obviously 20 or 28 |
| RAM, RAM per worker under load                        | Confirms memory isn't the real ceiling                  |
| Disk free, measured write rate                        | Multi-day logging/checkpoint volume                     |
| Default `torch.get_num_threads()` unconfigured        | Documents the oversubscription risk being fixed         |

### B2. Throughput vs concurrency (the sweep)

Run at reduced timesteps (cheap) purely to find the shape of the curve:

| **MeasurementWhy**                                      |                                              |
| ------------------------------------------------------- | -------------------------------------------- |
| Genomes/minute at concurrency ∈ {8, 12, 16, 20, 24, 28} | Finds the empirical sweet spot on *this* CPU |
| Per-genome latency at each level                        | Shows where contention starts                |
| CPU utilisation, memory bandwidth saturation            | Explains *why* the curve bends               |
| Point of diminishing returns                            | The number that sizes the launcher           |

**Do not assume 20 or 28 is optimal.** E-cores are slower per thread; hyperthread siblings share execution units. Let the measurement decide.

### B3. Real per-evaluation cost (expensive, bounded)

At the sweet-spot concurrency, full-length runs:

| **MeasurementWhy**                                       |                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| `t_PPO` per arm, mean and spread (n≈3 each)              | The absolute number everything else multiplies              |
| `t_PPO` **per task** (Walker and UpStepper separately)   | Harder terrain may simulate slower — do not assume transfer |
| `t_sentinel` (invalid genome: decode + validate, no PPO) | Confirms invalid evaluations really are \~free              |
| `t_develop` (NCA/rule development alone)                 | Confirms T_primary=100 is negligible vs PPO                 |

### B4. Controller budget determination (see D6)

| **MeasurementWhy**                                                  |                                            |
| ------------------------------------------------------------------- | ------------------------------------------ |
| Fitness-vs-timesteps learning curves to 2M, several bodies per task | Determines the timestep budget by evidence |
| Marginal fitness gain per 100k steps                                | Locates the plateau/knee                   |

### B5. Evaluation noise ⚠ (**often overlooked — do not skip**)

| **MeasurementWhy**                                                             |                                                                                                                                              |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Same body, same task, evaluated 5× with different PPO seeds → fitness variance | PPO is stochastic. The *same* body scores differently each run. This noise directly affects EA selection quality and every statistical test. |

**Why this matters:** if the between-arm difference you're trying to detect is smaller than the within-body evaluation noise, no amount of seeds will save you. This number determines whether your whole experiment is capable of answering its question, and it is cheap to obtain. It must go in the paper's methods.

### B6. Projection

Combine the above with **already-measured** per-arm validity rates:

```
expected_cost_per_eval(arm) = validity_rate × t_PPO + (1 − validity_rate) × t_sentinel

```

Output a table over candidate (population_size × generations × seeds) combinations → total wall-clock for the full matrix on this machine. **This table is the artifact you show your professor.**

### B7. Reliability

| **MeasurementWhy**                            |                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------ |
| Crash/failure rate over the pre-flight sample | Budget for retries                                                 |
| Memory high-water mark per worker             | Prevents OOM at full concurrency                                   |
| Thermal/clock behaviour under sustained load  | A 14700K under multi-day load may throttle; affects the projection |

---

## D6. Controller timestep budget: determined by learning curve, not chosen ⏳

**Rejected:** 5,000 (a pipeline-validation placeholder — far too small for meaningful locomotion learning). **Rejected as arbitrary:** 1,000,000 picked without justification. A reviewer will ask "why 1M?" and "we thought it seemed right" is not an answer.

**Decision:** determine it empirically in pre-flight (B4). Train several bodies per task to 2M timesteps, plot fitness vs timesteps, and select the **knee** — the point beyond which additional training yields negligible improvement.

**Selection rule (stated in advance):** the smallest budget at which median fitness improvement per additional 100k steps falls below 1% of the fitness achieved so far, sustained over three consecutive windows.

**Paper sentence:** *"The controller training budget was set to N timesteps, determined from learning curves on M representative morphologies per task as the point at which median marginal fitness improvement fell below 1% per 100k steps. Curves are reported in the supplementary material."*

**Critical fairness constraint:** the budget must be **identical across all arms** (it already is, enforced by `assert_comparable`). It may legitimately **differ between tasks** if their learning curves differ — that is not unfair, because arms are only ever compared within a task.

**Expected outcome:** somewhere between 500k and 1.5M. If the knee is below 1M, you save enormous compute *and* have evidence justifying it.

---

## D7. Tasks: sequential priority, not simultaneous (on one machine)

**Your question:** run both tasks in parallel, or finish one then the other?

**Key insight that makes this easy:** on a single machine, running two tasks "in parallel" **does not make anything finish sooner**. Total work is identical; the cores are the bottleneck either way. The only thing that changes is *which results exist if you stop early*.

**Decision (one machine): enqueue all Walker-v0 cells first, then all UpStepper-v0 cells.** The launcher keeps every core busy throughout — this is purely a queue ordering, not a throughput choice.

**Why:** if you run out of time, get interrupted, or lose the machine, you have **one complete, publishable task** rather than two half-finished ones. A paper with one complete task plus a documented partial second is publishable. A paper with two 60%- complete tasks is not.

**Decision (two machines): one task per machine.** Strictly better — genuine parallelism, identical code, just different `--tasks` arguments. Both machines must run the **same commit** (record the hash on each).

**Does parallel execution threaten validity or reproducibility? No.** Each cell is an independent, deterministically-seeded process. Fitness depends on the genome, the seed, and the task — never on wall-clock time or on what else is running. Concurrency affects *how long* a result takes, never *what* the result is. This is verified by an explicit test asserting identical outputs across different worker counts.

**One genuine caveat to disclose:** heavy concurrency can cause CPU thermal throttling, which changes *timing* but not *results*. Worth a sentence in the methods.

---

# PART C — INFRASTRUCTURE

## D8. Checkpointing: three levels

| **LevelWhat's savedFrequencyRecovers from** |                                                                          |                                  |                                              |
| ------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------- | -------------------------------------------- |
| **Per-generation** (inside a cell)          | population genomes, fitnesses, RNG state, generation index, elapsed time | every generation                 | crash mid-cell → lose ≤1 generation          |
| **Per-cell** (launcher manifest)            | status (pending/running/done/failed), output path, commit hash           | on every state change            | launcher crash / disconnect → resume exactly |
| **Campaign snapshot**                       | aggregate stats, partial figures, current best-per-arm                   | every N generations or T minutes | answers "how is it going?" at any moment     |

**Atomicity requirement:** every checkpoint writes to a temp file then renames. A partial write during a crash must never corrupt a checkpoint.

**Resumability requirement:** re-invoking the same command must skip completed cells, re-queue interrupted ones, and never silently re-run or skip. Verified by a kill-and-restart test.

## D9. Intermediate results and inference-while-running

At each campaign snapshot, automatically regenerate:

- Best-so-far fitness per (arm, task), median + IQR across completed seeds
- The dual-panel budget curves (spent vs valid evaluations)
- Current per-arm validity/diversity rates *as realised during evolution*
- The statistical comparison table over whatever seeds have completed **clearly marked PARTIAL, with n stated per cell**
- Best body renders (images) per arm

**Rule:** partial results are for monitoring and early drafting only. **No partial result goes in the paper without a completeness check.** Label every partial artifact with its completion fraction so a figure can never be mistaken for final.

## D10. Logging and metadata: save everything cheap, sample everything expensive

**Per evaluation (always):** genome hash, decoded body hash, validity + failure reason, fitness, PPO wall-clock, timesteps used, arm, strategy, task, seed, generation, evaluation index, worker id, timestamp.

**Per generation (always):** the above aggregated, plus population diversity (unique valid bodies), invalid-reason breakdown for that generation only, best/median/worst fitness, selection statistics.

**Per run (always):** full resolved config, commit hash, EvoGym commit, package versions, machine profile, start/end time, total evaluations.

**Sampled (expensive):** full genome vectors for the best k per generation; PPO learning curves for a sampled subset; body renders for best-per-generation.

**Explicitly NOT logged:** full PPO policy weights for every genome (enormous, almost never used). Save policy weights only for final best bodies per arm/task/seed.

**Format:** JSONL (append-only, survives crashes, streamable, greppable). One file per cell. Never a single global file — that would serialise writes across workers.

**Disk budget:** estimate from pre-flight's per-cell log size × cell count, ×3 safety margin. Confirm against the 294 GB free on `/` before launch. The unmounted 5.5 TB `sdb1` is available if needed.

## D11. Live monitoring: tmux + a status dashboard

**Decision: run under** **`tmux`****, with a dedicated status pane.**

`tmux` is non-negotiable for a multi-day run — it means an SSH disconnect, a closed laptop, or a dropped connection does not kill the experiment. Attach and detach at will.

**Terminal display must show, continuously:**

- Overall: cells done / running / pending / failed; % complete
- **Elapsed time and estimated time remaining** (ETA from measured mean cell time, with a confidence range, not a single misleading number)
- Per running cell: arm, strategy, task, seed, current generation, best fitness so far
- Throughput: evaluations/minute now vs. average
- Resource use: CPU %, RAM, disk free, per-core load
- Checkpoint status: last checkpoint time per cell
- Failures: count and the most recent error, prominently — never buried
- Time since last successful evaluation (detects a silent hang)

**Layout:** tmux with three panes — status dashboard (top), live log tail (bottom left), resource monitor (`htop`, bottom right).

## D12. Email notifications

**Send on:**

- Campaign start (with the full config and the projection, so you have a record)
- Each task's completion
- Every campaign snapshot (progress summary + ETA) — at most one per 6 hours
- **Any cell failure** (immediately, with the error)
- **Anomalies:** no successful evaluation in 30 minutes; disk below 20 GB; crash-loop detected (≥3 failures in 10 minutes)
- Campaign completion

**Do NOT send:** per-evaluation or per-generation emails. You will start ignoring them, and then miss the one that matters.

**Implementation:** SMTP via an app-specific password stored in an environment variable or a gitignored config file — **never committed to the repo**. Failure to send an email must never crash the run: wrap in try/except, log the failure, continue.

## D13. Robustness for an unattended multi-day run

| **RiskMitigation**                |                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| SSH disconnect                    | tmux (D11)                                                                              |
| Machine reboot / reclaimed by lab | Checkpoints (D8); results synced to `sdb1` and pushed to git regularly                  |
| Disk fills                        | Monitored; alert at 20 GB; logs are append-only JSONL, no unbounded growth              |
| A cell crash-loops                | Manifest marks failed after N retries and moves on; alert sent                          |
| Silent hang                       | "time since last evaluation" watchdog → alert                                           |
| Thermal throttling                | Monitored; affects timing only, not results                                             |
| OOM at full concurrency           | Per-worker memory high-water mark measured in pre-flight; concurrency sized with margin |

**Start-time recommendation:** begin a multi-day run in the **morning**, when you can watch the first hour. Most failures that occur at all occur early.

---

# PART D — DECISIONS STILL OPEN ⏳

## D14. The scoping gate (resolved immediately after pre-flight)

Pre-flight's projection table feeds one decision made with real numbers:

| **OptionBuysCosts**                           |                                        |              |
| --------------------------------------------- | -------------------------------------- | ------------ |
| More seeds (5 → 10)                           | Statistical power; interpretable nulls | 2×           |
| Second grid size (5×5 + 9×9)                  | Tests the actual mechanism (D3)        | \~2.5×       |
| Extra arms (`direct_repaired`, `nca_compact`) | Baseline credibility, capacity control | \~1.25× each |
| Larger timestep budget                        | Better-converged controllers           | linear       |

**Priority if forced to choose (recommendation, to be confirmed against real numbers):**

1. `direct_repaired` — cheapest defence against the "strawman baseline" objection
2. More seeds — without power, a null result is uninterpretable
3. Second grid size — highest scientific value, highest cost

**Contingency ladder if the projection exceeds available time:** reduce generations before reducing seeds; reduce seeds before dropping the second task; drop ablations before dropping any primary arm; **never** reduce seeds unevenly across arms (that breaks the comparison).

## D15. Population size, offspring size, evaluation budget ⏳

Remains genuinely open and is enforced as such in code (`OpenProfileDecisionError`). Resolved at D14 with the projection in hand. Starting point for discussion: population 20–24, generations 25–30, from the standard range in this literature — but confirmed against measured cost, not adopted by default.

## D16. Evaluation noise policy ⏳

Depends on pre-flight B5. If same-body fitness variance is **small** relative to between-arm differences: single PPO run per genome (standard practice, cheapest). If **large**: either average over 2–3 PPO seeds per genome (2–3× cost) or keep single evaluation and disclose the noise prominently as a limitation. **Final best bodies are always re-evaluated ≥5× and reported as a mean ± spread**, regardless — reporting a single lucky evaluation as "best fitness achieved" is a common and avoidable error.

---

# PART E — THE CAMPAIGN AT A GLANCE

```
STEP 1  Thread pinning + inner worker pool            ~1 day
STEP 2  PRE-FLIGHT (D5)                               ~half a day
        → machine profile, concurrency sweep, t_PPO per arm per task,
          learning curves, evaluation noise, projection table
STEP 3  SCOPING GATE (D14/D15/D16)                    decide together, with numbers
STEP 4  Full launcher + checkpointing + monitoring    ~1–2 days
STEP 5  LAUNCH: Walker-v0 to completion, then UpStepper-v0
STEP 6  Paper written in parallel from partial results (clearly labelled)
STEP 7  Final analysis on complete data; all figures regenerated from final data only

```

**What gets shown to the professor after Step 2:** the projection table, the learning curves justifying the timestep budget, the evaluation-noise figure, and the concurrency sweep — i.e. concrete evidence that the full experiment has a known, predictable duration.

---

# APPENDIX — Assumptions corrected in this document

1. **"Development adapts to the task"** — false; development runs before the task is seen. Corrected in D1.
2. **"Harder task ⇒ developmental encoding wins"** — difficulty is the wrong axis; regularity and scale drive indirect-encoding advantage, and the hardest tasks risk a floor effect where no algorithm succeeds. Corrected in D2.
3. **"We need results supporting statement X"** — replaced with a falsifiable hypothesis and pre-registered disconfirming conditions. Corrected in D1.
4. **"1M timesteps"** — not wrong, but unjustified as stated; now determined from measured learning curves. Corrected in D6.
5. **"Run tasks in parallel for efficiency"** — on one machine this yields no speedup; it only changes what exists if you stop early. Corrected in D7.
6. **"Pre-flight first, then parallelise"** — a serial measurement does not extrapolate to concurrent execution. Corrected in D4.