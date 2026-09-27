# ADR: Launcher scheduling — interleave replicate runs within each D7 task phase

**Status: ACCEPTED — approved by Tejeshwar, 2026-09-27. Nothing described here is implemented anywhere in the repository yet.**
Date: 2026-09-24. Owner of the decision: Tejeshwar (scoping gate, D14/D15). Basis: `results/preflight/20260922T122730Z_4be8402b/report/CAPACITY_PLANNING_REPORT.md`.

## Context

The full launcher is Step 4 of the D4 sequencing in `docs/final_decisions.md`. Its scheduling design is still open. Three facts from the code and the pre-flight data constrain it.

1. **A generation is a hard barrier inside one run.** In `evodevo/evolution/ea.py`, generation g+1's parents come from `sorted(population, key=fitness)[:mu]`, and that population includes every generation-g offspring. This is a real requirement of (μ+λ) selection, so every one of the λ evaluations must finish first. It is not an accident of the coding.
2. **Replicates share no state.** A "seed" in `projection.md` is an independent (arm, task, seed) run. Each has its own config (`dataclasses.replace(..., seed=seed)`), its own RNG (`np.random.default_rng(config.seed)`), and its own output directory. The fitness cache is off for the primary comparison and is created per run when it is on. Replicates can therefore be scheduled concurrently without changing any result.
3. **The D15 grid is λ-limited.** Every row of the current candidate grid has population = λ = 15, which is below the corrected optimal concurrency of 20. A single run can never occupy more than 15 of the 20 workers (75%), and it leaves the 5 extra workers idle. The batch also ends only when its slowest worker finishes, and 7 of the 15 workers are E-cores (2.3172× slower).

The existing code is fully serial. `benchmark.py` loops over `run_matrix` one run at a time, and `run_ea` evaluates one genome at a time through `tracker.evaluate`.

## Measured cost of the two designs (Monte Carlo, 500 trials, real measured inputs)

| scheduling | full campaign wall-clock, mean (p5–p95) |
|---|---|
| SERIAL_PER_RUN (one replicate at a time, width 15) | 381.2 h flat validity; 480.9 h historical curves; **480.0 h** (474.5–485.9) current-calibration curves |
| INTERLEAVED (all replicates of all arms in a task phase pooled on 20 workers) | 109.4 h flat validity; 151.9 h historical curves; **165.6 h** (164.2–167.0) current-calibration curves |
| Cost of not interleaving | +271.8 h (3.49×) flat; +328.9 h historical; **+314.5 h (2.90×)** current |

Under interleaving the pool realises 99.1% of its theoretical capacity. Under serial-per-run it realises 34% at width 15. See the report for the inputs, the sensitivity variants and the law cross-checks.

## Recommendation (for sign-off)

Build the full launcher with **interleaved cross-run scheduling inside each D7 task phase**:

- One shared pool of 20 workers, placed by `cpu_slots()` / `allocation_order()`.
- All (arm, seed) runs of the current task are pooled. Idle workers take ready evaluations greedily. Run r's generation g+1 batch becomes ready only when run r's own generation g has fully finished and been selected.
- The D7 ordering is unchanged: every Walker-v0 run completes before any UpStepper-v0 run starts.

## What this does NOT change

- **Not the (μ+λ) algorithm.** Each run keeps its own generation barrier, so selection, mutation and the evaluation-count semantics are identical. It changes only which worker evaluates which genome, and when.
- Results stay independent of scheduling, as D7 already states. That claim needs an explicit test asserting identical run outputs under serial and interleaved schedules.

## Consequences and required care

- **`ea.py` must be split into propose / evaluate / tell steps.** Today `run_ea` calls the evaluator one genome at a time, so no batch exists to hand to a pool. The λ offspring genomes depend only on the survivors and the RNG, never on the results of other offspring in the same generation, so they can be generated first and evaluated concurrently. The RNG draw order, genome-id order and `eval_index` order must be preserved so runs stay bit-reproducible. This is a code change to the shared loop and needs its own review; this ADR does not make it.
- **Fairness of partial results (traced; corrected twice).** An earlier draft claimed FIFO "leaves direct runs finishing late", and a later one compared a single direct run in the pool with a solo figure that was actually a 5-seed row under flat validity. The simulator (`scheduling_trace.py`, `scheduling_trace.json`, `round3_solo_and_inline.json`) shows: under FIFO the direct runs finish *before* the others (about 74.6 h against 82.7 h with the current-calibration validity curves; 38.8 h against 54.7 h with flat validity). What pooling costs a direct run is queueing: a single direct run takes **3.3 h solo with flat validity, but 13.2 h (historical curves) or 13.0 h (current curves) solo**, against **38.8 h / 69.3 h / 74.6 h pooled** — a slowdown of 11.6× / 5.3× / 5.8×. A direct generation with 15 near-instant invalid evaluations still waited about 10,000 s in the pool, because a sentinel evaluation needs a worker slot behind ~1,300 s PPO tasks. FIFO also leaves arms unevenly advanced at partial stops (direct about 2 generations ahead at 25% of the phase). Lowest-generation-first equalises progress at an identical makespan. Recommendation: use lowest-generation-first, or an equivalent per-run lock-step policy, if balanced partial results matter.
- **Resolving sentinel evaluations inline (measured; not recommended for wall-clock).** Letting invalid genomes skip the pool (decode + validate in the launcher) leaves the full-campaign interleaved total unchanged in every validity scenario (109.39 → 109.36 h flat; 151.99 → 152.00 h historical; 165.58 → 165.58 h current). It shortens direct's own pooled finish from 38.8 to 14.6 h under flat validity but only from 74.6 to 68.5 h (about 10% of its queueing delay) under the current-calibration curves. It still needs the batched dispatch of the refactor, so it is a small increment on top of it (launcher dispatch, exposing the validate stage in `ppo_runner.py`, accepting pre-resolved results in `eval_bookkeeping.py`), not an alternative to it. It does not change the recommendation.
- **Checkpointing** is naturally per (run, generation), which is where the barrier already sits.
- **Sensitivity (validity).** Validity is decided before PPO and independent of the task (`validate_body` runs first in `evaluate_body` and takes no task argument), so validity curves transfer across the PPO budget and across Walker-v0 / UpStepper-v0; the residual assumption is that the validity trajectory is not strongly affected by fitness-based choice among valid parents, which a no-PPO proxy reproduced for direct. The first-round figures assumed the random-genome validity rate in every generation (109.4 h interleaved, 381.2 h serial). Validity-by-generation **measured under the current calibrated configuration** (real `run_ea`, no PPO, 20 seeds per arm) gives about **165.6 h interleaved (p5–p95 164.2–167.0)** and **480.0 h serial-per-run (474.5–485.9)**, a serial penalty of +314.5 h. The historical (pre-calibration) run gave 151.9 h / 480.9 h. The hard ceiling (all post-generation-0 offspring valid) is 239.9 h interleaved. See the report's Round 3 section.

  **Accepted risks, not further investigated:**
  - (a) The validity-by-generation proxy used random fitness for valid genomes, so it cannot capture whether real PPO-based selection among valid parents shifts the trajectory differently. This is the largest remaining source of uncertainty in the wall-clock estimate.
  - (b) The post-PPO `simulation_instability` invalidation path was never observed in the historical logs (0 of 1,749 invalid records), but that data is Walker-v0 at 5,000 timesteps only; its rate at 1,000,000 timesteps or on UpStepper-v0 is unmeasured.
  - (c) Neither risk changes the interleaved-versus-serial recommendation: serial-per-run is 2.1–3.5× slower than interleaved in every scenario tested (flat 3.49×, historical 3.17×, current 2.90×, ceiling 2.14×). Only the absolute estimate (best current figure 165.6 h) carries the residual uncertainty.
  - (d) Closing (a) would need real PPO time and is not recommended before the campaign: the historical (151.9 h) and current (165.6 h) estimates already agree within about 9%, and the launcher's first real generations will show the true trajectory as a by-product.
- **Phase barrier.** The strict "Walker phase finishes before UpStepper starts" rule costs at most about 1% (the pool runs at 99% of capacity), so it is acceptable.

## Alternatives considered

- **Serial per run.** It is simplest, but it costs +272 h and leaves 25% of the workers idle by construction.
- **Asynchronous or steady-state EA.** It removes the barrier by changing the algorithm. That is out of scope and would invalidate the calibration chain.
- **Mixing tasks across phases.** D7 rejects it for partial-result safety.

## Sign-off

**This recommendation is decision-ready and does not depend on resolving the items in the Accepted Risks section above.**

- [x] Approved as the design for Step 4 (full launcher)
- [ ] Approved with changes: ______
- [ ] Rejected: ______
