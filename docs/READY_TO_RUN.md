# Ready to Run — How to Actually Run This, Today

This document is the step-by-step "how do I run this" guide, kept current.
For the full decision trail behind every value below, see `docs/decisions.md`.
For the consolidated resolved-configuration/readiness gate (every dynamics
field, cross-checked against the real production config path, plus what is
explicitly *not* covered), see `docs/PRE_PARALLELIZATION_READINESS.md` — that
document, not this one, is the authoritative gate for whether a real launch
is ready.

## What exists today

- **Four arms**: `direct`, `nca`, `rule_based`, `rule_based_monotone` — see
  the root `README.md` and `evodevo/encoders/*/resolved_config.py` for what
  each one is.
- **Two profiles** (`evodevo/experiments/benchmark.py::PROFILES`):
  `fast_dev` (today's default in `evodevo/experiments/configs/benchmark.yaml`
  — a pipeline-validation-scale budget: `ppo_timesteps=5000`,
  `population_size=15`, `offspring_size=15`, `eval_budget=300`) and
  `primary_matrix` (`ppo_timesteps=1,000,000`, the confirmed real target —
  but `population_size`/`offspring_size`/`eval_budget` are a deliberate,
  disclosed OPEN decision, not guessed anywhere in this codebase; attempting
  a real run under `primary_matrix` without explicitly supplying all three
  via `profile_overrides` raises `OpenProfileDecisionError`).
- **No real primary-matrix result exists yet.** `results/runs/` currently
  holds only a pre-architecture-redesign run (see
  `results/BENCHMARK_REPORT.md`'s own superseded banner) — real/current
  output has not been produced.

## Running the surrogate comparison (fast, no real PPO, safe to run anytime)

```bash
make benchmark SURROGATE=1
```

or directly:

```bash
python -m evodevo.experiments.benchmark \
  --config evodevo/experiments/configs/benchmark.yaml --surrogate
```

This exercises all four arms end-to-end (manifest, resumability, progress
reporting, report generation) using the fast physics-free evaluator. Safe to
run repeatedly; resumes any already-complete `(arm, task, seed)` cell rather
than re-running it.

## Running a real (non-surrogate) comparison

Against the current `fast_dev`-profile `benchmark.yaml`, a real run requires
an explicit acknowledgement that this is a pipeline-validation-scale budget,
not the scientific comparison:

```bash
python -m evodevo.experiments.benchmark \
  --config evodevo/experiments/configs/benchmark.yaml \
  --i-know-this-is-fast-dev
```

**Do not do this expecting a scientifically meaningful result** — it exists
to smoke-test the real-PPO path end-to-end at low cost, not to produce a
result worth citing. To run the actual primary matrix once the
`population_size`/`offspring_size`/`eval_budget` scoping decision is made:

```bash
python -m evodevo.experiments.benchmark \
  --config evodevo/experiments/configs/benchmark.yaml \
  --profile primary_matrix
  # plus whatever mechanism profile_overrides are supplied through once
  # population_size/offspring_size/eval_budget are decided -- see
  # docs/decisions.md's "Do NOT guess population_size/eval_budget/
  # offspring_size" row and docs/PRE_PARALLELIZATION_READINESS.md's
  # "NOT covered by this document" section.
```

## Config-hash verification (drift check, not a launch gate)

`evodevo/experiments/lock_hash.py` hashes the three resolved
`ExperimentConfig` objects for the shared-config fields (population_size,
eval_budget, etc. — not per-arm dynamics knobs, which are intentionally
outside this hash, same as they're outside the fairness check). Recompute
and compare before citing any result as reproducible:

```bash
python -m evodevo.experiments.lock_hash --config evodevo/experiments/configs/benchmark.yaml
```

Current value (`campaign` profile, as of 2026-09-27, commit `c7172d1`; re-locked after the
campaign decisions in `docs/decisions.md`: 4 arms x 2 tasks, per-task `ppo_timesteps`
1,000,000 / 1,200,000, per-task `ppo_seeds` 1 / 2, `ppo_seed_aggregation` max,
population 15, offspring 15, `eval_budget` 300):

```
e5c2aa7547d4e08e6e0dad9e57f5a6e928355b3c671606d867f7709b404af638
```

Previous value (`fast_dev` profile, 2026-09-19, commit `bbe54c6`; Walker-v0 only, 4 arms):
`c1685531d6e6543215b012b204d820c15e1a9284bbd6b0097cdc01aa1224f7ed`.

This is **not the same value** as either historical hash recorded in earlier
revisions of this document (`7e93f59a...`, `56b3ed67...`) — both were
computed against configurations that have since changed (the shared
`dev_steps` alone moved `39 -> 40 -> 100` across the T_primary corrections;
see `docs/decisions.md`). Any future change to `eval_budget`,
`ppo_timesteps`, `population_size`, `offspring_size`, `sentinel_fitness`,
`min_voxels`, `min_actuators`, `use_crossover`, `ppo_seeds`,
`ppo_seed_aggregation`, `dev_steps`, or `init_condition` invalidates the value above — re-run the command, don't
assume it still matches.

## Test suite

```bash
make test
```

Full suite (run this before citing any result). See `docs/
PRE_PARALLELIZATION_READINESS.md` for the exact pass/fail count as of its
own last regeneration.

## Historical record

Earlier revisions of this document (2026-09-02) recorded a full pre-launch
audit checklist (fairness re-verification, the NCA pre-redesign calibration
gate, data hygiene, unattended-run robustness, an isolated smoke test) and
concluded "READY TO RUN" for an 11-hour real-PPO primary comparison under
the pre-redesign architecture. That architecture (unbounded-residual NCA,
`row_major_last_wins` rule_based conflict resolution, uncalibrated mutation
defaults, no `rule_based_monotone` arm) no longer exists — it was
superseded by the entire N1-N17/R1-R8 redesign workstream recorded in
`docs/decisions.md`. That checklist's own audit disciplines (verify, don't
assume; re-run smoke tests against real code; disclose limitations plainly)
remain this project's standard and are what `docs/
PRE_PARALLELIZATION_READINESS.md` now applies to the current architecture.
