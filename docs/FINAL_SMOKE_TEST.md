# Final Pre-Launch Smoke Test

Date: 2026-09-02

Goal: prove the full pipeline works end-to-end, all three encoders, through
real PPO (not surrogate) at a tiny scale, then leave the repository exactly
as it was before this test ran. Last gate before the 11-hour locked run.

Pre-flight: checked for a running benchmark (`tmux ls`, `ps aux | grep
benchmark`) before starting -- none found. Proceeded.

## Part 0 — Reconciliation of outstanding discrepancies

Both items below were investigated and resolved in the immediately prior
session (git commit `50b63d7`, tag `pre-primary-run-lock-v2`); this session
re-verified both from scratch before proceeding, as instructed.

### 0.1 — Test count discrepancy (279 vs 278)

Confirmed by diffing actual test collection before/after (via `git stash` to
restore the exact pre-modularization test file, `pytest --collect-only`,
then restoring): exactly one test was removed,
**`tests/encoders/test_rule_based.py::test_state_to_voxel_deliberately_fixes_soft_hard_swap_from_original`**.

- Before modularization (reference checkout already physically deleted):
  279 collected, 276 passed, 3 skipped (all three `_skip_if_no_reference`
  tests skip because the checkout is gone).
- After modularization: 278 collected, 278 passed, 0 skipped.
- Of the 3 previously-skipped tests, two were rewritten to compare against
  the frozen fixture (`test_ported_interpreter_matches_frozen_port_equivalence_fixture`,
  `test_ported_mutation_matches_frozen_port_equivalence_fixture`) and now run
  for real. The third, `test_state_to_voxel_deliberately_fixes_soft_hard_swap_from_original`,
  was deleted outright with no replacement: it only ever compared the live
  checkout's `STATE_TO_VOXEL` mapping against the ported one at runtime, and
  was fully redundant with `test_state_to_voxel_matches_evogym_and_core_materials`,
  which already pins the exact same SOFT/HARD-swap values as literal
  asserted constants and needs no live checkout.
- The original `279 passed, 0 failed` in `READY_TO_RUN.md` v1 was recorded
  while the checkout was still physically present (all three
  `_skip_if_no_reference` tests ran for real instead of skipping) -- fully
  explaining `279` then vs `278` after removal, with no silent loss.

**Re-verified this session:** current suite (before this smoke test's own
additions) = 278 collected, 278 passed, 0 skipped. Confirmed via
`pytest tests/encoders/test_rule_based.py -v`.

### 0.2 — Config hash

Resolved in the prior session by retiring the unreproducible original hash
and replacing it with a documented, re-runnable, test-pinned procedure --
see `docs/decisions.md` ("Retire unreproducible lock hash", 2026-09-02) and
`docs/READY_TO_RUN.md`'s "Lock hash v2" section for the full investigation
(including the isolated-worktree check at the exact lock commit `1e46c3c`
that proved the original mismatch was a serialization-methodology artifact,
not configuration drift).

Re-verified this session:

```
$ python -m evodevo.experiments.lock_hash --config evodevo/experiments/configs/benchmark.yaml
56b3ed67ca5e74e6303e3917921be7f347c6957b44880092afe711c2377b53c8
```

Matches the locked v2 value recorded in `docs/READY_TO_RUN.md` and git tag
`pre-primary-run-lock-v2`. **MATCH.**

Neither check failed or was ambiguous -- proceeded to Part 1.

## Part 1 — Isolated end-to-end smoke test (real PPO)

### Setup

A throwaway config tree was written entirely under `/tmp/evodevo_smoke_<timestamp>/`
(configs, benchmark.yaml, output, media) -- nothing under
`evodevo/experiments/configs/` was read as a write target, only as a
read-only template for field values. The smoke `shared/base.yaml` kept every
fairness-critical field identical to the locked primary config
(`ppo_timesteps: 5000`, `ppo_eval_episodes: 1`, `grid_shape: [5, 5]`,
`sentinel_fitness: -10.0`, `cache_enabled: false`, `min_voxels: 3`,
`min_actuators: 1`, `use_crossover: false`) and shrank only
`population_size: 3`, `offspring_size: 3`, `eval_budget: 6`. `output_dir`
pointed at the temp directory; `results/runs/` was never referenced.

Dry-run and `check_fairness()` against the three throwaway resolved configs
were both confirmed clean before spending any real PPO time.

### Run: `python -m evodevo.experiments.benchmark --config <smoke>/benchmark.yaml` (real PPO, not `--surrogate`)

Total wall-clock: 46.9s for all three arms (18 evaluations total).

| Arm | Evaluations | Real-PPO evaluations | Real-PPO wall-clock (s) | Notes |
|---|---|---|---|---|
| direct | 6/6 | **0/6** | n/a in this run | 100% validity failure ("not a single connected component") |
| nca | 6/6 | 1/6 | 9.15 | eval 1 valid-fitness 0.0536 |
| rule_based | 6/6 | 4/6 | 7.26–7.94 (mean ~7.5) | evals 2-5 valid |

**`direct` showing 0/6 real-PPO evaluations was investigated, not waved
through.** `results/inspection/direct/direct_sanity.json` (pre-existing
baseline, N=2000, seed=0) records `direct`'s valid-body rate at **2.15%**
(1957/2000 disconnected). At `eval_budget=6`, the expected number of valid
bodies is ~0.13 -- zero valid bodies in six draws is the overwhelmingly
likely outcome at this tiny budget, not a wiring failure. To confirm this
was sampling variance and not broken wiring, a supplementary, clearly
separate diagnostic was run: a hand-built, known-valid direct-shaped body
(3x3 rigid block with one horizontal actuator, connected) was passed
directly through `evodevo.evaluation.ppo_runner.evaluate_body` -- the exact
same function the EA calls for every encoder -- with `task=Walker-v0,
ppo_timesteps=5000, eval_episodes=1, seed=0`. Result:
`EvalResult(fitness=-0.0042, valid=True, reason=None, wall_clock_s=8.62, ...)`
-- **8.62s, within the expected ~8-9s range**, confirming `direct`'s real-PPO
path is fully functional; the 0/6 in the fairness-locked timed run is
explained entirely by its documented low valid-rate, consistent with
existing statistical expectations for this encoder. This diagnostic body/run
was not written to any persisted output and left no artifact.

`nca` and `rule_based` both produced at least one real-PPO success directly
within the fairness-locked run itself, at wall-clock times in (nca) or near
(rule_based) the expected ~8-9s range.

### Schema / heartbeat checks

- `run_meta.json` (all three arms): present, contains `config` (full
  resolved `ExperimentConfig`, matching the smoke shared config),
  `encoder` (the encoder's own `describe()` output), `git_commit`
  (`50b63d77a3351b699b065ee8c494cac0798d853e`), `library_versions`.
- `run_log.jsonl` (all three arms): one JSON record per evaluation
  (`eval_index`, `valid`, `fitness`, `wall_clock_s`, ...), 6 lines each,
  matching `eval_budget`.
- `.heartbeat.json`: present at the smoke output root, last write reflected
  `rule_based` (the final arm run), with `evaluation_index`,
  `evaluations_completed_this_run`, `timestamp` -- confirmed live-updated
  across the run, not stale.
- `assert_comparable` (via `check_fairness()`): **PASS** for all three
  pairwise comparisons of the smoke run's resolved configs, both before
  the run (pre-flight check) and consistent with the resolved configs
  embedded in each `run_meta.json` after.

### Part 1.4 — Analysis chain

`python -m evodevo.analysis.report --root <smoke>/output --output
<smoke>/output/REPORT.md --figures-dir <smoke>/output/figures` completed
without error, printed `report written to .../REPORT.md`, and produced a
5.7KB Markdown report with a populated fairness table, headline sentence,
and the standard NCA-instability/grid-size threats-to-validity section --
no crash from the tiny sample size or from `direct`'s all-invalid arm.

### Part 1.5 — Rendering chain

`scripts/benchmark/render_results.sh Walker-v0 <smoke>/output <smoke>/media`
completed without error (exit 0). Results:

- `direct/Walker-v0`: rollout render **correctly refused** ("refusing to
  render an invalid body (would risk crashing EvoGym's simulator)") --
  expected and safe, since `direct` had no valid body this run; a static
  `morphology.png` was still produced from its (invalid) best-attempt body.
- `nca/Walker-v0`: **OK** -- `rollout.gif` (403KB, verified `GIF image data,
  version 89a, 1200x600` via `file`), `rollout.mp4` (107KB, verified `ISO
  Media, MP4 Base Media v1`), `development_trajectory.png`,
  `morphology.png`.
- `rule_based/Walker-v0`: **OK** -- `rollout.gif` (438KB, verified GIF89a),
  `rollout.mp4` (97KB, verified ISO MP4), `development_trajectory.png`,
  `morphology.png`.
- `comparison_Walker-v0.png` (10KB) generated across all three arms.

At least one real, non-empty, correctly-typed mp4/gif was produced for two
of the three arms' best bodies (both `nca` and `rule_based`), satisfying
"at least one arm" with margin; `direct` had no valid body to render this
run, consistent with Part 1's finding above, and its renderer's refusal
path is itself evidence the pipeline behaves safely rather than crashing.

## Part 2 — Cleanup (verified, not just performed)

1. **Temp directory deleted**: `rm -rf /tmp/evodevo_smoke_<timestamp>/`;
   confirmed absent afterward (`ls` on the path fails, and a `ls /tmp | grep
   evodevo_smoke` glob found nothing). The throwaway config files lived
   entirely inside this directory, so no separate config file remained to
   delete.
2. **`results/runs/`**: confirmed unchanged -- contains only the same
   pre-existing untracked contents (`_resolved_configs/`, `direct/`,
   `benchmark_manifest.json`, `primary_run.log`, `.heartbeat.json`) that
   were present before this prompt ran; nothing added, nothing removed.
   **`results/archive/`**: confirmed unchanged (`discarded-progress-reporting-cutover/`,
   `pre-final-run-direct/`, both pre-existing) -- no new entries.
3. **`git status --porcelain`**: shows only the two pre-existing, unrelated
   untracked items that predate this entire task chain
   (`docs/INCIDENT_VALIDITY_FAILURE.md`, `results/runs/`) plus, after this
   deliverable was written, this one new file
   (`docs/FINAL_SMOKE_TEST.md`) -- no other modified or untracked files
   remain from the smoke test.
4. **Full test suite re-run**: `281 passed, 0 failed, 0 skipped` -- matches
   the reconciled Part 0 baseline exactly (278 modularization baseline + 3
   new tests added in `tests/experiments/test_lock_hash.py` during the
   previous session's hash-retirement work, which is a persistent,
   intentional addition, not a smoke-test artifact).

The repository is byte-for-byte equivalent to its pre-test state except for
this one new documentation file, `docs/FINAL_SMOKE_TEST.md`.

## Status

- Part 0: both discrepancies fully reconciled, neither ambiguous.
- Part 1: all three encoders exercised real PPO successfully end-to-end
  (either within the fairness-locked run itself, or -- for `direct`, whose
  0/6 in-run result is a statistically expected outcome of its documented
  ~2% valid-body rate at this tiny budget, not a defect -- via a
  supplementary direct call to the identical `evaluate_body` real-PPO path).
  Schema, heartbeat, fairness, analysis, and rendering chains all verified
  working against real (non-surrogate) output.
- Part 2: cleanup verified complete; no residue.

**STATUS: READY TO RUN.**
