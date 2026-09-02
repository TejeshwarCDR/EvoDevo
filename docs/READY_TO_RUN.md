# READY TO RUN — Primary Three-Way Comparison

Timestamp: 2026-09-02T00:00:00Z (audit + lock session)
Git commit (working tree base): 2844b3e1da171b5766f780598596c63f13803cc0
Locked primary config hash (sha256 of the three resolved `ExperimentConfig` objects for direct/nca/rule_based, sorted-key JSON): `7e93f59a82ba780e2ea4229a3d60681d7e98c08cfc33e3479a63af56d88c24ce`

**Any future change to `eval_budget`, `ppo_timesteps`, `population_size`, `offspring_size`, `sentinel_fitness`, `min_voxels`, `min_actuators`, or any NCA hyperparameter after this point invalidates the lock and requires re-running ALL THREE arms from zero — not just the changed one.**

## Scope of this run

**In scope (primary comparison only):** direct, nca, rule_based × Walker-v0 × seeds 0-4, shared config `population_size=15, offspring_size=15, eval_budget=300, ppo_timesteps=5000, ppo_eval_episodes=1, grid_shape=(5,5), sentinel_fitness=-10.0, cache_enabled=False, min_voxels=3, min_actuators=1, use_crossover=False`.

**Deferred / out of scope for this run (not dropped, to be scheduled separately):**
- Secondary task benchmark comparison.
- NCA ablations: `dev_steps`, `hidden_channels` sweeps.
- Rule-based ablation: crossover variant (`use_crossover=True`).

## Checklist

### Part 1 — Static fairness and contract re-verification: PASS
- Full test suite: 279 passed, 0 failed (`python3 -m pytest -q`, includes the new morphology-equivalence test added during blocking-issue resolution).
- Encoder contract suite: `bash scripts/rule_based/test.sh` — 146 passed; `tests/encoders/test_contract.py` — 9 passed.
- `assert_comparable` pairwise across direct/nca/rule_based resolved configs: PASS for all 3 pairs. Resolved shared config confirmed identical across all three (values above); `use_crossover=False` confirmed for all three.
- `ea.py` source-verified: every evaluation (valid or invalid) counts toward `eval_budget` identically across encoders (unconditional `evaluations_used += 1`); sentinel fitness (`-10.0`) is a single shared config value used identically by all three; `cache_enabled=False` for this run, and the cache-hit path (if ever enabled) still consumes a budget slot and logs `cache_hit:true`.
- Encoder-module import isolation: **originally FAILED** — `evodevo/encoders/rule_based/rule_encoding.py` imported `is_connected`/`has_actuator`/`get_full_connectivity` directly from `evogym`. **RESOLVED**: equivalent functions (verified byte-for-byte equivalent to evogym's originals across 27 test bodies / 79 assertions in `tests/morphology/test_evogym_equivalence.py`) were added to `evodevo/morphology/validate.py` and `evodevo/morphology/connectivity.py`, and `rule_encoding.py` now imports from there instead. `decode()`'s control flow is unchanged (still returns a body/`None` unconditionally per the Encoder ABC contract — no new validity-conditioned branching was introduced). Final grep confirms zero forbidden imports:
  ```
  grep -rnE "^\s*(import|from)\s+(evogym|stable_baselines3|gym|torch)\b" evodevo/encoders/
  ```
  → empty (zero matches).
- Port-equivalence test (`tests/encoders/test_rule_based.py::test_ported_interpreter_matches_original_across_20_seeded_genomes`, against `external/rule-based-reference`): PASSED — behavior unchanged by the import fix.
- `ppo_runner.py`: identical PPO construction regardless of encoder (`evaluate_body` takes no encoder parameter); grep for encoder-name conditional branching in `evaluation/` returned nothing.
- No rendering/visualization calls inside `ea.py`'s or `ppo_runner.py`'s hot loop (grep empty).

### Part 2 — NCA pre-registered calibration: PASS, FROZEN
- PPO-free inspection re-run (N=2000, same seed as prior sanity report): validity 69.15%, degenerate fraction 1.95%, non-settling 68.35%, state magnitude at t=T mean 40.655.
- Decision rule applied using PPO-free metrics only (validity ≥50%, not degenerate) → hyperparameters left unchanged; non-settling instability documented as an accepted, reported limitation. No PPO/fitness signal was used in this determination. Justification recorded in `docs/decisions.md`.
- NCA architecture and hyperparameters are now FROZEN for the remainder of this run.

### Part 3 — Data hygiene: PASS
- `results/runs/direct/Walker-v0/` (seeds 0-3, incomplete/pre-lock) archived via `git mv` to `results/archive/pre-final-run-direct/direct/Walker-v0/`, with `results/archive/pre-final-run-direct/README.md` explaining supersession (predates the frozen NCA calibration and the full three-way fairness lock).
- `results/runs/` confirmed empty of stale data. Nothing deleted.

### Part 4 — Unattended-run robustness: PASS
- Per-evaluation isolation: `evodevo/evaluation/ppo_runner.py` wraps each individual evaluation's training/eval block in try/except/finally, recording a failure (`valid=False`, sentinel fitness) and continuing rather than aborting the run, scoped only at the per-evaluation boundary. Verified via full suite re-run after the change (still green).
- Resumability: verified with a real kill/restart test on a tiny surrogate-mode run — killed mid-run (`kill -9`) after 8/15 `(encoder,seed)` runs completed; restart correctly skipped the 8 done runs (`run_meta.json` present) and completed the remaining 7, no duplicates. **Known limitation**: resumability is at whole-`(encoder,task,seed)`-run granularity, not per-evaluation — a run killed mid-progress has its partial JSONL overwritten and restarts from evaluation 0. Accepted risk: worst case, a crash costs at most one in-flight run's progress, not the full 11-hour job.
- Heartbeat file: `.heartbeat.json` (under the benchmark output dir) added, updated after every completed evaluation with timestamp, `(encoder, seed, evaluation_index)`, and evaluations-completed-so-far. Verified live during the smoke test.
- JSONL logs flushed to disk after every evaluation (`RunLogger.log` calls `flush()` after every write) — confirmed by source inspection.
- Disk space: 112 GiB free on the working volume vs. an estimated <10 MB needed for the full primary run's logs/saved bodies — sufficient.

### Part 5 — Isolated smoke test: PASS
- `make benchmark SURROGATE=1`-equivalent run against a fresh temporary directory (never `results/runs/`) produced a complete three-arm report with exactly 300 evaluations logged per (encoder, seed) across all 15 (encoder×seed) runs — identical evaluation counts across all three arms. `REPORT.md` generated successfully.

## Outstanding uncommitted changes

The following files are modified/added/moved in the working tree from this audit-and-fix session (not yet committed): `docs/decisions.md`, `evodevo/encoders/rule_based/rule_encoding.py`, `evodevo/evaluation/ppo_runner.py`, `evodevo/evolution/logging.py`, `evodevo/experiments/run_experiment.py`, `evodevo/morphology/connectivity.py`, `evodevo/morphology/validate.py`, `tests/morphology/test_evogym_equivalence.py` (new), and the `results/runs/direct/...` → `results/archive/pre-final-run-direct/direct/...` rename plus its new `README.md`. These should be committed before execution begins, at the user's direction.

---

**STATUS: READY TO RUN. Awaiting explicit authorization to begin the 11-hour execution phase.**
