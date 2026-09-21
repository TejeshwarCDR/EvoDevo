# EvoDevo Pre-Flight Readiness

Generated 2026-09-21T02:46:06.668473+00:00 by `python -m evodevo.preflight.readiness`.

**The real pre-flight has NOT been executed as part of producing this document.** The only executions performed to produce this document (besides the test suite and a live, read-only B1 machine query) are the bounded `--self-check` and `--self-check-full` runs reported in Section 7. To run the real pre-flight, an operator runs (from the repo root, in tmux):

```
bash scripts/preflight/run_preflight.sh
```

which wraps:

```
python -m evodevo.preflight
```

## Changes since the previous readiness report

Closing the readiness gaps a review of the first PREFLIGHT_READINESS.md found, before the real pre-flight runs. Each item below is a full gate from that review; see Section 9 for the detailed write-up of the first three.

| # | gap | outcome |
|---|---|---|
| 1 | `evodevo/evaluation/ppo_runner.py` showed modified with no report (Prompt 2's own rule: stop and report, don't change silently) | Explained, classified **behaviour-neutral** (`_rollout_fitness()` extraction + additive `evaluate_body_with_curve()`), proven bit-identical against HEAD across 5 bodies by a new committed test. See Section 9. |
| 2 | self-check restricted to arm=direct, task=Walker-v0 -- first contact with 5 of 6 real combinations would happen hours into the real run | Added `--self-check-full` (same trivial settings, full 3-arm x 2-task grid, 10-minute budget). Root cause of the original 3-minute budget not fitting the full grid identified and fixed: `rule_based`'s decode() cost (~232ms/call) multiplied by previously-unreduced, CLI-unexposed `t_develop_n`/`t_sentinel_n` (now `--cost-develop-n`/`--cost-sentinel-n`, reduced to 5 for self-check). Measured at 64s against the 600s budget, zero partial/failed combinations. See Section 7 and Section 9. |
| 3 | no gate preventing a real launch against an uncommitted tree | `python -m evodevo.preflight` now refuses to create a run directory for a real launch when `git status --porcelain` (excluding results/) is non-empty, unless `--allow-dirty` is passed; self-check/self-check-full are exempt but always record `project_dirty` in their manifest. Covered by `tests/preflight/test_dirty_check.py` and `tests/preflight/test_cli.py`. |
| 4a | Section 8 reported evogym version as "unknown", disagreeing with B1's own correct report | Fixed to read the same source B1 uses (`importlib.metadata.version`, not `module.__version__`, which evogym does not define); a regression test asserts the two agree. |
| 4b | `coretemp_readable=true` appeared to contradict ENVIRONMENT_REPORT.md's "lm-sensors not installed" | Not a contradiction: both read `/sys/class/hwmon` directly via `psutil.sensors_temperatures()`; `lm-sensors` is only the separate, optional `sensors` CLI convenience tool. Verbatim sample confirmed non-null; all 56 samples in the checked run were non-null. See Section 9. |
| 4c | CPU frequency policy (scaling driver, per-core EPP, turbo state) was not recorded -- a measurement whose frequency policy is unrecorded cannot be reproduced | B1 now records scaling_driver, energy_performance_preference for one P-core and one E-core, intel_pstate no_turbo/status, and idle frequency per core; B7's D13 risk table surfaces all of it with a note that it must be IDENTICAL between this pre-flight and the real campaign. |
| 4d | `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` shows modified after every full-suite run | Confirmed deliberate: `test_order_independence_permutation_audit` documents this as an intentional side effect in its own docstring. Reported, not changed, per this gate's own instruction; reverted with `git checkout` after every full-suite run in this session. |
| 5 | work from Prompts 1-7 sat entirely uncommitted | Committed in 4 dependency-ordered commits (Step 0 env setup; the ppo_runner.py change, isolated and revertible alone; evodevo/parallel; evodevo/preflight) -- reordered from the prompt's literal (1)(2)(3)(4) numbering so the ppo_runner commit lands before evodevo/parallel, which depends on it (evaluator_entry.py's real-mode curve_request path), keeping every commit independently buildable. **`docs/final_decisions.md` was NOT committed** -- despite the instruction to copy an uploaded file there, no such file could be found anywhere on this machine or in this conversation's accessible context; this is flagged as a genuine blocker, not silently skipped or fabricated. |

## 1. Git state and config lock

- Commit: `0dbcb50566caeceefee51a000e67b5c779224624`
- Working tree clean: **False** -- see `git status --porcelain` output below

```
?? PREFLIGHT_READINESS.md
?? docs/PROJECT_LOOSE_ENDS_RESOLVED.md
```

- Config lock hash (`evodevo.experiments.lock_hash.compute_lock_hash_from_benchmark_config()`): `c1685531d6e6543215b012b204d820c15e1a9284bbd6b0097cdc01aa1224f7ed`

## 2. File inventory (everything created across Prompts 1-7)

| file | lines |
|---|---|
| `evodevo/parallel/__init__.py` | 0 |
| `evodevo/parallel/affinity.py` | 62 |
| `evodevo/parallel/evaluator_entry.py` | 290 |
| `evodevo/parallel/threading_control.py` | 201 |
| `evodevo/parallel/topology.py` | 345 |
| `evodevo/parallel/worker_pool.py` | 384 |
| `evodevo/preflight/__init__.py` | 0 |
| `evodevo/preflight/__main__.py` | 688 |
| `evodevo/preflight/context.py` | 50 |
| `evodevo/preflight/dashboard.py` | 275 |
| `evodevo/preflight/dirty_check.py` | 47 |
| `evodevo/preflight/events.py` | 84 |
| `evodevo/preflight/readiness.py` | 752 |
| `evodevo/preflight/report.py` | 230 |
| `evodevo/preflight/run_dir.py` | 200 |
| `evodevo/preflight/stages/__init__.py` | 268 |
| `evodevo/preflight/stages/b1_machine.py` | 496 |
| `evodevo/preflight/stages/b2_sweep.py` | 502 |
| `evodevo/preflight/stages/b3_cost.py` | 397 |
| `evodevo/preflight/stages/b4_learning_curves.py` | 362 |
| `evodevo/preflight/stages/b5_noise.py` | 421 |
| `evodevo/preflight/stages/b6_projection.py` | 449 |
| `evodevo/preflight/stages/b7_reliability.py` | 446 |
| `evodevo/preflight/telemetry.py` | 263 |
| `scripts/parallel/show_topology.py` | 104 |
| `scripts/preflight/run_preflight.sh` | 161 |
| `scripts/preflight/self_check.sh` | 141 |
| `tests/parallel/__init__.py` | 0 |
| `tests/parallel/test_affinity.py` | 63 |
| `tests/parallel/test_threading_control.py` | 152 |
| `tests/parallel/test_topology.py` | 149 |
| `tests/parallel/test_worker_pool.py` | 271 |
| `tests/preflight/__init__.py` | 0 |
| `tests/preflight/test_b1_b2.py` | 310 |
| `tests/preflight/test_b3_b5.py` | 395 |
| `tests/preflight/test_b4_b6_b7.py` | 574 |
| `tests/preflight/test_cli.py` | 293 |
| `tests/preflight/test_dashboard.py` | 150 |
| `tests/preflight/test_dirty_check.py` | 93 |
| `tests/preflight/test_events.py` | 110 |
| `tests/preflight/test_readiness.py` | 124 |
| `tests/preflight/test_report.py` | 88 |
| `tests/preflight/test_run_dir.py` | 88 |
| `tests/preflight/test_stages.py` | 360 |
| `tests/preflight/test_telemetry.py` | 139 |

**45 files, 10977 lines total.**

## 3. Test results

- Full suite: `918 passed, 33 warnings in 615.70s (0:10:15)`
- Exit code: 0

### Five specific required results

| test | result |
|---|---|
| cross-worker-count determinism (`tests/parallel/test_worker_pool.py::test_determinism_matches_single_worker`) | **PASS** -- `============================== 3 passed in 4.18s ===============================` |
| thread-pinning assertion (`tests/parallel/test_worker_pool.py::test_workers_report_single_thread_and_pinned_env_vars`) | **PASS** -- `============================== 1 passed in 1.83s ===============================` |
| affinity-distinctness (`tests/parallel/test_worker_pool.py::test_workers_get_distinct_cpu_affinity_sets`) | **PASS** -- `============================== 1 passed in 1.82s ===============================` |
| kill-and-resume (`tests/preflight/test_stages.py::test_kill_during_stage_is_rerun_on_resume_while_done_stage_is_skipped`) | **PASS** -- `============================== 1 passed in 0.15s ===============================` |
| knee-selection unit tests (`tests/preflight/test_b4_b6_b7.py -k select_knee`) | **PASS** -- `======================= 6 passed, 22 deselected in 0.07s =======================` |

## 4. Stage list (dependency order, cost, every default parameter value)

```
run directory: /tmp/tmpkmyzk32q/20260921T025631Z_0dbcb505
stage                          cost       depends_on                     description
------------------------------------------------------------------------------------
machine_profile                free       -                              Machine profile (D5 B1): topology, RAM/disk incl. measured write rate, unconfigured-default thread behaviour, environment provenance, accelerator presence, CPU model/freq/governor, and D13 risk state.
concurrency_sweep              cheap      machine_profile                Concurrency sweep (D5 B2): real-PPO throughput/efficiency vs. worker count across --sweep-levels, on a fixed genome workload, with randomised visit order and inter-level cooldown to control for thermal drift.
evaluation_cost                expensive  concurrency_sweep              Evaluation cost (D5 B3): t_PPO per arm x task at full ppo_timesteps under B2's chosen concurrency, plus t_sentinel (invalid-genome fast path) and t_develop (decode-only cost), and the t_PPO/t_sentinel ratio.
learning_curves                expensive  evaluation_cost                Learning curves (D5 B4): fitness-vs-timesteps per representative body per task (one PPO training run per body, periodic-eval callback), the D6 knee-selection rule applied per task, and a WARNING wherever no plateau was observed within curve_max_timesteps.
evaluation_noise               expensive  evaluation_cost                Evaluation noise (D5 B5): repeated-PPO-seed fitness spread for low/middle/high-fitness bodies per task, pooled-within-body sd as a fraction of B3's between-body spread, and a plain-language D16 decision-input block. Determines whether the campaign can answer its own question -- treated as first-class, not an appendix.
projection                     free       learning_curves,evaluation_noise Projection table (D5 B6): expected_cost_per_eval per arm (validity_rate x t_PPO + (1-validity_rate) x t_sentinel), projected wall-clock with a measured range for the full CLI-supplied candidate grid, and D14 option-cost deltas. Never picks a configuration.
reliability                    free       projection                     Reliability aggregation (D5 B7): failures by class, crash rate per 1000 evaluations, peak-RSS-implied max safe concurrency, whole-run thermal/frequency behaviour, disk projection with the D10 3x margin, and a risk table mirroring B1's D13 register.
```

### Every CLI flag's default value

```json
{
  "allow_dirty": false,
  "arms": "direct,nca,rule_based",
  "base_dir": "results/preflight",
  "concurrency": null,
  "cost_develop_n": 100,
  "cost_max_draws": 5000,
  "cost_reps": 3,
  "cost_sentinel_n": 100,
  "curve_bodies": 3,
  "curve_eval_interval": 100000,
  "curve_max_timesteps": 2000000,
  "dashboard_interval": 2.0,
  "disk_probe_mb": 512,
  "dry_run": false,
  "force": null,
  "no_dashboard": false,
  "noise_bodies": 3,
  "noise_reps": 5,
  "only": null,
  "ppo_timesteps": 1000000,
  "proj_arms": "direct,nca,rule_based",
  "proj_generations": "20",
  "proj_overhead": null,
  "proj_populations": "15",
  "proj_seeds": "5",
  "proj_tasks": "Walker-v0,UpStepper-v0",
  "ram_margin": 0.8,
  "run_id": null,
  "self_check": false,
  "self_check_full": false,
  "skip": null,
  "sweep_cooldown_s": 60.0,
  "sweep_genomes": 24,
  "sweep_levels": "8,12,16,20,24,28",
  "sweep_order": "shuffled",
  "sweep_timesteps": 20000,
  "sweep_tolerance": 0.05,
  "tasks": "Walker-v0,UpStepper-v0",
  "telemetry_interval": 5.0,
  "thermal_threshold_c": 85.0,
  "timeout_s": null
}
```

## 5. `python -m evodevo.preflight --dry-run` -- exact output

```
run directory: /tmp/tmphsgzdfyp/20260921T025631Z_0dbcb505
stage                          cost       depends_on                     description
------------------------------------------------------------------------------------
machine_profile                free       -                              Machine profile (D5 B1): topology, RAM/disk incl. measured write rate, unconfigured-default thread behaviour, environment provenance, accelerator presence, CPU model/freq/governor, and D13 risk state.
concurrency_sweep              cheap      machine_profile                Concurrency sweep (D5 B2): real-PPO throughput/efficiency vs. worker count across --sweep-levels, on a fixed genome workload, with randomised visit order and inter-level cooldown to control for thermal drift.
evaluation_cost                expensive  concurrency_sweep              Evaluation cost (D5 B3): t_PPO per arm x task at full ppo_timesteps under B2's chosen concurrency, plus t_sentinel (invalid-genome fast path) and t_develop (decode-only cost), and the t_PPO/t_sentinel ratio.
learning_curves                expensive  evaluation_cost                Learning curves (D5 B4): fitness-vs-timesteps per representative body per task (one PPO training run per body, periodic-eval callback), the D6 knee-selection rule applied per task, and a WARNING wherever no plateau was observed within curve_max_timesteps.
evaluation_noise               expensive  evaluation_cost                Evaluation noise (D5 B5): repeated-PPO-seed fitness spread for low/middle/high-fitness bodies per task, pooled-within-body sd as a fraction of B3's between-body spread, and a plain-language D16 decision-input block. Determines whether the campaign can answer its own question -- treated as first-class, not an appendix.
projection                     free       learning_curves,evaluation_noise Projection table (D5 B6): expected_cost_per_eval per arm (validity_rate x t_PPO + (1-validity_rate) x t_sentinel), projected wall-clock with a measured range for the full CLI-supplied candidate grid, and D14 option-cost deltas. Never picks a configuration.
reliability                    free       projection                     Reliability aggregation (D5 B7): failures by class, crash rate per 1000 evaluations, peak-RSS-implied max safe concurrency, whole-run thermal/frequency behaviour, disk projection with the D10 3x margin, and a risk table mirroring B1's D13 register.
```

## 6. B1 machine profile (collected live -- read-only, seconds, NOT a measurement run)

```json
{
  "accelerator": {
    "cuda_available": false,
    "device_name": null,
    "ppo_runner_device": "cpu",
    "torch_importable": true
  },
  "cpu": {
    "base_freq_mhz": 3400.0,
    "freq_policy": {
      "energy_performance_preference_e_core": "performance",
      "energy_performance_preference_p_core": "performance",
      "epp_e_core_cpu_id": 16,
      "epp_p_core_cpu_id": 0,
      "idle_freq_mhz_by_core": {
        "E_core_cpu16": 800.0,
        "P_core_cpu0": 800.0
      },
      "intel_pstate_no_turbo": false,
      "intel_pstate_status": "active",
      "scaling_driver": "intel_pstate"
    },
    "governor": "performance",
    "max_freq_mhz": 5600.0,
    "model": "Intel(R) Core(TM) i7-14700K"
  },
  "environment": {
    "evogym_commit": "f63c649c",
    "evogym_version": "2.0.0",
    "gymnasium_version": "1.3.0",
    "lock_hash": "c1685531d6e6543215b012b204d820c15e1a9284bbd6b0097cdc01aa1224f7ed",
    "lock_hash_error": null,
    "numpy_version": "1.26.4",
    "project_dirty": true,
    "project_dirty_files": [
      "results/rule_based/ORDER_INDEPENDENCE_AUDIT.md",
      "PREFLIGHT_READINESS.md",
      "docs/PROJECT_LOOSE_ENDS_RESOLVED.md"
    ],
    "project_git_commit": "0dbcb50566caeceefee51a000e67b5c779224624",
    "python_version": "3.10.20 (main, Mar  3 2026, 09:24:47) [GCC 13.3.0]",
    "stable_baselines3_version": "2.9.0",
    "torch_version": "2.14.0+cpu"
  },
  "memory_disk": {
    "disk_free_root_gb": 307.25,
    "disk_free_run_dir_gb": 307.25,
    "disk_probe_mb": 64,
    "disk_write_rate_mb_s": 459.9,
    "ram_available_mb": 58426.9,
    "ram_total_mb": 67170.1,
    "sdb1_free_gb": null,
    "sdb1_mount_point": null,
    "sdb1_mounted": false,
    "swap_total_mb": 32768.0,
    "swap_used_mb": 0.0
  },
  "risk_state": {
    "coretemp_readable": true,
    "nvidia_driver_loaded": false,
    "sdb1_mounted": false,
    "sleep_targets": {
      "hibernate.target": "static",
      "hybrid-sleep.target": "static",
      "sleep.target": "static",
      "suspend.target": "static"
    },
    "swap_in_use": false,
    "swap_total_mb": 32768.0,
    "swap_used_mb": 0.0,
    "ulimit_c_hard": -1,
    "ulimit_c_soft": 0,
    "ulimit_n_hard": 1048576,
    "ulimit_n_soft": 1048576,
    "unattended_upgrades_enabled": true
  },
  "self_check_stamp": null,
  "topology": {
    "detection_method": "cpu_core_cpu_atom",
    "e_core_count": 12,
    "e_core_ids": [
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27
    ],
    "hybrid": true,
    "logical_count": 28,
    "p_core_logical_ids": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15
    ],
    "p_core_physical_count": 8,
    "p_core_physical_ids": [
      0,
      2,
      4,
      6,
      8,
      10,
      12,
      14
    ],
    "physical_count": 20
  },
  "unconfigured_threading": {
    "env": {
      "MKL_DYNAMIC": null,
      "MKL_NUM_THREADS": null,
      "NUMEXPR_NUM_THREADS": null,
      "OMP_DYNAMIC": null,
      "OMP_NUM_THREADS": null,
      "OPENBLAS_NUM_THREADS": null,
      "VECLIB_MAXIMUM_THREADS": null
    },
    "torch_num_threads": 20
  }
}
```

## 7. Self-check results

**This is a bounded infrastructure self-check, NOT a measurement. None of the numbers below may ever be cited as real B1-B7 results -- every artifact each run produced is independently stamped `SELF-CHECK — NOT A MEASUREMENT` in both its JSON and its figure titles.**

### --self-check-full (recommended pre-launch check -- all 3 arms x 2 tasks)

- Run directory: `results/preflight/selfcheck_full_20260921T023129Z`
- project_dirty at run start: `False`
- Stages done: 7/7, failed: 0
- Total measured wall clock across stages: 63.2s

| stage | status | duration_s |
|---|---|---|
| concurrency_sweep | done | 1.6182889938354492 |
| evaluation_cost | done | 35.85692238807678 |
| evaluation_noise | done | 9.799874544143677 |
| learning_curves | done | 13.598326444625854 |
| machine_profile | done | 2.2656519412994385 |
| projection | done | 0.06274676322937012 |
| reliability | done | 0.004318714141845703 |

Per-arm x per-task detail (B3 `evaluation_cost`):

| arm | task | n_successful | n_failed | partial |
|---|---|---|---|---|
| direct | UpStepper-v0 | 1 | 0 | False |
| direct | Walker-v0 | 1 | 0 | False |
| nca | UpStepper-v0 | 1 | 0 | False |
| nca | Walker-v0 | 1 | 0 | False |
| rule_based | UpStepper-v0 | 1 | 0 | False |
| rule_based | Walker-v0 | 1 | 0 | False |

### --self-check (fast smoke test -- single arm/task)

- Run directory: `results/preflight/selfcheck_20260921T023250Z`
- project_dirty at run start: `True`
- Stages done: 7/7, failed: 0
- Total measured wall clock across stages: 19.1s

| stage | status | duration_s |
|---|---|---|
| concurrency_sweep | done | 1.592296838760376 |
| evaluation_cost | done | 4.454457521438599 |
| evaluation_noise | done | 4.527298927307129 |
| learning_curves | done | 6.211830139160156 |
| machine_profile | done | 2.28702712059021 |
| projection | done | 0.06478166580200195 |
| reliability | done | 0.002911806106567383 |

What both prove: every registered stage's `run(ctx)` executed to completion at trivial settings, every stage wrote its JSON output, every figure-producing stage rendered its figure, and the manifest/event-log/telemetry-sampler machinery all round-tripped correctly -- --self-check-full additionally proves this for every arm x task combination the real pre-flight will exercise, not just one.

## 8. Environment provenance

- python: `3.10.20`
- numpy: `1.26.4`
- torch: `2.14.0+cpu`
- gymnasium: `1.3.0`
- stable_baselines3: `2.9.0`
- evogym: `2.0.0`
- evogym (external/evogym) commit: `f63c649c688a`
- this repo's commit: `0dbcb50566ca`

## 9. Known gaps, deviations, and things deliberately not implemented

### The evodevo/evaluation/ppo_runner.py change

`git status` shows `evodevo/evaluation/ppo_runner.py` as modified against HEAD. Prompt 2's own rule was: if the worker pool needed ppo_runner.py behaviour changed, stop and report rather than change it silently. This subsection is that report, produced after the fact once the omission was flagged -- not before the change was made.

**What changed:** `evaluate_body()`'s own inline rollout-evaluation loop (reset, deterministic policy loop, accumulate reward per episode, mean across `eval_episodes`) was extracted verbatim into a new function, `_rollout_fitness(model, eval_env, eval_episodes)`; `evaluate_body()`'s own call site now calls it instead of inlining the loop, with the same arguments in the same place. Separately, two wholly new, additive things were introduced: a `CurveResult` dataclass and a new function `evaluate_body_with_curve()`, which trains PPO once to `max_timesteps` and periodically evaluates the policy every `eval_interval` timesteps via an SB3 callback (reusing `_rollout_fitness` for each checkpoint). Neither is called by `evaluate_body()` or any other pre-existing caller; `evaluate_body_with_curve()` is used only by the B4 `learning_curves` pre-flight stage.

**Why:** B4 needs a fitness-vs-timesteps curve per body, measured periodically DURING one PPO training run rather than one independent full training run per checkpoint (which would multiply cost by the checkpoint count for no additional information). Reusing `_rollout_fitness` for every checkpoint means each curve point is evaluated with the exact same methodology `evaluate_body()`'s own final evaluation uses, so a curve's own final point is directly comparable to a normal `evaluate_body()` call on the same body/task/seed/timesteps.

**Classification: (i) behaviour-neutral.** `evaluate_body()`'s own validation, sentinel fitness, seeding, PPO construction (`PPO("MlpPolicy", vec_env, seed=seed, verbose=0)`), `model.learn(total_timesteps=ppo_timesteps)`, evaluation-episode count, fitness computation, exception handling, and cache/threading/device selection are all untouched -- only the already-inline rollout loop moved into a function with an unchanged call site and unchanged arguments. `evaluate_body_with_curve()` is new, additive code `evaluate_body()` itself never reaches.

**Proof:** `tests/evaluation/test_ppo_runner.py::test_evaluate_body_is_bit_identical_between_head_and_working_tree` loads the git-committed HEAD version of this module (via `git show HEAD:...` into a separately-named module -- not `git stash`, which would touch the whole working tree) and the current working-tree version, and evaluates a fixed set of 5 bodies (2 valid -- GOOD_WALKER, GOOD_WALKER_2 -- and 3 invalid, each via a different `validate_body()` failure: too_few_voxels, disconnected, no_actuator) with both, at `ppo_timesteps=2000, eval_episodes=1, seed=0`. All 5 produced bit-identical `fitness`, `valid`, `reason`, and `body_hash` between the two versions. The test passed and is kept in the suite as a permanent regression guard against a future edit to this module silently changing `evaluate_body()`'s own behaviour.

### coretemp_readable=true vs. ENVIRONMENT_REPORT.md's "lm-sensors not installed"

Not a contradiction -- two different things. B1's `coretemp_readable` (b1_machine.py's `_coretemp_readable()`) and telemetry.py's per-sample `cpu_package_temp_c` both come from `psutil.sensors_temperatures()`, which on Linux reads DIRECTLY from the kernel's `/sys/class/hwmon/hwmon*/temp*_input` sysfs files (confirmed by reading psutil's own Linux implementation source, whose docstring states this explicitly: "lm-sensors on Ubuntu 16.04+ relies on /sys/class/hwmon" -- i.e. even the real `sensors` CLI tool reads the same underlying sysfs interface). The `coretemp` KERNEL DRIVER is already loaded and exposes these hwmon nodes on its own; the `lm-sensors` PACKAGE ENVIRONMENT_REPORT.md reports absent is only the separate `sensors` command-line binary, a human-facing formatting convenience over the same sysfs data -- never required for psutil (or any Python code) to read a temperature.

**Verbatim sample record**, first line of `results/preflight/selfcheck_full_20260920T143859Z/telemetry.jsonl` (the latest --self-check-full run at the time this was written):

```json
{"cpu_package_temp_c": 49.0, "cpu_percent_overall": 0.0, "disk_free_gb": 307.27, "load_avg_1m": 1.0439453125, "process_count": 494, "ram_available_mb": 59850.4, "ram_used_mb": 7319.7, "swap_used_mb": 0.0, "thermal_throttle_count": 0, "ts": "2026-09-20T14:38:59.980600+00:00", "ts_epoch": 1789915139.9806}
```

`cpu_package_temp_c` is non-null and checked across all 56 samples in that run's telemetry.jsonl -- zero were null. No D13 gap in this instance. This is stated as an observation about THIS run, not a guarantee: if a future run's telemetry.jsonl ever shows a null `cpu_package_temp_c`, that IS a D13 gap (a sensor that stopped being readable mid-run) and must be reported as one, not glossed over as "readable" because B1's one-time startup check happened to succeed.

### The results/rule_based/ORDER_INDEPENDENCE_AUDIT.md test side effect

`tests/encoders/test_rule_based_order_independence.py::test_order_independence_permutation_audit` DELIBERATELY rewrites `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` every time it runs -- its own docstring states this explicitly ("Writes results/rule_based/ORDER_INDEPENDENCE_AUDIT.md as a side effect so this evidence is inspectable without re-running the (slow, N=2000xT=100) audit"). Only the wall-clock timing line changes run to run; the permutation/growth-conflict fractions themselves are deterministic and pinned by a companion test, `test_no_stale_order_independence_citations`, which asserts every OTHER file's citation of those numbers still matches. This is a documented, intentional side effect, not a bug -- and per this prompt's own instruction, it is reported here, not changed. Practice throughout this whole project has been to `git checkout -- results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` after every full-suite run to keep the tree clean of this one regenerated timing line.

- `docs/final_decisions.md` still does not exist in this repository. A later instruction asked for an uploaded `final_decisions.md` to be copied to `docs/final_decisions.md` and committed, so every D-number cited in the code has a document to diff against -- but no such file could be found anywhere on this machine's filesystem or in the accessible conversation context (searched exhaustively). This is reported as a genuine, outstanding blocker, not silently skipped or fabricated: every "D-number" referenced throughout `evodevo/preflight/` (D5-D16) remains implemented from the wording given directly in each build prompt, cited as the authoritative source in each module's own docstring, with no separate document to diff it against until this file is actually provided.
- D14 (per-option cost deltas), D15 (the primary EA population/generations/seed grid), and D16 (single-eval vs. multi-seed per genome) are left open BY DESIGN: `projection.py` (B6) and `evaluation_noise.py` (B5) surface the evidence (deltas, decision_input text) but never pick a configuration. These are resolved by a human (named "Tejeshwar" in the source prompts) at a scoping gate after this pre-flight runs for real.
- The dashboard's ETA range (`dashboard.py`'s `Dashboard._eta_str`) is a crude heuristic -- elapsed / completed_fraction, presented as a [0.5x, 1.5x] band -- not a calibrated estimator. It gets meaningfully better once B2/B3 have real timing data; before that, it is a rough placeholder, labeled as such ("rough range, not a commitment").
- `--self-check` still restricts `--arms`/`--tasks`/`--proj-arms`/`--proj-tasks` to a single arm (`direct`) and single task (`Walker-v0`), kept as the fastest possible smoke test (well under 3 minutes). `--self-check-full` (added to close this gap) uses the identical trivial numeric settings across the FULL production grid (direct, nca, rule_based x Walker-v0, UpStepper-v0) instead, and is the one operators should run before a real launch -- see `scripts/preflight/self_check.sh --full`. Measured at 64s against a 10-minute budget, all 6 combinations succeeded with zero partial/failed results (see the self-check-full run directory cited in Section 7).
- **Why the full grid did not originally fit in 3 minutes:** NOT valid-genome search (direct, nca and rule_based all found their first valid random genome within a few dozen draws at worst, well under `--cost-max-draws`). The real cause was `rule_based`'s `decode()` cost -- measured directly at ~232ms/call versus `direct`'s ~0.001ms/call and `nca`'s ~1.8ms/call (a ~200,000x gap vs direct), because it runs an iterative rule-interpretation simulation over `dev_steps=100` growth steps per call, not a cheap array reshape. B3's `t_develop`/`t_sentinel` measurements (`_measure_develop`/`_measure_sentinel` in `b3_cost.py`) paid this cost `t_develop_n` (a FIXED 100 calls, always paid in full) plus up to `cost_max_draws` times (searching for `t_sentinel_n` invalid samples) -- for rule_based alone, up to roughly (100 + 200) x 0.232s =~ 70s of pure decode time, on top of the real PPO evaluations, per arm. `t_develop_n`/`t_sentinel_n` were not exposed as CLI flags at all before this fix, so self-check had no way to reduce them; they now are (`--cost-develop-n`/`--cost-sentinel-n`, both defaulting to the previous fixed value of 100 for production use), and self-check reduces both to 5.
- `report.py` regenerates figures by calling each stage module's own private `_make_figure`/`_write_markdown`/`_write_csv` functions directly from saved JSON; it does not independently reimplement any plotting logic, so a bug in a stage's own figure code is NOT caught twice.
- `readiness.py`'s wall-clock estimate (Section 10) is not derived from any real B2/B3 measurement (none exist yet) -- it is a rough, explicitly-caveated order-of-magnitude number from this machine's known real-PPO throughput at a small scale, meant to be superseded by B6's own `projection.md` once the real pre-flight actually runs.

## 10. Estimated pre-flight wall clock (range)

No real B2/B3 measurement exists yet (producing one is the pre-flight's own job), so this is a rough, explicitly-caveated planning number, not a measured estimate:

- **Assumption:** ~2.5-4ms of wall clock per PPO timestep, single evaluation, unconcurrent (from this machine's `test_real_ppo_smoke`-scale observations during earlier development), with no correction yet for B2's own concurrency overhead factor -- concurrent workers will not scale linearly, and B6's `projection.md` (produced by the real pre-flight itself) supersedes this number.
- At default settings (sweep: 6 levels x 24 genomes x 20,000 steps; cost: 3 arms x 2 tasks x 3 reps x 1,000,000 steps; curves: 3 bodies x 2 tasks x 2,000,000 steps; noise: 3 bodies x 2 tasks x 5 reps, timesteps from B3's own cost data) the dominant terms are B3 (~18M PPO-timesteps) and B4 (~12M PPO-timesteps); B2/B5 are smaller. At assumed 2.5-4ms/step *single-worker-equivalent*, and assuming B2's own measured concurrency roughly divides that by somewhere between 4x and 16x (a wide, deliberately conservative range, since this machine has never been measured under this exact load): **roughly 6-30 hours.**
- **The two phases are not the same order of magnitude, and an operator watching tmux should expect the pace to change sharply partway through:** B2 (`concurrency_sweep`, at default settings 6 levels x 24 genomes x 20,000 reduced timesteps, plus 60s inter-level cooldowns) is the cheap phase -- on the order of **minutes to about an hour**. B3/B4/B5 (`evaluation_cost`, `learning_curves`, `evaluation_noise`, at FULL campaign timesteps -- 1,000,000 for B3, 2,000,000 for B4) are the expensive phases and dominate the total -- on the order of **hours**, not minutes. If B2 finishes in the first hour and the dashboard's ETA suddenly looks far larger afterward, that is the expected transition into B3, not a stall.
- This range is wide on purpose -- it exists only to plan a tmux session and disk headroom around, not to promise a duration. Treat B6's `projection.md`, produced from real B2/B3 numbers, as authoritative once available.

## 11. Open questions for a human decision

1. D14/D15/D16 (option costs, the primary EA grid, single- vs multi-seed evaluation) -- resolved by Tejeshwar at the scoping gate, using this pre-flight's B5/B6 output as evidence.
2. **RESOLVED:** the self-check must honour the full arm/task grid. `--self-check-full` does this (all 3 arms x 2 tasks, same trivial numeric settings as `--self-check`, 10-minute budget; measured at 64s with zero partial/failed combinations) and is now the recommended pre-launch check -- see Section 9's own subsection on why the original 3-minute `--self-check` could not simply be widened to the full grid (rule_based's real decode() cost) and Section 7 for its own run's detail.
3. `--timeout-s` currently sends this process SIGTERM (the same path as an operator's own Ctrl-C/kill), stopping cleanly with a resumable manifest rather than continuing past the deadline -- confirm this is the intended semantics for an unattended, timed run.
4. The dashboard's ETA range is a crude [0.5x, 1.5x] heuristic around a linear extrapolation of completed-stage-fraction; confirm this rough band is acceptable for an operator watching tmux for several hours, or whether a tighter, cost-weighted estimator is worth building.
5. Disk headroom: B7's disk projection (D10's 3x margin) has not yet been checked against this machine's real available space under a full run's expected artifact volume -- worth confirming before an unattended multi-hour run starts.
