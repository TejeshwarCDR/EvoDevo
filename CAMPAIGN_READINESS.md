# EvoDevo Campaign Readiness

Generated 2026-09-28 by manual pre-launch validation (this document), following the
template and terminology of `PREFLIGHT_READINESS.md` (2026-09-21). **The real
campaign has NOT been started.** Only the bounded validation activity described
below was executed: the full test suite, two small real/surrogate mini-campaigns
(one real-PPO, one surrogate, both destroyed after use), a real-PPO serial vs.
interleaved determinism check, a notifier test, and a read-only plan/estimate
dump for the real `campaign` profile. Every validation artifact was generated
under `results/campaign_validation/` (untracked, deleted at the end — see
Section 7) and stamped, in this document, "VALIDATION — NOT A RESULT."

**Headline finding: the real campaign, as currently coded, cannot execute
through its own launcher.** `scripts/campaign/run_campaign.sh --mode real`
reproducibly crashes every worker process at startup (`BrokenProcessPool`).
Root cause, reproduction, and scope are in Section 6. This is the sole
blocking item; verdict is **NOT READY** (Section 12).

---

## 1. Git state and config lock

- HEAD at the start of this validation: `1a506e9fbc89d9b4a8ba44820e2f1add84831642`
  ("Label legacy snapshot summary and manifest as partial")
- Working tree: clean throughout (`git status --porcelain` empty before, during
  — modulo the deliberate `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md`
  side-effect from the test suite, reverted with `git checkout --` immediately
  after — and after this validation)
- `evodevo.experiments.lock_hash.compute_lock_hash_from_benchmark_config()` on
  the real `evodevo/experiments/configs/benchmark.yaml`:
  `e5c2aa7547d4e08e6e0dad9e57f5a6e928355b3c671606d867f7709b404af638`
- `CampaignPlan.from_benchmark(...).config_hash` (the scheduler/manifest-level
  hash, a different, complementary hash over the fully-resolved run plan):
  `d32171356cedb8e0e9e0e851aa3bd2234fbbdb03c0a17cf44e833dfeefd56fc4`
- This document itself is the only file committed as part of this validation
  (see Section 7); `docs/decisions.md` was read only, never modified.

## 2. Decisions locked (docs/decisions.md, all dated 2026-09-27, decided by Tejeshwar)

| Decision | Status | Outcome |
|---|---|---|
| D14 scoping gate | ACCEPTED | No options adopted (more seeds / second grid size / extra arms / larger budget all rejected — over the 432h budget or under-validated) |
| D15: population/offspring/budget/seeds/arms/task order | ACCEPTED | `population_size=15, offspring_size=15, eval_budget=300` genomes (20 generations); seeds 0-4; arms `direct, nca, rule_based, rule_based_monotone`; tasks Walker-v0 then UpStepper-v0 (D7 strict phases) |
| MONO: `rule_based_monotone` co-primary | ACCEPTED | Co-primary on both tasks, 5 seeds each, alongside the other 3 arms |
| D6 application: per-task `ppo_timesteps` | ACCEPTED | Walker-v0 = 1,000,000; UpStepper-v0 = 1,200,000 |
| D6 fallback rule for Walker-v0 | ACCEPTED (deviation from pre-registration, disclosed) | Written fallback rule applied because the registered rule returned no knee for Walker-v0; result 1,000,000 |
| D16 evaluation-noise policy | ACCEPTED (supersedes a same-day PROPOSED row) | Walker-v0 K=1; UpStepper-v0 K=2, genome fitness = MAX over completed PPO seeds, identical across arms; every per-seed value logged |
| Concurrency: 20 workers | ACCEPTED | 8 P-core primaries + 12 E-cores, `evodevo.parallel.topology.cpu_slots(topology, 20)`, from B2's measured throughput knee |
| **Budget guard, and its outcome** | ACCEPTED | Rule: if Step-4 p95 total > 432h, descope (a) monotone-Walker-only, (b) drop monotone, (c) UpStepper K=1. **Outcome: NONE applied.** p95 = 406.0h ≤ 432h (mean 403.2h, p5 400.6h). Confirmed independently in this validation: `CampaignPlan.from_benchmark(...).guard_outcome == "all four approved campaign arms"` |
| Infrastructure-failure policy | ACCEPTED | Crash/timeout retried up to 2x on a fresh worker; still-failing evaluations recorded `reason="infra_failure"`, sentinel for selection, counted and alerted separately if rate > 0.5% |

No row relevant to launch readiness is left OPEN. (D14/D15/D16 were themselves
the resolution of the earlier-disclosed OPEN items in `PREFLIGHT_READINESS.md`
Section 11.)

## 3. Pre-registered analysis plan

- Pre-registered in `docs/decisions.md` by commit `f2689b1dbdcba2745160bc4d2a14c83bc021b63a`
  ("Pre-register the campaign analysis plan in decisions.md"), **2026-09-27 17:16:48 +0530**.
- Campaign decisions locked by commit `a27c61a70d8b56a11a5376b4c4c6c1cf80fef970`
  ("Lock the campaign decisions: MONO, D14, D15, D16, concurrency, budget guard,
  infra failures"), **2026-09-27 20:57:57 +0530**.
- Campaign profile encoded by `c7172d1e4d8c2d9f58235c508245089cfd19d334` and the
  final capacity estimate by `77bc842291623b4a060151ddc2d5b05d6a7845a3`, both
  **2026-09-27 20:58:xx +0530**.
- All four commits predate this validation (2026-09-28) and predate any launch
  of the real campaign (which has not happened) — the analysis plan is fixed
  before data collection, as pre-registration requires.

## 4. Test results

- **Full suite (including slow/real-PPO tests): `1021 passed, 33 warnings in 820.19s (0:13:40)`, exit code 0.**
  Command: `.venv/bin/python -m pytest -q` (testpaths = `tests`, per `pytest.ini`;
  `slow` marker tests — real PPO training — included, not skipped, since evogym
  is built in this environment).
- One expected, documented side effect: `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md`
  is deliberately rewritten by `test_order_independence_permutation_audit` on
  every run (its own docstring says so); reverted with `git checkout --`
  immediately after, per standing practice recorded in `PREFLIGHT_READINESS.md`.
- Named results, all within the 1021 passed:
  - **Pinned EA bit-identity** (`evaluate_body()` HEAD vs. working tree):
    `tests/evaluation/test_ppo_runner.py::test_evaluate_body_is_bit_identical_between_head_and_working_tree`
    — passed (part of the full run; this specific guard was not re-isolated
    since HEAD was unmodified for this validation).
  - **K=1 bit-identity / real K=2 policy reload**:
    `tests/evaluation/test_multiseed.py::test_real_k2_saves_two_reloadable_policies`
    — passed (real PPO, K=2, both policies saved and reloaded, confirmed
    present in the suite's collected/executed test list for this run).
  - **Serial ≡ interleaved, surrogate**:
    `tests/campaign/test_scheduler_resume.py::test_surrogate_shared_phase_matches_serial_and_resume`
    — passed (pre-existing, committed).
  - **Serial ≡ interleaved, REAL** (this validation's own addition, not
    committed to the repo — see Section 5 below for the full run): passed,
    real PPO, 4 (arm, seed) combinations, 36 evaluations, fitness/valid/
    body_hash/reason bit-identical on every evaluation between `run_ea`
    (serial) and `CampaignScheduler` (4-worker interleaved), including
    checkpoint `evaluations_used` agreement.
  - **Resume ≡ uninterrupted, surrogate**: same
    `test_surrogate_shared_phase_matches_serial_and_resume` test, second half
    (re-running a completed phase is a no-op byte-for-byte) — passed. Also
    `tests/campaign/test_scheduler_resume.py::test_resume_repairs_log_rows_not_covered_by_checkpoint`
    (crash-injected resume) — passed.
  - **Resume ≡ uninterrupted, REAL**: NOT achieved through the campaign
    launcher — see Section 6 (blocking bug). Partially substituted by this
    validation's own real kill/resume test in **surrogate** mode through the
    actual `run_campaign.sh` (Section 5) plus the pre-existing, passing
    `tests/preflight/test_stages.py::test_kill_during_stage_is_rerun_on_resume_while_done_stage_is_skipped`
    and `evodevo/campaign/checkpoint.py`'s own real-PPO-agnostic
    checkpoint/reconciliation logic (exercised directly in Section 5's kill
    test, which is orchestration-layer, not PPO-content, correctness).
  - **Checkpoint round-trip**: `evodevo/campaign/checkpoint.py::write_checkpoint`/
    `latest_checkpoint` exercised by every scheduler test above and directly
    observed in this validation's Section 5 kill/resume run (checkpoint
    `gen_2.json` present, `run_log.jsonl` correctly truncated to the
    checkpoint-covered row count on resume).

## 5. Mini-campaign results per stage

### 5a. Real-PPO mini-campaign through `run_campaign.sh --mode real` — FAILED (launcher bug)

Config: 4 arms (`direct, nca, rule_based, rule_based_monotone`) × 2 seeds ×
2 tasks, `population_size=15, offspring_size=15, eval_budget=30` (2
generations), `ppo_timesteps=2000` both tasks, `ppo_seeds` K=1 (Walker-v0) /
K=2 (UpStepper-v0), 20 workers, `--mirror-dir` a temp directory, launched via
`bash scripts/campaign/run_campaign.sh --mode real ...` in a real, isolated
tmux session (`EVODEVO_TMUX_SOCKET=evodevo_validation`). `CampaignPlan`
resolved correctly (8 runs × 2 phases, correct K/timesteps per task, D7 order,
`guard_outcome="all four approved campaign arms"`) — the plan/config layer is
sound. The scheduler process launched, wrote its manifest, and began
dispatching PPO tasks (`dispatch_status.json` showed `tasks_running: 20`), but
**every one of the 20 spawned worker processes crashed immediately** with:

```
evodevo.parallel.threading_control.ThreadingConfiguredTooLateError: configure_threading(strict=True)
was called after numpy had already been imported. ...
concurrent.futures.process.BrokenProcessPool: A child process terminated abruptly,
the process pool is not usable anymore
```

Zero evaluations completed; `journal.jsonl` was never created; every
`run_log.jsonl` stayed at 0 rows. This is a genuine, reproducible defect —
see Section 6 for root cause. **The real-mode mini-campaign could not be
validated end-to-end through the real launcher.**

### 5b. Surrogate mini-campaign through `run_campaign.sh --mode surrogate` — PASSED (isolates the bug to real-mode worker bootstrap)

Same grid (4 arms × 2 seeds × 2 tasks, pop 15, eval_budget 30), `surrogate:
true`, launched the same way, in tmux, through the real launcher. Explicitly
labeled: **this validates orchestration only, not real PPO.**

- **D7 order confirmed**: `phase_status` timestamps —
  Walker-v0 `started_at=...086.37` → `ended_at=...116.61`; UpStepper-v0
  `started_at=...118.27` (after Walker-v0 ended) → `ended_at=...148.58`.
  Walker-v0 always completes before any UpStepper-v0 run starts.
- **Automatic Walker milestone package at the phase boundary**: manifest's
  `milestone_status["Walker-v0"]` = `{"ok": true, ...}`, generated with no
  operator action, immediately on Walker-v0 phase completion. Every file in
  `milestone.py`'s `REQUIRED` tuple present, plus per-arm best-body artifacts:
  `MILESTONE_REPORT.md`, `figures/{fitness_curves,budget_curves,validity_diversity}.{png,pdf}`,
  `stats/{omnibus,pairwise,final_fitness_summary,validity_diversity_by_generation,
  current_rates,final_best_morphology,evaluations_to_90pct_arm_median,
  rule_based_vs_monotone,tables}.{csv,md,tex}`, `best_bodies/{direct,nca,
  rule_based,rule_based_monotone}.{npz,png}` — `completion_fraction: 1.0`.
  Same for UpStepper-v0 (plus its K=2 `mean_of_2_*` sensitivity tables).
- **Mirror verified**: `evodevo.campaign.mirror.verify_checksum_manifest()`
  against the synced snapshot: `{"ok": true, "missing": [], "altered": [],
  "extra": [], "files": 130}`.
- **Every evaluation has a body and a genome; journal count**: not applicable
  in surrogate mode — `ResultJournal.append_task` is only called for real PPO
  tasks (`evodevo/campaign/scheduler.py::_phase_round`); surrogate results go
  straight to `run.tell()`. This is expected scheduler behavior, not a defect,
  but it means the journal-count-equals-expected-PPO-tasks check is
  necessarily unverified by the surrogate run — see Section 5a/6 for why the
  real-mode run that would have verified it could not complete.
- **Kill/resume, surrogate**: a second, larger (`eval_budget=3000`, same
  grid) surrogate run was launched the same way, killed mid-generation with
  `kill -KILL -- -$PGID` on the scheduler's own process group (recorded
  `scheduler.pgid`) after 3 generations had been told (checkpoints
  `gen_0.json`..`gen_2.json` present, 360/journal-equivalent rows written),
  confirmed dead (`ps -p $PGID` exit 1), then resumed with
  `run_campaign.sh --resume ...`. The resumed scheduler correctly picked up
  at generation 3: `run_log.jsonl` for `direct/Walker-v0/seed_0` held exactly
  45 rows (3 × 15, matching the last checkpoint), i.e. any partially-written
  rows from the killed generation were correctly truncated by the
  checkpoint-reconciliation logic in `CampaignScheduler._make_run`.
- **`stop_campaign.sh` against a live campaign**: run against the resumed,
  still-running surrogate campaign. Output: `"Stopped campaign
  VALIDATION-surrogate-1; no evodevo campaign processes remain."`, exit 0.
  Confirmed independently: `pgrep -af "campaign.(scheduler|monitor|dashboard)"`
  empty, `tmux has-session` returns "no server running" — **zero surviving
  processes.**

### 5c. Real-PPO serial ≡ interleaved determinism (Step 4 equivalent, real PPO)

Since the campaign launcher cannot run real mode (Section 6), this was
verified directly against `CampaignScheduler` and `run_ea`, bypassing only
the broken `-m evodevo.campaign.scheduler` subprocess entry point (invoked
in-process via `pytest`, exactly how the repo's own committed
`test_real_ppo_pool_smoke` already avoids the same bug — see Section 6).
Config: 2 arms (`direct, nca`) × 2 seeds, Walker-v0, `population_size=3,
offspring_size=3, eval_budget=9` (3 generations), `ppo_timesteps=2000`,
4 real workers. Result: **passed**. All 4 (arm, seed) combinations × 9
evaluations = 36 real evaluations compared field-by-field
(`fitness, valid, body_hash, reason`) between the campaign's 4-worker
interleaved run and a serial `run_ea` baseline reconstructed from the same
`ExperimentConfig`: **bit-identical on every field, every evaluation**,
including several genuinely-trained real-PPO fitness values for valid bodies
(e.g. `nca/seed_1`: `-0.0407354827..., -0.0478564176..., -0.0418130451...,
-0.0084011282..., 0.0056550879..., 0.0060542097...`, matched to full float
precision) — not only sentinel-fitness (invalid-genome) rows. Checkpoint
`evaluations_used` also matched. This directly confirms the scheduler's own
stated invariant ("completion order affects throughput only") holds under
real PPO, for the code paths that do work.

## 6. The real-mode launcher bug — root cause

**Reproduction**: `bash scripts/campaign/run_campaign.sh --mode real ...`
(exactly the documented/only way to launch a real campaign) → every worker
process raises `ThreadingConfiguredTooLateError` in
`evodevo/parallel/threading_control.py::configure_threading(strict=True)`,
called from `evodevo/parallel/worker_pool.py::_worker_initializer`, which is
supposed to be the FIRST thing any worker does, before numpy/torch are ever
imported (by design — see that module's own docstring). Every worker fails
identically; the pool immediately reports `BrokenProcessPool`, and the
scheduler process dies with an unhandled exception before completing even
generation 0 of the first phase.

**Root cause, isolated directly**:
```
$ .venv/bin/python -c "import sys; import evodevo.campaign; print('numpy' in sys.modules)"
True
```
`evodevo/campaign/__init__.py` does `from .scheduler import CampaignScheduler`
at package import time, which transitively imports
`evodevo.evaluation.ppo_runner` (hence numpy/torch) eagerly. Under
`multiprocessing`'s required `spawn` start method (this pool deliberately
never uses `fork` — see `worker_pool.py`'s own docstring on why), every
spawned worker must reconstruct the parent's `__main__` module before running
the pool's `initializer`. Because `run_campaign.sh` launches the scheduler as
`python -m evodevo.campaign.scheduler`, that module **is** `__main__` for
every worker — so reconstructing it re-runs `evodevo.campaign`'s eager
package import (and `scheduler.py`'s own top-level imports of `ppo_runner`,
`EvolutionRun`, etc.) unconditionally, before `_worker_initializer`'s
`configure_threading(strict=True)` ever gets to run. Confirmed by the
`RuntimeWarning` present in every failure: `'evodevo.campaign.scheduler'
found in sys.modules after import of package 'evodevo.campaign', but prior
to execution of 'evodevo.campaign.scheduler'`.

**Why the existing test suite (1021 passed) did not catch this**: the only
committed test that launches a real campaign end-to-end through
`run_campaign.sh` in a real tmux session is
`tests/campaign/test_operations.py::test_launcher_starts_and_cleans_tiny_surrogate_campaign`
— and, as its name says, it uses `--mode surrogate`, which never constructs
an `EvaluationPool` at all (surrogate evaluation runs inline in the scheduler
process). The suite's real-PPO worker-pool tests
(`tests/parallel/test_worker_pool.py::test_real_ppo_pool_smoke` and friends)
construct `EvaluationPool` directly from a pytest-collected test module,
never via `python -m evodevo.campaign.scheduler` — so `__main__` for their
spawned workers is pytest's own launcher (which does not eagerly import
`evodevo.campaign`), and the bug never triggers. The gap is precisely the
combination "real mode" + "launched via the actual CLI entry point" — a
combination no committed test exercises.

**Scope**: this affects every real-mode invocation of
`scripts/campaign/run_campaign.sh` (and `--resume`, which uses the identical
code path), independent of worker count, arm, task, or config — reproduced
identically at `--workers 20` (the mini-campaign) and confirmed by direct
invocation of `python -m evodevo.campaign.scheduler` outside tmux with a
60s timeout. It does **not** affect: surrogate-mode campaigns, the plan/
estimate layer (`CampaignPlan`, `resolve_configs`), `stop_campaign.sh`,
`status_campaign.sh`, the milestone/mirror/alert machinery, or PPO
training/evaluation itself when invoked through any other entry point (as
Section 5c demonstrates).

**No production code was modified as part of this validation** — diagnosing
and fixing this is engineering work outside a validation task's scope; it is
reported here, with full reproduction, root cause, and a concrete pointer
(`evodevo/campaign/__init__.py`'s eager `from .scheduler import
CampaignScheduler`, combined with `run_campaign.sh`'s `python -m
evodevo.campaign.scheduler` invocation), for the team to fix and re-validate.

## 7. Monitoring, alerts, and mirror evidence

- **Mirror**: verified in Section 5b (`verify_checksum_manifest` → `ok: true`,
  130/130 files). Sync also began correctly during the real-mode attempt
  (Section 5a) before the worker crash — `mirror_checksums.json` was written
  — confirming `sync()` itself does not depend on real vs. surrogate mode.
- **Alerts**: `phase_complete`, `milestone_failure`-style durable, deduplicated
  alert journaling exercised implicitly by the surrogate mini-campaign
  (`alerts.jsonl` populated with `phase_complete:Walker-v0` and
  `phase_complete:UpStepper-v0` informational events). Six-hour dedup,
  heartbeat, and thermal/disk/swap/infra-failure-rate conditions are
  exercised by the pre-existing, passing `tests/campaign/test_operations.py`
  suite (14 tests) rather than re-derived here.
- **Notifier test (Step 5)**: `python -m evodevo.campaign.alerts --test` was
  run. **No notifier is configured**: `campaign.local.yaml` does not exist
  anywhere in the repository or its parent directories. `Notifier.send()`
  guards both its delivery paths (`ntfy_topic_url`, `shell_command`) behind
  `if` checks against the (empty) loaded config, so the call is a silent
  no-op — **and the CLI still prints `"Test notification sent"` and exits 0**
  even though nothing was sent. This is a minor, disclosed observability gap:
  an operator running this exact command before launch could reasonably
  believe a real test notification went out when none did. No message was
  actually delivered anywhere.

## 8. Final estimate (real `campaign` profile, from the already-committed, pre-registered `results/campaign_prep/final_estimate/final_estimate.json`)

Guard outcome (independently reconfirmed against the live `benchmark.yaml` in
this validation, Section 1/6): **no descope applied — the campaign runs the
full 4-arm, 5-seed, both-task design.**

| | mean | p5 | median | p95 | sd |
|---|---|---|---|---|---|
| **Total** | 403.2 h (16.8 d) | 400.6 h | 403.2 h | **406.0 h (16.9 d)** | 1.6 h |
| Walker-v0 phase | 123.0 h (5.1 d) | 121.9 h | 123.0 h | 124.1 h | 0.7 h |
| UpStepper-v0 phase | 280.2 h (11.7 d) | 277.8 h | 280.1 h | 282.9 h | 1.5 h |

- **Walker milestone**: 30.5% of total runtime (of means: 30.50%; per-trial
  mean 30.50%, p5 30.24%, p95 30.76%) — i.e. the first, automatically-generated
  paper-ready milestone package lands roughly **5.1 days** into the campaign.
- **Sensitivities** (final configuration, still all under the 432h guard):
  pure-linear timestep scaling 409.0h; no measurement-condition correction
  425.2h; resampled per-seed validity trajectories 403.1h (p95 414.6h, wider
  spread as expected from resampling).
- **Illustrative calendar** (no specific launch date was given to this
  validation; shown for a launch at **00:00 on 2026-09-29**, i.e. the day
  after this report, purely as an example — recompute against the actual
  planned date before launch):
  - Launch: 2026-09-29 00:00
  - Walker-v0 milestone (mean): ~2026-10-04 03:00 (+123.0h)
  - Walker-v0 milestone (p95): ~2026-10-04 04:06 (+124.1h)
  - Campaign complete (mean): ~2026-10-15 19:10 (+403.2h)
  - Campaign complete (p95): ~2026-10-15 22:00 (+406.0h)

## 9. Disk projection vs. free space

From the already-committed `results/campaign_prep/disk/disk_estimate.json`
(40 runs, 6,000 Walker-v0 genomes, 6,000+12,000 UpStepper-v0 PPO trainings at
K=1/K=2): **total projected artifact volume ≈ 32.85 GB**; D10's 3× margin
rule requires **≈ 98.56 GB** free. That estimate recorded `free_bytes_root =
302.13 GB` at the time it was produced.

Re-measured live, in this validation (`df -h`):

| Mount | Device | Total | Free | 3× (98.56 GB) satisfied? |
|---|---|---|---|---|
| Main (`/`) | `/dev/sda7`, ext4 | 401 G | **281 G** | Yes (2.85×98.56GB headroom) |
| Candidate mirror (`/media/ubuntu/...`) | `/dev/sda3`, ntfs3 | 454 G | **129 G** | Yes, but narrowly (1.31×) |

**Gap, disclosed plainly**: no dedicated second physical disk is currently
mounted for mirroring (B1's own `sdb1_mounted: false` in
`PREFLIGHT_READINESS.md` still holds). The only other mounted volume,
`/dev/sda3`, is a **different partition of the same physical disk** (`sda`)
as the main filesystem, not independent storage — so today's only available
`--mirror-dir` target provides no real hardware-failure redundancy, only
protection against directory-level accidents (e.g. an operator `rm -rf`
inside the wrong tree). `--mirror-dir` is a required, operator-supplied flag
at launch time (not part of `benchmark.yaml`), so this is confirmed, not
assumed — an operator must explicitly point it somewhere before launch, and
should be told this is not a second disk.

## 10. Exact operator commands

```
# Launch (from repo root, config clean, in tmux)
bash scripts/campaign/run_campaign.sh \
  --config evodevo/experiments/configs/benchmark.yaml \
  --output-root results/campaign \
  --mirror-dir /path/to/a/real/second/disk \
  --workers 20 --mode real
# BLOCKED as of this validation -- see Section 6. Do not run until fixed.

# Attach to the running tmux session
tmux attach -t evodevo-campaign

# Status (one-shot)
bash scripts/campaign/status_campaign.sh --campaign latest --output-root results/campaign

# Graceful stop
bash scripts/campaign/stop_campaign.sh --campaign latest --output-root results/campaign --grace-s 120

# Resume after any interruption
bash scripts/campaign/run_campaign.sh --resume latest \
  --config evodevo/experiments/configs/benchmark.yaml \
  --output-root results/campaign \
  --mirror-dir /path/to/a/real/second/disk \
  --workers 20 --mode real

# Manual milestone regeneration for a phase (isolated subprocess, same
# mechanism the scheduler calls automatically at each phase boundary)
.venv/bin/python -m evodevo.campaign.milestone --campaign-root results/campaign/<ID> --task Walker-v0
```

## 11. Known gaps and accepted risks, stated plainly

1. **BLOCKING**: the real-mode campaign launcher is broken (Section 6). The
   real campaign cannot start successfully today.
2. The journal-count-equals-expected-PPO-tasks check (Step 2's own
   acceptance bullet) could not be verified against a real-mode run through
   the launcher, for the same reason (#1). It *is* structurally verified by
   `ResultJournal`'s own logic (`_phase_round` appends exactly one journal
   record per `(run_key, eval_index, ppo_k)` PPO task) and was observed
   correctly populating during Section 5c's in-process real-PPO run, but not
   from a `run_campaign.sh`-launched run.
3. No notifier is currently configured (`campaign.local.yaml` absent); the
   `--test` CLI reports success even when nothing is sent (Section 7) —
   worth fixing before relying on alerting during an unattended multi-day run.
4. The only available `--mirror-dir` target today is a different partition
   of the same physical disk as the main filesystem, not true off-disk
   redundancy (Section 9); its free space (129 GB) clears the 3× rule only
   narrowly (1.31×) and was not itself part of the original disk-estimate
   artifact's own accounting.
5. `PREFLIGHT_READINESS.md`'s own outstanding items are still open: `docs/
   final_decisions.md` still does not exist in this repository (Section 9 of
   that document); D14/D15/D16 are, however, now resolved in `docs/
   decisions.md` (Section 2 above), superseding that specific gap.
6. This validation's real-PPO evidence (Sections 4-5c) is necessarily at
   trivial scale (2,000 PPO timesteps vs. the real campaign's 1,000,000/
   1,200,000) and small grids (2-8 runs vs. 40) — appropriate for pipeline
   correctness, not for re-deriving any of the campaign's own capacity/
   timing numbers, which come from the separate, already-committed
   `results/campaign_prep/` measurement chain (Section 8).
7. A pre-existing, unrelated tmux session (`evodevo-preflight`, created
   2026-09-22, well before this validation) remains on the machine's default
   tmux socket. It was not created by this validation and was left
   untouched, since it is not part of this task's scope and destroying
   other work without being asked would be unsafe.

## 12. Verdict: **NOT READY**

**Blocking item**: Section 6's real-mode launcher bug. The real campaign
cannot be started with `scripts/campaign/run_campaign.sh --mode real` today —
every worker crashes at initialization, `BrokenProcessPool`, zero
evaluations complete. This is a simple, well-isolated, well-understood bug
(a package-level eager import defeating `spawn`-based worker bootstrap when
launched via `python -m evodevo.campaign.scheduler`), not a design or
capacity problem — everything else validated in this document (the plan/
estimate layer, D7 phase ordering, automatic milestone generation, mirror,
resume-after-kill, `stop_campaign.sh`, real-PPO determinism, real-PPO policy
save/reload, the full test suite) passed. Once the launcher bug is fixed,
re-run Section 5a (the real-mode mini-campaign) specifically — every other
section of this validation should carry forward unchanged.

**Do not launch the real campaign until Section 6 is fixed and Section 5a is
re-run successfully end-to-end through the real launcher.**
