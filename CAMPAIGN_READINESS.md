# EvoDevo Campaign Readiness

## Regenerated 2026-09-29 — final pre-launch status

**Verdict: READY, subject only to the operator-owned launch checklist. The real
campaign has NOT been started.** This current status supersedes the historical
2026-09-28 report retained below for provenance.

- Git readiness checkpoint: `d455a83` plus `fcdaae2`; tree clean after the
  report commit. The pre-registered plan remains unedited. Continuous
  per-run dispatch is ADR-conformant; the unchanged guard passed at 403.2 h
  mean / 405.9 h p95 (base p95 <= 432 h), Walker milestone 30.5%.
- Step 5 conformance passed: Walker 4,750.776 s vs 4,766.727 s simulated
  (0.335%), 99.692% vs 99.536% steady utilization; UpStepper 4,410.275 s vs
  4,458.073 s (1.072%), 98.495% vs 97.854%. Both are inside the 5% / 3-point
  gates. The remote-aware idle gate retains competing-work detection while
  allowing bounded AnyDesk/GNOME/Xorg/VS Code/Claude/Codex baseline activity.
- Step 6 re-proofs passed: 12,000 surrogate records and 480 real-PPO records
  matched serial references; 20 surrogate and 3 real SIGKILL resumes matched;
  10/10 graceful stops completed in 0.048–1.913 s with no survivors; crash,
  timeout, hash/config refusal paths passed.
- Step 7: 20 idle + 20 eight-busy-loop `tests/campaign` runs passed (40
  consecutive). The old intermittent failure was not reproduced and is
  explicitly unproven, not claimed fixed.
- Step 8: dashboard/heartbeat worker utilization, sustained low-utilization
  and >40 GiB worker-RSS alerts, NTFS-safe tree checks, transcript projection
  (0.13 MB/18 d), `--local-config`, commented local-config example, and
  dry-run operational facts are implemented. Long-task/RSS sampling limits
  are documented in `docs/decisions.md`.
- Step 9 real launcher/tmux/PPO mini validation passed: 480 body+genome
  records; 204 valid evaluations => exactly 305 journal tasks/policies; all
  policies reload; 20 distinct worker CPUs, spare scheduler siblings, one
  thread, inhibitor, phase/daily alerts, milestone package, and final
  761-file mirror/checksum verification passed. Full suite: 1089 passed,
  33 warnings, 891.19 s.
- Operator checklist before a real launch: confirm notifier delivery; reserve
  the machine/cooling; confirm backup mount persistence; make the machine
  otherwise idle (including stop AnyDesk); and end the old
  `evodevo-preflight` tmux session. The dry-run reports sshd active and
  `KillUserProcesses` unavailable from this session; verify that policy at
  launch. Use `scripts/campaign/run_campaign.sh --dry-run --mirror-dir
  /mnt/evodevo-backup --launch-at <next-00:00>` then the same launcher without
  `--dry-run`. Never use same-disk mirror or no-notify for the real campaign.


Regenerated 2026-09-28 after fixing the blocking launcher bug and every gap in
the previous report (commit `ab8f73d`), then re-running the full validation
**through the real launcher** (`scripts/campaign/run_campaign.sh` →
`python -m evodevo.campaign`, in real tmux, real PPO). **The real campaign has
NOT been started.** Every validation campaign was stamped
`VALIDATION - NOT A RESULT`, mirrored to the backup disk, verified, and then
deleted; their evidence is reproduced below.

**Verdict: NOT READY** — one blocking item (Section 12): the scheduler's real
dispatch policy is not the policy the pre-registered budget guard was
evaluated on; modelled on the real policy, the campaign's p95 is about
**667 h against the 432 h guard**. Everything else validated green.

---

## Changes since ab8f73d

### Commits (in order; each step committed on its own)

| commit | step | subject |
|---|---|---|
| `179155f` | 1 | Fix BrokenProcessPool: campaign package imported numpy before worker thread pinning |
| `5f95a8d` | 2 | Make notifier delivery truthful and refuse unnotified real launches |
| `3a3c539` | 3 | Require the mirror to be a dedicated mount on a separate physical disk |
| `2fbd136` | 4 | Add a write-nothing --dry-run to the campaign launcher |
| `6aa13b3` | 5a | Hold a systemd sleep/idle/lid inhibitor lock around the campaign scheduler |
| `39d1426` | 6 | Show the reached Walker-milestone time once the Walker phase ends |
| `2af7767` | 7 prep | Add an explicit artifact stamp for validation campaigns |
| `cb7180c` | 7 (found) | Fix a launcher race that could leave a "started" campaign not running |
| `d89f624` | 7 (found) | Assert per-slot CPU pinning in the CLI smoke; guard setsid --wait |
| `179ae00` | 7 (found) | Never "attach" to a stale campaign session whose scheduler is dead |
| `26faf94` | 7 (found) | Stop gracefully: campaign workers leave SIGINT to the scheduler |

No commit touches `docs/decisions.md` or the pre-registered analysis plan.
Scheduling, evaluation and result semantics were not changed: every commit is
entry-point/import, launcher, notifier, mirror, dashboard, telemetry, stamping
or signal-handling code. Because `26faf94` changes how pool workers are
started, serial ≡ interleaved was re-run on CLI output produced at `26faf94`
(Section 5, 7f) and that output was also shown identical to the output
produced before it.

### Root-cause postmortem: the spawn/import bug (Step 1)

**Reproduction (Step 1a, real CLI, 2-minute bound).** `python -m
evodevo.campaign.scheduler` on a tiny real plan (1 arm × 1 seed × Walker-v0,
500 timesteps, 2 workers) exited 1 with `BrokenProcessPool`; six workers
(initial pool plus retries) each raised `ThreadingConfiguredTooLateError`. An
import tracer injected via `PYTHONPATH` (repository untouched) recorded, in
**every spawned worker** (children of scheduler PID 409392), the first numpy
import with this stack:

```
multiprocessing/spawn.py:116 spawn_main -> :125 _main -> :234 prepare -> :258 _fixup_main_from_name
runpy.py:220 -> runpy.py:110 (import of the parent package)
evodevo/campaign/__init__.py:3        from .plan import CampaignPlan
evodevo/campaign/plan.py:10           from evodevo.experiments.benchmark import ...
evodevo/experiments/benchmark.py:25   import numpy
```

So: **module `evodevo.campaign/__init__.py` (via `plan` → `experiments.benchmark`),
in each spawned PPO worker process, during multiprocessing's `spawn`
re-import of the parent's main module (`__mp_main__` =
`evodevo.campaign.scheduler`)**, imported numpy before the pool initializer's
`configure_threading(1, strict=True)` ran. (The ab8f73d report named
`.scheduler` → `ppo_runner` as the first importer; the trace shows it is
`.plan` → `benchmark`, one line earlier.)

**Fix.** `evodevo/campaign/__init__.py` now imports nothing heavy (lazy
`__getattr__` re-exports). New entry point `evodevo/campaign/__main__.py`
whose first statement after importing `threading_control` is
`configure_threading(1, strict=True)` (load-bearing-import-order comment, as
in `evodevo.preflight.__main__`); all logic is in `main()`, which imports the
scheduler lazily and runs only under `if __name__ == "__main__"` (CPython's
spawn in fact skips re-importing a `pkg.__main__` main module). `scheduler.py`
refuses to run as `__main__`, naming the new entry point. `run_campaign.sh`,
`stop_campaign.sh`, `status_campaign.sh` and the tests use `python -m
evodevo.campaign` (`status` subcommand for the dashboard); nothing runs
`python -m evodevo.campaign.scheduler`.

**Audit of every other process/pool spawner** (`ProcessPoolExecutor`,
`multiprocessing`, `-m evodevo.`): the only process pool in the campaign
runtime is `evodevo/parallel/worker_pool.py`. `milestone.run_subprocess`
(plain `subprocess.run` of `-m evodevo.campaign.milestone`), `alerts.Notifier`
(`subprocess.run` of the operator's shell command), `mirror.sync` (`rsync`)
start ordinary subprocesses with no strict thread pinning; `dashboard`,
`monitor` and `resume` spawn nothing; the preflight pool is launched from
`evodevo.preflight.__main__`, which already had the correct pattern and whose
package `__init__` is empty. None shares the flaw.

**Why the earlier suite missed it.** The only test that launched a campaign
end-to-end through `run_campaign.sh` used `--mode surrogate`, which never
creates a worker pool; every real-PPO pool test constructed `EvaluationPool`
directly under pytest, whose `__main__` is pytest's launcher, so the faulty
re-import never happened. No test combined "real mode" with "the real CLI
entry point".

**Tests added** (`tests/campaign/test_entry_point.py`): import purity of
`evodevo.campaign` and everything imported before `configure_threading` (fresh
subprocess); import-order regression with a tracing `__import__` hook
(mirroring the preflight test); refusal when numpy is pre-loaded; refusal of
`-m evodevo.campaign.scheduler`; launch scripts use the package entry point;
**real-PPO CLI smoke** through `python -m evodevo.campaign` (1 arm × 1 seed ×
Walker-v0, population 15, eval_budget 15, ppo_timesteps 500, 4 workers): real
evaluations complete, no `BrokenProcessPool`, every journaled worker reports
torch threads = 1 and every thread env var = 1 (`*_DYNAMIC` = `FALSE`), and
worker slots have distinct CPU sets. Hand-run of the same command: 11 real
PPO tasks on 4 workers pinned to CPUs 0/2/4/6, all single-threaded.
**Non-vacuity, performed and reverted:** an eager `import numpy` added to
`evodevo/campaign/__init__.py` made the import-purity and import-order tests
fail (main tree; reverted with `git checkout -- evodevo/campaign/__init__.py`,
tree clean afterwards) and made the real-PPO CLI smoke fail with
`ThreadingConfiguredTooLateError` (in a throwaway clean worktree, since the
CLI refuses a dirty tree).

### New defects found during this validation, and fixed

1. **Launcher race (`cb7180c`).** `script -c 'setsid CMD'`: inside `script`
   the command is a process-group leader, so `setsid(1)` forks and exits at
   once; the pane closes and its SIGHUP could kill the child before it left
   the session — "Started campaign" printed, nothing running (reproduced 1 in
   12 surrogate launches; also why every transcript was empty, including the
   ab8f73d runs). Fixed with `setsid --wait`; 20/20 launches pass.
2. **Stale session "attach" (`179ae00`).** After a scheduler crash/SIGKILL the
   tmux session survives, and `run_campaign.sh --resume` printed "already
   running; attaching" and resumed nothing. The launcher now removes the stale
   session of the campaign being resumed (its recorded scheduler group is
   dead) and refuses any other launch with an explicit message.
3. **Graceful stop deadlock (`26faf94`).** `stop_campaign.sh` SIGINTs the
   scheduler's whole process group, workers included. A live stack dump
   (faulthandler injected via `PYTHONPATH`) showed a surviving worker blocked
   on the call-queue lock beside zombie workers, and the scheduler — already
   marked `stopped` — hung in `concurrent.futures.process._python_exit`
   joining them; the first real stop took **123 s and ended in SIGTERM**.
   Campaign pools now start workers with SIGINT ignored, so the scheduler
   drains in-flight tasks and shuts the pool down itself: real stops now take
   **4–5 s, exit 0**. The race is timing-dependent (a 4- and a 20-worker CLI
   test both passed on the old code), so non-vacuity rests on a deterministic
   pool-level test (workers survive a group SIGINT only when asked; fails with
   the ignore disabled).
4. **Dashboard** showed "Walker milestone: n/a" once the Walker phase ended;
   it now shows the reached time (`39d1426`).

### Every gap in the ab8f73d report and its resolution

| # | gap in ab8f73d | resolution | evidence |
|---|---|---|---|
| 1 | BLOCKING: real mode crashed every worker through the launcher | Fixed (`179155f`); root cause corrected to `__init__` → `plan` → `benchmark` | CLI smoke; real mini-campaign through the launcher (Section 5) |
| 2 | Journal count vs expected PPO tasks unverified on a real launcher run | Verified | 305 = 305 (Section 5) |
| 3 | No notifier; `alerts --test` printed "sent" and exited 0 with nothing configured | `--test` exits 2 with "NO NOTIFIER CONFIGURED - alerts go to alerts.jsonl only", 0 only after verified delivery; real launch refused without a notifier or explicit `--no-notify` (`5f95a8d`) | tests; Section 7. **Operator must still configure the real notifier** (checklist) |
| 4 | Only mirror candidate (`/dev/sda3`) was the same physical disk | Physical-disk + mountpoint check (`3a3c539`); `/mnt/evodevo-backup` = `/dev/sdb1` on disk `sdb`, results on `/dev/sda7` (disk `sda`); `sda3` now refused | Section 9 |
| 5 | "`docs/final_decisions.md` absent" | Stale claim copied from PREFLIGHT_READINESS.md: the file exists, committed in `71d946c` | `ls docs/final_decisions.md` |
| 6 | `run_campaign.sh --dry-run` did not exist | Added (`2fbd136`), writes nothing | Section 8/9, test |
| 7 | Real serial ≡ interleaved was only shown in-process | Shown on CLI-produced output vs serial `run_ea` | Section 5 (7f) |
| 8 | Real kill/resume through the launcher not achieved | Achieved: SIGKILL mid-UpStepper, `--resume latest`, bit-identical | Section 6 |
| 9 | Stop script only tested on surrogate | Tested on real runs; graceful-stop deadlock found and fixed | Section 6 |
| 10 | Real-PPO evidence only at trivial scale | Inherent to a bounded validation; still disclosed | Section 11 |
| 11 | Unrelated `evodevo-preflight` tmux session (2026-09-22) on the machine | Still present, still untouched (not created by this work) | Section 7 |

---

## 1. Git state

- HEAD validated: `26faf941a147f86bf621cde89f4500760bb6d0a6` ("Stop gracefully:
  campaign workers leave SIGINT to the scheduler"); this report is committed on
  top of it, alone.
- Working tree clean (`git status --porcelain` empty) before the report commit;
  the only transient change during testing was the documented
  `results/rule_based/ORDER_INDEPENDENCE_AUDIT.md` timing line, reverted after
  each full-suite run.
- Real campaign plan config hash (`CampaignPlan.config_hash`, the manifest
  lock): `d32171356cedb8e0e9e0e851aa3bd2234fbbdb03c0a17cf44e833dfeefd56fc4`
  (unchanged since ab8f73d). Benchmark lock hash (`lock_hash.py`):
  `e5c2aa7547d4e08e6e0dad9e57f5a6e928355b3c671606d867f7709b404af638`.
- Validation plan hash (all validation campaigns): `54180436e9ef…`.

## 2. Decisions locked

All campaign rows in `docs/decisions.md` are dated 2026-09-27 and ACCEPTED
(decided by Tejeshwar); unchanged by this work:

| decision | status | value |
|---|---|---|
| D14 scoping gate | ACCEPTED | no options adopted |
| D15 population/offspring/budget/seeds/arms/task order | ACCEPTED | 15 / 15 / 300 genomes (20 generations); seeds 0–4; `direct, nca, rule_based, rule_based_monotone`; Walker-v0 then UpStepper-v0 |
| MONO | ACCEPTED | `rule_based_monotone` co-primary on both tasks |
| D6 application | ACCEPTED | Walker-v0 1,000,000; UpStepper-v0 1,200,000 timesteps |
| D6 Walker-v0 fallback rule | ACCEPTED (disclosed deviation) | 1,000,000 |
| D16 evaluation noise | ACCEPTED (supersedes same-day PROPOSED row) | Walker-v0 K=1; UpStepper-v0 K=2, max over completed seeds |
| Concurrency | ACCEPTED | 20 workers: 8 P-core primaries + 12 E-cores |
| Budget guard and outcome | ACCEPTED | p95 > 432 h → descope (a)/(b)/(c); recorded outcome **none** (p95 406.0 h). See Section 12: that p95 was modelled on a dispatch policy the scheduler does not use |
| Infrastructure failures | ACCEPTED | 2 retries on fresh workers; then `infra_failure`, counted, alerted above 0.5% |

`python -m evodevo.campaign --dry-run` reproduces `guard outcome: all four
approved campaign arms` from the live config.

## 3. Pre-registered analysis plan

Pre-registered in `docs/decisions.md` by `f2689b1dbdcba2745160bc4d2a14c83bc021b63a`
on **2026-09-27 17:16:48 +0530**; decisions locked by `a27c61a` (2026-09-27
20:57:57 +0530). Both precede every commit of this work and any launch (none
has happened). Not edited.

## 4. Test results

- **Full suite at the validated HEAD `26faf94`: 1056 passed, 0 failed, 33 warnings in 840.22 s** (previous report:
  1021 passed; at `d89f624`, Step 7a: 1052 passed, 0 failed, 33 warnings,
  835.84 s). Slow (real-PPO) tests included.

Each named test re-run individually at `26faf94` (34 passed, 207 s):

| named check | test | result |
|---|---|---|
| pinned EA bit-identity (surrogate, 4 arms × 5 seeds) | `tests/evolution/test_propose_tell.py::test_pinned_pre_refactor_surrogate_bit_identity` | PASS (20/20) |
| pinned EA bit-identity (real PPO) | `…::test_pinned_pre_refactor_real_mode_bit_identity` | PASS (4/4 arms) |
| K=1 bit-identity (evaluate_body vs pre-refactor commit) | `tests/evaluation/test_ppo_runner.py::test_evaluate_body_is_bit_identical_against_pre_refactor_commit` | PASS |
| real K=2 policies saved and reloadable | `tests/evaluation/test_multiseed.py::test_real_k2_saves_two_reloadable_policies` | PASS |
| serial ≡ interleaved, surrogate | `tests/campaign/test_scheduler_resume.py::test_surrogate_shared_phase_matches_serial_and_resume` | PASS |
| serial ≡ interleaved, REAL | Step 7f: 16 serial `run_ea` runs vs CLI campaign output | **PASS** — 480/480 records identical on 14 fields |
| resume ≡ uninterrupted, surrogate | `…::test_resume_repairs_log_rows_not_covered_by_checkpoint` (+ second half of the test above) | PASS |
| resume ≡ uninterrupted, REAL | Step 7d: SIGKILL mid-UpStepper + `--resume latest` vs uninterrupted | **PASS** — 480/480 identical; 305 journal tasks identical |
| checkpoint round-trip | `tests/evolution/test_propose_tell.py::test_generation_boundary_checkpoint_round_trip` | PASS (4/4 arms) |
| real-PPO CLI smoke / graceful group stop | `tests/campaign/test_entry_point.py::test_real_ppo_cli_smoke`, `::test_real_cli_graceful_group_sigint_exits_promptly` | PASS / PASS |

Test flakiness, stated plainly: while fixing the launcher race, one full
`tests/campaign` run (1 of 18 after the fix) failed without its output being
captured. The most likely cause — the smoke test asserting distinct CPU sets
per worker **pid**, which a retried task on a rebuilt pool legitimately
violates — was corrected to the real per-slot invariant (`d89f624`); 17
further runs and every later run passed, but the original failure is not
proven to be that one.

## 5. Mini-campaign results (through `run_campaign.sh`, real tmux, real PPO)

Configuration (validation profile, same mechanism as the real one): 4 arms ×
2 seeds × Walker-v0 + UpStepper-v0, population 15, eval_budget 30
(2 generations), ppo_timesteps 2000, K = 1 / 2, 20 workers, mirror
`/mnt/evodevo-backup/validation` (disk `sdb`), notifier = shell command
appending to a file, `--stamp "VALIDATION - NOT A RESULT"`.

| campaign | HEAD | purpose | outcome |
|---|---|---|---|
| mini-1 | `d89f624` | uninterrupted reference | complete in **276 s** (Walker 74 s, UpStepper 111 s; 90-min limit) |
| mini-2 | `179ae00` | kill/resume | complete after SIGKILL + resume; identical to mini-1 |
| mini-3 | `179ae00` | stop script | stopped (123 s, SIGTERM — defect 3, fixed) |
| stop-4, stop-5 | `26faf94` | stop script re-run | stopped in 5.2 s and 4.2 s, exit 0 |
| mini-6 | `26faf94` | serial ≡ interleaved at final code | complete (Walker 73 s, UpStepper 110 s); identical to mini-1 |
| diag | `179ae00` | stack dump of the stop hang | stopped |

**7b checks on mini-1 (all PASS):**
- D7 phase order: plan `[Walker-v0, UpStepper-v0]`; Walker phase ended before
  UpStepper started; last Walker PPO completion < first UpStepper PPO completion.
- Walker milestone package generated automatically at the phase boundary
  (`milestone_status.Walker-v0.ok = true`, generated between Walker end and
  UpStepper start), status COMPLETE (240/240 evaluations), no required artifact
  missing, stamped. Inventory (36 files): `MILESTONE_REPORT.md`, `STAMP.txt`,
  `best_bodies/{direct,nca,rule_based,rule_based_monotone}.{npz,png}` and
  `*_rollout.{gif,mp4}` (trained-controller videos), `figures/{fitness_curves,
  budget_curves,validity_diversity}.{png,pdf}`, `stats/{current_rates,
  current_run_rates,evaluations_to_90pct_arm_median,final_best_morphology,
  final_fitness_summary,omnibus,pairwise,rule_based_vs_monotone,summary,
  validity_diversity_by_generation}.csv`, `stats/tables.{md,tex}`. The
  UpStepper package additionally has the four `mean_of_2_*` sensitivity files.
- Every evaluation record has a body and a genome: 480/480 (planned 480).
- Every valid PPO training has a policy that reloads: 305/305 policies loaded
  with `stable_baselines3.PPO.load`; 0 incomplete trainings.
- Journal count equals expected PPO tasks: 305 = Σ over valid, non-cached
  evaluations of K.
- Mirror: checksum manifest of the 760-file snapshot verifies on
  `/dev/sdb1` (disk `sdb`; results on `sda`); after the final sync only
  `manifest.json`, `alerts.jsonl`, `transcript.log` changed, as expected.
- Stamp: manifest `artifact_stamp`, `STAMP.txt`, every milestone figure title,
  report header, `tables.md` and `final_fitness_summary.csv`.

**7c live checks at 26 s (all PASS):** 20 workers with single-CPU affinities
`0,2,4,6,8,10,12,14` (P-core primaries, slots 0–7) then `16–27` (E-cores,
slots 8–19), each 1 thread; scheduler (and resource tracker) pinned to the
unused hyperthread siblings `1,3,5,7,9,11,13,15`; all journal records so far
show torch threads 1 and thread env 1/FALSE; `systemd-inhibit --list` showed
`sleep:idle:handle-lid-switch … EvoDevo campaign VALIDATION-NOT-A-RESULT-mini-1 block`
(manifest `launch_checks.sleep_inhibitor = systemd-inhibit`).

**7f serial ≡ interleaved (PASS):** mini-6 (CLI, 20 workers) vs 16 serial
`run_ea` runs rebuilt from mini-6's recorded plan (encoder construction
copied from `CampaignScheduler._make_run`, threads pinned to 1): all 16 runs
IDENTICAL on `eval_index, generation, genome_id, parent_id, fitness, valid,
reason, body_hash, per_seed_fitness, per_seed_completed, genome, body,
best_fitness_so_far, cache_hit` (480 records).

Per-run outcome of the validation grid (identical in mini-1, -2, -6 and serial;
VALIDATION - NOT A RESULT — 2 generations at 2000 timesteps mean nothing
scientifically):

| arm | Walker s0 / s1 best (valid of 30) | UpStepper s0 / s1 best (valid of 30) |
|---|---|---|
| direct | -0.074 (1) / 0.569 (2) | -0.093 (1) / -2.775 (2) |
| nca | 0.100 (21) / 0.038 (16) | 0.095 (20) / 0.037 (15) |
| rule_based | 0.012 (11) / 0.219 (20) | 0.016 (11) / 0.025 (20) |
| rule_based_monotone | 0.012 (12) / 0.034 (20) | 0.007 (12) / 0.032 (20) |

**Step 5b — worker RSS vs task sequence.** The PPO pool has **no worker
recycling**: `map_ppo_tasks` never recycles (`max_tasks_per_child` applies to
`map_tasks` only) and one pool serves the whole campaign. RSS (psutil RSS at the
end of each task; the journal field is named `peak_rss_mb` but is not a peak)
per worker in mini-1: 13–17 tasks each, starting 415–416 MB, plateauing at
422–425 MB; **largest growth 2.3%** (< 25%). No recycling implemented; nothing
changed.

## 6. Kill/resume and stop-script evidence

- **Kill/resume (mini-2).** SIGKILL to the scheduler's process group during
  UpStepper generation 0 (12 PPO tasks running, 36 queued, 48 UpStepper tasks
  journaled, no UpStepper checkpoint yet); 0 survivors in the group.
  `run_campaign.sh --resume latest` removed the stale tmux session ("Removing
  stale tmux session of VALIDATION-NOT-A-RESULT-mini-2: its scheduler is not
  running."), re-proposed generation 0 (genome hashes checked against the
  journal), reused the 48 journaled results and completed: journal 305 rows,
  305 unique task keys (nothing re-trained or duplicated). Final fitness,
  validity, body hashes (and genomes, bodies, per-seed fitness) **identical to
  the uninterrupted mini-1 on all 480 records**; the journal's per-task
  genome hash / fitness / completed also identical.
- **Stop script.** On `26faf94`, against running real campaigns (23 processes
  in the scheduler group: inhibitor, scheduler, tracker, 20 workers):
  `stop_campaign.sh` (defaults) → "no evodevo campaign processes remain", exit
  0, **4.2 s / 5.2 s**; afterwards `pgrep -f 'python.*-m evodevo\.'` = 0, spawn
  workers = 0, scheduler group empty, inhibitor locks = 0, tmux server gone,
  manifest `stopped`, transcript shows `Resume: …` and exit 0. (Before
  `26faf94`: 123 s, SIGTERM — defect 3.)

## 7. Monitoring, alerts and mirror evidence

- `python -m evodevo.campaign.alerts --test`: with nothing configured prints
  `NO NOTIFIER CONFIGURED - alerts go to alerts.jsonl only`, exit 2; with the
  file notifier: `Test notification delivered via shell_command`, exit 0, line
  written.
- Live campaigns delivered through the file notifier: 5 daily heartbeats (one
  per monitor start), `phase_complete:Walker-v0` ×3 and
  `phase_complete:UpStepper-v0` ×3.
- Heartbeat with a test override (interval 60 s, repeat window 30 s — the
  production ordering 24 h > 6 h preserved): checks at t = 0, 30, 59, 61, 100,
  125 s → 3 delivered (t = 0, 61, 125). (With only the interval overridden the
  6 h repeat window suppresses the second heartbeat — by design, not a bug.)
- De-duplication with an injected persistent `stall`: 43 journaled checks over
  7 h → exactly 3 notifications: `[alert]` at 0 h, `[alert]` at 6 h,
  `[cleared]` on recovery.
- Mock mirror disk full (12.5 GB < 50 GB): `[alert] disk_mirror: {"free_gb": 12.5}` delivered.
- Mirror: see Section 5; the scheduler and monitor re-check the mirror's mount
  and physical disk before every sync.

**Prompt-6 addenda audit (Step 6):**

| item | file | test | status |
|---|---|---|---|
| alert de-dup: first trigger, 6-hourly repeat, "cleared" notice | `evodevo/campaign/alerts.py` (`evaluate`, `REN_NOTIFY_S`) | `test_operations.py::test_alert_rate_limit_and_clear_are_durable`, `::test_two_alert_writers_share_one_rate_limit` | present; live-checked (7g) |
| 24-hourly heartbeat | `alerts.py` (`HEARTBEAT_S`, `heartbeat`) | `::test_heartbeat_and_phase_notice_deduplication`, `::test_all_alert_conditions_and_daily_heartbeat`, `::test_failed_heartbeat_is_journaled_and_retried` | present; live-checked (7g) |
| milestone-failure alert | `alerts.py::milestone_failure`, `milestone.py::run_subprocess` | `::test_milestone_package_and_subprocess_failure_isolated`, `test_scheduler_resume.py::test_milestone_failure_cannot_abort_completed_phase` | present |
| ETA "prior" (final_estimate.json) until 50 tasks per (arm, task), then "measured" | `dashboard.py` (`PRIOR_PATH`, `>= 50`) | `::test_dashboard_prior_then_per_arm_core_class_measured_eta`, `::test_initial_campaign_prior_uses_joint_p2_total_band` | present |
| stall threshold = 2 × E-core p95 of slowest (arm, task); completion rate < 25% of trailing median for 1 h | `monitor.py::slowest_ecore_p95_s`, `alerts.py::inspect` | `::test_stall_timer_starts_before_first_completed_task_and_ecore_threshold`, `::test_partial_stall_one_hour_window_survives_monitor_restart` | present |
| mirror mandatory by default | `run_campaign.sh`, `scheduler.cli_main` | `::test_run_script_help_and_existing_session_attach`; now also physical-disk/mountpoint (`test_mirror_disk.py`) | present, strengthened |
| dashboard: Walker-milestone ETA date/time; P2 estimate vs live projection with >432 h flag | `dashboard.py::collect/render` | `::test_dashboard_plain_and_tty_render_eta_range`, `::test_dashboard_prior_then_per_arm_core_class_measured_eta` (`projection_exceeds_432h`), new `::test_dashboard_shows_walker_milestone_time_after_the_phase_ends` | present; post-phase gap fixed (`39d1426`) |

No item is missing.

## 8. Final estimate

From the committed, pre-registered `results/campaign_prep/final_estimate/final_estimate.json`
(500 trials), as printed by `run_campaign.sh --dry-run` (Step 7h, launch = now,
2026-09-28 15:49 local):

| | mean | p5 | p95 |
|---|---|---|---|
| total | 403.2 h | 400.6 h | 406.0 h |
| Walker-v0 phase | 123.0 h | 121.9 h | 124.1 h |
| UpStepper-v0 phase | 280.2 h | 277.8 h | 282.9 h |

Walker milestone at 30.5% of runtime. Calendar from a launch at 00:00 on
2026-10-01 (`--launch-at 2026-10-01T00:00`): Walker milestone / Walker end
2026-10-06 01:55 (p5) · 02:58 (mean) · 04:06 (p95); campaign end
2026-10-17 16:33 (p5) · 19:09 (mean) · 21:57 (p95).

**This estimate does not model the scheduler that would run the campaign.**
`scripts/capacity_planning/final_estimate.py` → `simulate_campaign.run_interleaved`
simulates continuous greedy dispatch (any idle worker takes the next ready
task; each run advances to its next generation on its own). The real
scheduler has (1) a **global** generation barrier across all 20 runs of a
phase (`_phase_round` dispatches only the lowest generation), (2) PPO tasks
submitted in **waves of 20, each drained completely** before the next
(`map_ppo_tasks`), and (3) every run's propose/decode/sentinel work serial in
the one scheduler process. With 8 P-cores and 12 E-cores at a 2.3× speed
ratio, every wave waits for its slowest E-core task while P-cores idle.
Re-simulating with the same inputs, draws and trial count
(`final_estimate.py` functions unchanged; 200 trials):

| policy | total mean | p5 | p95 | Walker | UpStepper |
|---|---|---|---|---|---|
| modelled (reproduces the committed estimate) | 403.0 h | 400.2 h | 405.7 h | 123.0 h | 280.0 h |
| **real scheduler** | **662.1 h** | 656.4 h | **667.1 h** | 203.2 h | 458.9 h |

Ratio 1.64. The real mini-campaign corroborates it: measured E/P task-time
ratio 2.1–2.2, worker utilisation 43% (Walker) and 55% (UpStepper), P-cores
busy only 30–36% (at 2000 timesteps fixed per-generation costs also weigh
in, so this is indicative, not a calibration). **Projected completion on the
real policy:** from a 2026-10-01 00:00 launch, Walker milestone ≈ 2026-10-09
11:00, campaign end ≈ 2026-10-28 14:00 (mean) — see Section 12.

## 9. Disk projection vs free space

`results/campaign_prep/disk/disk_estimate.json`: projected 32.9 GB; D10 3×
margin requires 98.6 GB.

| disk | device → physical disk | free | 3× margin |
|---|---|---|---|
| main (`/home/ubuntu/projects/EvoDevo/results`) | `/dev/sda7` → `sda` | 300.0 GB | OK |
| mirror (`/mnt/evodevo-backup`, mountpoint) | `/dev/sdb1` (ntfs via fuseblk) → `sdb`, 5.5 TB | 5,936.9 GB | OK |

Same physical disk: **no**. `/media/ubuntu/A27C…` (`/dev/sda3`, the ab8f73d
candidate) is refused as the same disk as the results. rsync to the NTFS
backup works (ownership/permissions are flattened by ntfs-3g; content
checksums verify). Measured validation footprint: 95 MB per 480-evaluation
mini-campaign. The disk projection assumes the modelled wall-clock only for
telemetry volume; the longer real-policy duration adds telemetry/transcript
volume roughly in proportion (still far inside both disks).

## 10. Operator commands

```
# prerequisites: backup disk mounted at /mnt/evodevo-backup (separate disk),
# notifier configured in campaign.local.yaml and proven with:
.venv/bin/python -m evodevo.campaign.alerts --test

# dry run (writes nothing): plan, calendar, disk, mirror disk check, notifier, CPU placement
bash scripts/campaign/run_campaign.sh --dry-run --mirror-dir /mnt/evodevo-backup --launch-at 2026-10-01T00:00

# launch (in tmux; refuses a dirty tree, missing notifier, same-disk or unmounted mirror)
bash scripts/campaign/run_campaign.sh --mirror-dir /mnt/evodevo-backup

# attach
tmux attach -t evodevo-campaign

# status (one-shot)
bash scripts/campaign/status_campaign.sh --campaign latest

# graceful stop (drains in-flight PPO tasks, releases the inhibitor, removes the session)
bash scripts/campaign/stop_campaign.sh --campaign latest

# resume (also after a crash: a stale session of this campaign is removed automatically)
bash scripts/campaign/run_campaign.sh --resume latest --mirror-dir /mnt/evodevo-backup

# manual milestone regeneration for a phase (same isolated subprocess the scheduler uses)
.venv/bin/python -m evodevo.campaign.milestone --campaign <CAMPAIGN_ID> --output-root results/campaign --task Walker-v0
```

## 11. Known gaps and accepted risks

1. **BLOCKING — dispatch policy vs pre-registered estimate** (Section 8, 12).
2. Validation scale: real PPO at 2,000 timesteps, 2 generations, 2 seeds;
   the real campaign runs 1–1.2 M timesteps, 20 generations, 5 seeds. The RSS
   plateau (≤ 2.3% over ≤ 17 tasks/worker) cannot exclude slow growth over the
   ~300+ tasks each worker will run; the dashboard shows RAM/swap and the swap
   alert fires above 1 GB.
3. The operator's `campaign.local.yaml` has no notifier yet; the real launch
   is refused until one is configured (or `--no-notify` is passed, which this
   report does not recommend).
4. The backup disk is NTFS through FUSE; a sync failure there raises
   `mirror_failed` but a very slow FUSE write path was not load-tested beyond
   760-file snapshots.
5. One uncaptured intermittent `tests/campaign` failure (1 in 18 runs) —
   probable cause fixed (Section 4), not proven.
6. The graceful-stop deadlock is timing-dependent; its regression test is the
   deterministic pool-level check, while the CLI-level stop test passes on
   both old and new code.
7. `stats/tables.md` is written as CSV text because the optional `tabulate`
   package is not installed (pre-existing).
8. An unrelated tmux session `evodevo-preflight` (created 2026-09-22) remains;
   not created or touched by this work.

## 12. Verdict: **NOT READY**

**Blocking item — the budget guard was evaluated on a dispatch policy the
scheduler does not implement.** The pre-registered guard (p95 ≤ 432 h, else
descope) recorded "none applied" from an estimate that models continuous
dispatch; the actual scheduler drains waves of 20 behind a global generation
barrier. Modelled with identical inputs, that policy gives p95 ≈ 667 h
(mean ≈ 662 h), which would trigger the pre-registered descoping sequence,
and the real mini-campaign's 43–55% worker utilisation points the same way.
Resolving it needs a human decision, not a validation-time change (this task
forbids scheduling changes): either (a) change the scheduler to continuous
dispatch across generations/runs — a scheduling change that must then pass
serial ≡ interleaved and resume ≡ uninterrupted again and be re-estimated — or
(b) re-run `final_estimate.py` with the real dispatch policy and apply the
pre-registered guard mechanically to its p95.

Everything else required for launch is in place and validated through the
real launcher: the spawn/import fix, real-PPO mini-campaign, automatic Walker
milestone, kill/resume and stop, serial ≡ interleaved, notifier truthfulness,
physical-disk mirror, `--dry-run`, sleep inhibitor, pinning and single-thread
evidence, and the addenda audit.

## Pre-launch operator checklist

| # | item | status |
|---|---|---|
| 1 | Backup disk mounted at `/mnt/evodevo-backup` and on a different physical disk from `results/` (`run_campaign.sh --dry-run` → `same physical disk: False`) | verified 2026-09-28 (`sdb1`/`sdb` vs `sda7`/`sda`); re-check on launch day |
| 2 | Notifier configured in `campaign.local.yaml` and delivery proven (`python -m evodevo.campaign.alerts --test` exits 0 and the message arrives) | **not done** — operator |
| 3 | Blocking item in Section 12 resolved and the budget guard re-applied | **not done** — decision owner |
| 4 | `run_campaign.sh --dry-run` shows `LAUNCH CHECKS: PASS` and a calendar consistent with item 3 | pending items 2–3 |
| 5 | Cooling adequate for ~2–4 weeks at 20 busy cores (temperature alert fires at ≥ 95 °C sustained 10 min) | operator to confirm |
| 6 | Machine reserved for the campaign's duration (no other heavy jobs; unattended upgrades/reboots considered) | operator to confirm |
| 7 | Working tree clean at launch (the launcher refuses otherwise) | enforced |
