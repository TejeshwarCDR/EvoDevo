# EvoDevo campaign readiness

> **Production thermal update — 2026-09-30.** This report's retained
> 20-worker / 403.2-hour capacity discussion is historical validation
> evidence, not the launch configuration.  The active production contract is
> 18 workers: P CPUs 0,2,4,6,8,10,12,14 and E CPUs 16–25, with E26–27 reserved
> for scheduler/control work; turbo enabled; P policies 0–15 capped at 4.2
> GHz; RAPL PL1=125 W and PL2=160 W.  The matching 960-genome real-PPO run
> completed in 22.3 minutes at 42.96 genomes/min, 84 C peak / 77 C p95, with
> no throttle increment or worker failure.  It rate-scales the full campaign
> to approximately 434 h (18.1 days), with the Walker phase/milestone around
> 132 h (5.5 days).  Source: `results/thermal_worker_study/FINAL_SELECTION_20260930.md`.
> `scripts/campaign/run_production_campaign.sh` is the required power-held
> production entry point; it fails closed on any topology or control mismatch.

Fresh single-pass report, 2026-09-29. It describes only the current repository
state. The scientific campaign has not been started. Validation campaigns are
explicitly stamped `VALIDATION - NOT A RESULT`.

## 1. Current-state summary and traceability

| Current claim | Evidence commit | Detail in this report |
|---|---:|---|
| Production uses continuous, greedy dispatch with per-run generation barriers and strict D7 phase order | `f3d4e0e` (tests first in `f779833`; decision in `5452cfb`) | Sections 3–4 |
| The production scheduler and capacity model execute the same pure `DispatchCore` policy | `f3d4e0e`, `2cee966` | Sections 4–6 |
| Base estimate is 403.2 h mean, 400.6 h p5, 405.9 h p95; the 432 h guard applies no descope step | `2cee966` | Section 5 |
| Timed scheduler/simulator conformance is inside the 5% makespan and 3-point utilization gates | `3fc4e4c` | Section 6 |
| Equivalence, resume, graceful stop, task-local crash/timeout handling, and refusal paths pass | `725de97` plus fixes `133e4b0`, `cec5dde`, `7b47d67` | Section 7 |
| Real launcher/tmux/PPO mini-campaign and notification/dashboard checks pass | `d455a83`, `ca6923b`, `f44061d` | Sections 8–9 |
| Disk projection is 32.9 GB; the 3x requirement is 98.6 GB | estimate artifact committed through `2cee966` | Section 10 |

Repository reconciliation was performed before writing this report:

- HEAD was attached to `main` at `f44061de597ee6f0a2e5966c7f224711ab8b222a`,
  ahead of `origin/main`; the worktree was clean, and `git stash list` was empty.
- `git log --all` and every reflog do contain the dispatch work:
  `5452cfb` (decision), `f779833` (policy core and tests), `f3d4e0e`
  (production scheduler), and `2cee966` (shared-policy estimate).
- The contradictory old body was documentation drift. `2b5e624` prepended a
  current summary while retaining an obsolete report, and `f44061d` appended a
  full-suite result but copied the retained obsolete conclusion. That conclusion
  omitted dispatch commits already in its ancestry, the accepted rows in
  `docs/decisions.md`, the shared `DispatchCore`, the re-derived estimate, and
  the conformance/re-proof evidence. It was therefore wrong; the test run itself
  was not evidence for the obsolete scheduler description.

## 2. Git, plan, and locked configuration

- Pre-registered analysis-plan commit:
  `f2689b1dbdcba2745160bc4d2a14c83bc021b63a`. The original contract in that
  commit remains the reference. No change was made to it in this pass. The
  current file also contains a separately headed later D16 implementation note.
- Campaign-profile commit: `c7172d1`; configuration lock commit: `feed8ec`.
- Current campaign: four arms (`direct`, `nca`, `rule_based`,
  `rule_based_monotone`) x five seeds x two tasks; population/offspring 15/15;
  300 genomes per run; Walker-v0 first, then UpStepper-v0.
- PPO budgets: Walker-v0 1,000,000 steps and K=1; UpStepper-v0 1,200,000
  steps and K=2, aggregated by maximum over completed seeds. Twenty workers are
  assigned as 8 P-core primaries plus 12 E-cores.
- Neither `campaign.local.yaml` nor the pre-registered plan was read, edited, or
  rewritten during this reconciliation. No real campaign was launched.

## 3. Decisions in force

`docs/decisions.md` is the authoritative, append-only record. Its scientific
decisions before the campaign block remain in force, including the current
resolved architecture, readout, mutation, fairness, analysis, and statistical
protocol choices; superseded rows remain labeled as such. The campaign and
readiness decisions are all represented below.

| Date | Decision | Current resolution |
|---|---|---|
| 2026-09-19 | Interpreter | Python 3.10.20 in `.venv`; CPU-only PyTorch |
| 2026-09-27 | D6 task budgets | Walker 1.0M; UpStepper 1.2M; Walker fallback deviation recorded |
| 2026-09-27 | MONO | `rule_based_monotone` is co-primary on both tasks |
| 2026-09-27 | D14 | No optional scope added |
| 2026-09-27 | D15 | population 15, offspring 15, 300 genomes, seeds 0–4, four arms, Walker then UpStepper |
| 2026-09-27 | D16 | Walker K=1; UpStepper K=2; max over completed seeds; per-seed logging |
| 2026-09-27 | Concurrency | 20 workers |
| 2026-09-27 | Budget guard | 432 h p95; ordered fallbacks monotone Walker-only, drop monotone, then UpStepper K=1 |
| 2026-09-27 | Infrastructure failures | Two fresh-worker retries; then task-local `infra_failure` sentinel and alert above 0.5% |
| 2026-09-28 | Preflight provenance | Uncommitted preflight provenance disclosed; code subsequently committed |
| 2026-09-28 | Artifact policy | Save every valid PPO policy |
| 2026-09-28 | Walker budget limitation | Checkpoint-noise sensitivity disclosed; budget unchanged |
| 2026-09-28 | Dispatch policy | **ACCEPTED, Tejeshwar: continuous, per-run barrier (ADR-conformant)**; old global barrier/drained waves rejected |
| 2026-09-28 | Guard re-application | No fallback applied to the implemented continuous policy |
| 2026-09-28 | Validation idle gate | Three measured windows; bounded operator stack allowed, competing work refused |
| 2026-09-28 | Phase timeout sessions | Reuse pool, reset timeout state only after a drained D7 phase |
| 2026-09-29 | Conformance | Pass at fixed 5% makespan / 3-point utilization gates after one permitted rerun |
| 2026-09-29 | Graceful surrogate stop | Observe stop after each durably checkpointed generation |
| 2026-09-29 | Post-dispatch re-proof | Equivalence, recovery, stop, failure, and refusal evidence rerun |
| 2026-09-29 | Intermittent-suite investigation | Not reproduced in 40 consecutive runs; no unsupported fix claimed |
| 2026-09-29 | Operational hardening | Durable utilization/RSS telemetry and bounded alerts adopted |
| 2026-09-29 | Real mini-campaign | Full launcher/tmux/PPO validation passed |

The same-day D16 proposal is explicitly superseded by its accepted row. Older
scientific rows that contain words such as “OPEN” only describe historical
states and are explicitly superseded by later rows; they are not current
campaign choices.

## 4. Implemented dispatch policy

`evodevo/campaign/dispatch.py` is a pure, event-driven state machine with no
I/O. `DispatchCore.begin_phase` fixes the D7 phase and run set;
`report_ready` accepts exactly one complete generation batch per run;
`next_task` selects `(generation, tie_key, insertion sequence)` from a heap and
maintains a bounded `slots + prefetch` window; `complete` opens only that
task's owning run/generation barrier; `mark_run_complete` closes the phase.

Production imports that component in `evodevo/campaign/scheduler.py`. Each run
can prepare generation g+1 immediately after all of its own generation-g tasks
complete. `refill()` continuously submits available work and does not drain
fixed waves. Results are collected by task identity, journaled, then passed to
the correct run in candidate order, so completion order cannot affect results.
The outer loop still iterates campaign phases in plan order, preserving the
Walker-before-UpStepper D7 barrier.

The capacity model imports the same classes in
`scripts/capacity_planning/simulate_campaign.py`. Its production path calls the
same `begin_phase`, `report_ready`, `next_task`, `complete`, and
`mark_run_complete` transitions. It adds virtual workers and the measured
single scheduler server; it does not reimplement dispatch ordering. The old
global-barrier/drained-wave routine is retained only as a named historical
comparison.

Policy tests were committed before production integration (`f779833`) and
cover per-run barriers, work conservation/no waves, lowest-generation-first
priority and deterministic ties, D7 phase refusal/order, and result-order
independence. Failure tests additionally prove that one crashed or hung task
is retried and charged only to itself while healthy in-flight tasks remain
valid.

## 5. Re-derived estimate and mechanical guard

Source: `results/campaign_prep/final_estimate/final_estimate.json`, 500 trials,
Monte Carlo seed 20260927, production prefetch 4, measured scheduler serial
costs and journal fsync cost, and the same measured PPO/validity/P–E inputs.

| Phase | Mean | p5 | p95 |
|---|---:|---:|---:|
| Walker-v0 | 123.0 h | 121.9 h | 124.1 h |
| UpStepper-v0 | 280.2 h | 277.8 h | 282.8 h |
| Total | **403.2 h** | **400.6 h** | **405.9 h** |

Walker reaches its milestone at 30.5% of mean total runtime. Predicted total
worker utilization is 99.75%. For comparison only, the rejected historical
global-wave policy produces 662.3 h mean and 667.6 h p95 on the same draws.

The pre-registered rule is applied to the base estimate: **405.9 h <= 432 h**,
leaving 26.1 h headroom. Therefore none of the three fallback steps is applied,
and the full four-arm, five-seed, Walker K=1 / UpStepper K=2 design remains.
The t_PPO +10% sensitivity reaches 446.5 h p95, but sensitivities disclose
input uncertainty and do not replace the pre-registered base-case trigger.

## 6. Timed scheduler/simulator conformance

Evidence commit `3fc4e4c` records `validation-conformance-2`, run through the
real launcher and tmux after the validation idle gate passed. It exercised the
full 4-arm x 5-seed x 300-genome grid with scaled synthetic PPO durations,
Walker scale 1/100 and UpStepper K=2 scale 1/250.

| Phase | Real | Simulated | Makespan delta | Real / simulated steady utilization | Point delta |
|---|---:|---:|---:|---:|---:|
| Walker-v0 | 4,750.776 s | 4,766.727 s | 0.335% | 99.692% / 99.536% | 0.156 |
| UpStepper-v0 | 4,410.275 s | 4,458.073 s | 1.072% | 98.495% / 97.854% | 0.641 |

Both phases pass the fixed <=5% makespan and <=3-percentage-point utilization
gates. They completed 4,658 and 9,316 timed tasks, reached 24 in flight (20
slots plus prefetch 4), and preserved D7 order. The first attempt exposed a
stale cross-phase timeout session; `7b47d67` fixed it, a regression test was
added, and the one permitted rerun passed. Validation artifacts, notifications,
and the separate-disk 951-file mirror verified; no scientific result was made.

## 7. Equivalence, recovery, and refusal evidence

- Full-scale surrogate serial vs interleaved: all 12,000 normalized records,
  final populations, and `RunResult`s matched.
- Real-PPO CLI serial vs interleaved: 16 runs and all 480 records matched a
  single-threaded reference.
- Resume: 20 randomized surrogate SIGKILL points and 3 real-PPO SIGKILL points
  resumed equal to uninterrupted baselines with no duplicate completed journal
  identities. Real kills included cross-generation skew and active UpStepper.
- Graceful stop: 10/10 SIGINT trials completed in 0.048–1.913 s, left zero
  process-group survivors, and resumed identically.
- Targeted failure/refusal coverage passes for timeout continuation, one crash
  among 19 healthy tasks, two retries followed by `infra_failure`, alerting,
  genome-hash mismatch, config-hash mismatch, dirty-tree refusal, launcher
  cleanup, and process-group cleanup.

Evidence is recorded in `725de97`; the graceful-stop fix is `133e4b0`, and the
explicit genome-hash refusal test is `cec5dde`. `EvaluationPool` tracks and
replaces a failed future/worker individually, so a crash never assigns a retry
or failure to unrelated in-flight work.

## 8. Real launcher mini-campaign

Evidence commit `d455a83` records the completed `mini` validation using
`scripts/campaign/run_campaign.sh`, real tmux, real PPO, 20 workers, a temporary
file notifier, and a separate-disk mirror. It was stamped
`VALIDATION - NOT A RESULT`.

- Walker completed before UpStepper began. The Walker boundary automatically
  generated its complete stamped milestone package.
- All 480 planned evaluation records contain body and genome. The 204 valid
  evaluations imply exactly 305 PPO tasks under Walker K=1 / UpStepper K=2;
  journal rows and saved policies are both 305, and every policy reloads.
- Twenty distinct planned worker CPUs were observed; scheduler CPUs used spare
  siblings. Every journaled task recorded one Torch thread and single-thread
  environment settings. The sleep/idle/lid inhibitor was active.
- Phase-complete and daily-heartbeat messages reached the temporary notifier.
  The source/mirror checksum manifests verified all 761 files after final sync.

No real scientific campaign was started by this validation or by the present
reporting pass.

## 9. Notification and dashboard layer

Commit `ca6923b` added the shared status context and was covered by the final
suite in `f44061d`. Current verified behavior:

- campaign start is once-only; phase starts/completions are durable; the daily
  18:00 local heartbeat is resume-safe and does not replay missed days;
- completion delivery retries three times and writes the completion marker even
  if delivery fails; all messages carry phase/overall progress and ETA;
- the dashboard shows status, both progress values, ETA range, elapsed time,
  utilization target, and freshness; it renders a final static screen on normal
  or abnormal exit and has an ASCII fallback;
- alert rate limits, durable de-duplication, recovery notices, raw journal
  schema, sustained-low-utilization alert, and >40 GiB worker-RSS alert pass;
- telemetry is based on completed journal intervals, and RSS is the latest
  scheduler sample, so a long active task appears at the next completion/status
  refill rather than continuously.

## 10. Disk, transcript, and mirror

`results/campaign_prep/disk/disk_estimate.json` projects 32.85 GB and a 3x
requirement of 98.56 GB for 40 runs, 12,000 genomes, and up to 18,000 PPO
trainings/policies. The measured root free space in that artifact is 302.13 GB,
so the 3x rule passes. The conformance transcript projects to about 0.13 MB over
18 days, well below the 5 GB hardening threshold.

The validated mirror was `/dev/sdb1` on physical disk `sdb`, distinct from the
results filesystem on `sda`; content checksums passed. This is launch-time
state and must be rechecked immediately before the real launch. NTFS path
validation rejects forbidden characters and trailing dots/spaces.

## 11. Test results

The clean committed full-suite run recorded by `f44061d` is **1,093 passed,
33 warnings in 1,141.20 s (19:01), zero failures**. It included simulator,
campaign-launcher, real-PPO, worker/affinity, preflight, and visualization
coverage. Three launcher tests temporarily blocked by an untracked audit input
then passed 3/3 after a clean baseline was restored. The warnings were existing
dependency warnings or deliberately exercised missing-run/render warnings.

For this reconciliation, the current tree collected exactly 1,093 tests. A
fresh focused run passed 82 tests; three launch-path tests correctly refused the
then-dirty report-editing tree. Immediately after committing the report, those
three tests passed **3/3 in 61.22 s** from the clean state. Together the run
covered dispatch policy/model, failures, resume, operations, entry point,
worker pool, notification/dashboard, dry run, and mirror checks.

## 12. Exact operator commands

```bash
# Configure and prove a real notifier first (does not launch the campaign).
.venv/bin/python -m evodevo.campaign.alerts --test

# Read-only launch rehearsal; choose the intended midnight explicitly.
bash scripts/campaign/run_campaign.sh --dry-run \
  --mirror-dir /mnt/evodevo-backup \
  --launch-at 2026-09-30T00:00

# Real launch only after every item in Section 13 is closed.
bash scripts/campaign/run_campaign.sh \
  --mirror-dir /mnt/evodevo-backup

# Observe, attach, stop, or resume.
bash scripts/campaign/status_campaign.sh --campaign latest
tmux attach -t evodevo-campaign
bash scripts/campaign/stop_campaign.sh --campaign latest
bash scripts/campaign/run_campaign.sh --resume latest \
  --mirror-dir /mnt/evodevo-backup

# Regenerate a completed phase package if required.
.venv/bin/python -m evodevo.campaign.milestone \
  --campaign <CAMPAIGN_ID> --output-root results/campaign --task Walker-v0
```

The launcher refuses a dirty tree, missing required notifier, unmounted or
same-physical-disk mirror, and incompatible campaign/config hash. Do not use
`--no-notify` or a same-disk override for the scientific campaign.

## 13. Known gaps and operator gates

- Configure the operator-owned notifier in the ignored local configuration and
  confirm actual delivery. It was intentionally not touched here.
- On launch day, re-run the dry run and confirm the backup mount, distinct
  physical disks, free space, CPU placement, ssh/session policy, and a clean
  tree.
- Reserve the machine for the estimated duration, verify cooling and reboot/
  unattended-upgrade policy, stop avoidable operator applications (including
  AnyDesk), and close the pre-existing `evodevo-preflight` tmux session.
- The 500-trial interval conditions on measured input means; the +10% PPO-time
  sensitivity exceeds 432 h. Runtime utilization/ETA monitoring is therefore
  operationally important even though the registered base-case guard passes.
- Real-PPO validation used 2,000-step mini runs rather than 1.0M/1.2M scientific
  runs. Worker RSS plateaued in validation, but slow growth over hundreds of
  tasks per worker cannot be excluded.
- The earlier intermittent campaign-test failure was not reproduced in 20 idle
  plus 20 loaded runs; its cause remains unproven.
- The mirror is NTFS/FUSE. Checksums pass, but long-duration behavior was not
  load-tested beyond the validation snapshots.

## 14. Verdict: **NOT READY**

The dispatch-policy contradiction is resolved in favor of the current code:
continuous per-run dispatch exists, is committed, shares its policy core with
the estimator, passes conformance and re-proofs, and yields a base p95 of
405.9 h, so the 432 h guard applies no fallback. The remaining blockers are
operator-owned launch gates in Section 13—especially notifier delivery,
launch-day mirror/idle checks, machine reservation, cooling, and reboot policy.
Close those gates and rerun the dry run before starting the scientific campaign.
