# EvoDevo production launch readiness

**Authoritative launch document.** This file supersedes the retained
20-worker/403-hour narratives in historical reports. It describes the only
accepted production configuration, the safeguards that must remain in force,
and the checks required immediately before manual launch. A real launch was
attempted, but the scientific campaign **never started evaluation**.

## Final verdict

**READY FOR MANUAL LAUNCH.** The cross-UID guardian and stale scheduler-PID
defects are fixed in commit `f68e2b4922389945ecd2d3602f935871e7280e6a`.
Focused and campaign regression suites pass, the authoritative root-held dry
run reports `LAUNCH CHECKS: PASS`, and the repeated bounded real-host lifecycle
validation passes every live-control, acknowledgement, user-tmux, cleanup, and
restoration gate. No scientific evaluation started. This verdict authorizes
only a deliberate future manual launch after the final clean-HEAD dry run; it
does not itself start or resume the campaign.

## Exact production contract

| Item | Required value |
|---|---|
| CPU profile | `thermal-validated` |
| Workers | 18, exactly one logical CPU per worker |
| P-worker CPUs | `0,2,4,6,8,10,12,14` (physical primaries only) |
| E-worker CPUs | `16-25` |
| Scheduler/control CPUs | `26,27` (never assigned to workers) |
| Turbo | enabled (`intel_pstate/no_turbo=0`) |
| P-policy ceiling | 4,200,000 kHz for policies `0-15` |
| RAPL package limits | PL1 125,000,000 µW (125 W); PL2 160,000,000 µW (160 W) |
| Workload | 4 arms × 5 seeds × Walker-v0 then UpStepper-v0; 300 genomes/run |
| PPO budget | Walker 1,000,000 steps/K=1; UpStepper 1,200,000 steps/K=2, max aggregation |
| Required launcher | `scripts/campaign/run_production_campaign.sh` only |

The contract is defined in `evodevo/campaign/production.py`, persisted in a
production manifest, and validated on resume. `cpu_placement()` rejects a
different worker count or incompatible CPU topology; launch checks reject any
control mismatch. The production wrapper applies the controls as root, verifies
the actual sysfs values, starts the scheduler as `ubuntu`, holds the controls
for the scheduler lifetime, and restores original values on normal exit, error,
SIGINT, or SIGTERM. It removes inherited tmux/socket overrides, records and
verifies the live root guardian PID, and ignores SSH hangup while the scheduler
remains alive.

## Failed launch attempt and lifecycle fix

At `20260930T034016Z`, the privileged wrapper correctly applied and verified
turbo on, P policies 0–15 at 4,200,000 kHz, PL1 125,000,000 µW and PL2
160,000,000 µW. It then printed a successful start, immediately restored the
host controls, and exited. The tmux session disappeared; the run directory
contained only `transcript.log`; no scheduler or worker survived. No scientific
evaluation began.

The root cause was a startup race in `run_campaign.sh --wait-for-campaign`.
Immediately after asynchronous tmux window creation it called
`scheduler_alive()`. Absence of `scheduler.pgid`—normal while Python was still
starting—was incorrectly treated as scheduler termination. That returned the
user launcher to `with_cpu_power_cap.sh`, whose exit trap correctly restored
the controls. The delayed child then ran its independent production-power gate
against the restored 5.5–5.6 GHz / 253 W state and correctly refused launch.

The corrected launcher treats `scheduler.pgid` as a positive startup
acknowledgement. It waits while the scheduler tmux window is alive, fails and
restores if that window exits or startup times out, and only reports “Started”
after both the scheduler process group and root guardian are verified alive.
It then holds the user waiter and root guardian until the scheduler group
actually terminates. Restoration is idempotent and runs exactly once. HUP is
ignored after controls are applied; SIGINT/SIGTERM drive the controlled-stop
path. Stale scheduler/guardian PID files are removed before launch and the
guardian PID file is removed at exit.

## Thermal evidence and capacity

Authoritative raw evidence and selection matrix:
`results/thermal_worker_study/FINAL_SELECTION_20260930.md` and the committed
JSON telemetry beside it.

| Measured real-PPO validation | Result |
|---|---:|
| Profile/control | P8+E10, turbo on, 4.2 GHz P cap, PL1/PL2 125/160 W |
| Workload | 960 genomes × 20,000 PPO steps |
| Duration / throughput | 22.3 min / 42.96 genomes per min |
| Package temperature | 84 C maximum; 77 C p95 |
| Worker utilization | 97.1% |
| Worker failures / affinity | 0 / verified |
| Thermal throttle counter | no increment |
| Full campaign projection | about 434.4 h / 18.1 days |

The matrix acceptance gate was at least 20 minutes, at most 88 C peak, no
throttle increment, zero worker failures, verified affinity, and a 15–20 day
projection. The selected profile is the fastest measured configuration meeting
every gate. Relevant rejected alternatives include P8+E10 at PL1=135 W (89 C
abort), P7+E12 (89 C abort), P8+E12 (90 C abort), and turbo-disabled P8+E10
(24.7 projected days). The 20-worker historical configuration sustained
98–100 C and is prohibited.

The benchmark harness retains its one-second temperature/throttle/affinity/
utilization watchdog and automatic temperature abort. Production monitoring
continues to expose CPU temperature, thermal-throttle count, worker utilization
and worker failures through dashboard, journal and alerts; do not suppress a
thermal or failure alert to keep a run alive.

## Calendar and milestones

The 18.1-day number is a measured-rate-scaled operational projection, **not** a
new confidence interval. For a launch time `T`:

| Event | Projection |
|---|---:|
| Walker-v0 phase completion and milestone package | `T + about 132.5 h` (5.5 days; 30.5% of total) |
| UpStepper-v0 completion / campaign completion | `T + about 434.4 h` (18.1 days) |

`python -m evodevo.campaign --dry-run` calculates and prints the actual local
calendar from the proposed launch time. It also keeps the historical Monte
Carlo estimate separate from this thermal capacity projection.

## Latest verified machine state

The latest root-held production dry run applied and read back turbo on, all
sixteen 4.2 GHz P-policy limits, PL1=125 W and PL2=160 W before launch checks.
It then restored the original host settings after exit (P policy maxima
5.5–5.6 GHz, PL1/PL2 253 W); these restored values are expected when no
campaign is active.

| Check | Last verified result |
|---|---|
| Results filesystem | `/dev/sda7`, physical disk `sda` |
| Mirror | `/mnt/evodevo-backup` on `/dev/sdb1`, physical disk `sdb` |
| Mirror separation | pass — no shared physical disk |
| Free space at check | results 302.7 GB; mirror 5,937.2 GB |
| Disk requirement | 98.6 GB (3× the 32.9 GB projection) — pass |
| Notifier | `campaign.local.yaml` configured; supported delivery test passed via `ntfy_topic_url` |
| Process/session audit | no campaign, benchmark, or campaign tmux session running |
| Root-held dry run | pass on clean fix commit `f68e2b4922389945ecd2d3602f935871e7280e6a`; `LAUNCH CHECKS: PASS`; no campaign started; controls restored |
| Failed real launch | `20260930T034016Z`; controls restored; no scientific evaluation started |
| Post-fix host lifecycle validation | initial attempt **FAIL**; corrected repeat **PASS** on 2026-09-30 with full cleanup and restoration |

The current CLI checks idle/stale process state, a dedicated physically distinct
mirror, disk capacity, notifier configuration, git cleanliness, topology,
affinity, and production power state before it can create a scheduler.

## Safety and recovery operations

| Need | Command / behavior |
|---|---|
| Check a running campaign | `scripts/campaign/status_campaign.sh --campaign CAMPAIGN_ID` (executable mode enforced by test) |
| Dashboard/heartbeat | the launcher starts dashboard and monitor panes; durable start, phase, failure, thermal/throttle, utilization and daily heartbeat alerts use `campaign.local.yaml` |
| Controlled stop | `scripts/campaign/stop_campaign.sh --campaign CAMPAIGN_ID --grace-s 120` |
| Resume after a controlled stop/crash | `sudo scripts/campaign/run_production_campaign.sh --resume CAMPAIGN_ID --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml` |
| Power-control recovery | the root wrapper restores changed sysfs values in reverse order on exit or signal; never bypass it with a bare real-mode launcher |

The scheduler journals completed work, performs config/hash and production-
contract checks on resume, protects phase ordering, verifies worker affinity,
keeps worker and scheduler CPUs disjoint, detects failures/retries, checks
mirror/disk state, and retains sleep/idle/lid inhibition when systemd grants
the lock. Stop/recovery logic validates its target process group before
signalling it and refuses to kill an unrelated process group.

## Git and artifact integrity

The production implementation and raw thermal evidence baseline remains commit
`69b785671c46b64106d199d82127fd4ee84a3d4f`. The first lifecycle fix is committed
as `a69f4b5e3802940499ba9412447cc2f5afbc99b0`. The real-host corrections and
their regression coverage are committed as
`f68e2b4922389945ecd2d3602f935871e7280e6a`; the authoritative dry run and
bounded host validation both passed against that clean commit. The launcher
refuses a dirty working tree before creating a run. Thermal selection reports
and raw JSON telemetry are committed; large derived selection views are
reproducible and intentionally ignored.

Before launch, `git status --short` must be empty and the dry-run `git commit:`
line must match the intended reviewed commit. Never add local notifier settings
or credentials to Git.

## Verification record

| Verification | Result |
|---|---|
| Python compile/import and shell syntax | pass |
| Historical full repository suite (pre-fix) | **1,101 passed, 33 warnings, 907.39 s** |
| Final focused post-fix lifecycle suite | **8 passed in 10.05 s** |
| Cross-UID guardian/PID-cleanup focused suite | **18 passed in 10.66 s** |
| Campaign regression suite after second fix | **118 passed, 4 clean-tree integration cases deselected in 51.60 s** |
| Deferred clean-tree integration cases | **4 passed in 53.28 s** after commit |
| Relevant post-fix launcher/production suite | **80 passed, 3 deselected in 43.38 s**; deselected real-launch integration cases require a clean committed tree |
| Inclusive post-fix suite attempt | **80 passed, 3 refused by the expected dirty-tree gate**; those three started no campaign |
| Lifecycle assertions | delayed startup retains controls; child sees 4.2 GHz / 125 W / 160 W; guardian and user tmux remain live; completion/startup failure/controlled stop/SIGINT/SIGTERM restore once; HUP does not restore; no guardian/session remains; dry run starts no campaign |
| Supported notifier test | delivered successfully via configured notifier |
| Post-fix production power-control validation | **pass** on corrected real-host repeat; child accepted exact values and controls remained held through guardian age 84 s |
| Post-fix production dry run | pass on clean `f68e2b4922389945ecd2d3602f935871e7280e6a`; controls restored to PL1/PL2 253 W; no campaign started |
| Scientific campaign | failed launch attempt created no evaluation; not started |

The warnings in the full suite are existing deprecation, intentional safety-
clip, environment-rendering, and deliberate missing-artifact warnings; none
is a test failure. The final pre-launch dry run must still be repeated after
the final documentation commit so it attests to the exact clean `HEAD` to be
launched.

## Bounded real-host lifecycle validation — FAIL

The final bounded validation was attempted on 2026-09-30 against clean commit
`a69f4b5e3802940499ba9412447cc2f5afbc99b0`. It used the repository's
`campaign_validation` profile, `--mode validation`, and mandatory artifact
stamp `VALIDATION - NOT A RESULT`. A temporary Python adapter added the
existing hidden `--production-power` check to both the launch check and actual
validation scheduler; it did not change repository files or scientific code.
The exact launch was:

```bash
sudo env EVODEVO_CAMPAIGN_PYTHON=/tmp/evodevo-production-power-python \
  scripts/campaign/with_cpu_power_cap.sh \
  --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
  --run-as-user ubuntu -- \
  env EVODEVO_CAMPAIGN_PYTHON=/tmp/evodevo-production-power-python \
  scripts/campaign/run_campaign.sh \
  --campaign-id bounded-postfix-lifecycle \
  --config evodevo/experiments/configs/campaign_validation.yaml \
  --output-root results/campaign --workers 18 \
  --cpu-profile thermal-validated --mode validation \
  --no-mirror --no-notify --wait-for-campaign \
  --stamp 'VALIDATION - NOT A RESULT'
```

The adapter's complete behavior was: execute `.venv/bin/python` unchanged for
all commands except `-m evodevo.campaign`, for which it appended
`--production-power`. The bounded scheduler was terminated only through the
launcher's automatic call to the supported command:

```bash
scripts/campaign/stop_campaign.sh \
  --campaign bounded-postfix-lifecycle \
  --output-root results/campaign --grace-s 120
```

Start was `2026-09-30T04:14:17Z` and the final post-cleanup snapshot was
`2026-09-30T04:17:15Z`, a bounded wall duration of 178 seconds. Evidence while
the validation child was alive:

```text
applied reversible controls:
  intel_pstate/no_turbo=0
  constraint_0_power_limit_uw=125000000
  constraint_1_power_limit_uw=160000000
  policy0..policy15/scaling_max_freq=4200000

launch_checks.production_power.required=true
launch_checks.production_power.turbo_enabled=true
launch_checks.production_power.no_turbo=0
launch_checks.production_power.p_core_max_khz[0..15]=4200000
launch_checks.production_power.pl1_uw=125000000
launch_checks.production_power.pl2_uw=160000000

root PID 770848:
  bash scripts/campaign/with_cpu_power_cap.sh ...
scheduler acknowledgement:
  scheduler.pgid=770944
validation scheduler PID 770951:
  ... python -m evodevo.campaign ... --mode validation ... --production-power
ordinary ubuntu `tmux ls`:
  evodevo-campaign: 3 windows (created Wed Sep 30 09:44:33 2026)
worker slots:
  [0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23,24,25]
scheduler CPUs:
  [26,27]
```

The child therefore saw and accepted the exact required production values,
and the root guardian, scheduler acknowledgement, and ordinary user tmux
session were simultaneously present. Controls were still at 125/160 W after
the scheduler had been alive for 132 seconds, well beyond the original
delayed-`scheduler.pgid` race window. The lifecycle nevertheless failed at the
guardian acknowledgement step:

```text
Root control guardian is not alive; stopping scheduler before controls can
escape supervision.
```

The guardian was in fact alive and root-owned. `run_campaign.sh`, executing as
`ubuntu`, tests it with `kill -0`; Linux denies that signal probe across the
UID boundary. A direct control observation demonstrated the same semantics:

```text
pid1_owner=root 1 Ss systemd
ubuntu kill -0 1: Operation not permitted (exit 1)
```

Because the check failed before the PID-file write,
`control_guardian.pid` was never created. The supported stop path reported:

```text
Stopped campaign bounded-postfix-lifecycle; no evodevo campaign processes remain.
```

Final cleanup/restoration evidence was:

```text
guardian process: absent
ordinary ubuntu `tmux ls`: no server running on /tmp/tmux-1000/default
control_guardian.pid: absent
scheduler.pgid: EXISTS, value 770944                 <-- cleanup failure
campaign/scheduler/worker process audit: none remain
no_turbo=0
policy0..7=5500000
policy8..11=5600000
policy12..15=5500000
PL1=253000000
PL2=253000000
manifest status=stopped
manifest artifact_stamp='VALIDATION - NOT A RESULT'
manifest validation_profile stamp='VALIDATION - NOT A RESULT'
manifest completed_runs=0
```

**Result: FAIL.** The real scientific campaign was neither started nor
resumed. The only created run was the explicitly stamped synthetic validation;
it produced no completed run and no policy/training artifacts. Restoration and
process cleanup succeeded, but the required live root-guardian PID
acknowledgement failed and the supported stop path did not remove
`scheduler.pgid`. The readiness gate remains unsatisfied.

### Corrections after the failed validation

`scripts/campaign/verify_control_guardian.py` now distinguishes `ESRCH` from
`EPERM`: permission denial means the PID exists but cannot be signal-probed by
the caller. It then independently fails closed unless `/proc` proves all four
process UIDs equal the required UID (root in production), the command line is
the repository's `with_cpu_power_cap.sh`, and the guardian is an actual
ancestor of the user launcher. `run_campaign.sh` writes
`control_guardian.pid` only after all checks succeed; the unprivileged test
seam still requires the corresponding current UID and cannot affect the
production wrapper, which removes that environment variable.

`run_campaign.sh` now removes `scheduler.pgid` only after its recorded process
group has no non-zombie member. It refuses to remove malformed state or state
for a live group. `stop_campaign.sh` removes the file only after its existing
group-identity checks, signal escalation, final group-death check, and campaign
process audit all succeed. Both paths are idempotent.

Focused regression coverage reproduces `EPERM`, `ESRCH`, wrong owner, wrong
wrapper identity, wrong ancestry, successful PID persistence, normal
completion, startup failure, supported stop, SIGINT, and SIGTERM. It also
asserts no guardian/scheduler PID file survives, controls remain held until
the scheduler ends, and restoration occurs exactly once. Results before the
clean commit are **18 passed in 10.66 s**. The wider campaign suite is
**118 passed, 4 clean-tree integration cases deselected in 51.60 s**; those
four are intentionally deferred until after commit because the launcher
correctly refuses a dirty tree. These test results do not satisfy the
real-host gate by themselves.

## Corrected bounded real-host lifecycle validation — PASS

After commit `f68e2b4922389945ecd2d3602f935871e7280e6a`, the four integration cases
previously blocked by the dirty-tree gate passed (**4 passed in 53.28 s**). The
authoritative command then ran against that clean commit:

```bash
sudo scripts/campaign/run_production_campaign.sh --dry-run \
  --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml
```

It printed the exact P8+E10 placement, scheduler CPUs 26–27, distinct `sda` and
`sdb` storage, configured notifier, production power values, clean Git commit,
and `LAUNCH CHECKS: PASS`. It started no scheduler or campaign and restored the
host controls.

The bounded repeat used the same existing `campaign_validation` mechanism and
temporary power-attestation adapter documented in the failed attempt, with a
new campaign ID and a 10-second controlled-stop grace period. The exact command
was:

```bash
sudo env EVODEVO_CAMPAIGN_PYTHON=/tmp/evodevo-production-power-python \
  scripts/campaign/with_cpu_power_cap.sh \
  --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
  --run-as-user ubuntu -- \
  env EVODEVO_CAMPAIGN_PYTHON=/tmp/evodevo-production-power-python \
  scripts/campaign/run_campaign.sh \
  --campaign-id bounded-postfix-lifecycle-repeat \
  --config evodevo/experiments/configs/campaign_validation.yaml \
  --output-root results/campaign --workers 18 \
  --cpu-profile thermal-validated --mode validation \
  --no-mirror --no-notify --wait-for-campaign --grace-s 10 \
  --stamp 'VALIDATION - NOT A RESULT'
```

Start was `2026-09-30T04:30:28Z`; the final cleanup snapshot was
`2026-09-30T04:32:28Z`, a bounded duration of 120 seconds. While alive:

```text
control_guardian.pid=781352
scheduler.pgid=781463
guardian process:
  root uid=0 pid=781352 ... bash scripts/campaign/with_cpu_power_cap.sh ...
identity verifier:
  guardian PID 781352 verified (exists-but-signal-inaccessible)
validation child PID=781470, scheduler PGID=781463
ordinary ubuntu tmux ls:
  evodevo-campaign: 3 windows
worker CPUs:
  0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23,24,25
scheduler CPUs:
  26,27
no_turbo=0
policy0..15=4200000
PL1=125000000
PL2=160000000
child production_power.required=true
child production_power.turbo_enabled=true
child production_power.p_core_max_khz[0..15]=4200000
child production_power.pl1_uw=125000000
child production_power.pl2_uw=160000000
```

The second sample at guardian age 84 seconds still had 21 live scheduler-group
members, both PID files, the ordinary user tmux session, and every required
control value. This is well beyond the original startup acknowledgement race.
Termination used only:

```bash
scripts/campaign/stop_campaign.sh \
  --campaign bounded-postfix-lifecycle-repeat \
  --output-root results/campaign --grace-s 10
```

It returned after 13 seconds with:

```text
Stopped campaign bounded-postfix-lifecycle-repeat; no evodevo campaign processes remain.
```

Final evidence:

```text
guardian PID 781352: absent
scheduler PGID leader 781463: absent
validation child PID 781470: absent
ordinary ubuntu tmux ls: no server running on /tmp/tmux-1000/default
control_guardian.pid: absent
scheduler.pgid: absent
no_turbo=0
policy0..7=5500000
policy8..11=5600000
policy12..15=5500000
PL1=253000000
PL2=253000000
manifest status=stopped
manifest artifact_stamp='VALIDATION - NOT A RESULT'
manifest validation_profile stamp='VALIDATION - NOT A RESULT'
manifest completed_runs=0
policy/training artifacts=0
```

**Result: PASS.** The live guardian, acknowledgement, tmux, sustained control,
child attestation, supported stop, process cleanup, PID-file cleanup, and exact
restoration gates all passed. The real scientific campaign was not started or
resumed.

## Launch procedure

1. Confirm the intended launch commit includes the reviewed lifecycle fixes and
   this passing host evidence. Confirm `git status --short` prints nothing and
   no campaign/tmux session is active.
2. Run the exact root-held preflight below. It applies, verifies, and restores
   the production controls, but writes no campaign artifacts and starts no
   scheduler:

   ```bash
   sudo scripts/campaign/run_production_campaign.sh --dry-run \
     --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml
   ```

3. Read the output. It must show the exact CPU list above, turbo/control values,
   separate `sda`/`sdb` storage, sufficient disk, configured notifier, clean
   tree, the expected calendar, and `LAUNCH CHECKS: PASS`.
4. Confirm the corrected bounded validation recorded above remains applicable
   to the exact launcher code. If launcher/control code changes, repeat it
   before launch.
5. Update this document with both post-fix host results. Only then may the
   readiness verdict be reconsidered. When deliberately ready to begin the
   scientific campaign, rerun the same command **without** `--dry-run`:

   ```bash
   sudo scripts/campaign/run_production_campaign.sh \
     --mirror-dir /mnt/evodevo-backup --local-config campaign.local.yaml
   ```

Do not pass `--workers`, `--cpu-profile`, `--mode`, `--no-mirror`,
`--no-notify`, `--allow-same-disk-mirror`, or `--wait-for-campaign` to the
production wrapper: it rejects them to protect the fixed contract. Do not use
`run_campaign.sh` directly for this production run. Do not store credentials
in Git, source, manifests, shell history, or this document.
