# EvoDevo production launch readiness

**Authoritative launch document.** This file supersedes the retained
20-worker/403-hour narratives in historical reports. It describes the only
accepted production configuration, the safeguards that must remain in force,
and the checks required immediately before manual launch. A real launch was
attempted, but the scientific campaign **never started evaluation**.

## Final verdict

**NOT READY TO LAUNCH.** The real launch attempt `20260930T034016Z` exposed a
production-wrapper lifecycle race. No scientific evaluation started. The code
fix and focused simulated lifecycle tests pass, but launch readiness remains
blocked until the fix is committed and the corrected wrapper passes both a
fresh root-held production dry run and a bounded, non-scientific lifecycle
validation on the production host. Do not remove `--dry-run` before those two
checks are recorded here.

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

## Latest verified machine state (pre-fix; revalidation required)

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
| Root-held dry run | historic pre-fix pass only; a fresh post-fix run is required |
| Failed real launch | `20260930T034016Z`; controls restored; no scientific evaluation started |
| Post-fix host lifecycle validation | pending; required before readiness can be restored |

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
`69b785671c46b64106d199d82127fd4ee84a3d4f`. The lifecycle fix described above
is currently an uncommitted working-tree change and must be reviewed and
committed before host validation. The final root-held dry run reports the exact
`HEAD` it verified, and the launcher refuses a dirty working tree before
creating a run. Thermal selection reports and raw JSON telemetry are committed;
large derived selection views are reproducible and intentionally ignored.

Before launch, `git status --short` must be empty and the dry-run `git commit:`
line must match the intended reviewed commit. Never add local notifier settings
or credentials to Git.

## Verification record

| Verification | Result |
|---|---|
| Python compile/import and shell syntax | pass |
| Historical full repository suite (pre-fix) | **1,101 passed, 33 warnings, 907.39 s** |
| Focused post-fix lifecycle suite | **9 passed in 10.12 s** |
| Relevant post-fix launcher/production suite | **80 passed, 3 deselected in 43.38 s**; deselected real-launch integration cases require a clean committed tree |
| Inclusive post-fix suite attempt | **80 passed, 3 refused by the expected dirty-tree gate**; those three started no campaign |
| Lifecycle assertions | delayed startup retains controls; child sees 4.2 GHz / 125 W / 160 W; guardian and user tmux remain live; completion/startup failure/controlled stop/SIGINT/SIGTERM restore once; HUP does not restore; no guardian/session remains; dry run starts no campaign |
| Supported notifier test | delivered successfully via configured notifier |
| Post-fix production power-control validation | simulated sysfs lifecycle pass; fresh real root-held validation pending |
| Post-fix production dry run | pending |
| Scientific campaign | failed launch attempt created no evaluation; not started |

The warnings in the full suite are existing deprecation, intentional safety-
clip, environment-rendering, and deliberate missing-artifact warnings; none
is a test failure. The final pre-launch dry run must still be repeated after
the final documentation commit so it attests to the exact clean `HEAD` to be
launched.

## Launch procedure

1. Review and commit the lifecycle fix. Confirm `git status --short` prints
   nothing and no campaign/tmux session is active.
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
4. Record a bounded non-scientific lifecycle validation showing the corrected
   launcher keeps 4.2 GHz / 125 W / 160 W in force while its child is alive,
   exposes the tmux session to `ubuntu` via ordinary `tmux ls`, keeps the root
   guardian alive, and restores the exact original values after child exit.
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
