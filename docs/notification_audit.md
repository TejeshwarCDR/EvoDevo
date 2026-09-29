# EvoDevo Campaign Notification Audit

## Summary
- **Message format:** `"[{label}] {name}: {json(details)}"` — `label` ∈ `alert` | `information` | `cleared`. `details` is JSON with sorted keys.
- **Channels:** `ntfy_topic_url` (HTTP POST requires 2xx) and `shell_command` (run with message as single arg; exit 0 required). Both used if configured.
- **Durable journal:** `alerts.jsonl` — each event written with `timestamp`, `condition`, `status` (`active`/`clear`), `details`, `notified` (bool), optional `message`, `delivery` (`journal_only`), and `notification_error`.
- **Config lookup:** explicit `--config` > `EVODEVO_NOTIFY_CONFIG` env > `campaign.local.yaml`.
- **Rate & sustain constants:** `REN_NOTIFY_S = 6*3600` (6h re-notify), `HEARTBEAT_S = 24*3600` (24h), `TEMPERATURE_SUSTAIN_S = 600` (10m), `PARTIAL_STALL_S = 3600` (1h), `LOW_UTILIZATION_S = 7200` (2h), `WORKER_RSS_LIMIT_BYTES = 40 GiB`.
- **Dedup rules:** notify on first activation, at most once per `REN_NOTIFY_S` while active, and once on clear. Heartbeat at most once per `HEARTBEAT_S`. Journal-only entries still count toward dedup.


## Alerts (name → trigger, details, example, classification, timing, delivery)

1. **`stall`**
   - Trigger: `snapshot["seconds_since_last_task"]` present and `> 2 * p95_ecore_s` passed to `inspect()`.
   - Details: `{"seconds_since_task": <float>, "threshold_s": <float>}`
   - Example: `[alert] stall: {"seconds_since_task": 1234.0, "threshold_s": 600.0}`
   - Classification: progress warning (stall).
   - Timing: immediate; 6h re-notify while active; clear notice on recovery.
   - Delivery: posted to configured channels; if no notifier, journal-only.

2. **`temperature`**
   - Trigger: sustained temperature ≥ 95°C for at least `TEMPERATURE_SUSTAIN_S` (10 min). `throttled` boolean computed via `thermal_throttle_is_genuine`.
   - Details: `{"temperature_c": <value|null>, "throttled": <bool>}`
   - Example: `[alert] temperature: {"throttled": true, "temperature_c": 99}`
   - Classification: hardware/operational warning.
   - Timing: 10m sustain → active; 6h re-notify; clear when below sustain threshold.
   - Delivery: same semantics; fail recorded in `notification_error`.

3. **`disk_main`**
   - Trigger: `main_disk_free_gb` present and `< 50`.
   - Details: `{"free_gb": <float>}`
   - Example: `[alert] disk_main: {"free_gb": 12}`
   - Classification: resource warning.
   - Timing: immediate; 6h re-notify; clear on recovery.
   - Delivery: same.

4. **`disk_mirror`**
   - Trigger: `mirror_disk_free_gb` present and `< 50`.
   - Details: `{"free_gb": <float>}`
   - Example: `[alert] disk_mirror: {"free_gb": 8}`
   - Classification: mirror resource warning.
   - Timing & Delivery: as `disk_main`.

5. **`infra_failure_rate`**
   - Trigger: `completed_ppo > 0 and infra_failure_count / completed_ppo > 0.005` ( > 0.5% failure rate).
   - Details: `{"failures": <int>, "tasks": <int>}`
   - Example: `[alert] infra_failure_rate: {"failures": 10, "tasks": 1000}`
   - Classification: reliability failure.
   - Timing: immediate; 6h re-notify; clear on improvement.
   - Delivery: same.

6. **`swap`**
   - Trigger: `swap_used_gb` present and `> 1`.
   - Details: `{"swap_used_gb": <float>}`
   - Example: `[alert] swap: {"swap_used_gb": 1.5}`
   - Classification: memory-pressure warning.
   - Timing & Delivery: as above.

7. **`mirror_failed`**
   - Trigger: `mirror_ok` is False (mirror missing or sync reported not-ok).
   - Details: `{"last_mirror_success": <timestamp|null>}`
   - Example: `[alert] mirror_failed: {"last_mirror_success": 1680000000}`
   - Classification: data-protection / mirror failure.
   - Timing: immediate; shared dedupe across processes; 6h re-notify; clear on success.
   - Delivery: same.

8. **`partial_stall`**
   - Trigger: `throughput_now_per_h` < 25% of 24h median for at least `PARTIAL_STALL_S` (=1h).
   - Details: `{"current_per_h": <float>, "median_24h_per_h": <float>}`
   - Example: `[alert] partial_stall: {"current_per_h": 1, "median_24h_per_h": 20}`
   - Classification: performance degradation.
   - Timing: 1h sustain; durable across restarts (journal restores window if within cutoff); 6h re-notify; clear when recovered.
   - Delivery: same.

9. **`worker_utilization`**
   - Trigger: `worker_utilization_last_hour` below `UTILIZATION_FLOOR[phase]` for `LOW_UTILIZATION_S` (=2h), excluding final 5% of phase.
   - Details: `{"last_hour_busy_fraction": <float>, "floor": <float>, "phase": <str>, "below_floor": <bool>, "excluded_final_5pct": <bool>}`
   - Example: `[alert] worker_utilization: {"below_floor": true, "floor": 0.84, "last_hour_busy_fraction": 0.5, "phase": "Walker-v0", "excluded_final_5pct": false}`
   - Classification: utilization/performance warning.
   - Timing: 2h sustain; durable; 6h re-notify; clear when condition resolves or phase final 5% reached.
   - Delivery: same.

10. **`worker_rss`**
    - Trigger: `worker_rss_bytes` present and `> WORKER_RSS_LIMIT_BYTES` (40 GiB).
    - Details: `{"rss_bytes": <int>, "limit_bytes": 42949672960}`
    - Example: `[alert] worker_rss: {"limit_bytes": 42949672960, "rss_bytes": 44100000000}`
    - Classification: memory runaway / failure.
    - Timing & Delivery: immediate, 6h re-notify, clear on drop below limit.

11. **`daily_heartbeat`**
   - Trigger: `heartbeat(snapshot)` called when `_last_heartbeat` is None or ≥ `HEARTBEAT_S` (24h) since last heartbeat. `inspect()` calls `heartbeat()` when due.
    - Details:
      - `phase`, `per_arm_mean_generation` (dict), `completed_evaluations`, `eta_range_s`, `infra_failure_count`, `max_temperature_24h_c`, `main_disk_free_gb`, `mirror_disk_free_gb`, `last_mirror_success`.
    - Example: `[information] daily_heartbeat: { ... }`
    - Classification: informational progress heartbeat.
   - Timing: ≤ once per 24h; `informational=True` → label `information`; journaled even when notifier absent and counts toward `_last_heartbeat`.
    - Delivery: same.

12. **`phase_complete:{task}`**
    - Trigger: explicit `AlertManager.phase_complete(task)` on phase boundary or milestone completion.
    - Details: `{"task": <task>}`
    - Example: `[information] phase_complete:Walker-v0: {"task":"Walker-v0"}`
    - Classification: milestone/informational.
    - Timing: one durable notice per phase; journal prevents duplicate emits after restart for same phase.
    - Delivery: same.

13. **`milestone_failure:{task}`**
    - Trigger: `AlertManager.milestone_failure(task, missing, error)` when milestone packaging or subprocess fails.
    - Details: `{"task": <task>, "missing_artifacts": <list>, "error": <string|null>}`
    - Example: `[alert] milestone_failure:UpStepper-v0: {"task":"UpStepper-v0","missing_artifacts":["MILESTONE_REPORT.md"],"error":"..."}`
    - Classification: failure.
    - Timing: immediate; 6h re-notify; clear not automatic (separate success notice likely).
    - Delivery: same.

14. **`[test]` (CLI test)**
    - Trigger: `python -m evodevo.campaign.alerts --test` → `Notifier.send("[test] EvoDevo campaign notification")`.
    - Details: none (raw test message).
    - Example: `[test] EvoDevo campaign notification`
    - Classification: diagnostic/operator verification.
    - Timing: one-off CLI; requires configured notifier to exit 0.
    - Delivery: same; not journaled by `AlertManager` (main uses `Notifier` directly).

15. **Other `evaluate` calls**
    - Mirror, monitor, scheduler, and milestone code may call `evaluate(...)` for other or the same conditions (e.g., `mirror` module uses `mirror_failed`, `milestone` uses `milestone_failure`).
    - Dedup across processes maintained via file lock and journal.


## Delivery failure & journal behavior
- If any channel fails (HTTP non-2xx, shell exit non-zero, timeout, or notifier not configured), `evaluate()` writes the event with `notified: false` and records `notification_error`. Heartbeats and dedup counters are updated per journal so failed attempts are visible and retried per natural schedule.
- If notifier unconfigured, `AlertManager.evaluate` writes `"delivery": "journal_only"` and still updates `_last_notice` so restarts don't re-trigger alerts immediately.


## Launch / preflight behavior (impacting notifications)
- Real-launch preflight refuses to start a real campaign unless notifier configured or `--no-notify` explicitly passed; `run_campaign.sh` / `evodevo.campaign` / `launch_checks.py` enforce this (tests verify refusal and explicit `--no-notify` acceptance).


## Important campaign information NOT sent via ntfy
- Full `alerts.jsonl` log file contents (only summary messages are posted).
- Full monitor `snapshot` objects (only selected heartbeat fields are posted).
- Per-evaluation `run_log.jsonl` contents and evaluation-level records.
- Milestone package artifacts and large files: `MILESTONE_REPORT.md`, `package_manifest.json`, figures, CSVs, `best_bodies/*`, videos, binaries, trained policy files.
- Full subprocess stdout/stderr and full logs — only trimmed `error` tails appear in `milestone_failure`.
- Full telemetry series and raw temperature history (heartbeat may include `max_temperature_24h_c` only).
- Mirror sync details and file diffs — only `mirror_failed` + `last_mirror_success` timestamp are posted.
- Full configuration/policy files and local environment artifacts (not transmitted by default).
- Any large binary/model artifacts or media files.


If you want changes (different path or additional fields), tell me and I will update the file.
