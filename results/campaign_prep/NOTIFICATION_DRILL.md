# Notification/dashboard synthetic drill

The direct, journal-only drill used a temporary directory and deleted it on exit. It simulated launch, phase start, all ten existing alert conditions, heartbeat, stall recovery, and completion. It emitted all 14 logical notification types with `display_message` footers and wrote its temporary completion marker. No PPO, scheduler launch, tmux session, or `campaign.local.yaml` was used.

TTY RUNNING render:

```text
════════════════════════════════════════════════════════════════════════
RUNNING
Phase 20% | Overall 10% | ETA: 2026-01-01 15:30 IST (range 2026-01-01 15:00 IST–2026-01-01 16:00 IST)
Elapsed: 1:00:00 | utilization now: 90.0% | target: 84.7%
As of 2026-01-01 14:00:00 IST (4s ago)
════════════════════════════════════════════════════════════════════════
```

Non-TTY ASCII COMPLETE render:

```text
========================================================================
CAMPAIGN COMPLETE
Phase 100% | Overall 100% | ETA: COMPLETE
Elapsed: 1:00:00 | utilization now: 90.0% | target: 83.5%
As of 2026-01-01 14:00:00 IST (4s ago)
========================================================================
```
