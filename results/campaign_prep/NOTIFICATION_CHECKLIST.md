## 13. Notification & Dashboard Verification Checklist

Already everything is run — this checklist verifies the notification/dashboard layer only. Do not re-run the full campaign test suite; check notification delivery per this list if you want to test.

- [x] `campaign_started` fires once with the locked plan, guard outcome, estimate calendar, and first anchored heartbeat time.
- [x] `phase_started:{task}` fires once per phase and remains durable across restart.
- [x] `daily_heartbeat` fires at local 18:00, is resume-safe, and does not replay missed days.
- [x] `campaign_complete` retries delivery three times and writes `CAMPAIGN_COMPLETE.txt` even if every delivery fails.
- [x] Every old and new notification carries the common status footer with phase progress, overall progress, and ETA.
- [x] Dashboard shows the banner, both progress percentages, ETA range, elapsed time, utilization target, and freshness counter.
- [x] Dashboard repaints a final static screen on both normal and abnormal scheduler exit.
- [x] ASCII fallback renders without box-drawing or emoji.
- [x] Existing alert deduplication, rate limiting, trigger thresholds, and raw journal schema are unchanged; only `display_message` is added.
- [x] No existing test was broken silently; no exact raw alert-literal assertion existed, and heartbeat timing assertions were deliberately updated for the new 18:00 requirement.
