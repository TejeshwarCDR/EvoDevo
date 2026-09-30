#!/usr/bin/env bash
# The dashboard pane owns its own final frame; it must never freeze on a live frame.
set -u
campaign="$1"
output_root="$2"
pgid_file="$output_root/$campaign/scheduler.pgid"
while true; do
  .venv/bin/python -m evodevo.campaign status --campaign "$campaign" --output-root "$output_root" || true
  pgid="$(cat "$pgid_file" 2>/dev/null || true)"
  if [[ ! "$pgid" =~ ^[0-9]+$ ]] || ! ps -eo pgid=,stat= | awk -v group="$pgid" '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'; then
    status="$(python - "$output_root/$campaign/manifest.json" <<'PY'
import json, sys
try: print(json.load(open(sys.argv[1])).get("status", "stopped"))
except Exception: print("stopped")
PY
)"
    if [[ "$status" == complete ]]; then
      echo 'CAMPAIGN COMPLETE — see results/campaign/'"$campaign"'/CAMPAIGN_SUMMARY.md and CAMPAIGN_COMPLETE.txt — safe to close this tmux session'
    else
      echo 'CAMPAIGN STOPPED UNEXPECTEDLY — check alerts.jsonl and the log — resume with: scripts/campaign/run_campaign.sh --resume '"$campaign"
    fi
    exit 0
  fi
  sleep 10
done
