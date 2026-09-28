#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/campaign/stop_campaign.sh [--campaign ID|latest] [--output-root DIR] [--grace-s SEC]"
}

CAMPAIGN_ID="latest"
OUTPUT_ROOT="results/campaign"
GRACE_S=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --campaign) CAMPAIGN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --grace-s) GRACE_S="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ "$CAMPAIGN_ID" == "latest" ]]; then
  CAMPAIGN_ID="$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
fi
RUN_DIR="$OUTPUT_ROOT/$CAMPAIGN_ID"
PID_FILE="$RUN_DIR/scheduler.pgid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No scheduler process-group file at $PID_FILE"
  exit 0
fi
PGID="$(<"$PID_FILE")"
if [[ ! "$PGID" =~ ^[0-9]+$ || "$PGID" -le 1 ]]; then
  echo "Invalid scheduler process group: $PGID" >&2
  exit 1
fi
if [[ "$(ps -o pgid= -p "$PGID" 2>/dev/null | tr -d ' ')" != "$PGID" ]]; then
  echo "Scheduler process group $PGID is no longer live"
  exit 0
fi

group_live() {
  ps -eo pgid=,stat= | awk -v group="$PGID" '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

kill -INT -- "-$PGID" 2>/dev/null || true
deadline=$((SECONDS + GRACE_S))
while (( SECONDS < deadline )); do
  if ! group_live; then break; fi
  sleep 1
done
if group_live; then
  kill -TERM -- "-$PGID" 2>/dev/null || true
  sleep 3
fi
if group_live; then
  kill -KILL -- "-$PGID" 2>/dev/null || true
fi
TMUX_ARGS=()
if [[ -n "${EVODEVO_TMUX_SOCKET:-}" ]]; then
  TMUX_ARGS=(-L "$EVODEVO_TMUX_SOCKET")
fi
if tmux "${TMUX_ARGS[@]}" has-session -t evodevo-campaign 2>/dev/null; then
  tmux "${TMUX_ARGS[@]}" kill-session -t evodevo-campaign
fi
if group_live; then
  echo "ERROR: process group $PGID survived stop" >&2
  exit 1
fi
if pgrep -af "evodevo\.campaign\.(scheduler|monitor|dashboard).*${CAMPAIGN_ID}" >/dev/null; then
  echo "ERROR: evodevo campaign processes for $CAMPAIGN_ID remain:" >&2
  pgrep -af "evodevo\.campaign\.(scheduler|monitor|dashboard).*${CAMPAIGN_ID}" >&2 || true
  exit 1
fi
echo "Stopped campaign $CAMPAIGN_ID; no evodevo campaign processes remain."
