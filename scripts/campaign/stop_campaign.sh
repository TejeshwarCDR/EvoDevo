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
  CAMPAIGN_ID="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  CAMPAIGN_ID="${CAMPAIGN_ID##*/}"
fi
RUN_DIR="$OUTPUT_ROOT/$CAMPAIGN_ID"
PID_FILE="$RUN_DIR/scheduler.pgid"
PGID=""
if [[ -f "$PID_FILE" ]]; then PGID="$(<"$PID_FILE")"; fi
if [[ -n "$PGID" && ( ! "$PGID" =~ ^[0-9]+$ || "$PGID" -le 1 ) ]]; then
  echo "Invalid scheduler process group: $PGID" >&2
  exit 1
fi

group_live() {
  [[ -n "$PGID" ]] || return 1
  ps -eo pgid=,stat= | awk -v group="$PGID" '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'
}

group_owned() {
  [[ -n "$PGID" ]] || return 1
  [[ "$(ps -o pgid= -p "$PGID" 2>/dev/null | tr -d ' ')" == "$PGID" ]] || return 1
  ps -o args= -p "$PGID" 2>/dev/null | grep -E "evodevo\.campaign\.(scheduler|monitor).*${CAMPAIGN_ID}" >/dev/null
}

if group_live; then
  if ! group_owned; then
    echo "Refusing to signal process group $PGID: leader does not identify campaign $CAMPAIGN_ID" >&2
    exit 1
  fi
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
fi
TMUX_ARGS=()
if [[ -n "${EVODEVO_TMUX_SOCKET:-}" ]]; then
  TMUX_ARGS=(-L "$EVODEVO_TMUX_SOCKET")
fi
if tmux "${TMUX_ARGS[@]}" has-session -t evodevo-campaign 2>/dev/null; then
  SESSION_OWNER="$(tmux "${TMUX_ARGS[@]}" show-environment -t evodevo-campaign EVODEVO_CAMPAIGN_ID 2>/dev/null || true)"
  if [[ "$SESSION_OWNER" == "EVODEVO_CAMPAIGN_ID=$CAMPAIGN_ID" ]]; then
    tmux "${TMUX_ARGS[@]}" kill-session -t evodevo-campaign
  else
    echo "Leaving tmux session intact: it belongs to ${SESSION_OWNER:-an unlabelled campaign}, not $CAMPAIGN_ID" >&2
  fi
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
