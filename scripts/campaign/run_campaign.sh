#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/campaign/run_campaign.sh [--resume ID|latest] [--campaign-id ID] [--config FILE] [--output-root DIR] [--mirror-dir DIR|--no-mirror] [--workers N] [--mode real|surrogate] [--grace-s SEC] [--no-notify] [--allow-same-disk-mirror] [--dry-run [--launch-at ISO]]"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
RESUME=""
CAMPAIGN_ID=""
OUTPUT_ROOT="results/campaign"
CONFIG="evodevo/experiments/configs/benchmark.yaml"
MIRROR_DIR=""
NO_MIRROR=0
WORKERS=20
MODE=real
GRACE_S=120
NO_NOTIFY=0
ALLOW_SAME_DISK=0
DRY_RUN=0
LAUNCH_AT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --mirror-dir) MIRROR_DIR="$2"; shift 2 ;;
    --no-mirror) NO_MIRROR=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --grace-s) GRACE_S="$2"; shift 2 ;;
    --no-notify) NO_NOTIFY=1; shift ;;
    --allow-same-disk-mirror) ALLOW_SAME_DISK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --launch-at) LAUNCH_AT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  # Writes nothing: no run directory, tmux session, mirror copy or bytecode.
  DRY=(env PYTHONDONTWRITEBYTECODE=1 "EVODEVO_NOTIFY_CONFIG=${EVODEVO_NOTIFY_CONFIG:-$REPO_ROOT/campaign.local.yaml}"
       .venv/bin/python -m evodevo.campaign --dry-run --config "$CONFIG" --output-root "$OUTPUT_ROOT"
       --workers "$WORKERS" --mode "$MODE")
  if [[ -n "$RESUME" ]]; then DRY+=(--resume "$RESUME"); elif [[ -n "$CAMPAIGN_ID" ]]; then DRY+=(--campaign-id "$CAMPAIGN_ID"); fi
  if [[ "$NO_MIRROR" -eq 1 ]]; then DRY+=(--no-mirror); elif [[ -n "$MIRROR_DIR" ]]; then DRY+=(--mirror-dir "$MIRROR_DIR"); fi
  if [[ "$NO_NOTIFY" -eq 1 ]]; then DRY+=(--no-notify); fi
  if [[ "$ALLOW_SAME_DISK" -eq 1 ]]; then DRY+=(--allow-same-disk-mirror); fi
  if [[ -n "$LAUNCH_AT" ]]; then DRY+=(--launch-at "$LAUNCH_AT"); fi
  exec "${DRY[@]}"
fi
TMUX_ARGS=()
if [[ -n "${EVODEVO_TMUX_SOCKET:-}" ]]; then TMUX_ARGS=(-L "$EVODEVO_TMUX_SOCKET"); fi
if tmux "${TMUX_ARGS[@]}" has-session -t evodevo-campaign 2>/dev/null; then
  echo "Campaign session already running; attaching."
  if [[ -t 0 && -t 1 ]]; then tmux "${TMUX_ARGS[@]}" attach-session -t evodevo-campaign; fi
  exit 0
fi
if [[ -z "$MIRROR_DIR" && "$NO_MIRROR" -ne 1 ]]; then
  echo "--mirror-dir is required unless --no-mirror is explicit" >&2; exit 2
fi
if [[ -n "$MIRROR_DIR" && "$NO_MIRROR" -eq 1 ]]; then
  echo "--mirror-dir and --no-mirror are mutually exclusive" >&2; exit 2
fi
if [[ -n "$RESUME" && -n "$CAMPAIGN_ID" ]]; then
  echo "--resume and --campaign-id are mutually exclusive" >&2; exit 2
fi
if [[ "$RESUME" == latest ]]; then
  RESUME="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  RESUME="${RESUME#* }"
  RESUME="${RESUME##*/}"
  if [[ -z "$RESUME" ]]; then echo "no campaign to resume" >&2; exit 1; fi
fi
if [[ -z "$CAMPAIGN_ID" ]]; then CAMPAIGN_ID="${RESUME:-$(date -u +%Y%m%dT%H%M%SZ)}"; fi
if [[ -z "$RESUME" && -e "$OUTPUT_ROOT/$CAMPAIGN_ID/manifest.json" ]]; then
  echo "Campaign $CAMPAIGN_ID already exists; use --resume $CAMPAIGN_ID" >&2; exit 1
fi

ulimit -n 65535
RUN_DIR="$OUTPUT_ROOT/$CAMPAIGN_ID"
NOTIFY_CONFIG="${EVODEVO_NOTIFY_CONFIG:-$REPO_ROOT/campaign.local.yaml}"
CMD=(env "EVODEVO_NOTIFY_CONFIG=$NOTIFY_CONFIG" .venv/bin/python -m evodevo.campaign --config "$CONFIG" --output-root "$OUTPUT_ROOT" --workers "$WORKERS" --mode "$MODE" --grace-s "$GRACE_S")
if [[ -n "$RESUME" ]]; then CMD+=(--resume "$RESUME"); else CMD+=(--campaign-id "$CAMPAIGN_ID"); fi
if [[ "$NO_MIRROR" -eq 1 ]]; then CMD+=(--no-mirror); else CMD+=(--mirror-dir "$MIRROR_DIR"); fi
if [[ "$NO_NOTIFY" -eq 1 ]]; then CMD+=(--no-notify); fi
if [[ "$ALLOW_SAME_DISK" -eq 1 ]]; then CMD+=(--allow-same-disk-mirror); fi
# Every refusal (dirty tree, config-hash mismatch on resume, notifier, mirror
# disk) is decided by the Python entry point itself; run it here, before any
# tmux session exists, so the operator sees the reason.
if ! "${CMD[@]}" --check-only; then
  echo "Launch refused; nothing was started." >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
printf -v NOTIFY_QUOTED '%q' "$NOTIFY_CONFIG"
printf -v CMD_QUOTED '%q ' "${CMD[@]}"
printf -v TRANSCRIPT_QUOTED '%q' "$RUN_DIR/transcript.log"
tmux "${TMUX_ARGS[@]}" new-session -d -s evodevo-campaign -n dashboard
tmux "${TMUX_ARGS[@]}" set-environment -t evodevo-campaign EVODEVO_CAMPAIGN_ID "$CAMPAIGN_ID"
tmux "${TMUX_ARGS[@]}" send-keys -t evodevo-campaign:dashboard \
  ".venv/bin/python -m evodevo.campaign status --campaign '$CAMPAIGN_ID' --output-root '$OUTPUT_ROOT' --watch" C-m
tmux "${TMUX_ARGS[@]}" split-window -h -l 50 -t evodevo-campaign:dashboard \
  "tail -F '$RUN_DIR/transcript.log'"
tmux "${TMUX_ARGS[@]}" split-window -v -l 15 -t evodevo-campaign:dashboard.1 "top"
tmux "${TMUX_ARGS[@]}" new-window -d -t evodevo-campaign -n scheduler \
  "script -q -f $TRANSCRIPT_QUOTED -c 'setsid $CMD_QUOTED'"
tmux "${TMUX_ARGS[@]}" new-window -d -t evodevo-campaign -n monitor \
  "env EVODEVO_NOTIFY_CONFIG=$NOTIFY_QUOTED .venv/bin/python -m evodevo.campaign.monitor --campaign '$CAMPAIGN_ID' --output-root '$OUTPUT_ROOT'"
echo "Started campaign $CAMPAIGN_ID in tmux session evodevo-campaign."
if [[ -t 0 && -t 1 ]]; then tmux "${TMUX_ARGS[@]}" attach-session -t evodevo-campaign; fi
