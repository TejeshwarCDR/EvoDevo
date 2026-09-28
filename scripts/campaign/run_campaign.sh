#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/campaign/run_campaign.sh [--resume ID|latest] [--campaign-id ID] [--config FILE] [--output-root DIR] [--mirror-dir DIR|--no-mirror] [--workers N] [--mode real|surrogate] [--grace-s SEC]"
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
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
if [[ -n "$RESUME" && -n "$CAMPAIGN_ID" ]]; then
  echo "--resume and --campaign-id are mutually exclusive" >&2; exit 2
fi
if [[ "$RESUME" == latest ]]; then
  RESUME="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -printf '%h\n' 2>/dev/null | sort | tail -n 1)"
  RESUME="${RESUME##*/}"
  if [[ -z "$RESUME" ]]; then echo "no campaign to resume" >&2; exit 1; fi
fi
if [[ -z "$CAMPAIGN_ID" ]]; then CAMPAIGN_ID="${RESUME:-$(date -u +%Y%m%dT%H%M%SZ)}"; fi
if [[ -z "$RESUME" && -e "$OUTPUT_ROOT/$CAMPAIGN_ID/manifest.json" ]]; then
  echo "Campaign $CAMPAIGN_ID already exists; use --resume $CAMPAIGN_ID" >&2; exit 1
fi

if .venv/bin/python -c 'from evodevo.preflight.dirty_check import project_is_dirty; import os,sys; sys.exit(project_is_dirty(os.getcwd()))'; then
  :
else
  echo "Refusing to start on a dirty working tree" >&2; exit 1
fi
if [[ -n "$RESUME" ]]; then
  .venv/bin/python - "$CONFIG" "$OUTPUT_ROOT" "$RESUME" <<'PY'
import json, sys
from pathlib import Path
from evodevo.campaign.plan import CampaignPlan
from evodevo.experiments.benchmark import BenchmarkConfig
config, root, campaign = sys.argv[1:]
manifest = json.loads((Path(root) / campaign / "manifest.json").read_text())
expected = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(config)).config_hash
if manifest["config_hash"] != expected:
    raise SystemExit("Refusing config-hash mismatch")
PY
fi
ulimit -n 65535
RUN_DIR="$OUTPUT_ROOT/$CAMPAIGN_ID"
mkdir -p "$RUN_DIR"
CMD=(.venv/bin/python -m evodevo.campaign.scheduler --config "$CONFIG" --output-root "$OUTPUT_ROOT" --workers "$WORKERS" --mode "$MODE" --grace-s "$GRACE_S")
if [[ -n "$RESUME" ]]; then CMD+=(--resume "$RESUME"); else CMD+=(--campaign-id "$CAMPAIGN_ID"); fi
if [[ "$NO_MIRROR" -eq 1 ]]; then CMD+=(--no-mirror); else CMD+=(--mirror-dir "$MIRROR_DIR"); fi
printf -v CMD_QUOTED '%q ' "${CMD[@]}"
printf -v TRANSCRIPT_QUOTED '%q' "$RUN_DIR/transcript.log"
tmux "${TMUX_ARGS[@]}" new-session -d -s evodevo-campaign -n dashboard
tmux "${TMUX_ARGS[@]}" send-keys -t evodevo-campaign:dashboard \
  ".venv/bin/python -m evodevo.campaign.dashboard --campaign '$CAMPAIGN_ID' --output-root '$OUTPUT_ROOT' --watch" C-m
tmux "${TMUX_ARGS[@]}" split-window -h -l 50 -t evodevo-campaign:dashboard \
  "tail -F '$RUN_DIR/transcript.log'"
tmux "${TMUX_ARGS[@]}" split-window -v -l 15 -t evodevo-campaign:dashboard.1 "top"
tmux "${TMUX_ARGS[@]}" new-window -d -t evodevo-campaign -n scheduler \
  "script -q -f $TRANSCRIPT_QUOTED -c 'setsid $CMD_QUOTED'"
tmux "${TMUX_ARGS[@]}" new-window -d -t evodevo-campaign -n monitor \
  ".venv/bin/python -m evodevo.campaign.monitor --campaign '$CAMPAIGN_ID' --output-root '$OUTPUT_ROOT'"
echo "Started campaign $CAMPAIGN_ID in tmux session evodevo-campaign."
if [[ -t 0 && -t 1 ]]; then tmux "${TMUX_ARGS[@]}" attach-session -t evodevo-campaign; fi
