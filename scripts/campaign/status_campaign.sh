#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/campaign/status_campaign.sh [--campaign ID|latest] [--output-root DIR]"
}

CAMPAIGN_ID="latest"
OUTPUT_ROOT="results/campaign"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --campaign) CAMPAIGN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ "$CAMPAIGN_ID" == "latest" ]]; then
  CAMPAIGN_ID="$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  CAMPAIGN_ID="${CAMPAIGN_ID##*/}"
fi
if [[ -z "$CAMPAIGN_ID" || ! -f "$OUTPUT_ROOT/$CAMPAIGN_ID/manifest.json" ]]; then
  echo "No campaign manifest found" >&2
  exit 1
fi
exec .venv/bin/python -m evodevo.campaign.dashboard --campaign "$CAMPAIGN_ID" --output-root "$OUTPUT_ROOT"
