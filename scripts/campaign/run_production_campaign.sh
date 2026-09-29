#!/usr/bin/env bash
# Start or resume the validated thermal production campaign under a root-held
# power-control supervisor.  This is the ONLY production entry point; it
# deliberately delegates normal launch checks, tmux and recovery to
# run_campaign.sh while retaining the selected controls for the scheduler's
# full lifetime.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo scripts/campaign/run_production_campaign.sh [run_campaign options]

Fixed production contract (cannot be overridden):
  --mode real --workers 18 --cpu-profile thermal-validated
  turbo enabled, P policies 0-15 capped at 4200 MHz, RAPL PL1=125 W, PL2=160 W

Run this from a durable user tmux session.  The root parent remains alive
until the scheduler stops, then restores every altered CPU control.  Pass
--dry-run to verify the exact production launch without starting a campaign.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
for arg in "$@"; do
  [[ "$arg" == "--help" || "$arg" == "-h" ]] && { usage; exit 0; }
done
(( EUID == 0 )) || { echo "error: root is required; use sudo" >&2; exit 2; }
[[ -n "${SUDO_USER:-}" ]] || { echo "error: invoke through sudo so the campaign runs as its owner" >&2; exit 2; }

for arg in "$@"; do
  case "$arg" in
    --workers|--cpu-profile|--mode|--no-mirror|--no-notify|--allow-same-disk-mirror|--wait-for-campaign)
      echo "error: $arg is not permitted by the fixed production contract" >&2
      exit 2
      ;;
  esac
done

exec scripts/campaign/with_cpu_power_cap.sh \
  --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
  --run-as-user "$SUDO_USER" -- \
  scripts/campaign/run_campaign.sh --mode real --workers 18 \
  --cpu-profile thermal-validated --wait-for-campaign "$@"
