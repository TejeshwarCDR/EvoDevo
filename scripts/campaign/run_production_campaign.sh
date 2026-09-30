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

The root control guardian ignores SSH hangup and remains alive until the
scheduler stops, then restores every altered CPU control.  The campaign tmux
server is owned by the sudo-invoking user on that user's ordinary tmux socket.
Pass --dry-run to verify the exact production launch without starting one.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
for arg in "$@"; do
  [[ "$arg" == "--help" || "$arg" == "-h" ]] && { usage; exit 0; }
done
(( EUID == 0 )) || { echo "error: root is required; use sudo" >&2; exit 2; }
[[ -n "${SUDO_USER:-}" ]] || { echo "error: invoke through sudo so the campaign runs as its owner" >&2; exit 2; }
[[ "$SUDO_USER" == ubuntu ]] || { echo "error: production campaign must run as ubuntu (sudo invoker is $SUDO_USER)" >&2; exit 2; }

for arg in "$@"; do
  case "$arg" in
    --workers|--cpu-profile|--mode|--no-mirror|--no-notify|--allow-same-disk-mirror|--wait-for-campaign)
      echo "error: $arg is not permitted by the fixed production contract" >&2
      exit 2
      ;;
  esac
done

# Never permit test-only power paths, Python substitutions, custom tmux
# sockets, or an enclosing tmux client to leak through sudo into production.
# run_campaign.sh is executed after runuser drops to SUDO_USER, so its default
# tmux server/socket is both owned by and visible to that user via `tmux ls`.
unset EVODEVO_POWER_CONTROL_TEST_ROOT EVODEVO_ALLOW_UNPRIVILEGED_POWER_TEST
unset EVODEVO_POWER_CONTROL_TRACE EVODEVO_CAMPAIGN_PYTHON
unset EVODEVO_SCHEDULER_START_TIMEOUT_S EVODEVO_TMUX_SOCKET TMUX TMUX_TMPDIR

exec scripts/campaign/with_cpu_power_cap.sh \
  --turbo --p-max-mhz 4200 --p-cpus 0-15 --pl1-w 125 --pl2-w 160 \
  --run-as-user "$SUDO_USER" -- \
  env -u TMUX -u TMUX_TMPDIR -u EVODEVO_TMUX_SOCKET scripts/campaign/run_campaign.sh --mode real --workers 18 \
  --cpu-profile thermal-validated --wait-for-campaign "$@"
