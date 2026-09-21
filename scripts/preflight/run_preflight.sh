#!/usr/bin/env bash
# Operator launcher for the REAL pre-flight (python -m evodevo.preflight).
#
# Creates a tmux session named "evodevo-preflight" with a three-pane
# layout (D11): the live dashboard on top (a full pty-wrapped run of
# `python -m evodevo.preflight`, captured verbatim to a plain-text
# transcript under the run directory via `script`), a tail of the run's
# structured event log bottom-left, and htop (or top, if htop is not
# installed) bottom-right.
#
# Idempotent: if the session already exists, this attaches to it instead
# of starting a second pre-flight -- running this script twice in a row is
# always safe.
set -euo pipefail

SESSION_NAME="evodevo-preflight"
BASE_DIR="results/preflight"
RUN_ID=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/preflight/run_preflight.sh [--run-id ID] [--base-dir DIR] [-- EXTRA_ARGS...]

Starts (or attaches to) a tmux session named "evodevo-preflight" running the
real pre-flight harness (`python -m evodevo.preflight`) in a three-pane
layout: live dashboard (top), event-log tail (bottom-left), htop/top
(bottom-right). The full pty transcript is captured verbatim to
<run_dir>/logs/transcript.log via `script`.

Idempotent: if the session is already running, this attaches to it instead
of starting a second pre-flight -- safe to run more than once.

Options:
  --run-id ID     resume an existing run directory (its folder name under
                   --base-dir); default starts a new run
  --base-dir DIR  root directory pre-flight runs live under
                   (default: results/preflight)
  -h, --help      print this usage and exit without starting or attaching
                   to anything

Anything after a literal "--" is passed through unchanged to
`python -m evodevo.preflight` (e.g. --timeout-s 43200, --only machine_profile).

To attach to an already-running session from another terminal:
  tmux attach -t evodevo-preflight

To resume a pre-flight that was interrupted (Ctrl-C, tmux kill, machine
reboot), first find its run-id under --base-dir, then:
  scripts/preflight/run_preflight.sh --run-id <run-id>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' is already running -- attaching (NOT starting a second pre-flight)."
  echo "  (to resume a different/interrupted run instead, first: tmux kill-session -t $SESSION_NAME)"
  exec tmux attach -t "$SESSION_NAME"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

if [[ -z "$RUN_ID" ]]; then
  # Pre-create the run directory here (not inside python -m evodevo.preflight
  # itself) so this script knows the run-id -- and therefore the transcript
  # and event-log paths for the other two panes -- before the harness
  # process even starts. evodevo.preflight.run_dir is numpy/torch-free at
  # module scope (see its own module discipline), so this is a safe,
  # ordinary subprocess -- it does not touch evodevo.preflight.__main__ or
  # its own separate import-order requirement.
  RUN_ID="$("$PYTHON" -c "
from evodevo.preflight.run_dir import create_or_resume_run_dir
run_dir = create_or_resume_run_dir(base_dir='$BASE_DIR')
print(run_dir.run_id)
")"
  echo "new run: $RUN_ID"
else
  if [[ ! -d "$BASE_DIR/$RUN_ID" ]]; then
    echo "error: --run-id '$RUN_ID' does not exist under '$BASE_DIR' -- resume requires an existing run directory." >&2
    exit 1
  fi
  echo "resuming run: $RUN_ID"
fi

RUN_DIR="$BASE_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs"
TRANSCRIPT="$RUN_DIR/logs/transcript.log"
EVENTS="$RUN_DIR/events.jsonl"
touch "$EVENTS"

PREFLIGHT_CMD="$PYTHON -m evodevo.preflight --run-id $RUN_ID --base-dir $BASE_DIR"
for arg in "${EXTRA_ARGS[@]}"; do
  PREFLIGHT_CMD="$PREFLIGHT_CMD $(printf '%q' "$arg")"
done

echo "run directory: $RUN_DIR"
echo "transcript:     $TRANSCRIPT"
echo "command:        $PREFLIGHT_CMD"

if command -v htop >/dev/null 2>&1; then
  RESOURCE_CMD="htop"
else
  RESOURCE_CMD="top"
fi

tmux new-session -d -s "$SESSION_NAME" -n preflight -x 220 -y 50

# Top pane: the pre-flight itself, run under `script` for a full pty
# transcript (captures the dashboard's own ANSI redraws faithfully, not
# just plain text) written verbatim to $TRANSCRIPT.
tmux send-keys -t "$SESSION_NAME:preflight" \
  "cd '$REPO_ROOT' && script -qefc \"$PREFLIGHT_CMD\" '$TRANSCRIPT'" C-m

# Split: bottom-left = event-log tail, bottom-right = htop/top.
tmux split-window -v -t "$SESSION_NAME:preflight" -p 35
tmux send-keys -t "$SESSION_NAME:preflight" \
  "cd '$REPO_ROOT' && echo '-- tailing $EVENTS --' && tail -n 40 -f '$EVENTS'" C-m

tmux split-window -h -t "$SESSION_NAME:preflight"
tmux send-keys -t "$SESSION_NAME:preflight" "$RESOURCE_CMD" C-m

tmux select-pane -t "$SESSION_NAME:preflight.0"

echo ""
echo "started in tmux session '$SESSION_NAME'."
echo "attach with:  tmux attach -t $SESSION_NAME"
echo "resume with:  scripts/preflight/run_preflight.sh --run-id $RUN_ID"
echo ""

exec tmux attach -t "$SESSION_NAME"
