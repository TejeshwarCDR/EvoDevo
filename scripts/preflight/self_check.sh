#!/usr/bin/env bash
# THE ONLY EXECUTION THIS PROJECT'S "finish Step 2" build prompt permits:
# a bounded, trivial-settings run of every pre-flight stage, proving every
# code path executes and every artifact is written -- producing NO number
# that may ever be cited as a real measurement. See evodevo/preflight/
# __main__.py's SELF_CHECK_OVERRIDES/SELF_CHECK_FULL_OVERRIDES and
# stages/__init__.py's SELF_CHECK_STAMP for the mechanics this wraps.
#
# Usage:
#   scripts/preflight/self_check.sh          # --self-check, single arm/task, 3-minute budget
#   scripts/preflight/self_check.sh --full   # --self-check-full, all 3 arms x 2 tasks, 10-minute budget
#
# Must complete within its own time limit. If it does not, that is
# reported here as a blocking finding, not silently extended.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

FULL=0
if [[ "${1:-}" == "--full" ]]; then
  FULL=1
fi

if [[ $FULL -eq 1 ]]; then
  FLAG="--self-check-full"
  TIME_LIMIT_S=600
  GLOB="results/preflight/selfcheck_full_*/"
  LABEL="self-check-full (all 3 arms x 2 tasks)"
else
  FLAG="--self-check"
  TIME_LIMIT_S=180
  GLOB="results/preflight/selfcheck_*/"
  LABEL="self-check (single arm/task)"
fi

echo "== EvoDevo pre-flight ${LABEL} =="
echo "This is a bounded infrastructure check, NOT a measurement -- see"
echo "PREFLIGHT_READINESS.md Section 7 for what it proves. Its numbers must"
echo "never be cited as real B1-B7 results."
echo "time limit: ${TIME_LIMIT_S}s"
echo ""

START=$(date +%s)
timeout "${TIME_LIMIT_S}s" "$PYTHON" -m evodevo.preflight "$FLAG" --no-dashboard
STATUS=$?
END=$(date +%s)
ELAPSED=$((END - START))

if [[ $STATUS -eq 124 ]]; then
  echo "" >&2
  echo "BLOCKING FINDING: ${LABEL} did not complete within its own ${TIME_LIMIT_S}s budget." >&2
  echo "Per its own definition this is reported as a blocking finding, not extended." >&2
  # Even on timeout, try to report which stage was in flight -- that is
  # itself the "which stage/combination is the bottleneck" report step
  # 2b/2d asked for, straight from the manifest the harness already wrote.
  RUN_DIR=$(ls -dt $GLOB 2>/dev/null | head -1)
  if [[ -n "$RUN_DIR" ]]; then
    echo "run directory: $RUN_DIR" >&2
    echo "last known stage statuses:" >&2
    "$PYTHON" -c "
import json
with open('${RUN_DIR}manifest.json') as f:
    m = json.load(f)
for name, entry in sorted(m.get('stages', {}).items()):
    print(f'  {name}: {entry.get(\"status\")} (duration_s={entry.get(\"duration_s\")})')
" >&2
  fi
  exit 1
fi

if [[ $STATUS -ne 0 ]]; then
  echo "" >&2
  echo "BLOCKING FINDING: ${LABEL} exited with status $STATUS after ${ELAPSED}s." >&2
  exit "$STATUS"
fi

echo ""
echo "${LABEL} completed in ${ELAPSED}s (limit ${TIME_LIMIT_S}s)."

RUN_DIR=$(ls -dt $GLOB 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
  echo "BLOCKING FINDING: ${LABEL} exited 0 but no matching run directory was found (${GLOB})." >&2
  exit 1
fi
echo "run directory: $RUN_DIR"

UNSTAMPED=0
for f in "$RUN_DIR"stages/*.json; do
  [[ -e "$f" ]] || continue
  if ! grep -q "SELF-CHECK" "$f"; then
    echo "BLOCKING FINDING: $f is missing its SELF-CHECK stamp." >&2
    UNSTAMPED=1
  fi
done

for f in "$RUN_DIR"report/*.md; do
  [[ -e "$f" ]] || continue
  if ! grep -q "SELF-CHECK" "$f"; then
    echo "BLOCKING FINDING: $f is missing its SELF-CHECK stamp." >&2
    UNSTAMPED=1
  fi
done

N_FIGURES=$(ls "$RUN_DIR"report/*.png 2>/dev/null | wc -l)
echo "figures produced: ${N_FIGURES} (title-stamping verified by tests/preflight/test_cli.py, not by this shell script -- a PNG's embedded title text is not reliably greppable as plain text)"

echo ""
echo "-- per-stage / per-arm / per-task detail --"
"$PYTHON" -c "
import json, os

run_dir = '$RUN_DIR'
manifest = json.load(open(os.path.join(run_dir, 'manifest.json')))
for name, entry in sorted(manifest.get('stages', {}).items()):
    print(f\"{name}: {entry.get('status')} ({entry.get('duration_s')}s)\")
    path = os.path.join(run_dir, 'stages', f'{name}.json')
    if not os.path.exists(path):
        continue
    data = json.load(open(path))
    if name == 'evaluation_cost':
        for arm_id, arm in (data.get('arms') or {}).items():
            for task_name, task in (arm.get('tasks') or {}).items():
                print(f\"    arm={arm_id:<14} task={task_name:<14} n_successful={task.get('n_successful')} n_failed={task.get('n_failed')} partial={task.get('partial')}\")
    if name in ('learning_curves', 'evaluation_noise'):
        for task_name, task in (data.get('tasks') or {}).items():
            print(f\"    task={task_name:<14} partial={task.get('partial')}\")
"

if [[ $UNSTAMPED -ne 0 ]]; then
  exit 1
fi

echo ""
echo "every JSON/markdown artifact under $RUN_DIR is stamped SELF-CHECK -- NOT A MEASUREMENT."
exit 0
