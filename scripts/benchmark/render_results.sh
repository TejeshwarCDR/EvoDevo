#!/usr/bin/env bash
# For each arm's best evolved body (from results/runs/<encoder>/<task>/seed_*/),
# renders a static morphology PNG, a rollout gif+mp4, and (where available) a
# development-trajectory figure, writing to results/media/<encoder>/<task>/.
# Also writes a single side-by-side comparison figure of every arm's best
# morphology. Rendering never crashes this script -- failures are logged and
# skipped (see evodevo/visualization/render_results.py).
#
# Usage: scripts/benchmark/render_results.sh [TASK] [RUNS_ROOT] [OUTPUT_ROOT]
set -euo pipefail

TASK="${1:-Walker-v0}"
RUNS_ROOT="${2:-results/runs}"
OUTPUT_ROOT="${3:-results/media}"

echo "=== rendering results: task=${TASK} runs_root=${RUNS_ROOT} -> ${OUTPUT_ROOT} ==="
python3 -m evodevo.visualization.render_results \
    --runs-root "${RUNS_ROOT}" \
    --task "${TASK}" \
    --output-root "${OUTPUT_ROOT}"
