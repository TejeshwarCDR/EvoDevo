#!/usr/bin/env bash
# PPO-free characterization of the NCA encoder: samples N random genomes and
# reports validity rate (with failure-reason breakdown), unique-body count,
# degeneracy rate, and developmental settling behaviour (NCA exposes a
# develop() trajectory hook). No PPO/EA is run.
#
# Usage: scripts/nca/inspect.sh [N] [SEED]
set -euo pipefail

N="${1:-2000}"
SEED="${2:-0}"
OUTPUT_DIR="results/inspection/nca"

echo "=== inspecting nca encoder: n=${N} seed=${SEED} -> ${OUTPUT_DIR} ==="
python3 -m evodevo.experiments.inspect_encoder --encoder nca --n "${N}" --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
