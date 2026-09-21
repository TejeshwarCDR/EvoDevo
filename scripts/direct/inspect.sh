#!/usr/bin/env bash
# PPO-free characterization of the Direct encoder: samples N random genomes
# and reports validity rate (with failure-reason breakdown), unique-body
# count, and degeneracy rate. No PPO/EA is run.
#
# Usage: scripts/direct/inspect.sh [N] [SEED]
set -euo pipefail

# Raise the open-file limit for ~28 parallel workers x per-cell JSONL handles
# (1024 default is too low; hard limit on this machine permits 65535+).
ulimit -n 65535

N="${1:-2000}"
SEED="${2:-0}"
OUTPUT_DIR="results/inspection/direct"

echo "=== inspecting direct encoder: n=${N} seed=${SEED} -> ${OUTPUT_DIR} ==="
python3 -m evodevo.experiments.inspect_encoder --encoder direct --n "${N}" --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
