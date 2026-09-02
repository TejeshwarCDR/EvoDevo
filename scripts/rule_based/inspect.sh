#!/usr/bin/env bash
# PPO-free characterization of the Rule-Based encoder: samples N random
# genomes and reports validity rate (with failure-reason breakdown),
# unique-body count, and degeneracy rate. No PPO/EA is run.
#
# Note: RuleBasedEncoder does not currently expose a develop()
# trajectory hook (see evodevo/experiments/inspect_encoder.py), so no
# settling-behaviour section will appear in its report -- only NCA's does.
#
# Usage: scripts/rule_based/inspect.sh [N] [SEED]
set -euo pipefail

N="${1:-2000}"
SEED="${2:-0}"
OUTPUT_DIR="results/inspection/rule_based"

echo "=== inspecting rule_based encoder: n=${N} seed=${SEED} -> ${OUTPUT_DIR} ==="
python3 -m evodevo.experiments.inspect_encoder --encoder rule_based --n "${N}" --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
