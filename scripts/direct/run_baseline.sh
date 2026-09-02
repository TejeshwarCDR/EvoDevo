#!/usr/bin/env bash
# Direct-encoding scientific control: 5 seeds on Walker-v0 with real PPO.
# This is the hard gate at roadmap step 6 — do not proceed to the NCA encoder
# until this run produces a believable fitness curve.
set -euo pipefail

CONFIG="evodevo/experiments/configs/direct/walker.yaml"

for SEED in 0 1 2 3 4; do
    echo "=== direct encoder, Walker-v0, seed ${SEED} ==="
    python3 -m evodevo.experiments.run_experiment --config "${CONFIG}" --seed "${SEED}" "$@"
done
