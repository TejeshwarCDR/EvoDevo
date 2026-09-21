#!/usr/bin/env bash
# The head-to-head comparison: direct vs NCA, Walker-v0, seeds 0-4, identical
# budgets. assert_comparable() is enforced first so an accidental config
# asymmetry cannot silently invalidate the comparison.
set -euo pipefail

# Raise the open-file limit for ~28 parallel workers x per-cell JSONL handles
# (1024 default is too low; hard limit on this machine permits 65535+).
ulimit -n 65535

DIRECT_CONFIG="evodevo/experiments/configs/direct/walker.yaml"
NCA_CONFIG="evodevo/experiments/configs/nca/walker.yaml"

python3 -c "
from evodevo.core.config import ExperimentConfig
from evodevo.experiments.fairness import assert_comparable

direct_config = ExperimentConfig.from_method_yaml('${DIRECT_CONFIG}')
nca_config = ExperimentConfig.from_method_yaml('${NCA_CONFIG}')
assert_comparable(direct_config, nca_config)
print('fairness check passed: direct and NCA configs are comparable')
"

for SEED in 0 1 2 3 4; do
    echo "=== direct encoder, Walker-v0, seed ${SEED} ==="
    python3 -m evodevo.experiments.run_experiment --config "${DIRECT_CONFIG}" --seed "${SEED}" "$@"
    echo "=== nca encoder, Walker-v0, seed ${SEED} ==="
    python3 -m evodevo.experiments.run_experiment --config "${NCA_CONFIG}" --seed "${SEED}" "$@"
done
