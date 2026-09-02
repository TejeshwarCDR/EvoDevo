#!/usr/bin/env bash
# Single-method EA run for the NCA encoder over a seed range.
#
# Usage:
#   scripts/nca/run.sh [CONFIG] [SEED...] [-- EXTRA_RUN_EXPERIMENT_ARGS]
#   scripts/nca/run.sh                                # default config, seeds 0-4
#   scripts/nca/run.sh path/to/config.yaml 0 1 2       # explicit config + seeds
#   scripts/nca/run.sh "" 0 1 -- --surrogate           # default config, custom seeds, extra flags
#
# CONFIG defaults to evodevo/experiments/configs/nca/walker.yaml (pass ""
# to keep the default while still specifying seeds). SEEDS defaults to
# "0 1 2 3 4". Anything after "--" is passed through to
# `python -m evodevo.experiments.run_experiment` verbatim (e.g. --surrogate,
# --overwrite, --encoder-kwarg dev_steps=1).
set -euo pipefail

DEFAULT_CONFIG="evodevo/experiments/configs/nca/walker.yaml"
CONFIG="${1:-${DEFAULT_CONFIG}}"
[ -z "${CONFIG}" ] && CONFIG="${DEFAULT_CONFIG}"
[ "$#" -gt 0 ] && shift

SEEDS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    SEEDS+=("$1")
    shift
done
if [ "${#SEEDS[@]}" -eq 0 ]; then
    SEEDS=(0 1 2 3 4)
fi

EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA_ARGS=("$@")
fi

echo "=== nca encoder: config=${CONFIG} seeds=[${SEEDS[*]}] ==="
for SEED in "${SEEDS[@]}"; do
    echo "--- nca, seed ${SEED} ---"
    python3 -m evodevo.experiments.run_experiment --config "${CONFIG}" --seed "${SEED}" "${EXTRA_ARGS[@]}"
done
