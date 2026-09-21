#!/usr/bin/env bash
# Single-method EA run for the Rule-Based encoder over a seed range.
#
# Usage:
#   scripts/rule_based/run.sh [CONFIG] [SEED...] [-- EXTRA_RUN_EXPERIMENT_ARGS]
#   scripts/rule_based/run.sh                                # default config, seeds 0-4
#   scripts/rule_based/run.sh path/to/config.yaml 0 1 2       # explicit config + seeds
#   scripts/rule_based/run.sh "" 0 1 -- --surrogate           # default config, custom seeds, extra flags
#
# CONFIG defaults to evodevo/experiments/configs/rule_based/walker.yaml (pass
# "" to keep the default while still specifying seeds). SEEDS defaults to
# "0 1 2 3 4". Anything after "--" is passed through to
# `python -m evodevo.experiments.run_experiment` verbatim (e.g. --surrogate,
# --overwrite). Note: this runs the PRIMARY mutation-only arm (use_crossover
# is false in the shared base config) -- the rule_based_xover ablation arm is
# a separate, explicitly labelled run, not this script.
set -euo pipefail

# Raise the open-file limit for ~28 parallel workers x per-cell JSONL handles
# (1024 default is too low; hard limit on this machine permits 65535+).
ulimit -n 65535

DEFAULT_CONFIG="evodevo/experiments/configs/rule_based/walker.yaml"
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

echo "=== rule_based encoder: config=${CONFIG} seeds=[${SEEDS[*]}] ==="
for SEED in "${SEEDS[@]}"; do
    echo "--- rule_based, seed ${SEED} ---"
    python3 -m evodevo.experiments.run_experiment --config "${CONFIG}" --seed "${SEED}" "${EXTRA_ARGS[@]}"
done
