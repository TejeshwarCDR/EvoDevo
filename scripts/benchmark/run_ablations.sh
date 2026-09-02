#!/usr/bin/env bash
# Minimal ablation set: A1 (dev_steps), A2 (hidden_channels), E3 (second task).
# 3 seeds per arm, identical PPO budget/population/offspring/eval_budget as the
# main comparison (evodevo/experiments/configs/nca/walker.yaml /
# direct_walker.yaml) -- only the named hyperparameter is allowed to vary.
#
# A2's hidden_channels=3 cell is IDENTICAL to A1's dev_steps=10 cell (both are
# just the NCA encoder's defaults), so it is not re-run here; ablation_report.py
# reuses the "T10" tag's runs for that cell instead of duplicating the compute.
set -euo pipefail

NCA_WALKER_CONFIG="evodevo/experiments/configs/nca/walker.yaml"
DIRECT_WALKER_CONFIG="evodevo/experiments/configs/direct/walker.yaml"
DIRECT_SECOND_TASK_CONFIG="evodevo/experiments/configs/direct/upstepper.yaml"
NCA_SECOND_TASK_CONFIG="evodevo/experiments/configs/nca/upstepper.yaml"
SEEDS="0 1 2"

echo "### A1: NCA, dev_steps in {1, 5, 10, 20} ###"
for T in 1 5 10 20; do
    for SEED in ${SEEDS}; do
        echo "=== A1 dev_steps=${T} seed=${SEED} ==="
        python3 -m evodevo.experiments.run_experiment \
            --config "${NCA_WALKER_CONFIG}" --seed "${SEED}" \
            --encoder-kwarg "dev_steps=${T}" --tag "T${T}" "$@"
    done
done

echo "### A2: NCA, hidden_channels in {0, 3} ###"
echo "(hidden_channels=3 reuses the T10 tag's runs from A1 -- same hyperparameters)"
for SEED in ${SEEDS}; do
    echo "=== A2 hidden_channels=0 seed=${SEED} ==="
    python3 -m evodevo.experiments.run_experiment \
        --config "${NCA_WALKER_CONFIG}" --seed "${SEED}" \
        --encoder-kwarg "hidden_channels=0" --tag "H0" "$@"
done

echo "### E3: direct and NCA on a second task (UpStepper-v0) ###"
for SEED in ${SEEDS}; do
    echo "=== E3 direct UpStepper-v0 seed=${SEED} ==="
    python3 -m evodevo.experiments.run_experiment \
        --config "${DIRECT_SECOND_TASK_CONFIG}" --seed "${SEED}" "$@"
    echo "=== E3 nca UpStepper-v0 seed=${SEED} ==="
    python3 -m evodevo.experiments.run_experiment \
        --config "${NCA_SECOND_TASK_CONFIG}" --seed "${SEED}" "$@"
done
