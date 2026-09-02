#!/usr/bin/env bash
# Runs ONLY the Rule-Based encoder's tests: the contract suite parametrized
# to "rule_based", rule_based's own method-specific tests (including the
# port-equivalence test against external/rule-based-reference, if available),
# and the shared core modules it depends on (core/, morphology/, evaluation/,
# evolution/) -- surrogate evaluation only, no real PPO, so this completes
# in well under 60 seconds. Does NOT run direct's or nca's method-specific
# tests.
set -euo pipefail

echo "=== [rule_based] shared-core + method-specific tests (surrogate only, no real PPO) ==="
python3 -m pytest -m "not slow" \
    tests/core \
    tests/morphology \
    tests/evaluation \
    tests/evolution \
    tests/encoders/test_rule_based.py \
    "$@"

echo "=== [rule_based] contract suite (rule_based only) ==="
python3 -m pytest tests/encoders/test_contract.py -k rule_based "$@"
