#!/usr/bin/env bash
# Runs ONLY the Direct encoder's tests: the contract suite parametrized to
# "direct", direct's own method-specific tests, and the shared core modules
# it depends on (core/, morphology/, evaluation/, evolution/) -- surrogate
# evaluation only, no real PPO, so this completes in well under 60 seconds.
# Does NOT run nca's or rule_based's method-specific tests.
set -euo pipefail

echo "=== [direct] shared-core + method-specific tests (surrogate only, no real PPO) ==="
python3 -m pytest -m "not slow" \
    tests/core \
    tests/morphology \
    tests/evaluation \
    tests/evolution \
    tests/encoders/test_direct.py \
    "$@"

echo "=== [direct] contract suite (direct only) ==="
python3 -m pytest tests/encoders/test_contract.py -k direct "$@"
