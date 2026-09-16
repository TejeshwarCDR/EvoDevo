"""Enforces that evodevo/encoders/rule_based/resolved_config.py is the
single source of truth for rule_based's resolved dynamics settings, and
that it matches RuleBasedEncoder's own constructor defaults exactly --
mirrors tests/encoders/test_resolved_config_single_source.py (the NCA
counterpart)."""
from __future__ import annotations

import inspect
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOLVED_CONFIG_PATH = os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based", "resolved_config.py")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "evodevo", "analysis")

# Names that must never be independently redefined as a module-level
# constant outside resolved_config.py -- the rule_based counterpart of the
# NCA_LAMBDA_STAR/NCA_B_STAR guard.
FORBIDDEN_CONSTANT_NAMES = (
    "RULE_BASED_CONFLICT_RESOLUTION", "RULE_BASED_POSITION_PREDICATE",
    "RULE_BASED_BOUNDARY", "RULE_BASED_NEIGHBOURHOOD",
)

_ASSIGNMENT_PATTERN = re.compile(
    r"^(" + "|".join(FORBIDDEN_CONSTANT_NAMES) + r")\s*(:\s*\w+\s*)?="
)


def _module_level_assignments(py_path: str) -> list:
    hits = []
    with open(py_path) as f:
        for line_no, line in enumerate(f, start=1):
            if _ASSIGNMENT_PATTERN.match(line):
                hits.append((line_no, line.rstrip("\n")))
    return hits


def test_resolved_config_module_exists_and_defines_primary():
    assert os.path.exists(RESOLVED_CONFIG_PATH)
    from evodevo.encoders.rule_based.resolved_config import PRIMARY, RuleBasedResolvedConfig

    assert isinstance(PRIMARY, RuleBasedResolvedConfig)


def test_no_other_module_defines_an_independent_resolved_constant():
    violations = []
    for dirpath, _dirnames, filenames in os.walk(ANALYSIS_DIR):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            hits = _module_level_assignments(path)
            if hits:
                rel = os.path.relpath(path, REPO_ROOT)
                for line_no, line in hits:
                    violations.append(f"{rel}:{line_no}: {line}")

    assert not violations, (
        "Found an independent module-level definition of a resolved rule_based dynamics "
        "constant outside resolved_config.py -- the exact six-copies risk pattern flagged "
        "for NCA in results/NCA_READINESS.md Section 10 must not recur for rule_based:\n"
        + "\n".join(violations)
    )


def test_primary_matches_rule_based_encoder_constructor_defaults():
    """PRIMARY names the encoder's own defaults, it does not introduce a
    second, independently-maintained copy of them -- if this test ever
    fails, either PRIMARY or the encoder's own defaults changed without
    updating the other."""
    from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
    from evodevo.encoders.rule_based.resolved_config import PRIMARY

    sig = inspect.signature(RuleBasedEncoder.__init__)
    for field in ("conflict_resolution", "position_predicate", "boundary", "neighbourhood", "action_monotonicity"):
        assert sig.parameters[field].default == getattr(PRIMARY, field), (
            f"RuleBasedEncoder's own default for {field!r} "
            f"({sig.parameters[field].default!r}) no longer matches "
            f"resolved_config.PRIMARY.{field} ({getattr(PRIMARY, field)!r})"
        )


def test_monotone_ablation_differs_from_primary_only_in_action_monotonicity():
    from evodevo.encoders.rule_based.resolved_config import MONOTONE_ABLATION, PRIMARY

    assert MONOTONE_ABLATION.action_monotonicity == "monotone"
    assert PRIMARY.action_monotonicity == "unrestricted"
    for field in ("conflict_resolution", "position_predicate", "boundary", "neighbourhood"):
        assert getattr(MONOTONE_ABLATION, field) == getattr(PRIMARY, field)


def test_hard_max_iterations_and_rule_bounds_are_reexported_not_redefined():
    from evodevo.encoders.rule_based._rules.genome import MAX_RULES, MIN_RULES
    from evodevo.encoders.rule_based._rules.interpreter import HARD_MAX_ITERATIONS
    from evodevo.encoders.rule_based.resolved_config import (
        HARD_MAX_ITERATIONS as REEXPORTED_HARD_MAX_ITERATIONS,
        MAX_RULES as REEXPORTED_MAX_RULES,
        MIN_RULES as REEXPORTED_MIN_RULES,
    )

    assert REEXPORTED_HARD_MAX_ITERATIONS == HARD_MAX_ITERATIONS
    assert REEXPORTED_MAX_RULES == MAX_RULES
    assert REEXPORTED_MIN_RULES == MIN_RULES
