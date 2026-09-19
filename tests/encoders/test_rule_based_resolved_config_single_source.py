"""Enforces that evodevo/encoders/rule_based/resolved_config.py is the
single source of truth for rule_based's resolved dynamics settings, and
that it matches RuleBasedEncoder's own constructor defaults exactly --
mirrors tests/encoders/test_resolved_config_single_source.py (the NCA
counterpart)."""
from __future__ import annotations

import ast
import inspect
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOLVED_CONFIG_PATH = os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based", "resolved_config.py")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "evodevo", "analysis")
RULE_BASED_PKG_DIR = os.path.join(REPO_ROOT, "evodevo", "encoders", "rule_based")

# Files where hardcoding these exact literal values is a deliberate design
# choice, not a duplication risk -- checked by hand (see the module-wide
# audit report), not silently exempted:
#   - genome.py / interpreter.py / encoder.py (the RuleBasedEncoder class
#     itself): the ORIGIN of these defaults (resolved_config.py::PRIMARY is
#     documented to MIRROR these, not the reverse).
#   - rule_based_cycles.py: its own "current" default now imports PRIMARY;
#     its "pre-R2/R3 baseline" literals are a permanent historical constant
#     (row_major_last_wins/enabled), a DIFFERENT value from what this check
#     scans for, so it does not need an exemption.
_LITERAL_DUPLICATE_EXEMPT_BASENAMES = frozenset({
    "genome.py", "interpreter.py", "resolved_config.py", "encoder.py",
})

# kwarg name -> the exact literal value that must be sourced from PRIMARY
# rather than retyped -- the two rule_based dynamics fields found
# independently copied as bare call-site literals across several analysis
# modules (a different risk shape from FORBIDDEN_CONSTANT_NAMES below: no
# named constant was created, the literal was just retyped at each call
# site).
_WATCHED_LITERAL_KWARGS = {
    "position_predicate": "disabled",
    "conflict_resolution": "first_rule_index",
}


def _iter_py_files(*dirs):
    for d in dirs:
        for dirpath, _dirnames, filenames in os.walk(d):
            for fname in filenames:
                if fname.endswith(".py"):
                    yield os.path.join(dirpath, fname)


def _hardcoded_resolved_kwarg_literals(py_path: str) -> list:
    """AST-based (not regex/grep) so this cannot false-positive on a
    docstring or comment merely mentioning `position_predicate="disabled"`
    in prose -- only real ast.Call keyword arguments and function-signature
    parameter defaults count. Flags either:
      (a) a call site that passes one of _WATCHED_LITERAL_KWARGS as a bare
          string literal matching the watched value, instead of an
          attribute lookup (e.g. `PRIMARY.position_predicate`) or omitting
          the argument to inherit the callee's own default;
      (b) a function/method definition whose OWN parameter default for one
          of _WATCHED_LITERAL_KWARGS is that same bare literal, instead of
          PRIMARY's attribute -- the exact_cycle() gap found during the
          module-wide audit: a def-level default is a different AST shape
          from a call-site kwarg and needs its own check."""
    with open(py_path) as f:
        tree = ast.parse(f.read(), filename=py_path)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg not in _WATCHED_LITERAL_KWARGS:
                    continue
                if isinstance(kw.value, ast.Constant) and kw.value.value == _WATCHED_LITERAL_KWARGS[kw.arg]:
                    hits.append((node.lineno, kw.arg, kw.value.value))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            named_args = args.posonlyargs + args.args
            # defaults align to the LAST len(defaults) entries of named_args
            paired = list(zip(named_args[len(named_args) - len(args.defaults):], args.defaults))
            paired += list(zip(args.kwonlyargs, args.kw_defaults))
            for arg, default in paired:
                if default is None or arg.arg not in _WATCHED_LITERAL_KWARGS:
                    continue
                if isinstance(default, ast.Constant) and default.value == _WATCHED_LITERAL_KWARGS[arg.arg]:
                    hits.append((node.lineno, arg.arg, default.value))
    return hits

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


def test_no_module_hardcodes_a_resolved_dynamics_literal_instead_of_importing_primary():
    """Regression test for a real, found-and-fixed duplication pattern
    (module-wide silent-clip/duplicated-literal audit): several analysis
    modules called sample_developmental_rules(position_predicate="disabled")
    or RuleBasedEncoder(..., conflict_resolution="first_rule_index") as bare
    literals instead of PRIMARY.position_predicate/PRIMARY.conflict_
    resolution -- the exact "six independently-copied lambda*/B* literals"
    risk pattern already guarded for NCA, just via call-site kwargs instead
    of module-level constants, so the existing
    test_no_other_module_defines_an_independent_resolved_constant check
    above did not catch it. Fixed in rule_based_monotone_report.py,
    rule_based_cycles.py, rule_based_capacity_study.py, and
    rule_based_context_report.py; this test guards against recurrence."""
    violations = []
    for path in _iter_py_files(ANALYSIS_DIR, RULE_BASED_PKG_DIR):
        if os.path.basename(path) in _LITERAL_DUPLICATE_EXEMPT_BASENAMES:
            continue
        for line_no, kwarg, value in _hardcoded_resolved_kwarg_literals(path):
            rel = os.path.relpath(path, REPO_ROOT)
            violations.append(f"{rel}:{line_no}: {kwarg}={value!r}")

    assert not violations, (
        "Found a call site hardcoding a resolved rule_based dynamics value as a bare "
        "literal instead of importing it from resolved_config.py::PRIMARY -- the same "
        "duplicated-literal risk already fixed once for these exact fields:\n"
        + "\n".join(violations)
    )


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
