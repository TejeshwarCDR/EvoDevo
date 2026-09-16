"""Enforces that evodevo/encoders/nca/resolved_config.py is the ONLY place
the resolved NCA architecture's constants are defined -- mirrors
NCA_READINESS.md's own Section 9(c) hardcoded-literal-check style (a grep,
scoped to what it can actually prove, over module-level assignments rather
than raw numeric literals -- decay=0.2/state_bound=5.0 happen to equal
NCAEncoder's own class defaults, so grepping for those bare numbers would
false-positive on the encoder's legitimate default declaration and on
unrelated code; the precise, unambiguous, provable claim is that no OTHER
module defines its own top-level NCA_LAMBDA_STAR/NCA_B_STAR/
NCA_HIDDEN_CHANNELS/NCA_UPDATE_RULE constant, the exact pattern
results/NCA_READINESS.md Section 10 flagged across six files).
"""
from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOLVED_CONFIG_PATH = os.path.join(REPO_ROOT, "evodevo", "encoders", "nca", "resolved_config.py")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "evodevo", "analysis")

# The exact constant names that used to be independently copied across six
# files (results/NCA_READINESS.md Section 10): architecture_delta.py,
# regeneration.py, readout_study.py's own LAMBDA_STAR/B_STAR, depth_study.py,
# structural_report.py, mutation_calibration.py.
FORBIDDEN_CONSTANT_NAMES = (
    "NCA_LAMBDA_STAR", "NCA_B_STAR", "NCA_HIDDEN_CHANNELS", "NCA_UPDATE_RULE",
    "LAMBDA_STAR", "B_STAR",
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
    from evodevo.encoders.nca.resolved_config import PRIMARY, NCAResolvedConfig

    assert isinstance(PRIMARY, NCAResolvedConfig)


def test_no_other_module_defines_an_independent_resolved_constant():
    violations = []
    for dirpath, _dirnames, filenames in os.walk(ANALYSIS_DIR):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            if os.path.abspath(path) == os.path.abspath(RESOLVED_CONFIG_PATH):
                continue
            hits = _module_level_assignments(path)
            if hits:
                rel = os.path.relpath(path, REPO_ROOT)
                for line_no, line in hits:
                    violations.append(f"{rel}:{line_no}: {line}")

    assert not violations, (
        "Found independent module-level definitions of a resolved-config constant name "
        "outside evodevo/encoders/nca/resolved_config.py -- the six-copies risk flagged in "
        "results/NCA_READINESS.md Section 10 has recurred:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize("module_name", [
    "architecture_delta", "regeneration", "readout_study", "depth_study",
    "structural_report", "mutation_calibration",
])
def test_former_six_call_sites_import_primary_from_resolved_config(module_name):
    path = os.path.join(ANALYSIS_DIR, f"{module_name}.py")
    with open(path) as f:
        text = f.read()
    assert "from evodevo.encoders.nca.resolved_config import PRIMARY" in text, (
        f"{module_name}.py no longer imports PRIMARY from resolved_config.py"
    )
