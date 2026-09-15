"""Mechanically enforces that evodevo/analysis/structural.py -- the
fitness-blind measurement substrate -- never imports anything from the PPO,
EA, or fitness-evaluation modules. This is not "it happens to run fast
without physics"; it is a structural guarantee that every metric in that
module is computable from decode() alone."""
from __future__ import annotations

import ast
import os

STRUCTURAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evodevo", "analysis", "structural.py",
)

# Module prefixes that would pull in PPO/task/reward/EA machinery.
FORBIDDEN_PREFIXES = (
    "evodevo.evaluation",  # ppo_runner.py, cache.py -- PPO + fitness caching
    "evodevo.evolution",  # ea.py, logging.py -- the EA loop
    "evogym",  # the physics simulator itself
)
FORBIDDEN_SUBSTRINGS = ("ppo", "fitness", "reward")


def _imported_module_names(path: str) -> list:
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_structural_module_never_imports_ppo_ea_or_fitness_modules():
    names = _imported_module_names(STRUCTURAL_PATH)
    assert names, "expected at least one import statement in structural.py"

    for name in names:
        lowered = name.lower()
        assert not any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), (
            f"structural.py imports '{name}', which pulls in PPO/EA machinery -- "
            "this module must be computable from decode() alone"
        )
        assert not any(substr in lowered for substr in FORBIDDEN_SUBSTRINGS), (
            f"structural.py imports '{name}', which looks fitness/reward/PPO-related"
        )


def test_structural_report_module_also_stays_ppo_free():
    report_path = os.path.join(os.path.dirname(STRUCTURAL_PATH), "structural_report.py")
    names = _imported_module_names(report_path)
    for name in names:
        lowered = name.lower()
        assert not any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), (
            f"structural_report.py imports '{name}', which pulls in PPO/EA machinery"
        )
        assert not any(substr in lowered for substr in FORBIDDEN_SUBSTRINGS), (
            f"structural_report.py imports '{name}', which looks fitness/reward/PPO-related"
        )


def test_init_condition_study_module_also_stays_ppo_free():
    study_path = os.path.join(os.path.dirname(STRUCTURAL_PATH), "init_condition_study.py")
    names = _imported_module_names(study_path)
    for name in names:
        lowered = name.lower()
        assert not any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), (
            f"init_condition_study.py imports '{name}', which pulls in PPO/EA machinery"
        )
        assert not any(substr in lowered for substr in FORBIDDEN_SUBSTRINGS), (
            f"init_condition_study.py imports '{name}', which looks fitness/reward/PPO-related"
        )


def test_depth_study_module_also_stays_ppo_free():
    depth_path = os.path.join(os.path.dirname(STRUCTURAL_PATH), "depth_study.py")
    names = _imported_module_names(depth_path)
    for name in names:
        lowered = name.lower()
        assert not any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), (
            f"depth_study.py imports '{name}', which pulls in PPO/EA machinery"
        )
        assert not any(substr in lowered for substr in FORBIDDEN_SUBSTRINGS), (
            f"depth_study.py imports '{name}', which looks fitness/reward/PPO-related"
        )


def test_mutation_calibration_module_also_stays_ppo_free():
    calib_path = os.path.join(os.path.dirname(STRUCTURAL_PATH), "mutation_calibration.py")
    names = _imported_module_names(calib_path)
    for name in names:
        lowered = name.lower()
        assert not any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), (
            f"mutation_calibration.py imports '{name}', which pulls in PPO/EA machinery"
        )
        assert not any(substr in lowered for substr in FORBIDDEN_SUBSTRINGS), (
            f"mutation_calibration.py imports '{name}', which looks fitness/reward/PPO-related"
        )
