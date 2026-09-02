import ast
import re
from pathlib import Path

from evodevo.core.materials import (
    ACTUATORS,
    EMPTY,
    H_ACT,
    MATERIAL_NAMES,
    MATERIALS,
    N_MATERIALS,
    RIGID,
    SOFT,
    V_ACT,
)

EVODEVO_ROOT = Path(__file__).resolve().parents[2] / "evodevo"
MATERIALS_FILE = EVODEVO_ROOT / "core" / "materials.py"

# Any bare integer literal 0..5 assigned to a name matching this pattern would
# be a re-definition of the material vocabulary outside its single source of
# truth. This is deliberately narrow (name-based) rather than "any small int
# literal anywhere", since the latter would flag unrelated code (e.g. loop
# bounds, grid indices) that has nothing to do with materials.
_SUSPECT_NAME = re.compile(
    r"^(N_MATERIALS|EMPTY|RIGID|SOFT|H_ACT|V_ACT|FIXED|MATERIALS|MATERIAL_NAMES|ACTUATORS?(_IDS)?)$"
)


def _is_int_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def _module_defines_material_literal(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and _SUSPECT_NAME.match(target.id)):
                continue
            # A plain re-export/alias ("X = other_module.X" or "X = core.materials.RIGID")
            # is fine; only a literal redefinition of the vocabulary is a problem.
            # Restricted to INT constants (not any ast.Constant) so that an
            # unrelated symbolic enum sharing a name -- e.g. a string-valued
            # VoxelState.EMPTY = "empty" in the rule-based encoder's rule
            # language -- is not mistaken for a redefinition of the integer
            # material vocabulary. Bare dict/tuple/set literals are still
            # flagged (that's how MATERIALS itself would be redefined).
            if _is_int_constant(node.value):
                return True
            if isinstance(node.value, (ast.Dict, ast.Set, ast.List, ast.Tuple)):
                return True
    return False


def test_only_materials_py_defines_material_constants():
    offenders = []
    for path in EVODEVO_ROOT.rglob("*.py"):
        if path == MATERIALS_FILE:
            continue
        if _module_defines_material_literal(path):
            offenders.append(str(path))
    assert offenders == [], f"material constants redefined outside core/materials.py: {offenders}"


def test_grep_style_no_hardcoded_vocabulary_outside_materials():
    pattern = re.compile(r"H_ACT\s*=\s*3|N_MATERIALS\s*=\s*5")
    offenders = []
    for path in EVODEVO_ROOT.rglob("*.py"):
        if path == MATERIALS_FILE:
            continue
        if pattern.search(path.read_text()):
            offenders.append(str(path))
    assert offenders == []


def test_values_and_shapes():
    assert (EMPTY, RIGID, SOFT, H_ACT, V_ACT) == (0, 1, 2, 3, 4)
    assert N_MATERIALS == 5
    assert len(MATERIAL_NAMES) == N_MATERIALS
    assert MATERIAL_NAMES[H_ACT] == "H_ACT"
    assert MATERIAL_NAMES[V_ACT] == "V_ACT"
    assert ACTUATORS == frozenset({H_ACT, V_ACT})
    assert MATERIALS == {"EMPTY": 0, "RIGID": 1, "SOFT": 2, "H_ACT": 3, "V_ACT": 4}


def test_fixed_terrain_material_is_not_part_of_the_vocabulary():
    # FIXED=5 is EvoGym's terrain material; it must never be a valid robot voxel.
    assert 5 not in MATERIALS.values()
    assert 5 not in ACTUATORS
    assert N_MATERIALS == 5  # i.e. valid ids are exactly 0..4, never including 5
