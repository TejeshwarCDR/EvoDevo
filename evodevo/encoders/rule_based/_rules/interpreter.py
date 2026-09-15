# Ported from a standalone reference implementation's developmental_interpreter.py
# (commit 703e711, "Initial project implementation"; that checkout has since
# been permanently removed -- see docs/decisions.md). Original author: the
# project's Rule-Based teammate. This is a port, not a rewrite, with ONE
# deliberate, documented exception: the locally redefined EMPTY/SOFT/HARD/
# HORIZONTAL/VERTICAL numeric constants are now sourced from
# evodevo.core.materials (this project's single authoritative material
# vocabulary), and in doing so the original's SOFT=1/HARD=2 codes -- which
# did not match EvoGym's real material vocabulary and were a pre-existing
# bug -- are corrected to RIGID=1/SOFT=2, matching EvoGym. See the "Shared
# numeric voxel codes" comment below for the full explanation. The
# rule-matching/action-application algorithm, the two-pass synchronous
# update order, and the single-SOFT-seed-at-centre initial condition are
# all otherwise unchanged. This equivalence claim's permanent proof is the
# frozen fixture at tests/encoders/fixtures/rule_based_port_equivalence.json
# (see tests/encoders/test_rule_based.py) -- the original reference source
# is no longer present to re-verify against.
"""
Deterministic, bounded-iteration interpreter for the V2 developmental rule
language (rule_types.py).

    DevelopmentalRuleGenome  --develop_rules-->  final voxel body

This module is completely independent from V1's rule_interpreter.py (not
ported here, since V1 is a superseded earlier iteration).

DEVELOPMENTAL STATE vs. FINAL VOXEL TYPE
-----------------------------------------
The grid manipulated by this interpreter represents the CURRENT
DEVELOPMENTAL STATE at each iteration — not yet the final phenotype. Only
the grid returned after the last iteration is the final EvoGym voxel body
passed on to validation (see rule_encoding.py). The developmental state and
the final voxel type share the same 5-symbol vocabulary
(EMPTY/SOFT/HARD/ACTUATOR_HORIZONTAL/ACTUATOR_VERTICAL) and the same
underlying integer codes as evodevo.core.materials (0..4), purely so the
final grid is directly usable as an EvoGym body with no extra translation
step — this does not mean intermediate grids are themselves valid or
meaningful robots.

DETERMINISM
-----------
This interpreter never calls random.*. The same genome, shape, iteration
count, and seed_position always produce the same final grid.

SYNCHRONOUS UPDATES
--------------------
Each iteration first RESOLVES every cell's first-matching rule purely by
reading the CURRENT (pre-iteration) grid — no writes happen during this
resolution pass, so no rule ever sees another rule's effect from the same
iteration. Only after every cell's decision is resolved does the
iteration WRITE: first all self-directed actions (SET_EMPTY/SET_SOFT/
SET_HARD/SET_ACTUATOR_*/COPY_NEIGHBOR, which write to the acting cell
itself), then all GROW_DIRECTIONAL actions (which write to a NEIGHBOR
cell). Growth is applied after self-directed writes so that a growth
target's own same-iteration self-action (evaluated, like everything else,
against its pre-growth EMPTY state) can never immediately erase growth
that just arrived there in the same sweep — without this ordering, any
cell with a catch-all rule (e.g. ALWAYS) would silently undo every
neighbor's growth into it on every single iteration, making
GROW_DIRECTIONAL unusable. This two-pass write order is a deliberate,
documented resolution of an ordering ambiguity the task specification
does not otherwise address; it does not let any rule observe another
rule's effect mid-iteration (the resolution pass that decides every
cell's action is still 100% based on the untouched current grid).

RULE MATCHING
--------------
For each cell, in row-major order, the rules in genome order are checked;
the first rule whose condition matches the cell (evaluated against the
CURRENT grid) determines that cell's action for this iteration. If no
rule matches, the cell is unaffected this iteration (next_grid keeps
whatever current_grid already had there, unless a *different* cell's
GROW_DIRECTIONAL action targets it — see below).

ACTION SEMANTICS
-----------------
- SET_EMPTY / SET_SOFT / SET_HARD / SET_ACTUATOR_HORIZONTAL /
  SET_ACTUATOR_VERTICAL: unconditionally set the ACTING cell (i, j) in
  next_grid to the corresponding state (this can overwrite whatever was
  already at (i, j)).
- GROW_DIRECTIONAL(direction, voxel_type): let (ti, tj) be the neighbor of
  (i, j) in `direction`. If (ti, tj) is inside the grid AND is EMPTY in
  the CURRENT grid, set next_grid[ti, tj] = voxel_type. If (ti, tj) is
  out of bounds, or is not EMPTY in the current grid (an already-occupied
  cell can never be overwritten by growth), this is a no-op. Because
  cells are processed in row-major order and every write always checks
  the CURRENT (pre-iteration) grid rather than an in-progress next_grid,
  it is possible for two different acting cells in the same iteration to
  both target the same empty neighbor; this implementation resolves that
  deterministically by row-major write order (the later-processed acting
  cell's write to next_grid[ti, tj] simply overwrites the earlier one) —
  documented here explicitly since the task's specification does not
  otherwise define same-target growth collisions.
- COPY_NEIGHBOR(direction): let (si, sj) be the neighbor of (i, j) in
  `direction`. If (si, sj) is inside the grid, set next_grid[i, j] =
  current_grid[si, sj] (copying EMPTY is allowed — this can erase an
  already-occupied acting cell, unlike GROW_DIRECTIONAL's protection,
  since COPY_NEIGHBOR is a direct "set the acting cell" action, not a
  growth-into-empty-space action). If (si, sj) is out of bounds, this is
  a no-op (next_grid[i, j] keeps its current_grid value).

BOUNDED GRID
-------------
The grid never resizes and growth targets outside [0, rows) x [0, cols)
are always ignored (no wraparound).

BOUNDED ITERATIONS
--------------------
DEFAULT_MAX_ITERATIONS caps the default iteration count; HARD_MAX_ITERATIONS
is enforced as an absolute ceiling — requesting more raises ValueError.
There is no recursion anywhere in this module.
"""

from typing import Optional, Tuple

import numpy as np

from evodevo.core.materials import EMPTY, H_ACT, RIGID, SOFT, V_ACT
from evodevo.encoders.rule_based._rules.types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)

# Shared numeric voxel codes: sourced from evodevo.core.materials (this
# project's single authoritative vocabulary) rather than redefined locally.
#
# DELIBERATE DEVIATION FROM THE ORIGINAL (the one place this port is not
# byte-for-byte behavior-preserving): the original module's own constants
# were EMPTY=0, SOFT=1, HARD=2, HORIZONTAL=3, VERTICAL=4. That does NOT
# match EvoGym's actual material vocabulary (evogym.utils.VOXEL_TYPES, and
# this project's evodevo.core.materials): EMPTY is 0, RIGID is 1, SOFT is 2,
# H_ACT is 3, V_ACT is 4 -- SOFT and RIGID/HARD are swapped between the two. This
# was a pre-existing bug in the original reference implementation: every
# body it ever generated and fed into real EvoGym physics would have had
# its soft and rigid voxels simulated backwards from what its own rule
# names ("SET_SOFT" / "SET_HARD") suggest (is_connected/has_actuator are
# unaffected, since they only distinguish EMPTY vs. non-EMPTY and
# H_ACT/V_ACT). This has been corrected here to use EvoGym's real
# convention, per an explicit decision recorded in docs/decisions.md --
# see tests/encoders/test_rule_based.py for the equivalence test that
# proves this SOFT/HARD swap is the *only* behavioural difference from the
# original (every other rule-matching/growth/synchronicity semantic is
# unchanged). HORIZONTAL/VERTICAL are unaffected by this fix.
HARD = RIGID
HORIZONTAL = H_ACT
VERTICAL = V_ACT

STATE_TO_VOXEL = {
    VoxelState.EMPTY: EMPTY,
    VoxelState.SOFT: SOFT,
    VoxelState.HARD: HARD,
    VoxelState.ACTUATOR_HORIZONTAL: HORIZONTAL,
    VoxelState.ACTUATOR_VERTICAL: VERTICAL,
}

_SET_ACTION_TO_VOXEL = {
    ActionType.SET_EMPTY: EMPTY,
    ActionType.SET_SOFT: SOFT,
    ActionType.SET_HARD: HARD,
    ActionType.SET_ACTUATOR_HORIZONTAL: HORIZONTAL,
    ActionType.SET_ACTUATOR_VERTICAL: VERTICAL,
}

_DIRECTION_OFFSET = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

DEFAULT_MAX_ITERATIONS = 5
# Raised from 10 to 40 so this arm's `dev_steps`/`developmental_iterations`
# can legitimately be exercised as a crossed experimental factor up to the
# same hard cap as evodevo/analysis/structural.py's T_MAX_DEFAULT (the
# depth-selection study's settling/light-cone probes already ran the
# interpreter to 40 steps via repeated iterations=1 calls, which this
# ceiling never gated -- this only affects a single from-scratch call with
# a large `iterations` value, e.g. RuleBasedEncoder.decode() at deep T).
# Purely a safety-ceiling value, not part of the rule language/vocabulary.
HARD_MAX_ITERATIONS = 40


def _in_bounds(i: int, j: int, shape: Tuple[int, int]) -> bool:
    rows, cols = shape
    return 0 <= i < rows and 0 <= j < cols


def _condition_matches(rule: DevelopmentalRule, i: int, j: int, grid: np.ndarray, shape: Tuple[int, int]) -> bool:
    rows, cols = shape

    if rule.condition_type == ConditionType.ALWAYS:
        return True

    if rule.condition_type == ConditionType.CELL_TYPE:
        return grid[i][j] == STATE_TO_VOXEL[rule.condition_params["state"]]

    if rule.condition_type == ConditionType.NEIGHBOR_COUNT:
        threshold = rule.condition_params["threshold"]
        count = 0
        for di, dj in _DIRECTION_OFFSET.values():
            ni, nj = i + di, j + dj
            if _in_bounds(ni, nj, shape) and grid[ni][nj] != EMPTY:
                count += 1
        return count >= threshold

    if rule.condition_type == ConditionType.EMPTY_NEIGHBOR:
        direction = rule.condition_params["direction"]
        di, dj = _DIRECTION_OFFSET[direction]
        ni, nj = i + di, j + dj
        return _in_bounds(ni, nj, shape) and grid[ni][nj] == EMPTY

    if rule.condition_type == ConditionType.POSITION:
        position = rule.condition_params["position"]
        is_boundary = i == 0 or i == rows - 1 or j == 0 or j == cols - 1
        if position == PositionType.BOUNDARY:
            return is_boundary
        if position == PositionType.INTERIOR:
            return not is_boundary
        if position == PositionType.TOP:
            return i == 0
        if position == PositionType.BOTTOM:
            return i == rows - 1
        if position == PositionType.LEFT:
            return j == 0
        if position == PositionType.RIGHT:
            return j == cols - 1
        raise ValueError(f"Unsupported PositionType: {position}")

    raise ValueError(f"Unsupported condition_type: {rule.condition_type}")


def _apply_action(rule: DevelopmentalRule, i: int, j: int, current_grid: np.ndarray, next_grid: np.ndarray, shape: Tuple[int, int]) -> None:
    if rule.action_type in _SET_ACTION_TO_VOXEL:
        next_grid[i][j] = _SET_ACTION_TO_VOXEL[rule.action_type]
        return

    if rule.action_type == ActionType.GROW_DIRECTIONAL:
        direction = rule.action_params["direction"]
        voxel_type = rule.action_params["voxel_type"]
        di, dj = _DIRECTION_OFFSET[direction]
        ti, tj = i + di, j + dj
        if _in_bounds(ti, tj, shape) and current_grid[ti][tj] == EMPTY:
            next_grid[ti][tj] = STATE_TO_VOXEL[voxel_type]
        return

    if rule.action_type == ActionType.COPY_NEIGHBOR:
        direction = rule.action_params["direction"]
        di, dj = _DIRECTION_OFFSET[direction]
        si, sj = i + di, j + dj
        if _in_bounds(si, sj, shape):
            next_grid[i][j] = current_grid[si][sj]
        return

    raise ValueError(f"Unsupported action_type: {rule.action_type}")


def develop_rules(
    genome,
    shape: Tuple[int, int] = (5, 5),
    iterations: int = DEFAULT_MAX_ITERATIONS,
    seed_position: Optional[Tuple[int, int]] = None,
    initial_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Deterministically develops a DevelopmentalRuleGenome into a final
    voxel grid, using `iterations` synchronous update steps starting from
    an initial grid (see module docstring for exact update semantics).

    If `initial_grid` is given, it is used as-is (copied) as the t=0 grid --
    this is how the crossed `init_condition` experimental factor (see
    evodevo.core.initial_conditions) reaches this interpreter without it
    knowing anything about named conditions. `seed_position` is ignored
    when `initial_grid` is given. If `initial_grid` is None (the historical
    default, still used by the frozen equivalence fixture in
    tests/encoders/fixtures/rule_based_port_equivalence.json), the original
    single-SOFT-cell-at-`seed_position` (default: centre) grid is built.

    iterations=0 returns the initial grid unchanged (no rules are ever
    evaluated). iterations must be in [0, HARD_MAX_ITERATIONS].
    """

    if not (0 <= iterations <= HARD_MAX_ITERATIONS):
        raise ValueError(
            f"iterations must be in [0, {HARD_MAX_ITERATIONS}], got {iterations}"
        )

    rows, cols = shape

    if initial_grid is not None:
        grid = np.array(initial_grid, dtype=int, copy=True)
        if grid.shape != shape:
            raise ValueError(f"initial_grid shape {grid.shape} does not match shape {shape}")
    else:
        grid = np.zeros(shape, dtype=int)
        if seed_position is None:
            seed_position = (rows // 2, cols // 2)
        si, sj = seed_position
        if not _in_bounds(si, sj, shape):
            raise ValueError(f"seed_position {seed_position} is outside shape {shape}")
        grid[si][sj] = SOFT

    for _ in range(iterations):
        # Two sub-passes are used to apply this iteration's decisions,
        # both computed purely from the CURRENT (pre-iteration) grid, so
        # this remains fully synchronous — no rule ever sees another
        # rule's effect from the same iteration. First every cell's
        # first-matching rule is resolved (read-only, no writes yet).
        # Then all SELF-directed actions (SET_*/COPY_NEIGHBOR, which
        # write to the acting cell itself) are applied. Finally all
        # GROW_DIRECTIONAL actions (which write to a NEIGHBOR cell) are
        # applied. Growth is applied last so that a growth target's own
        # same-iteration self-action (e.g. an unrelated cell's ALWAYS
        # rule deciding to stay/become EMPTY, evaluated against its
        # pre-growth EMPTY state) can never immediately overwrite growth
        # that just arrived there — without this ordering, any cell with
        # a catch-all self-rule would silently erase every neighbor's
        # growth into it every iteration, making GROW_DIRECTIONAL
        # unusable. This two-pass order is a deliberate, documented
        # resolution of an ambiguity the task's specification does not
        # otherwise address, not a change to any explicitly specified
        # semantics.
        decisions = []
        for i in range(rows):
            for j in range(cols):
                for rule in genome.rules:
                    if _condition_matches(rule, i, j, grid, shape):
                        decisions.append((i, j, rule))
                        break

        next_grid = grid.copy()

        for i, j, rule in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                _apply_action(rule, i, j, grid, next_grid, shape)

        for i, j, rule in decisions:
            if rule.action_type == ActionType.GROW_DIRECTIONAL:
                _apply_action(rule, i, j, grid, next_grid, shape)

        grid = next_grid

    return grid
