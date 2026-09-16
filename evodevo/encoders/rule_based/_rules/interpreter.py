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

VALID_CONFLICT_RESOLUTIONS = ("row_major_last_wins", "no_op", "lowest_material_id", "first_rule_index")
VALID_COPY_NEIGHBOR_READS = ("pre_iteration", "live")

# Chosen on evidence (N=2000, T=100, results/rule_based/CONFLICT_POLICY.md), not in
# advance -- see that report for the four-way comparison this default was selected
# from. row_major_last_wins (the historical, order-DEPENDENT behaviour, see
# results/rule_based/ORDER_INDEPENDENCE_AUDIT.md) remains fully selectable and is
# NOT the default any more.
DEFAULT_CONFLICT_RESOLUTION = "first_rule_index"

DEFAULT_MAX_ITERATIONS = 5
# Raised from 10 to 40 so this arm's `dev_steps`/`developmental_iterations`
# can legitimately be exercised as a crossed experimental factor up to the
# same hard cap as evodevo/analysis/structural.py's T_MAX_DEFAULT (the
# depth-selection study's settling/light-cone probes already ran the
# interpreter to 40 steps via repeated iterations=1 calls, which this
# ceiling never gated -- this only affects a single from-scratch call with
# a large `iterations` value, e.g. RuleBasedEncoder.decode() at deep T).
# Purely a safety-ceiling value, not part of the rule language/vocabulary.
#
# Raised again from 40 to 100 to support
# evodevo/analysis/rule_based_settling_audit.py's T_max=100 non-settling/
# cycle audit (results/rule_based/SETTLING_AUDIT.md) -- again purely a
# ceiling increase, no change to condition/action semantics, synchronicity,
# or the genome representation.
#
# Raised once more from 100 to 105 so evodevo/analysis/rule_based_readout_policy.py's
# T-shift sweep can query T_primary=100 shifted by +5 (105) without the query
# itself being clipped by this ceiling -- that clipping was silently
# producing a spurious fraction_changed=0.0 at T=100's shift=+1/+5 columns
# (T+shift got clipped to the SAME trajectory endpoint as the T=100 base
# point, trivially "unchanged"), not a genuine stability measurement. Found
# and fixed 2026-09-16 (docs/decisions.md's "T=40-staleness audit" row).
# Purely a ceiling increase, same as the two prior raises.
HARD_MAX_ITERATIONS = 105

# --------------------------------------------------------------------------
# Locality axes (2026-09-16, "restore Proposition R3" follow-up): the
# rule-based counterparts of NCAEncoder's own `boundary`/`neighbourhood`
# axes (evodevo/encoders/nca/encoder.py). See docs/rule-language-spec.md
# and results/rule_based/LOCALITY_STUDY.md.
# --------------------------------------------------------------------------

VALID_BOUNDARIES = ("no_op", "toroidal")
# "no_op" (default): an off-grid neighbor read is treated as the
# DISTINGUISHED OFF_GRID category (docs/rule-language-spec.md section D --
# NOT folded into EMPTY), and an off-grid GROW_DIRECTIONAL target is
# silently discarded -- this is the interpreter's ENTIRE historical
# behaviour, preserved byte-for-byte as the default.
# "toroidal" (NCA's other boundary option): the grid wraps at every edge --
# every neighbor read and growth target is always in-bounds (mod rows/cols),
# so OFF_GRID never occurs under this setting.

VALID_NEIGHBOURHOODS = ("von_neumann", "moore")
# "von_neumann" (default, matches EvoGym's own cardinal connectivity --
# see docs/decisions.md's "Neighbourhood type" row, not re-justified here):
# NEIGHBOR_COUNT examines the 4 cardinal neighbors only (the historical
# behaviour). "moore" (NCA's other neighbourhood option): NEIGHBOR_COUNT
# examines all 8 neighbors (4 cardinal + 4 diagonal) instead. This axis
# affects ONLY NEIGHBOR_COUNT's perception -- it does NOT add diagonal
# `Direction` values to EMPTY_NEIGHBOR/GROW_DIRECTIONAL/COPY_NEIGHBOR
# (those remain 4-directional; extending them would be an action/condition
# VOCABULARY change, explicitly out of scope for this locality-only
# follow-up).
_MOORE_DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

VALID_ACTION_MONOTONICITY = ("unrestricted", "monotone")
# "unrestricted" (default): every action applies exactly as documented
# above -- the interpreter's entire historical behaviour, unchanged.
# "monotone" (the rule-based counterpart of NCAEncoder's own
# `spectral_normalise` certified ablation, see resolved_config.py's
# CERTIFIED_ABLATION and Corollary 3): fixes a TOTAL ORDER (rank) over the
# 5 materials, EMPTY lowest, and makes a SELF-affecting action (SET_*,
# COPY_NEIGHBOR) a no-op for a cell whenever it would DECREASE that cell's
# rank. Rank chosen (lowest to highest): EMPTY, SOFT, RIGID, H_ACT, V_ACT --
# justified as "commitment/structural complexity": EMPTY is definitionally
# the least-committed state (also the only state GROW_DIRECTIONAL may ever
# target, so it must be the rank-0 floor for growth to remain compatible
# with monotonicity, proven below); SOFT is deformable/least-structured
# material next; RIGID is a fixed, undeformable structure, a stronger
# commitment than SOFT; H_ACT/V_ACT (actuators) are the most specialised
# voxel roles (rigid structure PLUS an actuation direction), ranked above
# plain RIGID. Any other fixed total order with EMPTY lowest would equally
# satisfy the proposition below -- see results/rule_based/
# MONOTONE_ABLATION.md for the full statement and proof.
RANK = {EMPTY: 0, SOFT: 1, RIGID: 2, H_ACT: 3, V_ACT: 4}


def _in_bounds(i: int, j: int, shape: Tuple[int, int]) -> bool:
    rows, cols = shape
    return 0 <= i < rows and 0 <= j < cols


def _resolve_neighbor(i: int, j: int, di: int, dj: int, shape: Tuple[int, int], boundary: str):
    """Returns the (ni, nj) coordinate of the neighbor of (i, j) offset by
    (di, dj), or None if that neighbor is OFF_GRID under `boundary`.
    "toroidal" always returns a wrapped, in-bounds coordinate (never None).
    "no_op" returns None for an out-of-bounds offset (the historical
    behaviour, unchanged)."""
    rows, cols = shape
    ni, nj = i + di, j + dj
    if boundary == "toroidal":
        return ni % rows, nj % cols
    return (ni, nj) if _in_bounds(ni, nj, shape) else None


def _condition_matches(
    rule: DevelopmentalRule, i: int, j: int, grid: np.ndarray, shape: Tuple[int, int],
    boundary: str = "no_op", neighbourhood: str = "von_neumann",
) -> bool:
    rows, cols = shape

    if rule.condition_type == ConditionType.ALWAYS:
        return True

    if rule.condition_type == ConditionType.CELL_TYPE:
        return grid[i][j] == STATE_TO_VOXEL[rule.condition_params["state"]]

    if rule.condition_type == ConditionType.NEIGHBOR_COUNT:
        threshold = rule.condition_params["threshold"]
        offsets = _DIRECTION_OFFSET.values() if neighbourhood == "von_neumann" else (
            tuple(_DIRECTION_OFFSET.values()) + _MOORE_DIAGONAL_OFFSETS
        )
        count = 0
        for di, dj in offsets:
            neighbor = _resolve_neighbor(i, j, di, dj, shape, boundary)
            if neighbor is not None and grid[neighbor[0]][neighbor[1]] != EMPTY:
                count += 1
        return count >= threshold

    if rule.condition_type == ConditionType.EMPTY_NEIGHBOR:
        direction = rule.condition_params["direction"]
        di, dj = _DIRECTION_OFFSET[direction]
        neighbor = _resolve_neighbor(i, j, di, dj, shape, boundary)
        return neighbor is not None and grid[neighbor[0]][neighbor[1]] == EMPTY

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


def _apply_self_action(
    rule: DevelopmentalRule, i: int, j: int, pre_grid: np.ndarray, next_grid: np.ndarray,
    shape: Tuple[int, int], copy_neighbor_reads: str, boundary: str = "no_op",
    action_monotonicity: str = "unrestricted",
) -> None:
    """Pass 2a: SET_*/COPY_NEIGHBOR only (never GROW_DIRECTIONAL). Always
    writes to the ACTING cell (i, j) alone -- Pass 1 guarantees at most one
    decision per cell, so distinct calls to this function never write to
    the same target, making this pass order-independent by construction
    regardless of `conflict_resolution` (which only concerns Pass 2b/growth
    writes).

    `action_monotonicity="monotone"` (see VALID_ACTION_MONOTONICITY/RANK):
    a SET_*/COPY_NEIGHBOR write is applied only if it does not DECREASE
    `RANK[pre_grid[i][j]]` -- the acting cell's OWN rank at the START of
    this iteration (never `next_grid`, so this check is itself
    order-independent, exactly like every other read in this function).
    Otherwise the write is a no-op for this cell THIS iteration (the cell
    keeps whatever `next_grid[i][j]` already held -- its pre-iteration
    value, since `next_grid` starts as `pre_grid.copy()`)."""
    if rule.action_type in _SET_ACTION_TO_VOXEL:
        new_value = _SET_ACTION_TO_VOXEL[rule.action_type]
        if action_monotonicity == "monotone" and RANK[new_value] < RANK[int(pre_grid[i][j])]:
            return
        next_grid[i][j] = new_value
        return

    if rule.action_type == ActionType.COPY_NEIGHBOR:
        direction = rule.action_params["direction"]
        di, dj = _DIRECTION_OFFSET[direction]
        neighbor = _resolve_neighbor(i, j, di, dj, shape, boundary)
        if neighbor is not None:
            si, sj = neighbor
            # `copy_neighbor_reads="pre_iteration"` (the only behaviour that
            # ever existed before this parameter, and the default): read
            # from `pre_grid`, the untouched start-of-iteration grid --
            # never from `next_grid`, so this can never observe another
            # cell's same-iteration self-write.
            # `copy_neighbor_reads="live"`: read from whatever `next_grid`
            # holds AT THE MOMENT this call runs, i.e. a partially-updated
            # grid -- reintroduces an order-dependency deliberately, for
            # comparison purposes only; NOT the default, see
            # results/rule_based/CONFLICT_POLICY.md.
            source_grid = pre_grid if copy_neighbor_reads == "pre_iteration" else next_grid
            new_value = int(source_grid[si][sj])
            if action_monotonicity == "monotone" and RANK[new_value] < RANK[int(pre_grid[i][j])]:
                return
            next_grid[i][j] = new_value
        return

    raise ValueError(f"Unsupported action_type for a self-action: {rule.action_type}")


def _growth_target(rule: DevelopmentalRule, i: int, j: int, pre_grid: np.ndarray, shape: Tuple[int, int], boundary: str = "no_op"):
    """Returns (ti, tj) if this GROW_DIRECTIONAL decision is a LIVE growth
    candidate (in-bounds/wrapped target, EMPTY in the pre-iteration grid),
    else None. The emptiness check is always against `pre_grid` (B.3's
    point (iii)), never against any partially-written grid. Under
    `boundary="toroidal"`, the target always exists (wrapped); under the
    default `boundary="no_op"`, an off-grid target is discarded (returns
    None), the historical behaviour.

    NOTE (action_monotonicity, results/rule_based/MONOTONE_ABLATION.md):
    GROW_DIRECTIONAL needs NO monotonicity gate at all -- it is
    UNCONDITIONALLY rank-non-decreasing by construction, for every genome,
    with no special-casing: this function only ever returns a target whose
    pre-iteration value is EMPTY (RANK 0, the floor of the fixed order),
    and `voxel_type` is drawn from GROWABLE_STATES (types.py), which
    EXCLUDES EMPTY by construction -- so every growth write's RANK is
    >= 1 > 0 = RANK[EMPTY]. This is the base case Proposition R4's proof
    rests on."""
    direction = rule.action_params["direction"]
    di, dj = _DIRECTION_OFFSET[direction]
    neighbor = _resolve_neighbor(i, j, di, dj, shape, boundary)
    if neighbor is not None and pre_grid[neighbor[0]][neighbor[1]] == EMPTY:
        return neighbor
    return None


def _resolve_growth_conflicts(growth_candidates: list, conflict_resolution: str) -> dict:
    """Pass 2b's conflict resolution: `growth_candidates` is a list of
    (ti, tj, i, j, rule, rule_index) tuples -- one per GROW_DIRECTIONAL
    decision whose target passed `_growth_target`'s pre-iteration-emptiness
    check, in Pass-1 enumeration (`π`) order. Returns {(ti, tj): voxel_type}
    for every target that ends up with a winning write (targets resolved to
    "no write" under `no_op` are simply absent from the returned dict).

    Grouping by target is done here (not by the caller) so every policy
    below sees the FULL set of competitors for that target, independent of
    `π` -- this is the collect-then-resolve structure B.3 specifies.
    """
    if conflict_resolution not in VALID_CONFLICT_RESOLUTIONS:
        raise ValueError(f"unsupported conflict_resolution '{conflict_resolution}'; must be one of {VALID_CONFLICT_RESOLUTIONS}")

    by_target: dict = {}
    for ti, tj, i, j, rule, rule_index in growth_candidates:
        by_target.setdefault((ti, tj), []).append((i, j, rule, rule_index))

    resolved = {}
    for target, candidates in by_target.items():
        if len(candidates) == 1:
            _, _, rule, _ = candidates[0]
            resolved[target] = STATE_TO_VOXEL[rule.action_params["voxel_type"]]
            continue

        # >=2 competing writes to the SAME target this iteration -- a
        # genuine growth conflict (results/rule_based/ORDER_INDEPENDENCE_
        # AUDIT.md: not rare, 30.40% of genomes have at least one).
        if conflict_resolution == "row_major_last_wins":
            # `candidates` preserves Pass-1's (π-order) relative sequence
            # (each appended in that order above) -- reproduces the
            # historical, order-DEPENDENT "later acting cell wins" rule
            # byte-for-byte.
            _, _, rule, _ = candidates[-1]
            resolved[target] = STATE_TO_VOXEL[rule.action_params["voxel_type"]]

        elif conflict_resolution == "no_op":
            # Order-independent by construction: depends only on the SET of
            # competitors (>=2), never on which one or in what order.
            # Competing simultaneous growth cancels out -- contact
            # inhibition -- target stays EMPTY (absent from `resolved`).
            pass

        elif conflict_resolution == "lowest_material_id":
            # Order-independent: a pure min() over the competing writes'
            # numeric material codes -- the SET of (material_id) values is
            # independent of π, so the min is too.
            _, _, rule, _ = min(candidates, key=lambda c: STATE_TO_VOXEL[c[2].action_params["voxel_type"]])
            resolved[target] = STATE_TO_VOXEL[rule.action_params["voxel_type"]]

        elif conflict_resolution == "first_rule_index":
            # Order-independent: min() over rule_index, a property of WHICH
            # rule matched, not of enumeration order. Full tie-break chain,
            # as specified: first_rule_index -> lowest_material_id -> (both
            # exhausted only if every tied candidate has BOTH the same rule
            # index AND the same material, which happens automatically
            # whenever candidates share a rule index, since a rule's
            # action_params -- hence its voxel_type -- are fixed per rule;
            # so this chain always terminates in a deterministic pick, never
            # needs a THIRD no_op fallback in practice, but the code is
            # written to not assume that and just pick deterministically
            # among the fully-tied remainder).
            min_rule_index = min(c[3] for c in candidates)
            tied = [c for c in candidates if c[3] == min_rule_index]
            if len(tied) > 1:
                min_material = min(STATE_TO_VOXEL[c[2].action_params["voxel_type"]] for c in tied)
                tied = [c for c in tied if STATE_TO_VOXEL[c[2].action_params["voxel_type"]] == min_material]
            _, _, rule, _ = tied[0]
            resolved[target] = STATE_TO_VOXEL[rule.action_params["voxel_type"]]

    return resolved


def develop_rules(
    genome,
    shape: Tuple[int, int] = (5, 5),
    iterations: int = DEFAULT_MAX_ITERATIONS,
    seed_position: Optional[Tuple[int, int]] = None,
    initial_grid: Optional[np.ndarray] = None,
    conflict_resolution: str = DEFAULT_CONFLICT_RESOLUTION,
    copy_neighbor_reads: str = "pre_iteration",
    boundary: str = "no_op",
    neighbourhood: str = "von_neumann",
    action_monotonicity: str = "unrestricted",
    _cell_order: Optional[list] = None,
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

    `conflict_resolution` (see VALID_CONFLICT_RESOLUTIONS): how Pass 2b
    resolves a GROWTH CONFLICT (>=2 different acting cells' GROW_DIRECTIONAL
    targeting the same empty neighbor in one iteration -- see
    results/rule_based/ORDER_INDEPENDENCE_AUDIT.md, not rare: 30.40% of
    sampled genomes have at least one across a T=100 development).
    "row_major_last_wins" is the HISTORICAL, order-DEPENDENT behaviour
    (reproduces every version of this function before this parameter
    existed, byte-for-byte, including the permanently-frozen port-
    equivalence fixture at tests/encoders/fixtures/
    rule_based_port_equivalence.json when explicitly requested -- it is
    NOT the default any more, see DEFAULT_CONFLICT_RESOLUTION and
    results/rule_based/CONFLICT_POLICY.md for the evidence the new default
    was chosen from). The other three ("no_op", "lowest_material_id",
    "first_rule_index") resolve every conflict as a function of the SET of
    competing writes only, independent of `_cell_order`/enumeration order
    -- see `_resolve_growth_conflicts`'s own docstring for the exact
    tie-break chain each uses.

    `copy_neighbor_reads` (see VALID_COPY_NEIGHBOR_READS): "pre_iteration"
    (default, the only behaviour that ever existed before this parameter)
    reads COPY_NEIGHBOR's source from the untouched start-of-iteration
    grid; "live" reads from the grid as partially updated by other
    self-actions applied earlier in the same Pass 2a sweep, deliberately
    reintroducing an order-dependency, provided for comparison only.

    `boundary` (see VALID_BOUNDARIES) and `neighbourhood` (see
    VALID_NEIGHBOURHOODS): the rule-based counterparts of NCAEncoder's own
    same-named axes -- see this module's own VALID_BOUNDARIES/
    VALID_NEIGHBOURHOODS docstrings above for the full semantics, and
    docs/rule-language-spec.md / results/rule_based/LOCALITY_STUDY.md for
    the evidence and rationale. Both default to the interpreter's entire
    historical behaviour ("no_op", "von_neumann").

    `action_monotonicity` (see VALID_ACTION_MONOTONICITY/RANK above): the
    rule-based counterpart of NCAEncoder's `spectral_normalise` certified
    ablation. "unrestricted" (default): every action applies exactly as
    documented, unchanged. "monotone": a SELF-affecting action (SET_*,
    COPY_NEIGHBOR) is a no-op for a cell whenever it would DECREASE that
    cell's RANK; GROW_DIRECTIONAL needs no gate at all (see
    `_growth_target`'s own NOTE -- it is unconditionally rank-non-
    decreasing by construction). This TERMINATION-GUARANTEES development
    (Proposition R4: Phi(s) = sum of RANK over all 25 cells is
    non-decreasing every step and bounded above by 25*4=100, so no genome
    can cycle with period > 1 under this setting) at a measured
    expressivity cost -- see results/rule_based/MONOTONE_ABLATION.md.

    `_cell_order`: TEST-ONLY hook (leading underscore, not part of the
    public API, no production caller ever passes it). When None (the
    default, used by every real caller), Pass 1 enumerates cells in the
    standard row-major order, exactly as before this parameter existed --
    behaviourally IDENTICAL to every prior version of this function. When
    given, it must be a permutation of [(i, j) for i in range(rows) for j
    in range(cols)]; Pass 1 (and, under `conflict_resolution=
    "row_major_last_wins"` only, the order growth-conflict writes are
    resolved in) uses this order instead. Added specifically to let
    tests/encoders/test_rule_based_order_independence.py empirically check
    order-(in)dependence, without changing default behaviour for anything
    else. Under any of the three ORDER-INDEPENDENT `conflict_resolution`
    policies, `_cell_order` has NO effect on the final grid whatsoever (by
    construction, verified by that same test suite) -- it only still
    affects `row_major_last_wins`, by design.
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

    if conflict_resolution not in VALID_CONFLICT_RESOLUTIONS:
        raise ValueError(f"unsupported conflict_resolution '{conflict_resolution}'; must be one of {VALID_CONFLICT_RESOLUTIONS}")
    if copy_neighbor_reads not in VALID_COPY_NEIGHBOR_READS:
        raise ValueError(f"unsupported copy_neighbor_reads '{copy_neighbor_reads}'; must be one of {VALID_COPY_NEIGHBOR_READS}")
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"unsupported boundary '{boundary}'; must be one of {VALID_BOUNDARIES}")
    if neighbourhood not in VALID_NEIGHBOURHOODS:
        raise ValueError(f"unsupported neighbourhood '{neighbourhood}'; must be one of {VALID_NEIGHBOURHOODS}")
    if action_monotonicity not in VALID_ACTION_MONOTONICITY:
        raise ValueError(f"unsupported action_monotonicity '{action_monotonicity}'; must be one of {VALID_ACTION_MONOTONICITY}")

    for _ in range(iterations):
        # B.3's two-pass operator (docs/rule-language-spec.md), implemented
        # as three sub-passes here:
        #   Pass 1  (MATCH, read-only): every cell's first-matching rule is
        #           resolved purely against the pre-iteration `grid` --
        #           order-independent by construction (each cell's decision
        #           depends only on `grid`, never on another cell's
        #           decision).
        #   Pass 2a (SELF-WRITE): SET_*/COPY_NEIGHBOR decisions write to
        #           their own acting cell only -- order-independent by
        #           construction (at most one decision per cell, so no two
        #           self-actions ever share a write target).
        #   Pass 2b (GROWTH-WRITE): GROW_DIRECTIONAL decisions are first
        #           COLLECTED (not applied) as (target, candidate) pairs,
        #           checked for pre-iteration emptiness, THEN resolved per
        #           target by `conflict_resolution` -- this is the only
        #           potentially order-dependent step, and only under
        #           conflict_resolution="row_major_last_wins" (see that
        #           policy's own docstring in _resolve_growth_conflicts).
        #           Applied after Pass 2a so a growth target's own
        #           same-iteration self-action can never silently erase
        #           growth that just arrived there (growth checks
        #           pre-iteration emptiness and unconditionally overwrites
        #           whatever Pass 2a wrote, if any -- without this, any
        #           catch-all self-rule would make GROW_DIRECTIONAL
        #           unusable).
        cell_order = _cell_order if _cell_order is not None else [(i, j) for i in range(rows) for j in range(cols)]
        decisions = []
        for i, j in cell_order:
            for rule_index, rule in enumerate(genome.rules):
                if _condition_matches(rule, i, j, grid, shape, boundary=boundary, neighbourhood=neighbourhood):
                    decisions.append((i, j, rule, rule_index))
                    break

        next_grid = grid.copy()

        # Pass 2a
        for i, j, rule, _rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                _apply_self_action(
                    rule, i, j, grid, next_grid, shape, copy_neighbor_reads, boundary=boundary,
                    action_monotonicity=action_monotonicity,
                )

        # Pass 2b: collect growth candidates (whose target is empty in the
        # PRE-ITERATION grid -- never `next_grid`), then resolve conflicts.
        growth_candidates = []
        for i, j, rule, rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                continue
            target = _growth_target(rule, i, j, grid, shape, boundary=boundary)
            if target is not None:
                ti, tj = target
                growth_candidates.append((ti, tj, i, j, rule, rule_index))

        resolved_writes = _resolve_growth_conflicts(growth_candidates, conflict_resolution)
        for (ti, tj), voxel_code in resolved_writes.items():
            next_grid[ti][tj] = voxel_code

        grid = next_grid

    return grid
