"""Exact (not sampled) rule reachability, coverage, and redundancy analysis
for the V2 developmental rule language, exploiting the fact that every
condition in the vocabulary inspects only a LOCAL CONTEXT -- (self, up,
down, left, right) -- which is small enough to enumerate exhaustively (see
docs/rule-language-spec.md section D: |C| = 5 * 6^4 = 6480, OFF_GRID
distinguished, not folded into EMPTY).

This has no analogue for the direct or NCA arms (their genome/phenotype
spaces are continuous or astronomically large) -- it is the concrete
content of the claim that the rule-based encoding is INTERPRETABLE: for
any genome, we can say EXACTLY which rules can ever fire, on exactly which
inputs, and whether two rules are behaviourally redundant, without
sampling or approximation.

SCOPE NOTE: `ConditionType.POSITION` depends on the acting cell's ABSOLUTE
grid coordinates, which are NOT part of the local context this module
enumerates (see docs/rule-language-spec.md's own correction: POSITION is
what breaks locality/translation-equivariance, R3). This module's static
coverage/redundancy functions therefore only support genomes with NO
POSITION rules -- exactly the population `position_predicate="disabled"`
(the encoder's own default since the R3 fix) actually produces. A genome
containing POSITION raises `PositionNotSupportedError` rather than
silently giving a wrong answer.

Two distinct notions of "dead rule", NOT conflated (see module docstring's
own CONTEXT_ANALYSIS.md companion report):
  - STATIC coverage: exact, over ALL of C, computed in closed form.
  - DYNAMIC coverage: over contexts ACTUALLY ENCOUNTERED during real
    development from the project's own center-seed initial condition,
    within a T-step horizon -- a genome can be statically alive (some
    context in C would trigger it) but dynamically dead (that context
    never actually arises from this initial condition within T steps).
"""
from __future__ import annotations

from typing import List, NamedTuple

import numpy as np

from evodevo.core.materials import EMPTY, N_MATERIALS
from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome
from evodevo.encoders.rule_based._rules.interpreter import (
    DEFAULT_CONFLICT_RESOLUTION,
    _apply_self_action,
    _DIRECTION_OFFSET,
    _growth_target,
    _in_bounds,
    _resolve_growth_conflicts,
)
from evodevo.encoders.rule_based._rules.types import ConditionType, Direction, VoxelState

OFF_GRID = -1  # distinguished sentinel, never a valid material code (0..4)

# Context field order: (self, up, down, left, right) -- matches
# _DIRECTION_OFFSET's own key order for the 4 neighbor fields.
_NEIGHBOR_DIRECTIONS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)

NEIGHBOR_VALUES = tuple(range(OFF_GRID, N_MATERIALS))  # (-1, 0, 1, 2, 3, 4) -- 6 symbols
SELF_VALUES = tuple(range(N_MATERIALS))  # (0, 1, 2, 3, 4) -- 5 symbols, self is never off-grid

CONTEXT_SPACE_SIZE = len(SELF_VALUES) * len(NEIGHBOR_VALUES) ** 4  # 5 * 6^4 = 6480


class Context(NamedTuple):
    self_: int
    up: int
    down: int
    left: int
    right: int


class PositionNotSupportedError(ValueError):
    """Raised when a genome contains a ConditionType.POSITION rule --
    POSITION depends on absolute grid coordinates, not local context, so
    this module's exact enumeration does not apply to it (see module
    docstring's SCOPE NOTE)."""


def _check_no_position(genome: DevelopmentalRuleGenome) -> None:
    for rule in genome.rules:
        if rule.condition_type == ConditionType.POSITION:
            raise PositionNotSupportedError(
                "context_analysis only supports genomes with no ConditionType.POSITION rules -- "
                "POSITION depends on absolute grid coordinates, not local context (see this module's "
                "SCOPE NOTE and docs/rule-language-spec.md's R3 section)."
            )


def enumerate_contexts() -> List[Context]:
    """The complete, exact local-context space: every (self, up, down,
    left, right) combination, self in {EMPTY, RIGID, SOFT, H_ACT, V_ACT}
    (5 symbols, a cell is never off-grid with respect to itself), each
    neighbor in {OFF_GRID, EMPTY, RIGID, SOFT, H_ACT, V_ACT} (6 symbols).
    len(result) == CONTEXT_SPACE_SIZE == 6480, verified by
    tests/encoders/test_context_analysis.py."""
    contexts = []
    for s in SELF_VALUES:
        for up in NEIGHBOR_VALUES:
            for down in NEIGHBOR_VALUES:
                for left in NEIGHBOR_VALUES:
                    for right in NEIGHBOR_VALUES:
                        contexts.append(Context(s, up, down, left, right))
    return contexts


# ---------------------------------------------------------------------------
# Vectorized condition matching over the WHOLE context space at once
# ---------------------------------------------------------------------------

_NEIGHBOR_FIELD = {Direction.UP: "up", Direction.DOWN: "down", Direction.LEFT: "left", Direction.RIGHT: "right"}


def _context_arrays():
    """The context space as 5 parallel numpy arrays (self, up, down, left,
    right), each of length CONTEXT_SPACE_SIZE, for vectorized condition
    evaluation (one boolean mask computed per rule via a handful of numpy
    ops, instead of a 6480-iteration Python loop per rule per genome)."""
    self_vals, up_vals, down_vals, left_vals, right_vals = np.meshgrid(
        np.array(SELF_VALUES), np.array(NEIGHBOR_VALUES), np.array(NEIGHBOR_VALUES),
        np.array(NEIGHBOR_VALUES), np.array(NEIGHBOR_VALUES), indexing="ij",
    )
    return (
        self_vals.reshape(-1), up_vals.reshape(-1), down_vals.reshape(-1),
        left_vals.reshape(-1), right_vals.reshape(-1),
    )


_SELF_ARR, _UP_ARR, _DOWN_ARR, _LEFT_ARR, _RIGHT_ARR = _context_arrays()
_NEIGHBOR_ARR = {"up": _UP_ARR, "down": _DOWN_ARR, "left": _LEFT_ARR, "right": _RIGHT_ARR}

assert _SELF_ARR.shape[0] == CONTEXT_SPACE_SIZE, "context array construction does not match |C|"


def _condition_mask(rule) -> np.ndarray:
    """Boolean mask (length CONTEXT_SPACE_SIZE) of every context this
    rule's CONDITION matches -- vectorized, exact, no sampling."""
    ct = rule.condition_type
    if ct == ConditionType.ALWAYS:
        return np.ones(CONTEXT_SPACE_SIZE, dtype=bool)
    if ct == ConditionType.CELL_TYPE:
        from evodevo.encoders.rule_based._rules.interpreter import STATE_TO_VOXEL
        target = STATE_TO_VOXEL[rule.condition_params["state"]]
        return _SELF_ARR == target
    if ct == ConditionType.NEIGHBOR_COUNT:
        threshold = rule.condition_params["threshold"]
        count = np.zeros(CONTEXT_SPACE_SIZE, dtype=int)
        for field in ("up", "down", "left", "right"):
            arr = _NEIGHBOR_ARR[field]
            count += ((arr != OFF_GRID) & (arr != EMPTY)).astype(int)
        return count >= threshold
    if ct == ConditionType.EMPTY_NEIGHBOR:
        direction = rule.condition_params["direction"]
        field = _NEIGHBOR_FIELD[direction]
        return _NEIGHBOR_ARR[field] == EMPTY
    if ct == ConditionType.POSITION:
        raise PositionNotSupportedError("POSITION cannot be evaluated from local context alone")
    raise ValueError(f"Unsupported condition_type: {ct}")


# ---------------------------------------------------------------------------
# Coverage / dead rules / effective rule count
# ---------------------------------------------------------------------------


def rule_coverage(genome: DevelopmentalRuleGenome) -> dict:
    """For each rule index k, the EXACT set of contexts for which rule k
    is the FIRST match (first-match/genome-order semantics, identical to
    the interpreter's own matching -- see interpreter.py's own module
    docstring). Returns {"coverage_masks": [bool array per rule],
    "coverage_counts": [int per rule], "context_space_size": int}.
    Raises PositionNotSupportedError if the genome contains a POSITION
    rule (see module SCOPE NOTE)."""
    _check_no_position(genome)

    already_matched = np.zeros(CONTEXT_SPACE_SIZE, dtype=bool)
    coverage_masks = []
    for rule in genome.rules:
        matches = _condition_mask(rule)
        owns = matches & ~already_matched
        coverage_masks.append(owns)
        already_matched = already_matched | owns

    return {
        "coverage_masks": coverage_masks,
        "coverage_counts": [int(m.sum()) for m in coverage_masks],
        "context_space_size": CONTEXT_SPACE_SIZE,
        "unreachable_contexts": int((~already_matched).sum()),  # no rule matches at all
    }


def dead_rules(genome: DevelopmentalRuleGenome) -> List[int]:
    """Rule indices with EMPTY static coverage -- PROVABLY unreachable
    (some earlier rule in genome order matches every context this rule's
    own condition would ever match), not merely unobserved in a sample."""
    coverage = rule_coverage(genome)
    return [k for k, count in enumerate(coverage["coverage_counts"]) if count == 0]


def effective_rule_count(genome: DevelopmentalRuleGenome) -> int:
    """Rules with non-empty static coverage."""
    coverage = rule_coverage(genome)
    return sum(1 for count in coverage["coverage_counts"] if count > 0)


# ---------------------------------------------------------------------------
# Action redundancy
# ---------------------------------------------------------------------------


def _action_effect_on_self(rule, context_arrays=None) -> np.ndarray:
    """The ACTING CELL'S OWN resulting value under this rule's action,
    evaluated over the WHOLE context space (independent of whether the
    rule's own CONDITION would match -- this function asks "if this action
    were applied here, what would happen to the acting cell", the
    ingredient action_redundancy needs to compare two actions' semantics
    directly). SET_*: constant. COPY_NEIGHBOR(dir): the named neighbor's
    value (falls back to unchanged/self if that neighbor is OFF_GRID,
    matching the interpreter's own `_in_bounds` guard -- a copy from an
    off-grid neighbor is a no-op in the real interpreter). GROW_DIRECTIONAL:
    the acting cell is NEVER written by its own growth action (growth
    writes to a NEIGHBOR) -- its own value is unchanged (= self)."""
    from evodevo.encoders.rule_based._rules.interpreter import _SET_ACTION_TO_VOXEL

    if rule.action_type in _SET_ACTION_TO_VOXEL:
        value = _SET_ACTION_TO_VOXEL[rule.action_type]
        return np.full(CONTEXT_SPACE_SIZE, value, dtype=int)

    from evodevo.encoders.rule_based._rules.types import ActionType

    if rule.action_type == ActionType.COPY_NEIGHBOR:
        direction = rule.action_params["direction"]
        field = _NEIGHBOR_FIELD[direction]
        neighbor = _NEIGHBOR_ARR[field]
        # off-grid neighbor -> no-op -> acting cell keeps its own value
        return np.where(neighbor == OFF_GRID, _SELF_ARR, neighbor)

    if rule.action_type == ActionType.GROW_DIRECTIONAL:
        return _SELF_ARR.copy()  # growth never writes the acting cell itself

    raise ValueError(f"Unsupported action_type: {rule.action_type}")


def action_redundancy(genome: DevelopmentalRuleGenome) -> dict:
    """For each pair of rules (i, j), i < j: the fraction of the FULL
    context space C where applying rule i's action and rule j's action
    (independently -- NOT restricted to either rule's own coverage,
    "the contexts they each own" is read as "evaluated over C, the space
    both rules' conditions are defined over") would produce the SAME
    resulting value for the acting cell (see `_action_effect_on_self`).
    Returns {"pairs": {(i, j): fraction}, "by_action_type_pair": {(type_i,
    type_j): [fractions]}}."""
    _check_no_position(genome)

    effects = [_action_effect_on_self(rule) for rule in genome.rules]
    pairs = {}
    by_type_pair: dict = {}
    n = len(genome.rules)
    for i in range(n):
        for j in range(i + 1, n):
            frac = float(np.mean(effects[i] == effects[j]))
            pairs[(i, j)] = frac
            type_pair = tuple(sorted((genome.rules[i].action_type.value, genome.rules[j].action_type.value)))
            by_type_pair.setdefault(type_pair, []).append(frac)

    return {"pairs": pairs, "by_action_type_pair": by_type_pair}


# ---------------------------------------------------------------------------
# Dynamic coverage: contexts actually observed during real development
# ---------------------------------------------------------------------------


def _extract_context(grid: np.ndarray, i: int, j: int, shape) -> Context:
    rows, cols = shape
    vals = {"up": OFF_GRID, "down": OFF_GRID, "left": OFF_GRID, "right": OFF_GRID}
    for direction, field in _NEIGHBOR_FIELD.items():
        di, dj = _DIRECTION_OFFSET[direction]
        ni, nj = i + di, j + dj
        if _in_bounds(ni, nj, shape):
            vals[field] = int(grid[ni][nj])
    return Context(int(grid[i][j]), vals["up"], vals["down"], vals["left"], vals["right"])


def dynamic_coverage(
    genome: DevelopmentalRuleGenome, initial_grid: np.ndarray, shape, iterations: int,
    conflict_resolution: str = DEFAULT_CONFLICT_RESOLUTION, copy_neighbor_reads: str = "pre_iteration",
) -> dict:
    """Re-runs the REAL interpreter algorithm (using interpreter.py's own
    private helpers directly, so this is guaranteed faithful to
    develop_rules' actual behaviour -- boundary="no_op" always, matching
    the encoder's default) while additionally recording, at every
    iteration and every cell, the LOCAL CONTEXT actually present and which
    rule index (if any) matched it. Returns {"rule_fire_counts": [int per
    rule], "n_cells_evaluated": int, "n_no_match": int} -- a genome's
    DYNAMIC coverage is `rule_fire_counts[k] > 0`, a WEAKER, sample-
    dependent notion than static coverage (see module docstring)."""
    _check_no_position(genome)

    rows, cols = shape
    grid = np.array(initial_grid, dtype=int, copy=True)
    rule_fire_counts = [0] * len(genome.rules)
    n_cells_evaluated = 0
    n_no_match = 0

    from evodevo.encoders.rule_based._rules.interpreter import _condition_matches
    from evodevo.encoders.rule_based._rules.types import ActionType

    for _ in range(iterations):
        decisions = []
        for i in range(rows):
            for j in range(cols):
                n_cells_evaluated += 1
                matched = False
                for rule_index, rule in enumerate(genome.rules):
                    if _condition_matches(rule, i, j, grid, shape):
                        decisions.append((i, j, rule, rule_index))
                        rule_fire_counts[rule_index] += 1
                        matched = True
                        break
                if not matched:
                    n_no_match += 1

        next_grid = grid.copy()
        for i, j, rule, _rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                _apply_self_action(rule, i, j, grid, next_grid, shape, copy_neighbor_reads)

        growth_candidates = []
        for i, j, rule, rule_index in decisions:
            if rule.action_type != ActionType.GROW_DIRECTIONAL:
                continue
            target = _growth_target(rule, i, j, grid, shape)
            if target is not None:
                ti, tj = target
                growth_candidates.append((ti, tj, i, j, rule, rule_index))

        resolved_writes = _resolve_growth_conflicts(growth_candidates, conflict_resolution)
        for (ti, tj), voxel_code in resolved_writes.items():
            next_grid[ti][tj] = voxel_code

        grid = next_grid

    return {
        "rule_fire_counts": rule_fire_counts,
        "n_cells_evaluated": n_cells_evaluated,
        "n_no_match": n_no_match,
    }
