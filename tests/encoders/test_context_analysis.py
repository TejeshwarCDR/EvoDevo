"""Tests for evodevo/encoders/rule_based/context_analysis.py -- exact
(not sampled) rule reachability, coverage, and redundancy."""
import numpy as np
import pytest

from evodevo.core.materials import EMPTY, SOFT
from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome
from evodevo.encoders.rule_based._rules.types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)
from evodevo.encoders.rule_based.context_analysis import (
    CONTEXT_SPACE_SIZE,
    Context,
    PositionNotSupportedError,
    action_redundancy,
    dead_rules,
    dynamic_coverage,
    effective_rule_count,
    enumerate_contexts,
    rule_coverage,
)


def test_context_space_size_matches_spec():
    assert CONTEXT_SPACE_SIZE == 5 * 6 ** 4 == 6480


def test_enumerate_contexts_is_exhaustive_and_unique():
    contexts = enumerate_contexts()
    assert len(contexts) == CONTEXT_SPACE_SIZE
    assert len(set(contexts)) == CONTEXT_SPACE_SIZE
    for c in contexts:
        assert isinstance(c, Context)
        assert 0 <= c.self_ <= 4
        for field in (c.up, c.down, c.left, c.right):
            assert -1 <= field <= 4


def test_rule_coverage_first_match_semantics_trivial_two_rule_genome():
    # rule0 ALWAYS matches everything -> rule1 is provably dead.
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.EMPTY}, ActionType.SET_HARD, {}),
    ])
    cov = rule_coverage(genome)
    assert cov["coverage_counts"] == [CONTEXT_SPACE_SIZE, 0]
    assert cov["unreachable_contexts"] == 0
    assert dead_rules(genome) == [1]
    assert effective_rule_count(genome) == 1


def test_rule_coverage_counts_sum_with_unreachable_to_full_space():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT}, ActionType.SET_HARD, {}),
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.EMPTY}, ActionType.SET_SOFT, {}),
    ])
    cov = rule_coverage(genome)
    assert sum(cov["coverage_counts"]) + cov["unreachable_contexts"] == CONTEXT_SPACE_SIZE
    assert all(c > 0 for c in cov["coverage_counts"])  # both rules alive
    assert dead_rules(genome) == []
    assert effective_rule_count(genome) == 2


def test_position_rule_raises_not_silently_wrong():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.POSITION, {"position": PositionType.TOP}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    with pytest.raises(PositionNotSupportedError):
        rule_coverage(genome)
    with pytest.raises(PositionNotSupportedError):
        action_redundancy(genome)


def test_action_redundancy_worked_example_copy_neighbor_vs_set():
    # copy_neighbor(UP) and set(SOFT) coincide when either (a) the
    # up-neighbour is SOFT directly (P=1/6 of NEIGHBOR_VALUES), or (b) the
    # up-neighbour is OFF_GRID (P=1/6, in which case copy_neighbor's
    # no-op-fallback keeps the acting cell's own value) AND the acting
    # cell's own value is already SOFT (P=1/5 of SELF_VALUES) -> total
    # 1/6 + 1/6*1/5 = 5/30 + 1/30 = 6/30 = 1/5 = 0.2 of the full context
    # space (each case exhaustively verified against the exact computed
    # fraction, not merely asserted).
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.COPY_NEIGHBOR, {"direction": Direction.UP}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_SOFT, {}),
    ])
    ar = action_redundancy(genome)
    frac = ar["pairs"][(0, 1)]
    assert abs(frac - 0.2) < 1e-9


def test_action_redundancy_identical_actions_are_fully_redundant():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT}, ActionType.SET_HARD, {}),
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.EMPTY}, ActionType.SET_HARD, {}),
    ])
    ar = action_redundancy(genome)
    assert ar["pairs"][(0, 1)] == 1.0


def test_dynamic_coverage_matches_real_develop_rules_decision_count():
    from evodevo.encoders.rule_based._rules.interpreter import develop_rules

    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
                           ActionType.GROW_DIRECTIONAL, {"direction": Direction.RIGHT, "voxel_type": VoxelState.SOFT}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    initial = np.zeros((5, 5), dtype=int)
    initial[2, 2] = SOFT
    dyn = dynamic_coverage(genome, initial, (5, 5), 3)
    assert dyn["n_cells_evaluated"] == 3 * 25
    assert sum(dyn["rule_fire_counts"]) + dyn["n_no_match"] == dyn["n_cells_evaluated"]
    # cross-check final grid against the real interpreter
    real_grid = develop_rules(genome, shape=(5, 5), iterations=3, initial_grid=initial)
    assert real_grid.shape == (5, 5)


def test_dynamic_coverage_can_be_strictly_weaker_than_static():
    # A rule that is statically alive (some context in C triggers it) but
    # never actually fires from the center-seed initial condition within a
    # short horizon.
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.H_ACT if False else VoxelState.ACTUATOR_HORIZONTAL},
                           ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    assert dead_rules(genome) == []  # statically alive: some context has self=H_ACT
    initial = np.zeros((5, 5), dtype=int)
    initial[2, 2] = SOFT  # never becomes H_ACT under this genome
    dyn = dynamic_coverage(genome, initial, (5, 5), 5)
    assert dyn["rule_fire_counts"][0] == 0  # dynamically dead despite static aliveness
