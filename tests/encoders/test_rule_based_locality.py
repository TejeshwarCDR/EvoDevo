"""Tests for the 2026-09-16 "restore Proposition R3" follow-up:
position_predicate/boundary/neighbourhood axes on RuleBasedEncoder and the
interpreter."""
import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import EMPTY, SOFT
from evodevo.encoders.rule_based._rules.genome import (
    DevelopmentalRuleGenome,
    mutate_developmental_rules,
    sample_developmental_rules,
)
from evodevo.encoders.rule_based._rules.interpreter import develop_rules
from evodevo.encoders.rule_based._rules.types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder


def _factors(**overrides):
    params = dict(dev_steps=10, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


# ---------------------------------------------------------------------------
# position_predicate
# ---------------------------------------------------------------------------


def test_default_position_predicate_is_disabled():
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.position_predicate == "disabled"


def test_sample_developmental_rules_never_draws_position_when_disabled():
    for seed in range(300):
        import random
        random.seed(seed)
        genome = sample_developmental_rules(position_predicate="disabled")
        assert all(r.condition_type != ConditionType.POSITION for r in genome.rules)


def test_sample_developmental_rules_can_draw_position_when_enabled():
    import random
    found = False
    for seed in range(300):
        random.seed(seed)
        genome = sample_developmental_rules(position_predicate="enabled")
        if any(r.condition_type == ConditionType.POSITION for r in genome.rules):
            found = True
            break
    assert found, "position_predicate='enabled' never drew a POSITION condition in 300 draws"


def test_mutation_never_introduces_position_when_disabled():
    import random
    random.seed(0)
    genome = sample_developmental_rules(position_predicate="disabled")
    for seed in range(300):
        random.seed(seed + 1000)
        genome = mutate_developmental_rules(genome, mutation_rate=0.5, position_predicate="disabled")
        assert all(r.condition_type != ConditionType.POSITION for r in genome.rules)


def test_deserialized_position_genome_still_decodes_regardless_of_position_predicate():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.POSITION, {"position": PositionType.TOP}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    encoder = RuleBasedEncoder(factors=_factors(), position_predicate="disabled")
    serialized = encoder.serialize_genome(genome)
    restored = encoder.deserialize_genome(serialized)
    body = encoder.decode(restored)
    assert body.shape == (5, 5)
    # row 0 (top) should have been SET_SOFT by the deserialized POSITION rule
    assert (body[0] == SOFT).all()


def test_mutate_condition_params_on_existing_position_rule_does_not_raise():
    # _mutate_condition_params re-rolls params for the EXISTING condition
    # type -- if that type is POSITION (from a deserialized genome), this
    # must not be blocked by position_predicate="disabled".
    import random
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.POSITION, {"position": PositionType.TOP}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    for seed in range(50):
        random.seed(seed)
        genome = mutate_developmental_rules(genome, mutation_rate=1.0, position_predicate="disabled")
        assert len(genome.rules) >= 2


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


def test_default_boundary_is_no_op():
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.boundary == "no_op"


def test_boundary_no_op_discards_off_grid_growth():
    # A cell at the top-left corner growing UP (off-grid) must be a no-op.
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
                           ActionType.GROW_DIRECTIONAL, {"direction": Direction.UP, "voxel_type": VoxelState.SOFT}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    initial = np.zeros((5, 5), dtype=int)
    initial[0, 0] = SOFT
    result = develop_rules(genome, shape=(5, 5), iterations=1, initial_grid=initial, boundary="no_op")
    # no wraparound growth into (4, 0)
    assert result[4, 0] == EMPTY


def test_boundary_toroidal_wraps_growth():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
                           ActionType.GROW_DIRECTIONAL, {"direction": Direction.UP, "voxel_type": VoxelState.SOFT}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    initial = np.zeros((5, 5), dtype=int)
    initial[0, 0] = SOFT
    result = develop_rules(genome, shape=(5, 5), iterations=1, initial_grid=initial, boundary="toroidal")
    assert result[4, 0] == SOFT


# ---------------------------------------------------------------------------
# neighbourhood
# ---------------------------------------------------------------------------


def test_default_neighbourhood_is_von_neumann():
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.neighbourhood == "von_neumann"


def test_neighbourhood_moore_counts_diagonal_neighbors():
    # Cell (2,2) with all 4 diagonal neighbors occupied (but 0 cardinal
    # neighbors occupied): von_neumann sees count=0 (threshold=1 fails),
    # moore sees count=4 (threshold=1 succeeds).
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.NEIGHBOR_COUNT, {"threshold": 1},
                           ActionType.SET_HARD, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    initial = np.zeros((5, 5), dtype=int)
    initial[1, 1] = SOFT
    initial[1, 3] = SOFT
    initial[3, 1] = SOFT
    initial[3, 3] = SOFT
    von_neumann_result = develop_rules(genome, shape=(5, 5), iterations=1, initial_grid=initial, neighbourhood="von_neumann")
    moore_result = develop_rules(genome, shape=(5, 5), iterations=1, initial_grid=initial, neighbourhood="moore")
    assert von_neumann_result[2, 2] == EMPTY  # SET_EMPTY branch (no cardinal neighbor occupied)
    from evodevo.core.materials import RIGID
    assert moore_result[2, 2] == RIGID  # SET_HARD branch (4 diagonal neighbors occupied)
