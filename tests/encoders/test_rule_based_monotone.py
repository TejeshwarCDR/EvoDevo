"""Tests for the "action_monotonicity" ablation (Proposition R4) --
termination-guaranteed development, the rule-based counterpart of NCA's
spectral_normalise certified ablation."""
import random

import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import N_MATERIALS, SOFT
from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome, sample_developmental_rules
from evodevo.encoders.rule_based._rules.interpreter import RANK, develop_rules
from evodevo.encoders.rule_based._rules.types import ActionType, ConditionType, DevelopmentalRule, VoxelState
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

SHAPE = (5, 5)
PHI_MAX = 25 * (N_MATERIALS - 1)  # 25 cells * max rank 4 = 100


def _factors(**overrides):
    params = dict(dev_steps=10, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


def _initial_grid():
    grid = np.zeros(SHAPE, dtype=int)
    grid[2, 2] = SOFT
    return grid


def _phi(grid: np.ndarray) -> int:
    return int(sum(RANK[int(v)] for v in grid.flatten()))


def test_rank_is_a_total_order_with_empty_lowest():
    from evodevo.core.materials import EMPTY
    assert RANK[EMPTY] == 0
    assert len(set(RANK.values())) == N_MATERIALS  # all distinct -> total order


def test_default_action_monotonicity_is_unrestricted():
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.action_monotonicity == "unrestricted"


def test_set_empty_never_fires_on_occupied_cell_under_monotone():
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_HARD, {}),
    ])
    initial = np.zeros(SHAPE, dtype=int)
    initial[2, 2] = SOFT
    result = develop_rules(genome, shape=SHAPE, iterations=1, initial_grid=initial, action_monotonicity="monotone")
    # SET_EMPTY (rank 0) can never win over SOFT (rank 1) -- the occupied
    # cell must NOT become EMPTY under monotone (rule 0 matches ALWAYS but
    # is gated as a no-op; rule 1 never even gets evaluated for this cell
    # since rule 0 already "matched" in Pass 1 -- the GATE is on the
    # WRITE, not the match, so the cell simply stays SOFT this step).
    assert result[2, 2] != 0


def test_grow_directional_always_permitted_under_monotone():
    from evodevo.encoders.rule_based._rules.types import Direction

    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(
            ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
            ActionType.GROW_DIRECTIONAL, {"direction": Direction.RIGHT, "voxel_type": VoxelState.HARD},
        ),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
    ])
    initial = np.zeros(SHAPE, dtype=int)
    initial[2, 2] = SOFT
    result = develop_rules(genome, shape=SHAPE, iterations=1, initial_grid=initial, action_monotonicity="monotone")
    assert result[2, 3] != 0  # grew into the empty neighbor, unaffected by monotone gating


def test_phi_non_decreasing_zero_violations_small_n_smoke():
    # Small-N smoke test (fast, always run) -- the full N=2000 x T=100
    # verification (this task's actual acceptance requirement) lives in
    # evodevo/analysis/rule_based_monotone_report.py, committed to
    # results/rule_based/MONOTONE_ABLATION.md, since it is too slow (~200k
    # single-step develop_rules calls) for the routine test suite.
    n_violations = 0
    n_checked = 0
    for idx in range(50):
        random.seed(idx)
        genome = sample_developmental_rules(position_predicate="disabled")
        grid = _initial_grid()
        phi_prev = _phi(grid)
        for t in range(100):
            grid = develop_rules(genome, shape=SHAPE, iterations=1, initial_grid=grid, action_monotonicity="monotone")
            phi = _phi(grid)
            n_checked += 1
            if phi < phi_prev:
                n_violations += 1
            phi_prev = phi
            assert phi <= PHI_MAX
    assert n_violations == 0, f"{n_violations}/{n_checked} Phi-decrease violations found under monotone"
