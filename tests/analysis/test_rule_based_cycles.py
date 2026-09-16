"""Tests for evodevo/analysis/rule_based_cycles.py -- exact (mu, p)
cycle detection (Proposition R1)."""
import numpy as np

from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome
from evodevo.encoders.rule_based._rules.types import ActionType, ConditionType, DevelopmentalRule, VoxelState
from evodevo.analysis.rule_based_cycles import (
    exact_cycle,
    run,
    summarize,
    w_divisibility_check,
)


def test_exact_cycle_fixed_point():
    # ALWAYS -> SET_SOFT: every cell becomes SOFT on step 1 and stays SOFT
    # forever -- mu should be small (settles almost immediately) and p=1.
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_HARD, {}),
    ])
    result = exact_cycle(genome, conflict_resolution="first_rule_index", position_predicate="disabled")
    assert result["cap_hit"] is False
    assert result["p"] == 1


def test_exact_cycle_period_two_by_construction():
    # cell_type(EMPTY) -> SET_SOFT; cell_type(SOFT) -> SET_EMPTY: every
    # cell alternates EMPTY<->SOFT every step, forever -- an exact,
    # hand-verifiable period-2 cycle with mu=0 (immediately periodic).
    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.EMPTY}, ActionType.SET_SOFT, {}),
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT}, ActionType.SET_EMPTY, {}),
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_HARD, {}),
    ])
    result = exact_cycle(genome, conflict_resolution="first_rule_index", position_predicate="disabled")
    assert result["cap_hit"] is False
    assert result["p"] == 2


def test_run_and_summarize_small_n():
    r = run(n=20, seed=0, conflict_resolution="first_rule_index", position_predicate="disabled")
    s = summarize(r)
    assert s["n"] == 20
    assert s["n_resolved"] + s["n_cap_hit"] == 20
    assert s["fraction_fixed_point"] + s["fraction_cycling"] == (
        (s["n_fixed_point"] + s["n_cycling"]) / 20
    )
    # every resolved genome's p appears in period_counts
    assert sum(s["period_counts"].values()) == s["n_resolved"]


def test_w_divisibility_check_small_n():
    r = run(n=50, seed=0, conflict_resolution="first_rule_index", position_predicate="disabled")
    w = w_divisibility_check(r, w=6)
    assert 0.0 <= w["fraction_not_divide_w"] <= 1.0
    assert w["n_not_divide_w"] <= w["n_cycling"]


def test_run_reproduces_same_draw_order_as_settling_audit_module():
    from evodevo.analysis.rule_based_settling_audit import _build_encoder, DEFAULT_SEED as AUDIT_SEED

    encoder = _build_encoder(dev_steps=10, seed=AUDIT_SEED)
    rng = np.random.default_rng(AUDIT_SEED)
    expected_first = encoder.random_genome(rng)

    from evodevo.core.factors import ExperimentFactors
    from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

    factors = ExperimentFactors(dev_steps=10, init_condition="center_seed", seed=AUDIT_SEED)
    cycles_encoder = RuleBasedEncoder(factors=factors, position_predicate="enabled")
    rng2 = np.random.default_rng(AUDIT_SEED)
    actual_first = cycles_encoder.random_genome(rng2)

    assert expected_first.rules == actual_first.rules
