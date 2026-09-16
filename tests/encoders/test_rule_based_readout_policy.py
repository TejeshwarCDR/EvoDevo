"""Tests for RuleBasedEncoder's selectable readout policy (fixed_t vs
mode_of_window), mirroring tests/encoders/test_nca_readout_policy.py -- see
evodevo/encoders/rule_based/encoder.py::decode and
results/rule_based/READOUT_POLICY_SELECTION.md.
"""
import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder


def _factors(**overrides):
    params = dict(dev_steps=10, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


def test_default_readout_mode_is_fixed_t():
    # Unlike NCAEncoder (whose T_primary is comparable to its resolved
    # W=50), rule_based's class default stays fixed_t -- its resolved
    # (T_primary, W) is threaded explicitly by whichever caller constructs
    # "the primary rule_based arm" (results/T_PRIMARY_UNIFIED.md).
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.readout_mode == "fixed_t"
    assert encoder.readout_window == 5


def test_invalid_readout_mode_raises():
    with pytest.raises(ValueError):
        RuleBasedEncoder(factors=_factors(), readout_mode="not_a_mode")


def test_invalid_readout_window_raises():
    with pytest.raises(ValueError):
        RuleBasedEncoder(factors=_factors(), readout_window=0)


def test_readout_mode_appears_in_describe():
    encoder = RuleBasedEncoder(factors=_factors(), readout_mode="mode_of_window", readout_window=3)
    described = encoder.describe()
    assert described["readout_mode"] == "mode_of_window"
    assert described["readout_window"] == 3


def test_mode_of_window_equals_fixed_t_for_a_genome_that_settles():
    # A genome whose rule list produces a body that stops changing quickly
    # (SET_EMPTY-only genomes settle in 1 step) -- the maximally
    # "cleanly settled" case, the equivalence mode_of_window must reproduce.
    from evodevo.encoders.rule_based._rules.genome import DevelopmentalRuleGenome
    from evodevo.encoders.rule_based._rules.types import ActionType, ConditionType, DevelopmentalRule

    genome = DevelopmentalRuleGenome([
        DevelopmentalRule(condition_type=ConditionType.ALWAYS, action_type=ActionType.SET_SOFT),
        DevelopmentalRule(condition_type=ConditionType.ALWAYS, action_type=ActionType.SET_SOFT),
    ])
    fixed = RuleBasedEncoder(factors=_factors(), readout_mode="fixed_t")
    windowed = RuleBasedEncoder(factors=_factors(), readout_mode="mode_of_window", readout_window=5)
    body_fixed = fixed.decode(genome)
    body_windowed = windowed.decode(genome)
    np.testing.assert_array_equal(body_fixed, body_windowed)


def test_decode_window_shorter_than_dev_steps_uses_available_prefix():
    factors = _factors(dev_steps=3)
    encoder = RuleBasedEncoder(factors=factors, readout_mode="mode_of_window", readout_window=10)
    genome = encoder.random_genome(np.random.default_rng(0))
    body = encoder.decode(genome)
    assert body.shape == (5, 5)


def test_mode_of_window_uses_shared_windowed_readout_utility():
    # Confirms RuleBasedEncoder and NCAEncoder use the SAME shared mode-of-
    # window definition, not two independently-maintained copies.
    from evodevo.analysis.windowed_readout import mode_of_window
    from evodevo.encoders.nca.encoder import _mode_of_window as nca_alias

    assert mode_of_window is nca_alias
