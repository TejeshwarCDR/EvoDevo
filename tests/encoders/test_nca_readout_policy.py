"""Tests for NCAEncoder's selectable readout policy (fixed_t vs
mode_of_window), added alongside the existing fixed-T argmax readout --
see evodevo/encoders/nca/encoder.py::decode and
results/nca/READOUT_POLICY_SELECTION.md.
"""
import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca.encoder import NCAEncoder, _mode_of_window


def _factors(**overrides):
    params = dict(dev_steps=40, init_condition="center_seed", seed=0, hidden_channels=0)
    params.update(overrides)
    return ExperimentFactors(**params)


def test_default_readout_mode_is_mode_of_window():
    # NCAEncoder's own CLASS default is mode_of_window/50, matching the
    # resolved PRIMARY (results/nca/READOUT_POLICY_SELECTION.md,
    # resolved_config.py::PRIMARY) -- the same "class default equals
    # PRIMARY" convention already used for update_rule/decay/state_bound
    # (docs/decisions.md's Outcome B row). fixed_t remains fully selectable
    # (test_fixed_t_still_selectable below). Callers that sweep `dev_steps`
    # across shallow values (depth_study.py, structural_report.py) must
    # pin readout_mode="fixed_t" explicitly -- see decode()'s CAVEAT
    # docstring for why mode_of_window at T << W is not meaningful there.
    encoder = NCAEncoder(factors=_factors())
    assert encoder.readout_mode == "mode_of_window"
    assert encoder.readout_window == 50


def test_fixed_t_still_selectable():
    encoder = NCAEncoder(factors=_factors(), readout_mode="fixed_t", readout_window=10)
    assert encoder.readout_mode == "fixed_t"
    assert encoder.readout_window == 10


def test_invalid_readout_mode_raises():
    with pytest.raises(ValueError):
        NCAEncoder(factors=_factors(), readout_mode="not_a_mode")


def test_invalid_readout_window_raises():
    with pytest.raises(ValueError):
        NCAEncoder(factors=_factors(), readout_window=0)


def test_mode_of_window_constant_window_returns_that_body():
    a = np.array([[0, 1], [2, 3]])
    window = [a.copy() for _ in range(5)]
    result = _mode_of_window(window)
    np.testing.assert_array_equal(result, a)


def test_mode_of_window_picks_strict_majority():
    a = np.array([[0, 0], [0, 0]])
    b = np.array([[1, 1], [1, 1]])
    window = [a, a, a, b, b]
    result = _mode_of_window(window)
    np.testing.assert_array_equal(result, a)


def test_mode_of_window_tie_break_is_lexicographically_smallest():
    a = np.array([[1, 0], [0, 0]])  # flattens to (1,0,0,0)
    b = np.array([[0, 1], [0, 0]])  # flattens to (0,1,0,0) -- lexicographically smaller
    window = [a, b]  # tied 1-1
    result = _mode_of_window(window)
    np.testing.assert_array_equal(result, b)


def test_mode_of_window_deterministic_no_rng_dependence():
    a = np.array([[2, 1], [0, 3]])
    b = np.array([[1, 2], [3, 0]])
    window = [a, b, a]
    r1 = _mode_of_window(window)
    r2 = _mode_of_window(window)
    np.testing.assert_array_equal(r1, r2)


def test_mode_of_window_equals_fixed_t_for_a_genome_that_settles():
    # zero-weight genome under unbounded_residual: state never changes from
    # the fixed initial condition (see tests/encoders/test_nca.py's own
    # zero-weight residual-property test) -- a maximally "cleanly settled"
    # genome, the case the windowed readout must reproduce exactly.
    factors = _factors()
    fixed = NCAEncoder(factors=factors, update_rule="unbounded_residual", readout_mode="fixed_t")
    windowed = NCAEncoder(factors=factors, update_rule="unbounded_residual", readout_mode="mode_of_window", readout_window=10)
    genome = np.zeros(fixed.genome_length())
    body_fixed = fixed.decode(genome)
    body_windowed = windowed.decode(genome)
    np.testing.assert_array_equal(body_fixed, body_windowed)


def test_readout_mode_appears_in_describe():
    encoder = NCAEncoder(factors=_factors(), readout_mode="mode_of_window", readout_window=20)
    described = encoder.describe()
    assert described["readout_mode"] == "mode_of_window"
    assert described["readout_window"] == 20


def test_decode_window_shorter_than_dev_steps_uses_available_prefix():
    # dev_steps < readout_window: the window should just use whatever
    # trajectory is available (dev_steps+1 snapshots), not crash.
    factors = _factors(dev_steps=3)
    encoder = NCAEncoder(factors=factors, readout_mode="mode_of_window", readout_window=10)
    genome = encoder.random_genome(np.random.default_rng(0))
    body = encoder.decode(genome)
    assert body.shape == (5, 5)
