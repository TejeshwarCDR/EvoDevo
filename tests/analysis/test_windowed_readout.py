"""Tests for evodevo/analysis/windowed_readout.py, including the
validity_aware tie-break added by the 2026-09-16 rule_based tie-break fix
(results/rule_based/TIEBREAK_VALIDITY_CHECK.md)."""
import numpy as np

from evodevo.analysis.windowed_readout import mode_of_window, windowed_body_at


def test_mode_of_window_constant_window_returns_that_body():
    a = np.array([[0, 1], [2, 3]])
    window = [a.copy() for _ in range(5)]
    result = mode_of_window(window)
    np.testing.assert_array_equal(result, a)


def test_mode_of_window_default_tie_break_is_lexicographically_smallest():
    a = np.array([[1, 0], [0, 0]])  # flattens to (1,0,0,0)
    b = np.array([[0, 1], [0, 0]])  # flattens to (0,1,0,0) -- lexicographically smaller
    window = [a, b]  # tied 1-1
    result = mode_of_window(window)
    np.testing.assert_array_equal(result, b)


def test_validity_aware_prefers_the_single_valid_tied_candidate():
    # A 5x5 grid, all EMPTY (0) -- invalid (too few voxels). A 5x5 grid with
    # a connected 3-voxel SOFT (2) actuator-free... actually needs an
    # actuator to be valid; build a minimal valid body with >=3 voxels and
    # >=1 actuator, connected.
    invalid = np.zeros((5, 5), dtype=int)
    valid = np.zeros((5, 5), dtype=int)
    valid[2, 1] = 2  # SOFT
    valid[2, 2] = 2  # SOFT
    valid[2, 3] = 3  # H_ACT (actuator)

    # lexicographic tie-break would pick `invalid` (all zeros is
    # lexicographically smallest), but validity_aware should override to `valid`
    # since exactly one of the two tied candidates validates.
    window = [invalid, valid]
    default_result = mode_of_window(window, validity_aware=False)
    np.testing.assert_array_equal(default_result, invalid)

    aware_result = mode_of_window(window, validity_aware=True)
    np.testing.assert_array_equal(aware_result, valid)


def test_validity_aware_falls_back_to_lexicographic_when_both_tied_are_valid():
    valid_a = np.zeros((5, 5), dtype=int)
    valid_a[2, 1] = 2
    valid_a[2, 2] = 2
    valid_a[2, 3] = 3

    valid_b = np.zeros((5, 5), dtype=int)
    valid_b[1, 1] = 2
    valid_b[1, 2] = 2
    valid_b[1, 3] = 3

    window = [valid_a, valid_b]
    aware_result = mode_of_window(window, validity_aware=True)
    default_result = mode_of_window(window, validity_aware=False)
    np.testing.assert_array_equal(aware_result, default_result)


def test_validity_aware_falls_back_to_lexicographic_when_neither_tied_is_valid():
    invalid_a = np.zeros((5, 5), dtype=int)
    invalid_b = np.zeros((5, 5), dtype=int)
    invalid_b[0, 0] = 2  # still invalid: single voxel, no actuator

    window = [invalid_a, invalid_b]
    aware_result = mode_of_window(window, validity_aware=True)
    default_result = mode_of_window(window, validity_aware=False)
    np.testing.assert_array_equal(aware_result, default_result)


def test_validity_aware_no_effect_when_there_is_a_real_majority_not_a_tie():
    a = np.zeros((5, 5), dtype=int)
    b = np.zeros((5, 5), dtype=int)
    b[0, 0] = 1
    window = [a, a, a, b]  # a is a strict majority, not a tie
    default_result = mode_of_window(window, validity_aware=False)
    aware_result = mode_of_window(window, validity_aware=True)
    np.testing.assert_array_equal(default_result, a)
    np.testing.assert_array_equal(aware_result, a)


def test_windowed_body_at_passes_through_validity_aware():
    invalid = np.zeros((5, 5), dtype=int)
    valid = np.zeros((5, 5), dtype=int)
    valid[2, 1] = 2
    valid[2, 2] = 2
    valid[2, 3] = 3
    trajectory = [invalid, valid]
    result = windowed_body_at(trajectory, t=1, w=2, validity_aware=True)
    np.testing.assert_array_equal(result, valid)
