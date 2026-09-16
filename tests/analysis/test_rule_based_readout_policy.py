"""Regression test for the silent-clip bug that once corrupted a T=105
shift query back to T=100 (docs/decisions.md's "Fix the silent-clip bug
class systemically" row): evodevo/analysis/rule_based_readout_policy.py's
TRAJECTORY_LENGTH must RAISE, not silently clip, if the module's own
required trajectory length ever exceeds HARD_MAX_ITERATIONS again."""
import pytest

from evodevo.analysis.rule_based_readout_policy import _required_trajectory_length


def test_required_trajectory_length_matches_max_candidate_plus_max_shift():
    assert _required_trajectory_length((10, 20, 40, 100), (-5, -1, 1, 5), hard_max_iterations=105) == 105


def test_required_trajectory_length_raises_instead_of_silently_clipping():
    with pytest.raises(ValueError, match="exceeds HARD_MAX_ITERATIONS"):
        _required_trajectory_length((10, 20, 40, 100), (-5, -1, 1, 5), hard_max_iterations=100)


def test_module_level_trajectory_length_is_within_the_real_hard_cap():
    """The actual module-level constant, computed at import time against
    the real HARD_MAX_ITERATIONS, must not have silently raised or clipped
    -- if HARD_MAX_ITERATIONS is ever lowered below what this module needs,
    importing it must fail loudly (covered by the two tests above), not
    silently succeed with a truncated trajectory."""
    from evodevo.analysis.rule_based_readout_policy import (
        HARD_MAX_ITERATIONS, SHIFTS, T_PRIMARY_CANDIDATES, TRAJECTORY_LENGTH,
    )

    assert TRAJECTORY_LENGTH == max(T_PRIMARY_CANDIDATES) + max(SHIFTS)
    assert TRAJECTORY_LENGTH <= HARD_MAX_ITERATIONS
