import numpy as np
import pytest

from evodevo.core.initial_conditions import get_initial_condition
from evodevo.core.materials import N_MATERIALS, SOFT

GRID_SHAPE = (5, 5)
CONDITIONS = ["center_seed", "full_nonsymmetric", "full_uniform"]


@pytest.mark.parametrize("name", CONDITIONS)
def test_roundtrip_consistency(name):
    condition = get_initial_condition(name)
    state = condition.as_state_tensor(GRID_SHAPE, N_MATERIALS)
    argmax_grid = np.argmax(state[..., :N_MATERIALS], axis=-1).astype(int)
    np.testing.assert_array_equal(argmax_grid, condition.as_discrete_grid(GRID_SHAPE))


@pytest.mark.parametrize("name", CONDITIONS)
def test_roundtrip_consistency_with_hidden_channels(name):
    condition = get_initial_condition(name)
    channels = N_MATERIALS + 3
    state = condition.as_state_tensor(GRID_SHAPE, channels)
    assert state.shape == (5, 5, channels)
    argmax_grid = np.argmax(state[..., :N_MATERIALS], axis=-1).astype(int)
    np.testing.assert_array_equal(argmax_grid, condition.as_discrete_grid(GRID_SHAPE))


def test_center_seed_is_a_single_soft_voxel():
    grid = get_initial_condition("center_seed").as_discrete_grid(GRID_SHAPE)
    assert grid[2, 2] == SOFT
    assert np.count_nonzero(grid) == 1


def test_full_uniform_is_all_soft():
    grid = get_initial_condition("full_uniform").as_discrete_grid(GRID_SHAPE)
    assert np.all(grid == SOFT)


def test_full_nonsymmetric_is_deterministic():
    a = get_initial_condition("full_nonsymmetric").as_discrete_grid(GRID_SHAPE)
    b = get_initial_condition("full_nonsymmetric").as_discrete_grid(GRID_SHAPE)
    np.testing.assert_array_equal(a, b)


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        get_initial_condition("bogus")


def test_full_uniform_fails_to_differentiate_under_a_symmetric_deterministic_rule():
    """Validates the locality argument: a perfectly symmetric starting grid
    under a rule that only distinguishes EMPTY vs non-EMPTY neighbour counts
    (no direction/position asymmetry) cannot break the initial symmetry --
    every cell of full_uniform starts identical to every other, so any such
    rule must produce another perfectly uniform grid."""
    grid = get_initial_condition("full_uniform").as_discrete_grid(GRID_SHAPE)
    assert np.all(grid == grid[0, 0])
