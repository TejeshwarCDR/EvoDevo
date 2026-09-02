import numpy as np
import pytest

from evodevo.morphology.validate import validate_body

try:
    from evogym import has_actuator, is_connected

    HAS_EVOGYM = True
except ImportError:
    HAS_EVOGYM = False

# A known-good, hand-written 5x5 walker: connected, has actuators.
GOOD_WALKER = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)


def test_all_empty_rejected():
    body = np.zeros((5, 5), dtype=int)
    result = validate_body(body)
    assert not result.valid


def test_disconnected_two_blob_rejected():
    body = np.zeros((5, 5), dtype=int)
    body[0, 0] = 1
    body[0, 1] = 3  # actuator to satisfy that check, blob 1 in top-left corner
    body[4, 4] = 1
    body[4, 3] = 3  # blob 2 in bottom-right corner, disconnected from blob 1
    result = validate_body(body)
    assert not result.valid
    assert any("connected" in r for r in result.reasons)


def test_no_actuator_rejected():
    body = np.zeros((5, 5), dtype=int)
    body[1, 1] = 1
    body[1, 2] = 2
    body[1, 3] = 1
    result = validate_body(body)
    assert not result.valid
    assert any("actuator" in r for r in result.reasons)


def test_value_five_rejected():
    body = GOOD_WALKER.copy()
    body[0, 0] = 5
    result = validate_body(body)
    assert not result.valid
    assert any("0..4" in r for r in result.reasons)


def test_wrong_shape_rejected():
    body = np.zeros((4, 5), dtype=int)
    result = validate_body(body)
    assert not result.valid
    assert any("shape" in r for r in result.reasons)


def test_known_good_walker_accepted():
    result = validate_body(GOOD_WALKER)
    assert result.valid
    assert result.reasons == []


def test_single_voxel_rejected_by_min_voxels():
    body = np.zeros((5, 5), dtype=int)
    body[2, 2] = 3  # single actuator voxel: connected, has actuator, but < min_voxels
    result = validate_body(body, min_voxels=3)
    assert not result.valid
    assert any("min_voxels" in r for r in result.reasons)


def test_non_integer_dtype_rejected():
    body = GOOD_WALKER.astype(float)
    result = validate_body(body)
    assert not result.valid
    assert any("dtype" in r for r in result.reasons)


@pytest.mark.skipif(not HAS_EVOGYM, reason="evogym is not built/importable")
def test_connectivity_matches_evogym():
    assert is_connected(GOOD_WALKER) == True
    result = validate_body(GOOD_WALKER)
    assert result.valid


@pytest.mark.skipif(not HAS_EVOGYM, reason="evogym is not built/importable")
def test_actuator_check_matches_evogym():
    assert has_actuator(GOOD_WALKER) == True
