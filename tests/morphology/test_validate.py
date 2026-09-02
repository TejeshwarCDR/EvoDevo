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


def test_min_actuators_default_matches_original_has_actuator_behaviour():
    # default min_actuators=1: identical to the pre-parameterization behaviour.
    result = validate_body(GOOD_WALKER, min_actuators=1)
    assert result.valid
    default_result = validate_body(GOOD_WALKER)
    assert default_result.valid == result.valid
    assert default_result.reasons == result.reasons


def test_min_actuators_stricter_than_default_rejects_single_actuator_body():
    # GOOD_WALKER has 4 actuator (H_ACT) voxels; requiring more actuators than
    # present must fail with a min_actuators reason.
    result = validate_body(GOOD_WALKER, min_actuators=100)
    assert not result.valid
    assert any("min_actuators" in r for r in result.reasons)


def test_min_actuators_matches_rule_based_policy_of_two():
    # The rule-based arm's own validation floor (>= 2 actuators, from the
    # inventory report) can be expressed with this single function instead of
    # a separate hard-coded check.
    body = np.zeros((5, 5), dtype=int)
    body[1, 1] = 1
    body[1, 2] = 3  # exactly one actuator: connected, but below a floor of 2
    result = validate_body(body, min_voxels=1, min_actuators=2)
    assert not result.valid
    assert any("min_actuators" in r for r in result.reasons)

    body[2, 1] = 4  # add a second actuator, adjacent so still connected
    result = validate_body(body, min_voxels=1, min_actuators=2)
    assert result.valid


def test_zero_actuators_reason_text_unchanged_for_no_actuator_categorization():
    # evodevo/experiments/nca_sanity.py categorizes failure reasons by
    # substring match on "no actuator"; this must keep working for the
    # actuator_count == 0 case regardless of the min_actuators parameterization.
    body = np.zeros((5, 5), dtype=int)
    body[1, 1] = 1
    body[1, 2] = 1
    body[1, 3] = 1
    result = validate_body(body)
    assert not result.valid
    assert any("no actuator (material 3 or 4) present" == r for r in result.reasons)


@pytest.mark.skipif(not HAS_EVOGYM, reason="evogym is not built/importable")
def test_connectivity_matches_evogym():
    assert is_connected(GOOD_WALKER) == True
    result = validate_body(GOOD_WALKER)
    assert result.valid


@pytest.mark.skipif(not HAS_EVOGYM, reason="evogym is not built/importable")
def test_actuator_check_matches_evogym():
    assert has_actuator(GOOD_WALKER) == True
