import numpy as np

from evodevo.morphology.connectivity import largest_component_size
from evodevo.morphology.metrics import body_metrics
from evodevo.morphology.validate import validate_body

FULLY_CONNECTED = np.array(
    [
        [0, 1, 2, 1, 0],
        [0, 3, 1, 3, 0],
        [1, 1, 1, 1, 1],
        [0, 3, 1, 3, 0],
        [0, 1, 2, 1, 0],
    ],
    dtype=int,
)

TWO_BLOBS = np.zeros((5, 5), dtype=int)
TWO_BLOBS[0, 0] = 1
TWO_BLOBS[0, 1] = 3
TWO_BLOBS[4, 4] = 1
TWO_BLOBS[4, 3] = 3


def test_largest_component_size_single_component():
    occupied = FULLY_CONNECTED != 0
    assert largest_component_size(occupied) == int(occupied.sum())


def test_largest_component_size_two_blobs_returns_bigger_one():
    occupied = TWO_BLOBS != 0
    # each blob has 2 occupied voxels, so the largest component is 2, not 4.
    assert largest_component_size(occupied) == 2


def test_largest_component_size_empty_grid_is_zero():
    occupied = np.zeros((5, 5), dtype=bool)
    assert largest_component_size(occupied) == 0


def test_validate_and_metrics_agree_on_largest_component():
    """validate.py and metrics.py must derive from the same canonical
    implementation and therefore agree, since they operate on the same
    occupied mask for the same body."""
    occupied = FULLY_CONNECTED != 0
    from_connectivity = largest_component_size(occupied)
    from_metrics = body_metrics(FULLY_CONNECTED)["largest_component_size"]
    assert from_connectivity == from_metrics

    result = validate_body(FULLY_CONNECTED)
    assert result.valid
