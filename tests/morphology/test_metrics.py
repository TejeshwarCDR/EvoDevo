import numpy as np

from evodevo.morphology.metrics import body_hash, body_metrics

# Fully left-right and top-bottom symmetric body.
SYMMETRIC_BODY = np.array(
    [
        [0, 1, 2, 1, 0],
        [0, 3, 1, 3, 0],
        [1, 1, 1, 1, 1],
        [0, 3, 1, 3, 0],
        [0, 1, 2, 1, 0],
    ],
    dtype=int,
)

# Asymmetric body: no mirror symmetry in either axis.
ASYMMETRIC_BODY = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
    ],
    dtype=int,
)


def test_symmetric_body_scores_perfect_symmetry():
    m = body_metrics(SYMMETRIC_BODY)
    assert m["horizontal_symmetry_score"] == 1.0
    assert m["vertical_symmetry_score"] == 1.0


def test_asymmetric_body_scores_below_perfect_symmetry():
    m = body_metrics(ASYMMETRIC_BODY)
    assert m["horizontal_symmetry_score"] < 1.0
    assert m["vertical_symmetry_score"] < 1.0


def test_body_metrics_counts():
    m = body_metrics(SYMMETRIC_BODY)
    assert m["empty_count"] == int(np.sum(SYMMETRIC_BODY == 0))
    assert m["rigid_count"] == int(np.sum(SYMMETRIC_BODY == 1))
    assert m["soft_count"] == int(np.sum(SYMMETRIC_BODY == 2))
    assert m["h_act_count"] == int(np.sum(SYMMETRIC_BODY == 3))
    assert m["v_act_count"] == int(np.sum(SYMMETRIC_BODY == 4))
    assert m["occupied_count"] == SYMMETRIC_BODY.size - m["empty_count"]
    expected_fraction = (m["h_act_count"] + m["v_act_count"]) / SYMMETRIC_BODY.size
    assert m["actuator_fraction"] == expected_fraction


def test_largest_component_size():
    m = body_metrics(SYMMETRIC_BODY)
    occupied = int(np.sum(SYMMETRIC_BODY != 0))
    assert m["largest_component_size"] == occupied  # fully connected


def test_hash_stability():
    h1 = body_hash(SYMMETRIC_BODY)
    h2 = body_hash(SYMMETRIC_BODY.copy())
    assert h1 == h2
    assert isinstance(h1, str)


def test_hash_differs_for_different_bodies():
    h1 = body_hash(SYMMETRIC_BODY)
    h2 = body_hash(ASYMMETRIC_BODY)
    assert h1 != h2
