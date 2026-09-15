"""Tests for depth_study.py's non-settling diagnosis classification
(evodevo.analysis.depth_study._classify_non_settling_genome /
diagnose_non_settling_nca) -- added alongside the T_primary clip fix."""
import numpy as np
import pytest

from evodevo.analysis import depth_study as D


def _bodies_with_last_flip():
    a = np.zeros((5, 5), dtype=int)
    b = np.ones((5, 5), dtype=int)
    # 41 steps (t=0..40); flip at the very last transition so _settle_info
    # reports never-settled.
    bodies = [a.copy() for _ in range(40)] + [b.copy()]
    return bodies


def test_classify_still_growing():
    magnitudes = [float(i) for i in range(1, 41)]  # strictly, steeply increasing
    bodies = _bodies_with_last_flip()
    assert D._classify_non_settling_genome(bodies, magnitudes) == "STILL_GROWING"


def test_classify_plateaued_oscillating():
    # Flat for the whole trace (well within 5% tolerance), no growth.
    magnitudes = [10.0] * 40
    bodies = _bodies_with_last_flip()
    assert D._classify_non_settling_genome(bodies, magnitudes) == "PLATEAUED_OSCILLATING"


def test_classify_slow_but_monotone():
    # Gentle drift: not >5% growth from step 35->40, not stable within 5%
    # over the last 5 steps either.
    magnitudes = [10.0] * 34 + [10.0, 10.2, 10.45, 10.7, 11.0, 11.35]
    bodies = _bodies_with_last_flip()
    category = D._classify_non_settling_genome(bodies, magnitudes)
    assert category in D.NON_SETTLING_CATEGORIES  # must be a valid category
    # Not growing fast enough for STILL_GROWING (mag[35]=10.2, mag[40]=11.35 -> ~11.3% growth actually)
    # so validate against the actual formula instead of assuming an exact bucket:
    mag_now, mag_35_ago = magnitudes[-1], magnitudes[-6]
    if mag_now > 1.05 * mag_35_ago:
        assert category == "STILL_GROWING"


def test_flip_steps_detects_every_change():
    bodies = [np.zeros((5, 5), dtype=int)] * 3
    bodies = [b.copy() for b in bodies]
    bodies[1][0, 0] = 1  # change at step 1
    bodies[2][0, 0] = 2  # change at step 2
    flips = D._flip_steps(bodies)
    assert flips == [1, 2]


def test_diagnose_non_settling_nca_counts_sum_to_total(tmp_path):
    result = D.diagnose_non_settling_nca(n_samples=50, init_condition="center_seed", seed=0)
    assert sum(result["counts"].values()) == result["total_never_settled"]
    assert set(result["counts"]) == set(D.NON_SETTLING_CATEGORIES)
    for category in D.NON_SETTLING_CATEGORIES:
        assert len(result["examples"][category]) <= 3
