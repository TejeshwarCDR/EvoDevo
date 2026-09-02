import numpy as np

from evodevo.seeding import make_rng, seed_everything


def test_make_rng_reproducible():
    rng1 = make_rng(42)
    rng2 = make_rng(42)
    assert np.array_equal(rng1.random(10), rng2.random(10))


def test_make_rng_different_seeds_differ():
    rng1 = make_rng(1)
    rng2 = make_rng(2)
    assert not np.array_equal(rng1.random(10), rng2.random(10))


def test_seed_everything_runs():
    seed_everything(42)
    import random

    a = random.random()
    seed_everything(42)
    b = random.random()
    assert a == b
