import importlib.util

import numpy as np
import pytest

from evodevo.evaluation.cache import FitnessCache
from evodevo.evaluation.ppo_runner import (
    DEFAULT_SENTINEL_FITNESS,
    evaluate_body,
    evaluate_body_surrogate,
    make_evaluator,
)

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

INVALID_BODY = np.zeros((5, 5), dtype=int)  # all-empty: fails validation

HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None


def test_surrogate_deterministic():
    r1 = evaluate_body_surrogate(GOOD_WALKER, seed=0)
    r2 = evaluate_body_surrogate(GOOD_WALKER, seed=0)
    assert r1.fitness == r2.fitness
    assert r1.valid and r2.valid


def test_surrogate_invalid_body_returns_sentinel():
    result = evaluate_body_surrogate(INVALID_BODY)
    assert not result.valid
    assert result.fitness == DEFAULT_SENTINEL_FITNESS
    assert result.reason is not None


def test_surrogate_custom_sentinel():
    result = evaluate_body_surrogate(INVALID_BODY, sentinel_fitness=-99.0)
    assert result.fitness == -99.0


def test_make_evaluator_surrogate():
    fn = make_evaluator("surrogate")
    assert fn is evaluate_body_surrogate


def test_make_evaluator_real():
    fn = make_evaluator("real")
    assert fn is evaluate_body


def test_make_evaluator_unknown_raises():
    with pytest.raises(ValueError):
        make_evaluator("bogus")


def test_cache_hit_and_miss():
    cache = FitnessCache()
    result = evaluate_body_surrogate(GOOD_WALKER, seed=0)

    fetched = cache.get(result.body_hash)
    assert fetched is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.put(result.body_hash, result)
    fetched = cache.get(result.body_hash)
    assert fetched is result
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.hit_rate == 0.5


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_ppo_smoke():
    result = evaluate_body(
        GOOD_WALKER,
        task="Walker-v0",
        ppo_timesteps=2000,
        eval_episodes=1,
        seed=0,
    )
    assert result.valid
    assert np.isfinite(result.fitness)
