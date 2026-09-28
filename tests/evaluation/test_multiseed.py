import os

import numpy as np
import pytest

from evodevo.evaluation.ppo_runner import (
    PPOTaskResult, aggregate_seed_results, evaluate_body, ppo_seed, validate_and_decode,
)
from evodevo.morphology.validate import validate_body
from tests.evolution.test_propose_tell import ARMS, _encoder
from tests.evaluation.test_ppo_runner import GOOD_WALKER, HAS_SIMULATOR


def _result(value, completed=True):
    return PPOTaskResult(value if completed else None, completed, None, 0, None, None, [], None, None, None, None)


def test_seed_derivation_is_stable_and_distinct():
    assert ppo_seed(0, 0) == 0
    assert [ppo_seed(7, k) for k in range(4)] == [ppo_seed(7, k) for k in range(4)]
    assert len({ppo_seed(7, k) for k in range(4)}) == 4


def test_aggregation_rules_and_failures():
    values = [_result(1.0), _result(3.0)]
    assert aggregate_seed_results(values, "max") == (3.0, True, False)
    assert aggregate_seed_results(values, "mean") == (2.0, True, False)
    assert aggregate_seed_results([_result(1.0), _result(None, False)], "max") == (1.0, True, True)
    assert aggregate_seed_results([_result(None, False)], "max") == (None, False, False)


@pytest.mark.parametrize("arm", ARMS)
def test_validate_and_decode_matches_embedded_validation_for_500_genomes(arm):
    encoder = _encoder(arm)
    rng = np.random.default_rng(123)
    for _ in range(500):
        genome = encoder.random_genome(rng)
        body, _connections, sentinel = validate_and_decode(genome, encoder)
        embedded = validate_body(encoder.decode(genome))
        assert np.array_equal(body, encoder.decode(genome))
        assert (sentinel is None) == embedded.valid


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_k2_saves_two_reloadable_policies(tmp_path):
    result = evaluate_body(GOOD_WALKER, "UpStepper-v0", 2000, 1, 0,
                           output_dir=str(tmp_path), ppo_seeds=2, ppo_seed_aggregation="max")
    assert result.valid
    assert len(result.per_seed_fitness) == 2
    assert result.fitness == max(result.per_seed_fitness)
    assert len(result.policy_paths) == 2
    assert all(os.path.getsize(path) > 0 for path in result.policy_paths)
    from stable_baselines3 import PPO
    for path in result.policy_paths:
        PPO.load(path)
