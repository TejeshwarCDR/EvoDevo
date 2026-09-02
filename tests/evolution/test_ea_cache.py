import numpy as np

from evodevo.core.config import ExperimentConfig
from evodevo.encoders.direct import DirectEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea


def _make_config(**overrides):
    defaults = dict(
        seed=0,
        encoder="direct",
        task="dummy-task",
        population_size=5,
        offspring_size=5,
        eval_budget=17,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class _AlwaysSameGenomeDirectEncoder(DirectEncoder):
    """Deterministic stand-in that always decodes to the same body, so the
    body-hash cache is guaranteed to be hit on every evaluation after the
    first regardless of RNG draws."""

    def random_genome(self, rng):
        return np.zeros(self.genome_length(), dtype=int)

    def mutate(self, genome, rng):
        return genome.copy()


def test_cache_disabled_by_default():
    config = _make_config()
    assert config.cache_enabled is False


def test_cache_hit_still_consumes_budget_and_is_flagged():
    encoder = _AlwaysSameGenomeDirectEncoder()
    config = _make_config(cache_enabled=True, eval_budget=10, population_size=3, offspring_size=3)
    result = run_ea(encoder, evaluate_body_surrogate, config)

    # budget accounting must be identical to the non-cached path: every
    # evaluate_genome() call -- hit or miss -- increments evaluations_used.
    assert result.evaluations_used == 10
    assert len(result.records) == 10

    # every body after the first evaluation is a repeat (same genome every
    # time), so there must be at least one recorded cache hit.
    assert any(r["cache_hit"] for r in result.records)
    # the very first evaluation of a never-before-seen body cannot be a hit.
    assert result.records[0]["cache_hit"] is False

    assert result.cache_hit_rate is not None
    assert 0.0 < result.cache_hit_rate <= 1.0


def test_cache_enabled_and_disabled_consume_identical_evaluations_for_same_seed():
    """This is the core fairness guarantee: enabling the cache must never
    change how many evaluations a run consumes for a given seed/config,
    only whether some of those evaluations were served from cache."""
    encoder_a = _AlwaysSameGenomeDirectEncoder()
    encoder_b = _AlwaysSameGenomeDirectEncoder()

    config_cached = _make_config(seed=42, cache_enabled=True, eval_budget=13, population_size=4, offspring_size=4)
    config_uncached = _make_config(seed=42, cache_enabled=False, eval_budget=13, population_size=4, offspring_size=4)

    result_cached = run_ea(encoder_a, evaluate_body_surrogate, config_cached)
    result_uncached = run_ea(encoder_b, evaluate_body_surrogate, config_uncached)

    assert result_cached.evaluations_used == result_uncached.evaluations_used == 13
    assert result_uncached.cache_hit_rate is None

    # fitness values themselves must be identical either way -- the cache only
    # changes whether the evaluator function is actually re-invoked, not what
    # it would have returned.
    fitness_cached = [r["fitness"] for r in result_cached.records]
    fitness_uncached = [r["fitness"] for r in result_uncached.records]
    assert fitness_cached == fitness_uncached


def test_records_always_carry_cache_hit_field_even_when_disabled():
    encoder = DirectEncoder()
    config = _make_config(cache_enabled=False, eval_budget=5, population_size=5, offspring_size=5)
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert all(r["cache_hit"] is False for r in result.records)
    assert result.cache_hit_rate is None
