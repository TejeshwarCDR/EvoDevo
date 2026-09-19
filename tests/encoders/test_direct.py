"""DirectEncoder-SPECIFIC behaviour only.

Generic Encoder-contract properties (decode shape/dtype/values, decode
determinism, random_genome reproducibility/seed-sensitivity, mutate
reproducibility, serialize/deserialize round-tripping, describe()
completeness, the physics/RL-free module rule, and the
does-not-resample-until-valid property) are covered once for every
registered encoder by tests/encoders/test_contract.py -- they are
deliberately not repeated here.
"""
import warnings

import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.encoders.direct import DirectEncoder


def _factors(**overrides):
    params = dict(dev_steps=1, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


def test_genome_length():
    enc = DirectEncoder(factors=_factors())
    assert enc.genome_length() == 25


def test_decode_is_a_pure_reshape_of_the_genome():
    # DirectEncoder-specific: unlike a developmental encoder, decode() does
    # no transformation at all -- genotype IS phenotype, just reshaped.
    enc = DirectEncoder(factors=_factors())
    genome = np.arange(25) % 5
    body = enc.decode(genome)
    assert np.array_equal(body, genome.reshape(5, 5))


def test_mutation_rate_overflow_warns_loudly_instead_of_silently_clipping():
    # mutation_rate * factors.mutation_scale > 1.0 is clipped to 1.0 (a
    # per-cell mutation probability cannot exceed 100%) -- but must warn,
    # not silently substitute a different effective rate than requested.
    with pytest.warns(UserWarning, match="exceeds 1.0"):
        enc = DirectEncoder(factors=_factors(mutation_scale=5.0), mutation_rate=0.5)
    assert enc.mutation_rate == 1.0


def test_mutation_rate_within_bounds_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc = DirectEncoder(factors=_factors(mutation_scale=1.0), mutation_rate=0.1)
    assert enc.mutation_rate == pytest.approx(0.1)


def test_mutate_rate_zero_is_identity():
    enc = DirectEncoder(factors=_factors(), mutation_rate=0.0)
    rng = np.random.default_rng(0)
    genome = np.arange(25) % 5
    mutated = enc.mutate(genome, rng)
    assert np.array_equal(mutated, genome)


def test_mutate_rate_one_flips_every_voxel_and_is_reproducible():
    enc = DirectEncoder(factors=_factors(), mutation_rate=1.0)
    genome = np.arange(25) % 5
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    mutated1 = enc.mutate(genome, rng1)
    mutated2 = enc.mutate(genome, rng2)
    assert np.array_equal(mutated1, mutated2)


def test_mutate_does_not_modify_input_in_place():
    enc = DirectEncoder(factors=_factors(), mutation_rate=1.0)
    genome = np.arange(25) % 5
    original = genome.copy()
    rng = np.random.default_rng(1)
    enc.mutate(genome, rng)
    assert np.array_equal(genome, original)


def test_get_encoder_registry():
    enc = get_encoder("direct", factors=_factors())
    assert isinstance(enc, DirectEncoder)
    assert enc.name == "direct"


# --- Regression: pin known (seed -> body) pairs so a future refactor of
# this file cannot silently change DirectEncoder's decode/mutate behaviour.
# These values were captured from the encoder as it existed at the time
# Prompt 3 (encoder plugin architecture) was applied; they must remain
# bit-identical, since that prompt is explicitly moves/refactoring-only.


def test_regression_random_genome_pinned_for_seed_0():
    enc = DirectEncoder(factors=_factors())
    rng = np.random.default_rng(0)
    genome = enc.random_genome(rng)
    expected = np.array(
        [1, 0, 0, 0, 3, 4, 1, 2, 0, 4, 3, 0, 3, 0, 2, 0, 3, 0, 0, 0, 0, 0, 1, 1, 1],
        dtype=int,
    )
    np.testing.assert_array_equal(genome, expected)
    body = enc.decode(genome)
    np.testing.assert_array_equal(body, expected.reshape(5, 5))


def test_regression_random_genome_pinned_for_seed_123():
    enc = DirectEncoder(factors=_factors())
    rng = np.random.default_rng(123)
    genome = enc.random_genome(rng)
    expected = np.array(
        [1, 0, 0, 0, 0, 3, 4, 0, 3, 3, 0, 0, 3, 0, 2, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0],
        dtype=int,
    )
    np.testing.assert_array_equal(genome, expected)


def test_regression_mutate_pinned_for_seed_99():
    enc = DirectEncoder(factors=_factors(), mutation_rate=1.0)
    genome = np.arange(25) % 5
    rng = np.random.default_rng(99)
    mutated = enc.mutate(genome, rng)
    expected = np.array(
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 1, 0, 2, 1, 0, 4, 0, 2],
        dtype=int,
    )
    np.testing.assert_array_equal(mutated, expected)
