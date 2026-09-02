import json

import numpy as np

from evodevo.encoders import get_encoder
from evodevo.encoders.direct import DirectEncoder


def test_genome_length():
    enc = DirectEncoder()
    assert enc.genome_length() == 25


def test_decode_shape_and_dtype():
    enc = DirectEncoder()
    genome = np.arange(25) % 5
    body = enc.decode(genome)
    assert body.shape == (5, 5)
    assert np.issubdtype(body.dtype, np.integer)


def test_decode_deterministic():
    enc = DirectEncoder()
    genome = np.arange(25) % 5
    body1 = enc.decode(genome)
    body2 = enc.decode(genome)
    assert np.array_equal(body1, body2)


def test_mutate_rate_zero_is_identity():
    enc = DirectEncoder(mutation_rate=0.0)
    rng = np.random.default_rng(0)
    genome = np.arange(25) % 5
    mutated = enc.mutate(genome, rng)
    assert np.array_equal(mutated, genome)


def test_mutate_rate_one_is_reproducible():
    enc = DirectEncoder(mutation_rate=1.0)
    genome = np.arange(25) % 5
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    mutated1 = enc.mutate(genome, rng1)
    mutated2 = enc.mutate(genome, rng2)
    assert np.array_equal(mutated1, mutated2)


def test_mutate_does_not_modify_input_in_place():
    enc = DirectEncoder(mutation_rate=1.0)
    genome = np.arange(25) % 5
    original = genome.copy()
    rng = np.random.default_rng(1)
    enc.mutate(genome, rng)
    assert np.array_equal(genome, original)


def test_random_genome_reproducible_with_same_seed():
    enc = DirectEncoder()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    g1 = enc.random_genome(rng1)
    g2 = enc.random_genome(rng2)
    assert np.array_equal(g1, g2)
    assert g1.shape == (25,)
    assert set(np.unique(g1).tolist()) <= {0, 1, 2, 3, 4}


def test_describe_json_serialisable():
    enc = DirectEncoder()
    d = enc.describe()
    json.dumps(d)  # must not raise
    assert d["name"] == "direct"


def test_get_encoder_registry():
    enc = get_encoder("direct")
    assert isinstance(enc, DirectEncoder)
    assert enc.name == "direct"


def test_get_encoder_unknown_raises():
    try:
        get_encoder("nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass
