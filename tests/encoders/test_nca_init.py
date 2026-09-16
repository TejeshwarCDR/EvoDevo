"""Tests for the layer-wise (per-fan-in) variance scaling applied when
unpacking the NCA genome -- see evodevo/encoders/nca/encoder.py module
docstring and results/nca/INIT_STUDY.md.
"""
import warnings

import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca import N_MATERIALS, NCAEncoder


def _factors(**overrides):
    params = dict(dev_steps=10, init_condition="full_nonsymmetric", seed=0, hidden_channels=3)
    params.update(overrides)
    return ExperimentFactors(**params)


def _preactivation_variances(encoder, n=2000, seed=0):
    """Returns (layer1_variance, layer2_variance) of the pre-activation
    (pre-tanh / pre-output) values over n random genomes, with the layer's
    input standardised to unit variance per element (the standard
    fan-in-scaling analysis: this isolates what the unpacking's variance
    scaling itself contributes, independent of the actual grid state's own
    scale -- see results/nca/INIT_STUDY.md)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(n, encoder.input_dim))  # layer 1 input: perception vector
    x2 = rng.normal(size=(n, encoder.hidden_width))  # layer 2 input: hidden activations

    layer1_samples = np.empty((n, encoder.hidden_width))
    layer2_samples = np.empty((n, encoder.channels))
    for i in range(n):
        genome = encoder.random_genome(rng)
        w1, b1, w2, b2 = encoder._unpack(genome)
        layer1_samples[i] = x1[i] @ w1 + b1
        layer2_samples[i] = x2[i] @ w2 + b2

    return float(np.var(layer1_samples)), float(np.var(layer2_samples))


# --- genome length unchanged -------------------------------------------------


def test_genome_length_unchanged_at_792_for_h3_h16():
    encoder = NCAEncoder(factors=_factors(hidden_channels=3), hidden_width=16)
    channels = N_MATERIALS + 3
    input_dim = 5 * channels
    expected = input_dim * 16 + 16 + 16 * channels + channels
    assert expected == 792  # sanity: matches the documented figure
    assert encoder.genome_length() == expected


# --- pre-activation variance near 1.0 at init_gain=1.0 -----------------------


def test_preactivation_variance_near_unit_at_both_layers():
    encoder = NCAEncoder(factors=_factors(), init_gain=1.0, update_rule="unbounded_residual")
    var1, var2 = _preactivation_variances(encoder, n=2000, seed=1)
    assert 0.5 <= var1 <= 2.0, f"layer 1 pre-activation variance {var1} not within [0.5, 2.0] of 1.0"
    assert 0.5 <= var2 <= 2.0, f"layer 2 pre-activation variance {var2} not within [0.5, 2.0] of 1.0"


# --- init_scale deprecated alias --------------------------------------------


def test_init_scale_alias_raises_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        NCAEncoder(factors=_factors(), init_scale=0.5)


def test_init_scale_alias_maps_directly_to_init_gain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        encoder = NCAEncoder(factors=_factors(), init_scale=0.5)
    assert encoder.init_gain == 0.5


def test_init_gain_without_init_scale_raises_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        NCAEncoder(factors=_factors(), init_gain=2.0)  # must not raise


# --- random_genome independent of init_gain ---------------------------------


def test_random_genome_distribution_independent_of_init_gain():
    encoder_a = NCAEncoder(factors=_factors(), init_gain=0.1)
    encoder_b = NCAEncoder(factors=_factors(), init_gain=100.0)
    genome_a = encoder_a.random_genome(np.random.default_rng(0))
    genome_b = encoder_b.random_genome(np.random.default_rng(0))
    np.testing.assert_array_equal(genome_a, genome_b)


# --- round-trip: unpack(pack(W)) == W ----------------------------------------


def test_pack_unpack_round_trip():
    encoder = NCAEncoder(factors=_factors(), init_gain=1.0, bias_scale=0.1, spectral_normalise=False)
    rng = np.random.default_rng(11)

    w1 = rng.normal(size=(encoder.input_dim, encoder.hidden_width))
    b1 = rng.normal(size=encoder.hidden_width)
    w2 = rng.normal(size=(encoder.hidden_width, encoder.channels))
    b2 = rng.normal(size=encoder.channels)

    genome = encoder._pack(w1, b1, w2, b2)
    assert genome.shape == (encoder.genome_length(),)

    w1_rt, b1_rt, w2_rt, b2_rt = encoder._unpack(genome)
    np.testing.assert_allclose(w1_rt, w1, atol=1e-10)
    np.testing.assert_allclose(b1_rt, b1, atol=1e-10)
    np.testing.assert_allclose(w2_rt, w2, atol=1e-10)
    np.testing.assert_allclose(b2_rt, b2, atol=1e-10)


@pytest.mark.parametrize("init_gain,bias_scale", [(1.0, 0.1), (0.25, 0.5), (4.0, 0.01)])
def test_pack_unpack_round_trip_across_gain_settings(init_gain, bias_scale):
    encoder = NCAEncoder(factors=_factors(), init_gain=init_gain, bias_scale=bias_scale)
    rng = np.random.default_rng(3)
    w1 = rng.normal(size=(encoder.input_dim, encoder.hidden_width))
    b1 = rng.normal(size=encoder.hidden_width)
    w2 = rng.normal(size=(encoder.hidden_width, encoder.channels))
    b2 = rng.normal(size=encoder.channels)

    genome = encoder._pack(w1, b1, w2, b2)
    w1_rt, b1_rt, w2_rt, b2_rt = encoder._unpack(genome)
    np.testing.assert_allclose(w1_rt, w1, atol=1e-8)
    np.testing.assert_allclose(b1_rt, b1, atol=1e-8)
    np.testing.assert_allclose(w2_rt, w2, atol=1e-8)
    np.testing.assert_allclose(b2_rt, b2, atol=1e-8)


# --- mutation_sigma is untouched by this change ------------------------------


def test_default_mutation_sigma_not_retuned():
    encoder = NCAEncoder(factors=_factors())
    assert encoder.mutation_sigma == pytest.approx(0.1)
