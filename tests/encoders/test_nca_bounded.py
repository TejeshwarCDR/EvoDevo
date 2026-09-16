"""Tests for the bounded_leaky NCA update rule and its variant selection
mechanism (update_rule, boundary, update_prob, spectral_normalise).

See evodevo/encoders/nca/encoder.py module docstring and
results/depth/DEPTH_STUDY.md "Diagnosis of non-settling genomes" for the
motivation: the old unbounded_residual rule left most genomes diverging
rather than converging.
"""
import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca import NCAEncoder


def _factors(**overrides):
    params = dict(dev_steps=10, init_condition="full_nonsymmetric", seed=0, hidden_channels=3)
    params.update(overrides)
    return ExperimentFactors(**params)


# --- Proposition 1: boundedness --------------------------------------------


@pytest.mark.parametrize("boundary", ["zero_pad", "replicate", "toroidal"])
@pytest.mark.parametrize("hidden_channels", [0, 3, 6])
def test_bounded_leaky_state_stays_within_bound_at_every_step(boundary, hidden_channels):
    state_bound = 5.0
    encoder = NCAEncoder(
        factors=_factors(hidden_channels=hidden_channels, dev_steps=100),
        update_rule="bounded_leaky",
        state_bound=state_bound,
        boundary=boundary,
    )
    init_inf_norm = np.max(np.abs(encoder._initial_state))
    limit = max(init_inf_norm, state_bound) + 1e-9

    rng = np.random.default_rng(0)
    for _ in range(500):
        # extreme init_gain: sample genomes at a large scale to stress-test
        # the bound under aggressive raw network outputs.
        scale = rng.choice([0.01, 1.0, 50.0, 500.0])
        genome = rng.normal(scale=scale, size=encoder.genome_length())

        w1, b1, w2, b2 = encoder._unpack(genome)
        state = encoder._initial_state.copy()
        for _ in range(encoder.dev_steps):
            state = encoder._step(state, w1, b1, w2, b2)
            assert np.max(np.abs(state)) <= limit, (
                f"state exceeded bound: max|s|={np.max(np.abs(state))} > {limit} "
                f"(boundary={boundary}, hidden_channels={hidden_channels}, scale={scale})"
            )


# --- unbounded_residual reproduces old behaviour exactly --------------------


def test_unbounded_residual_reproduces_pinned_decode_seed_5():
    # See tests/encoders/test_nca.py::test_regression_decode_pinned_for_seed_5
    # for the pinned value and the two rounds of updates it has been through.
    encoder = NCAEncoder(factors=_factors(), update_rule="unbounded_residual")
    rng = np.random.default_rng(5)
    genome = encoder.random_genome(rng)
    body = encoder.decode(genome)
    expected = np.array(
        [
            [1, 3, 3, 3, 0],
            [1, 1, 1, 0, 4],
            [1, 4, 1, 3, 3],
            [1, 1, 2, 1, 1],
            [1, 2, 4, 4, 3],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(body, expected)


def test_unbounded_residual_step_matches_state_plus_delta():
    encoder = NCAEncoder(factors=_factors(hidden_channels=0), update_rule="unbounded_residual")
    rng = np.random.default_rng(1)
    state = rng.normal(size=(5, 5, encoder.channels))
    genome = encoder.random_genome(rng)
    w1, b1, w2, b2 = encoder._unpack(genome)

    perception = encoder._perceive(state)
    flat = perception.reshape(-1, encoder.input_dim)
    hidden = np.tanh(flat @ w1 + b1)
    delta = (hidden @ w2 + b2).reshape(state.shape)
    expected = state + delta

    actual = encoder._step(state, w1, b1, w2, b2)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


# --- synchronicity preserved for update_prob=1.0 -----------------------------


def test_synchronicity_matches_synchronous_not_sequential_update_default_rule():
    encoder = NCAEncoder(
        factors=_factors(hidden_channels=0, dev_steps=1),
        hidden_width=1,
        update_rule="unbounded_residual",
        update_prob=1.0,
    )
    channels = encoder.channels

    rng = np.random.default_rng(42)
    state = rng.normal(scale=0.1, size=(5, 5, channels))

    own_ch0_idx = 0
    left_block_index = 3
    left_ch0_idx = left_block_index * channels + 0

    w1 = np.zeros((encoder.input_dim, 1))
    w1[own_ch0_idx, 0] = -0.001
    w1[left_ch0_idx, 0] = 0.001
    b1 = np.zeros(1)
    w2 = np.zeros((1, channels))
    w2[0, 0] = 1000.0
    b2 = np.zeros(channels)

    new_state = encoder._step(state, w1, b1, w2, b2)

    expected_sync = state[:, :, 0].copy()
    expected_sync[:, 1:] = state[:, :-1, 0]
    expected_sync[:, 0] = 0.0
    np.testing.assert_allclose(new_state[:, :, 0], expected_sync, atol=1e-5)


# --- decode() determinism at update_prob=1.0 ---------------------------------


def test_decode_deterministic_across_repeated_calls_default_update_prob():
    encoder = NCAEncoder(factors=_factors())
    genome = encoder.random_genome(np.random.default_rng(3))
    body1 = encoder.decode(genome)
    body2 = encoder.decode(genome)
    np.testing.assert_array_equal(body1, body2)


# --- zero-weight geometric decay toward zero ---------------------------------


@pytest.mark.parametrize("decay", [0.1, 0.2, 0.5])
def test_bounded_leaky_zero_weights_decays_geometrically(decay):
    encoder = NCAEncoder(
        factors=_factors(dev_steps=20),
        update_rule="bounded_leaky",
        decay=decay,
        state_bound=5.0,
    )
    genome = np.zeros(encoder.genome_length())
    w1, b1, w2, b2 = encoder._unpack(genome)

    state = encoder._initial_state.copy()
    # with all weights zero, f_theta(p) == 0 everywhere, so tanh(0) == 0 and
    # the update reduces to s_{t+1} = (1 - decay) * s_t.
    expected = state.copy()
    for _ in range(20):
        state = encoder._step(state, w1, b1, w2, b2)
        expected = (1.0 - decay) * expected
        np.testing.assert_allclose(state, expected, atol=1e-10)


# --- spectral_normalise satisfies the contraction certificate ---------------


def test_spectral_normalise_satisfies_contraction_bound():
    state_bound = 5.0
    encoder = NCAEncoder(
        factors=_factors(hidden_channels=3),
        update_rule="bounded_leaky",
        state_bound=state_bound,
        spectral_normalise=True,
    )
    rng = np.random.default_rng(7)
    for _ in range(1000):
        genome = rng.normal(scale=rng.choice([0.1, 1.0, 10.0]), size=encoder.genome_length())
        w1, b1, w2, b2 = encoder._unpack(genome)
        sigma1 = np.linalg.svd(w1, compute_uv=False)[0]
        sigma2 = np.linalg.svd(w2, compute_uv=False)[0]
        certificate = state_bound * np.sqrt(5.0) * sigma1 * sigma2
        assert certificate <= 0.9 + 1e-9


def test_spectral_normalise_false_by_default_leaves_weights_unscaled_beyond_init_gain():
    # spectral_normalise=False means _unpack applies only the (fixed,
    # deterministic) init_gain/bias_scale layer-wise scaling and nothing
    # data-dependent -- i.e. the genome-to-weight map is exactly linear,
    # regardless of how large the genome's values are.
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky")
    assert encoder.spectral_normalise is False
    genome = np.random.default_rng(0).normal(scale=100.0, size=encoder.genome_length())
    w1, b1, w2, b2 = encoder._unpack(genome)
    w1_size = encoder._sizes[0]
    fan_in1 = encoder.input_dim
    expected_w1 = genome[:w1_size].reshape(encoder.input_dim, encoder.hidden_width) * np.sqrt(
        encoder.init_gain / fan_in1
    )
    np.testing.assert_allclose(w1, expected_w1)


# --- boundary rules affect only the perception gather ------------------------


def test_replicate_boundary_hand_computed():
    encoder = NCAEncoder(factors=_factors(hidden_channels=0), grid_shape=(3, 3), boundary="replicate")
    channels = 5
    state = np.arange(3 * 3 * channels, dtype=float).reshape(3, 3, channels)
    perception = encoder._perceive(state)

    expected_corner = np.concatenate(
        [
            state[0, 0],
            state[0, 0],  # up: clamps to self row
            state[1, 0],  # down
            state[0, 0],  # left: clamps to self col
            state[0, 1],  # right
        ]
    )
    np.testing.assert_array_equal(perception[0, 0], expected_corner)


def test_toroidal_boundary_hand_computed():
    encoder = NCAEncoder(factors=_factors(hidden_channels=0), grid_shape=(3, 3), boundary="toroidal")
    channels = 5
    state = np.arange(3 * 3 * channels, dtype=float).reshape(3, 3, channels)
    perception = encoder._perceive(state)

    expected_corner = np.concatenate(
        [
            state[0, 0],
            state[2, 0],  # up: wraps to last row
            state[1, 0],  # down
            state[0, 2],  # left: wraps to last col
            state[0, 1],  # right
        ]
    )
    np.testing.assert_array_equal(perception[0, 0], expected_corner)


def test_boundary_zero_padding_hand_computed_default():
    encoder = NCAEncoder(factors=_factors(hidden_channels=0), grid_shape=(3, 3))
    channels = 5
    state = np.arange(3 * 3 * channels, dtype=float).reshape(3, 3, channels)
    perception = encoder._perceive(state)

    expected_corner = np.concatenate(
        [
            state[0, 0],
            np.zeros(channels),
            state[1, 0],
            np.zeros(channels),
            state[0, 1],
        ]
    )
    np.testing.assert_array_equal(perception[0, 0], expected_corner)


def test_invalid_update_rule_raises():
    with pytest.raises(ValueError):
        NCAEncoder(factors=_factors(), update_rule="not_a_rule")


def test_invalid_boundary_raises():
    with pytest.raises(ValueError):
        NCAEncoder(factors=_factors(), boundary="not_a_boundary")


# --- update_prob < 1.0: stochastic masking requires an explicit rng ---------


def test_update_prob_below_one_requires_rng_for_decode():
    encoder = NCAEncoder(factors=_factors(dev_steps=5), update_prob=0.5)
    genome = np.random.default_rng(0).normal(scale=0.1, size=encoder.genome_length())
    with pytest.raises(ValueError):
        encoder.decode(genome)


def test_update_prob_below_one_holds_state_where_masked_out():
    encoder = NCAEncoder(
        factors=_factors(hidden_channels=0, dev_steps=1),
        update_prob=0.0,  # no cell ever updates
        update_rule="unbounded_residual",
    )
    rng = np.random.default_rng(2)
    genome = encoder.random_genome(rng)
    body = encoder.decode(genome, rng=np.random.default_rng(0))
    expected = np.argmax(encoder._initial_state[:, :, :5], axis=-1).astype(int)
    np.testing.assert_array_equal(body, expected)


def test_update_prob_one_ignores_rng_argument_and_stays_deterministic():
    encoder = NCAEncoder(factors=_factors())
    genome = encoder.random_genome(np.random.default_rng(4))
    body_no_rng = encoder.decode(genome)
    body_with_rng = encoder.decode(genome, rng=np.random.default_rng(999))
    np.testing.assert_array_equal(body_no_rng, body_with_rng)


# --- default update_rule is bounded_leaky ------------------------------------


def test_default_update_rule_is_bounded_leaky():
    encoder = NCAEncoder(factors=_factors())
    assert encoder.update_rule == "bounded_leaky"
    assert encoder.update_prob == 1.0
    assert encoder.spectral_normalise is False
