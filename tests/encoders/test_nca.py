"""NCAEncoder-SPECIFIC behaviour only.

Generic Encoder-contract properties (decode shape/dtype/values, decode
determinism, random_genome reproducibility/seed-sensitivity, mutate
reproducibility, serialize/deserialize round-tripping, describe()
completeness, the physics/RL-free module rule, and the
does-not-resample-until-valid property) are covered once for every
registered encoder by tests/encoders/test_contract.py -- they are
deliberately not repeated here.
"""
import numpy as np
import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.encoders.nca import N_MATERIALS, NCAEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea


def _factors(**overrides):
    """Reproduces the NCA encoder's pre-factors historical default
    (dev_steps=10, the fixed non-symmetric initial state) unless overridden."""
    params = dict(dev_steps=10, init_condition="full_nonsymmetric", seed=0, hidden_channels=3)
    params.update(overrides)
    return ExperimentFactors(**params)


def test_synchronicity_matches_synchronous_not_sequential_update():
    # hidden_channels=0 keeps C small and the network trivial to hand-design.
    encoder = NCAEncoder(factors=_factors(hidden_channels=0, dev_steps=1), hidden_width=1)
    channels = encoder.channels  # 5

    rng = np.random.default_rng(42)
    state = rng.normal(scale=0.1, size=(5, 5, channels))

    # perception blocks are [own, up, down, left, right]; we wire the network
    # to compute delta_channel0 = left_neighbour.channel0 - own.channel0, so
    # new.channel0 = old.channel0 + delta = left_neighbour(t).channel0 exactly
    # (a rightward shift-by-one of channel 0, with zero padding at the left edge).
    own_ch0_idx = 0
    left_block_index = 3  # own=0, up=1, down=2, left=3, right=4
    left_ch0_idx = left_block_index * channels + 0

    w1 = np.zeros((encoder.input_dim, 1))
    w1[own_ch0_idx, 0] = -0.001
    w1[left_ch0_idx, 0] = 0.001
    b1 = np.zeros(1)
    w2 = np.zeros((1, channels))
    w2[0, 0] = 1000.0  # cancels the 0.001 scaling; tanh is ~linear at this input scale
    b2 = np.zeros(channels)

    genome = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
    assert genome.shape[0] == encoder.genome_length()

    new_state = encoder._step(state, w1, b1, w2, b2)

    # Correct synchronous result: every cell reads its LEFT neighbour's OLD
    # value; the whole row shifts right by one in a single step.
    expected_sync = state[:, :, 0].copy()
    expected_sync[:, 1:] = state[:, :-1, 0]
    expected_sync[:, 0] = 0.0
    np.testing.assert_allclose(new_state[:, :, 0], expected_sync, atol=1e-5)

    # A sequential (asynchronous), in-place, left-to-right sweep would instead
    # read already-updated neighbour values, collapsing the whole row to the
    # left-edge padding value (0) within a single "step" -- a different and
    # wrong answer that the synchronous implementation must not reproduce.
    async_state = state[:, :, 0].copy()
    for r in range(5):
        for c in range(5):
            left_val = async_state[r, c - 1] if c > 0 else 0.0
            async_state[r, c] = left_val
    assert not np.allclose(async_state, new_state[:, :, 0], atol=1e-5)


def test_boundary_zero_padding_hand_computed():
    encoder = NCAEncoder(factors=_factors(hidden_channels=0), grid_shape=(3, 3))
    channels = 5
    state = np.arange(3 * 3 * channels, dtype=float).reshape(3, 3, channels)
    perception = encoder._perceive(state)

    # corner (0, 0): up and left neighbours are out of bounds.
    expected_corner = np.concatenate(
        [
            state[0, 0],
            np.zeros(channels),  # up: out of bounds
            state[1, 0],  # down
            np.zeros(channels),  # left: out of bounds
            state[0, 1],  # right
        ]
    )
    np.testing.assert_array_equal(perception[0, 0], expected_corner)

    # edge (0, 1): top row, middle column -> only up is out of bounds.
    expected_edge = np.concatenate(
        [
            state[0, 1],
            np.zeros(channels),  # up: out of bounds
            state[1, 1],  # down
            state[0, 0],  # left
            state[0, 2],  # right
        ]
    )
    np.testing.assert_array_equal(perception[0, 1], expected_edge)


@pytest.mark.parametrize(
    "hidden_channels,hidden_width",
    [(0, 4), (3, 16), (5, 8), (1, 1)],
)
def test_genome_length_matches_analytic_parameter_count(hidden_channels, hidden_width):
    encoder = NCAEncoder(factors=_factors(hidden_channels=hidden_channels), hidden_width=hidden_width)
    channels = N_MATERIALS + hidden_channels
    input_dim = 5 * channels
    expected = input_dim * hidden_width + hidden_width + hidden_width * channels + channels
    assert encoder.genome_length() == expected


def test_residual_property_zero_weights_equals_fixed_initial_state():
    encoder = NCAEncoder(factors=_factors(dev_steps=10))
    genome = np.zeros(encoder.genome_length())
    body = encoder.decode(genome)
    expected = np.argmax(encoder._initial_state[:, :, :N_MATERIALS], axis=-1).astype(int)
    np.testing.assert_array_equal(body, expected)


def test_develop_returns_dev_steps_plus_one_trajectory_entries():
    encoder = NCAEncoder(factors=_factors(dev_steps=7))
    rng = np.random.default_rng(3)
    genome = encoder.random_genome(rng)
    trajectory = encoder.develop(genome, return_trajectory=True)
    assert len(trajectory) == 8
    for body in trajectory:
        assert body.shape == (5, 5)
        assert np.issubdtype(body.dtype, np.integer)


def test_get_encoder_nca_registry():
    encoder = get_encoder("nca", factors=_factors())
    assert isinstance(encoder, NCAEncoder)
    assert encoder.name == "nca"


def test_unsupported_neighbourhood_raises():
    with pytest.raises(ValueError):
        NCAEncoder(factors=_factors(neighbourhood="moore"))


def test_ea_runs_unmodified_with_nca_encoder_in_surrogate_mode():
    encoder = NCAEncoder(factors=_factors())
    config = ExperimentConfig(
        seed=0,
        encoder="nca",
        task="dummy-task",
        population_size=4,
        offspring_size=4,
        eval_budget=12,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 12
    assert len(result.records) == 12


# --- Regression: pin known (seed -> body) pairs so a future refactor of
# this file cannot silently change NCAEncoder's decode/mutate behaviour.
# These values were captured from the encoder as it existed at the time
# Prompt 3 (encoder plugin architecture) was applied; they must remain
# bit-identical, since that prompt is explicitly moves/refactoring-only.


def test_regression_decode_pinned_for_seed_5():
    # Pinned values updated for the ExperimentFactors refactor: the fixed
    # non-symmetric initial state (now evodevo.core.initial_conditions.
    # FullNonsymmetric) draws its material logits independently of hidden
    # channels so the material portion is channel-count-invariant (required
    # for as_state_tensor/as_discrete_grid round-trip consistency across
    # every caller) -- this changes the interleaving of the underlying RNG
    # draws versus the old single (rows, cols, channels) draw, hence new
    # pinned bodies, not a change to the update rule itself.
    encoder = NCAEncoder(factors=_factors())
    rng = np.random.default_rng(5)
    genome = encoder.random_genome(rng)
    body = encoder.decode(genome)
    expected = np.array(
        [
            [1, 3, 4, 4, 4],
            [1, 4, 1, 1, 1],
            [1, 4, 1, 1, 3],
            [4, 4, 2, 1, 1],
            [1, 1, 1, 4, 3],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(body, expected)


def test_regression_mutate_pinned_for_seed_9():
    # See test_regression_decode_pinned_for_seed_5: pinned values updated for
    # the channel-count-invariant FullNonsymmetric initial state.
    encoder = NCAEncoder(factors=_factors(), mutation_sigma=0.2, mutation_fraction=0.5)
    genome = np.zeros(encoder.genome_length())
    rng = np.random.default_rng(9)
    mutated = encoder.mutate(genome, rng)
    body = encoder.decode(mutated)
    expected_body = np.array(
        [
            [4, 4, 4, 4, 4],
            [3, 4, 4, 4, 4],
            [0, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [2, 2, 2, 2, 2],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(body, expected_body)
    assert np.count_nonzero(mutated) == 396
    assert mutated.sum() == pytest.approx(0.05458646500500386)
