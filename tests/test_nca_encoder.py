import numpy as np
import pytest

from evodevo.config import ExperimentConfig
from evodevo.encoders import get_encoder
from evodevo.encoders.nca import N_MATERIALS, NCAEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea


def test_decode_shape_dtype_and_values():
    encoder = NCAEncoder()
    rng = np.random.default_rng(0)
    genome = encoder.random_genome(rng)
    body = encoder.decode(genome)
    assert body.shape == (5, 5)
    assert np.issubdtype(body.dtype, np.integer)
    assert set(np.unique(body).tolist()) <= set(range(N_MATERIALS))


def test_decode_deterministic():
    encoder = NCAEncoder()
    rng = np.random.default_rng(1)
    genome = encoder.random_genome(rng)
    body1 = encoder.decode(genome)
    body2 = encoder.decode(genome)
    assert np.array_equal(body1, body2)


def test_synchronicity_matches_synchronous_not_sequential_update():
    # hidden_channels=0 keeps C small and the network trivial to hand-design.
    encoder = NCAEncoder(hidden_channels=0, hidden_width=1, dev_steps=1)
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
    encoder = NCAEncoder(grid_shape=(3, 3), hidden_channels=0)
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
    encoder = NCAEncoder(hidden_channels=hidden_channels, hidden_width=hidden_width)
    channels = N_MATERIALS + hidden_channels
    input_dim = 5 * channels
    expected = input_dim * hidden_width + hidden_width + hidden_width * channels + channels
    assert encoder.genome_length() == expected


def test_residual_property_zero_weights_equals_fixed_initial_state():
    encoder = NCAEncoder(dev_steps=10)
    genome = np.zeros(encoder.genome_length())
    body = encoder.decode(genome)
    expected = np.argmax(encoder._initial_state[:, :, :N_MATERIALS], axis=-1).astype(int)
    np.testing.assert_array_equal(body, expected)


def test_develop_returns_dev_steps_plus_one_trajectory_entries():
    encoder = NCAEncoder(dev_steps=7)
    rng = np.random.default_rng(3)
    genome = encoder.random_genome(rng)
    trajectory = encoder.develop(genome, return_trajectory=True)
    assert len(trajectory) == 8
    for body in trajectory:
        assert body.shape == (5, 5)
        assert np.issubdtype(body.dtype, np.integer)


def test_mutate_does_not_resample_until_valid_and_is_reproducible():
    encoder = NCAEncoder(mutation_sigma=0.2, mutation_fraction=0.5)
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    genome = np.zeros(encoder.genome_length())
    mutated1 = encoder.mutate(genome, rng1)
    mutated2 = encoder.mutate(genome, rng2)
    np.testing.assert_array_equal(mutated1, mutated2)
    assert not np.array_equal(mutated1, genome)


def test_describe_json_serialisable():
    import json

    encoder = NCAEncoder()
    json.dumps(encoder.describe())


def test_get_encoder_nca_registry():
    encoder = get_encoder("nca")
    assert isinstance(encoder, NCAEncoder)
    assert encoder.name == "nca"


def test_unsupported_neighbourhood_raises():
    with pytest.raises(ValueError):
        NCAEncoder(neighbourhood="moore")


def test_ea_runs_unmodified_with_nca_encoder_in_surrogate_mode():
    encoder = NCAEncoder()
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
