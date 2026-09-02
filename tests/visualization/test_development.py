import numpy as np

from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.visualization.development import plot_development_trajectory


def test_direct_encoder_returns_none_gracefully():
    encoder = DirectEncoder()
    genome = encoder.random_genome(np.random.default_rng(0))
    result = plot_development_trajectory(encoder, genome)
    assert result is None


def test_nca_encoder_produces_a_trajectory_figure():
    encoder = NCAEncoder(dev_steps=3)
    genome = encoder.random_genome(np.random.default_rng(0))
    axes = plot_development_trajectory(encoder, genome, title="nca test")
    assert axes is not None
    assert len(axes) == 4  # dev_steps + 1


def test_rule_based_encoder_produces_a_trajectory_figure():
    encoder = RuleBasedEncoder(developmental_iterations=2)
    genome = encoder.random_genome(np.random.default_rng(0))
    axes = plot_development_trajectory(encoder, genome, title="rule_based test")
    assert axes is not None
    assert len(axes) == 3  # developmental_iterations + 1


def test_base_encoder_default_develop_trajectory_is_none():
    # Direct's ABC-default develop_trajectory() (never overridden) must
    # return None rather than raising -- this is what plot_development_
    # trajectory relies on to handle a non-developmental encoder gracefully.
    encoder = DirectEncoder()
    genome = encoder.random_genome(np.random.default_rng(1))
    assert encoder.develop_trajectory(genome) is None
