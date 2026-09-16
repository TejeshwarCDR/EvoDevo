"""Tests for the Proposition 2 convergence certificate
(evodevo/encoders/nca/certificate.py).
"""
import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.nca import NCAEncoder
from evodevo.encoders.nca.certificate import (
    contraction_constant,
    is_certified_contractive,
    lipschitz_constant,
)
from evodevo.experiments.inspect_encoder import _settling_step


def _factors(**overrides):
    # dev_steps=200: the certificate only guarantees GEOMETRIC convergence in
    # the limit, at rate L(theta); when L is close to 1 (e.g. decay=0.2 with
    # a near-zero contraction constant gives L=1-decay=0.8), channels that
    # start near a readout tie can keep flipping the argmax for well over
    # T_max=40 (DEPTH_STUDY's horizon, chosen for a different question)
    # before the state has shrunk enough to resolve the tie -- 0.8^200 is
    # far below floating-point noise, so this horizon is generous enough
    # that "never settled" is a genuine signal, not an artifact of a too-short
    # T_max.
    params = dict(dev_steps=200, init_condition="full_nonsymmetric", seed=0, hidden_channels=3)
    params.update(overrides)
    return ExperimentFactors(**params)


def _settles(encoder, genome):
    trajectory = encoder.develop(genome, return_trajectory=True)
    _, never_settled = _settling_step(trajectory)
    return not never_settled


# --- hand-constructed genomes -----------------------------------------------


def test_tiny_weight_genome_is_certified_and_settles():
    # Exact zero weights: K(theta)=0 (certainly certified), and the update
    # reduces to s_{t+1} = (1-decay)*s_t -- a uniform scalar shrink that
    # preserves every cell's channel ordering, so the argmax readout is
    # already at its fixed point at t=0 and never changes.
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", spectral_normalise=False)
    genome = np.zeros(encoder.genome_length())

    assert is_certified_contractive(genome, encoder)
    assert _settles(encoder, genome)


def test_near_zero_weight_genome_is_certified_and_settles_given_enough_steps():
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", spectral_normalise=False)
    rng = np.random.default_rng(0)
    genome = rng.normal(scale=1.0, size=encoder.genome_length()) * 1e-4

    assert is_certified_contractive(genome, encoder)
    assert _settles(encoder, genome)


def test_large_weight_genome_is_not_certified():
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", spectral_normalise=False)
    rng = np.random.default_rng(0)
    genome = rng.normal(scale=1.0, size=encoder.genome_length()) * 1e4

    assert not is_certified_contractive(genome, encoder)


# --- zero false positives ----------------------------------------------------


def test_zero_false_positives_over_2000_genomes():
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", spectral_normalise=False)
    rng = np.random.default_rng(42)
    n = 2000
    false_positives = []
    for i in range(n):
        # mix of scales so certification isn't a rare event
        scale = rng.choice([0.01, 0.1, 1.0, 5.0])
        genome = rng.normal(scale=scale, size=encoder.genome_length())
        if is_certified_contractive(genome, encoder):
            if not _settles(encoder, genome):
                false_positives.append((i, scale))

    assert false_positives == [], (
        f"{len(false_positives)} certified genome(s) failed to settle -- this would indicate a bug "
        f"in the certificate or the update rule, not benign noise: {false_positives[:5]}"
    )


# --- lambda-independence: L < 1 <=> state_bound*K < 1, independent of decay -


@pytest.mark.parametrize("decay", [0.1, 0.2, 0.5, 1.0])
def test_lambda_independence_property(decay):
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", decay=decay, spectral_normalise=False)
    rng = np.random.default_rng(7)
    for _ in range(500):
        scale = rng.choice([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        genome = rng.normal(scale=scale, size=encoder.genome_length())

        certified = is_certified_contractive(genome, encoder)
        L = lipschitz_constant(genome, encoder)
        assert (L < 1.0) == certified, (
            f"L={L} certified={certified} decay={decay} -- L<1 must agree with "
            "state_bound*K<1 regardless of decay"
        )


def test_lipschitz_constant_matches_closed_form():
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", decay=0.3, spectral_normalise=False)
    rng = np.random.default_rng(1)
    genome = rng.normal(size=encoder.genome_length())
    K = contraction_constant(genome, encoder)
    expected_L = (1.0 - encoder.decay) + encoder.decay * encoder.state_bound * K
    assert lipschitz_constant(genome, encoder) == pytest.approx(expected_L)


def test_contraction_constant_is_nonnegative_and_scales_with_weight_magnitude():
    encoder = NCAEncoder(factors=_factors(), update_rule="bounded_leaky", spectral_normalise=False)
    rng = np.random.default_rng(2)
    genome = rng.normal(size=encoder.genome_length())
    K_base = contraction_constant(genome, encoder)
    K_scaled = contraction_constant(genome * 3.0, encoder)
    assert K_base >= 0.0
    # w1, w2 both scale linearly with the genome, so K (product of two
    # spectral norms) scales quadratically.
    assert K_scaled == pytest.approx(K_base * 9.0, rel=1e-6)
