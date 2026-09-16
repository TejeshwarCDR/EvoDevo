"""Per-genome convergence certificate for the bounded_leaky NCA update rule
(Proposition 2).

The bounded_leaky update is
    s_{t+1} = (1 - decay) * s_t + decay * state_bound * tanh(f_theta(p_t))
where f_theta(p) = W2 @ tanh(W1 @ p + b1) + b2 is the perception network and
p_t is the (Lipschitz-1, since it is a coordinate gather/concatenation of at
most 5 blocks of the state) perception of s_t.

tanh is 1-Lipschitz, so f_theta is Lipschitz in its input with constant
sigma_max(W1) * sigma_max(W2) (the spectral norms of the two linear layers
composed through a 1-Lipschitz nonlinearity), and the perception gather
itself has Lipschitz constant at most sqrt(5) (5 concatenated coordinate
projections). Composing:

    K(theta) = sqrt(5) * sigma_max(W1) * sigma_max(W2)

is a (generally loose, sufficient-not-necessary) Lipschitz bound on
s -> state_bound * tanh(f_theta(perceive(s))) w.r.t. s. Substituting into the
bounded_leaky update's contraction w.r.t. two states s, s':

    |s_{t+1} - s'_{t+1}| <= (1 - decay) |s_t - s'_t| + decay * state_bound * K(theta) * |s_t - s'_t|
                          = L(theta) |s_t - s'_t|

where L(theta) = (1 - decay) + decay * state_bound * K(theta) is the
per-step Lipschitz constant of the whole update map. The map is a
contraction (and development is certified to converge geometrically to a
unique fixed point) whenever L(theta) < 1, which -- since decay in (0, 1]
enters linearly -- is equivalent to state_bound * K(theta) < 1 regardless of
decay's value (decay only sets the convergence RATE, not whether the map
contracts at all; see is_certified_contractive vs lipschitz_constant, and
the lambda-independence property verified in
tests/encoders/test_nca_certificate.py).

The certificate is sufficient, not necessary: many genomes with
state_bound * K(theta) >= 1 still settle in practice (the Lipschitz bound
above is loose -- it does not use the actual trajectory's state magnitudes,
only worst-case spectral norms). See results/nca/CONVERGENCE.md for the
measured gap between "certified" and "empirically settles".
"""
from __future__ import annotations

import numpy as np


def _spectral_norms(genome: np.ndarray, encoder) -> tuple:
    w1, b1, w2, b2 = encoder._unpack(genome)
    sigma1 = np.linalg.svd(w1, compute_uv=False)[0] if w1.size else 0.0
    sigma2 = np.linalg.svd(w2, compute_uv=False)[0] if w2.size else 0.0
    return float(sigma1), float(sigma2)


def contraction_constant(genome: np.ndarray, encoder) -> float:
    """K(theta) = sqrt(5) * sigma_max(W1) * sigma_max(W2), from the unpacked
    (post-variance-scaling, post-spectral-normalisation-if-enabled) weights."""
    sigma1, sigma2 = _spectral_norms(genome, encoder)
    return float(np.sqrt(5.0) * sigma1 * sigma2)


def is_certified_contractive(genome: np.ndarray, encoder) -> bool:
    """True iff state_bound * K(theta) < 1 -- the update map is certified to
    be a contraction (Banach fixed point: development converges geometrically
    to a unique fixed point), independent of decay."""
    K = contraction_constant(genome, encoder)
    return encoder.state_bound * K < 1.0


def lipschitz_constant(genome: np.ndarray, encoder) -> float:
    """L(theta) = (1 - decay) + decay * state_bound * K(theta): the per-step
    Lipschitz constant of the full bounded_leaky update map, i.e. the
    predicted geometric convergence rate |s_t - s_T| ~ L^t. L < 1 iff
    state_bound * K(theta) < 1, for any decay in (0, 1] -- decay only
    interpolates between the two endpoints (decay -> 0: L -> 1, no progress;
    decay -> 1: L -> state_bound * K(theta), the raw contraction constant)."""
    K = contraction_constant(genome, encoder)
    decay = encoder.decay
    return float((1.0 - decay) + decay * encoder.state_bound * K)
