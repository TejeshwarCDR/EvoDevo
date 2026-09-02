"""Paired statistical comparison across seeds between two (or more, pairwise)
encoding arms.

IMPORTANT -- read before citing any number from this module: with roughly
5 seeds per arm (this project's actual seed count), no statistical test
here can support a strong claim. A p-value near 1, a wide confidence
interval, or a confidence interval that includes zero must be reported as
"no detectable difference" -- NOT as evidence that the two arms are
equivalent. Absence of evidence is not evidence of absence at n=5. Every
number this module produces is indicative at best, never conclusive.

Uses only numpy (no scipy dependency): Cliff's delta as a distribution-free
effect size, a Mann-Whitney U statistic with a normal-approximation
two-sided p-value (accurate only for larger samples -- treat the p-value
as a rough indicator at n~5, since the test has very low power to detect
anything short of a large effect at this sample size), and a bootstrap
confidence interval for the difference in means.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import erfc, sqrt
from typing import Sequence

import numpy as np


@dataclass
class PairedComparison:
    encoder_a: str
    encoder_b: str
    metric: str
    n_seeds_a: int
    n_seeds_b: int
    mean_a: float
    mean_b: float
    mean_difference: float  # mean_a - mean_b
    ci_low: float
    ci_high: float
    ci_level: float
    cliffs_delta: float
    mann_whitney_u: float
    mann_whitney_p_approx: float
    overlapping_ci_zero: bool
    verdict: str


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]: (P(a>b) - P(a<b)) over all
    pairs. 0 means fully overlapping distributions; +/-1 means fully
    separated. Makes no distributional assumption and is well-behaved at
    very small sample sizes, unlike a standardized mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    more = float(np.sum(a[:, None] > b[None, :]))
    less = float(np.sum(a[:, None] < b[None, :]))
    return (more - less) / (len(a) * len(b))


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Returns (U statistic, two-sided normal-approximation p-value).

    The normal approximation is only accurate for larger samples; at n~5
    per arm this p-value is a rough indicator, not an exact test -- the
    test has very low power at this sample size to detect anything short
    of a large, consistent effect.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    more = float(np.sum(a[:, None] > b[None, :]))
    ties = float(np.sum(a[:, None] == b[None, :]))
    u = more + 0.5 * ties

    mu = n_a * n_b / 2.0
    sigma = sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    if sigma == 0:
        return float(u), float("nan")
    z = (u - mu) / sigma
    p = erfc(abs(z) / sqrt(2))  # two-sided normal-approximation p-value
    return float(u), float(p)


def bootstrap_ci_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple:
    """Bootstrap confidence interval for mean(a) - mean(b). Each arm's
    seeds are resampled with replacement independently (seeds are
    independent RNG streams per arm, not literally paired across arms)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        resample_a = rng.choice(a, size=len(a), replace=True)
        resample_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = resample_a.mean() - resample_b.mean()
    alpha = 1.0 - ci
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def compare_two_arms(
    seed_values_a: Sequence[float],
    seed_values_b: Sequence[float],
    encoder_a: str,
    encoder_b: str,
    metric: str = "final_best_fitness",
    n_boot: int = 10000,
    ci: float = 0.95,
    bootstrap_seed: int = 0,
) -> PairedComparison:
    a = np.asarray(seed_values_a, dtype=float)
    b = np.asarray(seed_values_b, dtype=float)

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    ci_low, ci_high = bootstrap_ci_mean_difference(a, b, n_boot=n_boot, ci=ci, seed=bootstrap_seed)
    delta = cliffs_delta(a, b)
    u, p = mann_whitney_u(a, b)

    overlapping = ci_low <= 0.0 <= ci_high
    if overlapping:
        verdict = "no detectable difference"
    elif mean_a > mean_b:
        verdict = f"{encoder_a} > {encoder_b} (CI excludes zero)"
    else:
        verdict = f"{encoder_b} > {encoder_a} (CI excludes zero)"

    return PairedComparison(
        encoder_a=encoder_a,
        encoder_b=encoder_b,
        metric=metric,
        n_seeds_a=len(a),
        n_seeds_b=len(b),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_difference=mean_a - mean_b,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        cliffs_delta=delta,
        mann_whitney_u=u,
        mann_whitney_p_approx=p,
        overlapping_ci_zero=overlapping,
        verdict=verdict,
    )


def three_way_comparison(
    values_by_encoder: dict,
    metric: str = "final_best_fitness",
    **kwargs,
) -> list:
    """values_by_encoder: {encoder: [per-seed value, ...]}, e.g. from
    evodevo.analysis.aggregate.per_seed_finals(). Returns one
    PairedComparison per unordered pair of encoders (sorted for
    deterministic ordering)."""
    results = []
    for enc_a, enc_b in itertools.combinations(sorted(values_by_encoder), 2):
        results.append(
            compare_two_arms(
                values_by_encoder[enc_a],
                values_by_encoder[enc_b],
                enc_a,
                enc_b,
                metric=metric,
                **kwargs,
            )
        )
    return results
