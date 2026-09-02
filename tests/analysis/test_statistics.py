import numpy as np
import pytest

from evodevo.analysis.statistics import (
    PairedComparison,
    bootstrap_ci_mean_difference,
    cliffs_delta,
    compare_two_arms,
    mann_whitney_u,
    three_way_comparison,
)


def test_cliffs_delta_fully_separated_is_plus_or_minus_one():
    a = [10, 11, 12]
    b = [1, 2, 3]
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)


def test_cliffs_delta_identical_distributions_is_near_zero():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert cliffs_delta(a, b) == pytest.approx(0.0)


def test_mann_whitney_u_fully_separated_groups():
    a = [10, 11, 12, 13, 14]
    b = [1, 2, 3, 4, 5]
    u, p = mann_whitney_u(a, b)
    assert u == pytest.approx(25.0)  # every pair a_i > b_j: n_a * n_b
    assert p < 0.05


def test_bootstrap_ci_mean_difference_is_reproducible_with_same_seed():
    a = [10, 12, 11, 9, 13]
    b = [1, 2, 3, 2, 1]
    ci1 = bootstrap_ci_mean_difference(a, b, n_boot=500, seed=42)
    ci2 = bootstrap_ci_mean_difference(a, b, n_boot=500, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_excludes_zero_for_a_clearly_separated_pair():
    a = [10, 12, 11, 9, 13]
    b = [1, 2, 3, 2, 1]
    lo, hi = bootstrap_ci_mean_difference(a, b, n_boot=2000, seed=0)
    assert lo > 0.0  # a is clearly larger than b; CI should not include 0


def test_bootstrap_ci_includes_zero_for_overlapping_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=1.0, size=5)
    b = rng.normal(loc=0.0, scale=1.0, size=5)
    lo, hi = bootstrap_ci_mean_difference(a, b, n_boot=5000, seed=1)
    assert lo <= 0.0 <= hi


def test_compare_two_arms_returns_paired_comparison_with_expected_verdict():
    result = compare_two_arms(
        seed_values_a=[10, 12, 11, 9, 13],
        seed_values_b=[1, 2, 3, 2, 1],
        encoder_a="direct",
        encoder_b="nca",
        n_boot=2000,
        bootstrap_seed=0,
    )
    assert isinstance(result, PairedComparison)
    assert result.encoder_a == "direct"
    assert result.encoder_b == "nca"
    assert result.n_seeds_a == 5
    assert result.mean_difference > 0
    assert not result.overlapping_ci_zero
    assert "direct > nca" in result.verdict


def test_compare_two_arms_overlapping_reports_no_detectable_difference():
    result = compare_two_arms(
        seed_values_a=[5, 6, 7, 4, 5],
        seed_values_b=[5, 4, 6, 7, 5],
        encoder_a="direct",
        encoder_b="nca",
        n_boot=2000,
        bootstrap_seed=0,
    )
    assert result.overlapping_ci_zero
    assert result.verdict == "no detectable difference"


def test_three_way_comparison_returns_one_result_per_unordered_pair():
    values = {
        "direct": [1, 2, 3],
        "nca": [10, 11, 12],
        "rule_based": [5, 6, 7],
    }
    results = three_way_comparison(values, n_boot=500, bootstrap_seed=0)
    pairs = {(r.encoder_a, r.encoder_b) for r in results}
    assert pairs == {("direct", "nca"), ("direct", "rule_based"), ("nca", "rule_based")}
    assert len(results) == 3
