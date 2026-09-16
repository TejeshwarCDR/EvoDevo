"""Correctness tests for evodevo/analysis/mutation_calibration.py -- small
N, not the full N=2000 acceptance run (verified manually/by the committed
results/mutation/MUTATION_CALIBRATION.md)."""
import numpy as np
import pytest

from evodevo.analysis import mutation_calibration as M


def test_sample_displacements_returns_n_values_in_valid_range():
    for arm in M.PARAM_GRIDS:
        lo, hi = M.PARAM_GRIDS[arm]
        encoder = M._build_encoder(arm, (lo + hi) / 2, seed=0)
        rng = np.random.default_rng(0)
        distances = M.sample_displacements(encoder, 20, rng)
        assert len(distances) == 20
        assert np.all(distances >= 0)
        assert np.all(distances <= 25)


def test_distribution_stats_reports_more_than_the_mean():
    distances = np.array([0, 1, 1, 2, 3, 5, 8])
    stats = M._distribution_stats(distances)
    assert stats["mean"] == pytest.approx(distances.mean())
    assert stats["median"] == pytest.approx(np.median(distances))
    assert "iqr" in stats and "q25" in stats and "q75" in stats
    assert stats["iqr"] == pytest.approx(stats["q75"] - stats["q25"])


def test_interpolate_param_for_target_is_monotonic_direction():
    sweep = {
        "rows": [
            {"param": 0.1, "mean": 1.0},
            {"param": 0.2, "mean": 2.0},
            {"param": 0.3, "mean": 3.0},
        ]
    }
    param = M._interpolate_param_for_target(sweep, 2.5)
    assert 0.2 <= param <= 0.3


def test_choose_d_star_falls_back_to_common_range_when_band_unreachable():
    # Simulate nca's grid floor already exceeding the recommended 2-3 band.
    sweeps = {
        "direct": {"rows": [{"param": 0.01, "mean": 0.1}, {"param": 0.4, "mean": 6.0}]},
        "nca": {"rows": [{"param": 0.01, "mean": 3.5}, {"param": 0.5, "mean": 18.0}]},
        "rule_based": {"rows": [{"param": 0.02, "mean": 0.5}, {"param": 0.5, "mean": 10.0}]},
    }
    d_star, ranges, common_lo, common_hi = M._choose_d_star(sweeps)
    assert common_lo == 3.5  # nca's floor dominates
    assert d_star >= common_lo
    assert d_star <= common_hi


def test_choose_d_star_uses_recommended_band_when_jointly_achievable():
    sweeps = {
        "direct": {"rows": [{"param": 0.01, "mean": 0.1}, {"param": 0.4, "mean": 6.0}]},
        "nca": {"rows": [{"param": 0.01, "mean": 0.5}, {"param": 0.5, "mean": 18.0}]},
        "rule_based": {"rows": [{"param": 0.02, "mean": 0.2}, {"param": 0.5, "mean": 10.0}]},
    }
    d_star, ranges, common_lo, common_hi = M._choose_d_star(sweeps)
    assert 2.0 <= d_star <= 3.0


@pytest.mark.parametrize("arm", sorted(M.PARAM_GRIDS))
def test_build_encoder_uses_correct_mutation_param_name(arm):
    lo, hi = M.PARAM_GRIDS[arm]
    encoder = M._build_encoder(arm, lo, seed=0)
    if arm == "nca":
        assert encoder.mutation_sigma == pytest.approx(lo)
    else:
        assert encoder.mutation_rate == pytest.approx(lo)


def test_build_encoder_nca_uses_bounded_leaky_at_calibrated_lambda_b_and_h():
    encoder = M._build_encoder("nca", 0.02, seed=0)
    assert encoder.update_rule == M.PRIMARY.update_rule == "bounded_leaky"
    assert encoder.decay == pytest.approx(M.PRIMARY.decay)
    assert encoder.state_bound == pytest.approx(M.PRIMARY.state_bound)
    assert encoder.hidden_channels == M.PRIMARY.hidden_channels


def test_nca_displacement_floor_is_far_below_old_unbounded_floor():
    # The old unbounded architecture's grid floor (mutation_sigma=0.01)
    # produced a mean displacement of ~3.44 cells (docs/decisions.md,
    # 2026-09-15 "Mutation tuning procedure" row). Under bounded_leaky at
    # (lambda*, B*) and the layer-wise-scaled genome, the same floor should
    # be materially flatter -- this is the central empirical claim this
    # re-run exists to check.
    encoder = M._build_encoder("nca", M.PARAM_GRIDS["nca"][0], seed=0)
    rng = np.random.default_rng(0)
    distances = M.sample_displacements(encoder, 300, rng)
    assert distances.mean() < 1.5  # well below the old 3.44-cell floor
