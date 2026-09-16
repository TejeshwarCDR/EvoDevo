"""Smoke/correctness tests for evodevo/analysis/dynamics_classification.py --
small N, not the full N=2000 acceptance run (that is the committed
results/nca/DYNAMICS_CLASSIFICATION.md)."""
import numpy as np

from evodevo.analysis import dynamics_classification as D


def test_presaturation_slope_excludes_leading_zero_and_post_saturation_plateau():
    # distances: zero for 2 steps (perturbation not yet propagated), then a
    # clean e^0.1 growth for a while, then a flat plateau at state_bound scale.
    state_bound = 5.0
    real_slope = 0.5  # steep enough to actually cross 0.1*state_bound=0.5 within the window
    d = [0.0, 0.0]
    val = 1e-6
    for _ in range(30):
        val *= np.exp(real_slope)
        d.append(val)
    d += [d[-1]] * 20  # plateau (saturated)
    slope, t_star = D._presaturation_slope(np.array(d), state_bound, 0.1, 5)
    assert slope is not None
    assert abs(slope - real_slope) < 0.01
    # t_star should land before the plateau begins
    assert t_star < len(d) - 15


def test_presaturation_slope_unmeasurable_when_saturates_almost_immediately():
    state_bound = 5.0
    d = [0.0, 0.0, 0.0, 10.0] + [10.0] * 20  # jumps straight past threshold
    slope, t_star = D._presaturation_slope(np.array(d), state_bound, 0.1, 5)
    assert slope is None


def test_presaturation_slope_never_saturates_uses_full_window():
    state_bound = 5.0
    d = [0.0] + [1e-6] * 50  # flat, tiny, never approaches 0.1*state_bound
    slope, t_star = D._presaturation_slope(np.array(d), state_bound, 0.1, 5)
    assert t_star == len(d) - 1


def test_run_dynamics_classification_small_n_smoke():
    result = D.run_dynamics_classification(n=20, seed=0)
    assert result["n"] == 20
    assert result["n_unmeasurable"] >= 0
    assert len(result["new_slopes"]) == result["n_still_no_period_wide"]
    assert len(result["k_thetas"]) == result["n_still_no_period_wide"]
    # pearson_r_slope_vs_k is either a float in [-1, 1] or None (degenerate)
    r = result["pearson_r_slope_vs_k"]
    if r is not None:
        assert -1.0 <= r <= 1.0


def test_render_report_contains_old_and_new_sections():
    result = D.run_dynamics_classification(n=20, seed=0)
    report = D.render_report(result)
    assert "Methodology correction" in report
    assert "K(theta)-vs-slope correlation" in report
    assert "Old vs new classification" in report
