"""Correctness tests for evodevo/analysis/timescale_calibration.py -- small
N smoke tests for the sweep/verification plumbing, plus hand-computed cases
for the Wasserstein-1 / KS distance helpers. The full N=2000 acceptance run
is the committed results/nca/TIMESCALE_CALIBRATION.md, not re-run here."""
import numpy as np
import pytest

from evodevo.analysis import timescale_calibration as T


# --- distance helpers --------------------------------------------------------


def test_wasserstein_1_identical_distributions_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert T.wasserstein_1(a, a) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_1_hand_computed_shift():
    # Two point masses at 0 and one at 5: shifting the whole distribution by
    # a constant c gives W1 = |c| exactly.
    a = np.array([0.0, 0.0, 5.0])
    b = a + 3.0
    assert T.wasserstein_1(a, b) == pytest.approx(3.0, abs=1e-9)


def test_wasserstein_1_hand_computed_equal_size_matches_mean_abs_sorted_diff():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    # equal-size samples: W1 equals the mean absolute difference of sorted pairs
    expected = np.mean(np.abs(np.sort(a) - np.sort(b)))
    assert T.wasserstein_1(a, b) == pytest.approx(expected, abs=1e-9)


def test_wasserstein_1_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(loc=1.0, size=60)
    assert T.wasserstein_1(a, b) == pytest.approx(T.wasserstein_1(b, a), abs=1e-9)


def test_ks_statistic_identical_distributions_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert T.ks_statistic(a, a) == pytest.approx(0.0, abs=1e-9)


def test_ks_statistic_fully_separated_distributions_is_one():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 10.0, 10.0])
    assert T.ks_statistic(a, b) == pytest.approx(1.0, abs=1e-9)


def test_ks_statistic_hand_computed():
    # a: uniform on {1,2,3,4}; b: uniform on {2,3,4,5}. CDFs differ by 1/4 at
    # every point of divergence -- max |F_a - F_b| = 0.25.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    assert T.ks_statistic(a, b) == pytest.approx(0.25, abs=1e-9)


# --- tail fractions -----------------------------------------------------------


def test_tail_fractions_excludes_censored_values():
    censor = 40
    steps = np.array([1.0, 3.0, 3.0, 40.0, 40.0])  # 2 censored (never-settled)
    fractions = T._tail_fractions(steps, [2, 5], censor_value=censor)
    assert fractions[2] == pytest.approx(1 / 5)  # only the single "1"
    assert fractions[5] == pytest.approx(3 / 5)  # the "1" and both "3"s


# --- small-N smoke test of the sweep/verification pipeline -------------------


def test_run_sweep_smoke_small_n(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "LAMBDAS", [0.2, 1.0])
    monkeypatch.setattr(T, "STATE_BOUNDS", [2, 5])

    sweep = T.run_sweep(n=20, seed=0)

    assert len(sweep["grid"]) == 4  # 2 lambdas x 2 state_bounds
    for row in sweep["grid"]:
        assert row["wasserstein_1"] >= 0.0
        assert 0.0 <= row["ks_statistic"] <= 1.0
        assert 0.0 <= row["never_settled_fraction"] <= 1.0

    best = sweep["best"]
    assert best["wasserstein_1"] == min(row["wasserstein_1"] for row in sweep["grid"])
    assert (best["decay"], best["state_bound"]) in [(d, b) for d in [0.2, 1.0] for b in [2, 5]]

    verification = T.run_verification(best, n=20, seed=T.VERIFICATION_SEED, rb_steps=sweep["rb_steps"])
    assert verification["seed"] == T.VERIFICATION_SEED
    assert len(verification["steps"]) == 20
    assert verification["wasserstein_1"] >= 0.0


def test_never_modifies_rule_based_target(tmp_path, monkeypatch):
    """The rule_based settling distribution is measured once per run_sweep
    call and must be identical (not resampled) across every grid cell's
    comparison -- i.e. run_sweep only ever calls settling_profile for
    rule_based a single time."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "LAMBDAS", [0.2])
    monkeypatch.setattr(T, "STATE_BOUNDS", [5])

    call_count = {"rule_based": 0}
    original = T.S.settling_profile

    def counting_settling_profile(encoder, *args, **kwargs):
        if encoder.name == "rule_based":
            call_count["rule_based"] += 1
        return original(encoder, *args, **kwargs)

    monkeypatch.setattr(T.S, "settling_profile", counting_settling_profile)

    T.run_sweep(n=10, seed=0)
    assert call_count["rule_based"] == 1
