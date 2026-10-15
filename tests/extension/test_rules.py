"""Pre-registered extension rules (evodevo/extension/rules.py) on synthetic inputs."""
import math

import pytest

from evodevo.extension import rules


def curve(fn, interval=100_000, max_t=2_000_000):
    """Checkpoints at the first 2048-step rollout boundary past each 100k."""
    points = []
    for i in range(max_t // interval):
        t = math.ceil((i + 1) * interval / 2048) * 2048
        points.append((t, float(fn((i + 1) * interval))))
    return points


def plateau(level, knee):
    return lambda t: level * min(1.0, t / knee)


def test_constants_are_the_registered_values():
    assert rules.CURVE_MAX_TIMESTEPS == 2_000_000 and rules.CURVE_EVAL_INTERVAL == 100_000
    assert rules.QUALIFYING_FRACTION_OF_BEST == 0.10 and rules.MIN_QUALIFYING_BODIES == 2
    assert (rules.BUDGET_FLOOR, rules.BUDGET_CAP, rules.NON_LEARNING_BUDGET) == (500_000, 1_500_000, 1_000_000)
    assert rules.FALLBACK_CANDIDATES == (1_000_000, 1_200_000, 1_500_000, 2_000_000)
    assert rules.FALLBACK_FRACTION_OF_FINAL == 0.85
    assert rules.NOISE_RATIO_THRESHOLD == 0.33 and rules.MAX_OF_2_REQUIRED_IMPROVEMENT == 0.5
    assert rules.GUARD_THRESHOLD_H == 360.0 and rules.GUARD_TIMESTEP_CEILING == 1_200_000
    assert rules.ESTIMATE_MIN_RELATIVE_ERROR == 0.10


def test_registered_knee_branch_rounds_to_the_checkpoint():
    bodies = {"a": curve(plateau(5.0, 800_000)), "b": curve(plateau(3.0, 800_000)),
              "c": curve(lambda t: 0.0)}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "registered_knee"
    assert decision.qualifying_bodies == ["a", "b"]
    assert decision.ppo_timesteps == 800_000
    assert decision.ppo_timesteps % 100_000 == 0
    assert not decision.disclosed_floor and decision.clipped is None


def test_early_knee_is_raised_to_the_floor():
    bodies = {"a": curve(plateau(5.0, 200_000)), "b": curve(plateau(4.0, 200_000))}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "registered_knee"
    assert decision.ppo_timesteps == 500_000 and decision.clipped == "floor"


def test_late_knee_is_capped_and_disclosed():
    bodies = {"a": curve(plateau(5.0, 1_600_000)), "b": curve(plateau(4.0, 1_600_000))}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "registered_knee"
    assert decision.ppo_timesteps == 1_500_000 and decision.clipped == "cap" and decision.disclosed_floor


def test_non_learning_task_gets_the_walker_budget():
    bodies = {"a": curve(lambda t: -0.5), "b": curve(lambda t: -0.2), "c": curve(lambda t: -1.0)}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "non_learning" and decision.ppo_timesteps == 1_000_000 and decision.disclosed_floor


def test_single_learner_is_non_learning():
    bodies = {"a": curve(lambda t: t / 1e5), "b": curve(lambda t: 0.01), "c": curve(lambda t: 0.0)}
    assert rules.decide_budget("X", bodies).branch == "non_learning"


def test_incomplete_curves_are_excluded():
    short = curve(plateau(5.0, 800_000))[:10]
    bodies = {"a": curve(plateau(5.0, 800_000)), "b": short, "c": curve(plateau(4.0, 800_000))}
    decision = rules.decide_budget("X", bodies)
    assert decision.details["excluded_incomplete_bodies"] == ["b"]
    assert decision.qualifying_bodies == ["a", "c"]


def test_fallback_picks_the_smallest_candidate_meeting_both_conditions():
    # Still rising 2%/100k at the end (no knee), 90% of final at 1.2M, order stable.
    rise = lambda scale: (lambda t: scale * (0.6 + 0.4 * t / 2e6) * (1 + 0.05 * math.sin(t / 7e4)))
    bodies = {"a": curve(lambda t: 4 * (t / 2e6) ** 0.35), "b": curve(lambda t: 2 * (t / 2e6) ** 0.35)}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "fallback"
    table = {row["T"]: row for row in decision.details["fallback_table"]}
    assert table[1_000_000]["fractions_of_2M"]["a"] < 0.85
    assert decision.ppo_timesteps == min(t for t, row in table.items() if row["fraction_ok"] and row["order_ok"])


def test_fallback_without_a_passing_candidate_is_a_capped_floor():
    bodies = {"a": curve(lambda t: (t / 2e6) ** 3 * 4), "b": curve(lambda t: (t / 2e6) ** 3 * 2)}
    decision = rules.decide_budget("X", bodies)
    assert decision.branch == "fallback"
    assert decision.details["fallback_choice"] == 2_000_000
    assert decision.ppo_timesteps == 1_500_000 and decision.clipped == "cap" and decision.disclosed_floor


def test_noise_rule_reproduces_both_original_d16_decisions():
    # Walker-like: within/between ~0.16 -> K=1.
    walker = {"a": [1.0, 1.05, 0.98], "b": [2.0, 2.1, 1.95], "c": [3.0, 2.9, 3.1]}
    assert rules.decide_seeds("W", walker).ppo_seeds == 1
    # UpStepper-like: occasional collapse dominates -> max-of-2 halves the ratio -> K=2.
    up = {"a": [1.6, 0.14, 1.62], "b": [1.0, 1.02, 0.2], "c": [1.3, 1.31, 1.29]}
    decision = rules.decide_seeds("U", up)
    assert decision.details["ratio_single"] > 0.33
    assert decision.details["ratio_max_of_2"] <= 0.5 * decision.details["ratio_single"]
    assert decision.ppo_seeds == 2


def test_noise_that_max_of_2_cannot_fix_stays_k1():
    # Symmetric noise: max-of-2 does not halve the ratio.
    sym = {"a": [0.0, 2.0, 1.0], "b": [0.5, 2.5, 1.5], "c": [1.0, 3.0, 2.0]}
    decision = rules.decide_seeds("S", sym)
    assert decision.details["ratio_single"] > 0.33 and decision.ppo_seeds == 1


def test_indistinguishable_bodies_or_too_few_bodies_give_k1():
    assert rules.decide_seeds("Z", {"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]}).branch == "insufficient_spread"
    assert rules.decide_seeds("Z", {"a": [1.0, 2.0], "b": [3.0]}).branch == "insufficient_bodies"


def test_projection_band_uses_the_larger_of_10pct_and_backtest_error():
    band = rules.project_phase_band(100.0, 0.75, 1.4, 1, 0.04)
    assert band["mean"] == pytest.approx(105.0)
    assert band["p95"] == pytest.approx(115.5) and band["p5"] == pytest.approx(94.5)
    wide = rules.project_phase_band(100.0, 0.75, 1.4, 2, -0.2)
    assert wide["mean"] == pytest.approx(210.0) and wide["p95"] == pytest.approx(252.0)
    with pytest.raises(ValueError):
        rules.project_phase_band(100.0, 0.0, 1.0, 1, 0.0)


def _project(per_mtimestep_h):
    return lambda task, t, k: {q: per_mtimestep_h[task] * t / 1e6 * k for q in ("p5", "mean", "p95")}


def test_guard_within_threshold_changes_nothing():
    decided = {"A": {"ppo_timesteps": 1_000_000, "ppo_seeds": 2}, "B": {"ppo_timesteps": 1_500_000, "ppo_seeds": 1}}
    final, steps, total, outcome = rules.apply_guard(["A", "B"], decided, _project({"A": 50, "B": 50}))
    assert final == decided and outcome == "within_guard" and [s["step"] for s in steps] == ["none"]


def test_guard_cuts_k_latest_task_first_then_timesteps_then_proceeds():
    decided = {"A": {"ppo_timesteps": 1_500_000, "ppo_seeds": 2},
               "B": {"ppo_timesteps": 1_500_000, "ppo_seeds": 2},
               "C": {"ppo_timesteps": 1_000_000, "ppo_seeds": 1}}
    final, steps, total, outcome = rules.apply_guard(["A", "B", "C"], decided, _project({"A": 130, "B": 130, "C": 130}))
    assert [s["step"] for s in steps] == ["none", "a_K1_B", "a_K1_A", "b_T1.2M_B", "b_T1.2M_A"]
    assert final == {"A": {"ppo_timesteps": 1_200_000, "ppo_seeds": 1},
                     "B": {"ppo_timesteps": 1_200_000, "ppo_seeds": 1},
                     "C": {"ppo_timesteps": 1_000_000, "ppo_seeds": 1}}
    assert outcome == "over_guard_proceed_with_later_completion"
    assert total["p95"] == pytest.approx(130 * 1.2 * 2 + 130)


def test_guard_stops_as_soon_as_it_is_within():
    decided = {"A": {"ppo_timesteps": 1_000_000, "ppo_seeds": 2}, "B": {"ppo_timesteps": 1_000_000, "ppo_seeds": 2}}
    final, steps, total, outcome = rules.apply_guard(["A", "B"], decided, _project({"A": 100, "B": 100}))
    assert [s["step"] for s in steps] == ["none", "a_K1_B"]
    assert final["A"]["ppo_seeds"] == 2 and final["B"]["ppo_seeds"] == 1 and outcome == "within_guard"
