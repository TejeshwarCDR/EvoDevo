"""Unit tests for evodevo/analysis/stats_protocol.py -- the generic
statistical-protocol implementation locked by docs/analysis-protocol.md."""
import numpy as np
import pytest

from evodevo.analysis.stats_protocol import (
    ABLATION_WITHIN_ARM,
    ARM_VS_OWN_RANDOM_SEARCH,
    CROSS_ARM,
    Cell,
    enumerate_planned_comparisons,
    holm_bonferroni,
    median_iqr_curve,
    run_protocol,
    vargha_delaney_a12,
)


def test_vargha_delaney_a12_fully_separated_is_one():
    assert vargha_delaney_a12([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert vargha_delaney_a12([1, 2, 3], [10, 11, 12]) == pytest.approx(0.0)


def test_vargha_delaney_a12_identical_distributions_is_half():
    assert vargha_delaney_a12([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_holm_bonferroni_matches_hand_worked_example():
    # Classic textbook example: p = [0.01, 0.04, 0.03, 0.005], m=4, alpha=0.05.
    # Sorted: 0.005(x4=0.02), 0.01(x3=0.03), 0.03(x2=0.06), 0.04(x1=0.04) ->
    # cummax: 0.02, 0.03, 0.06, 0.06 (last one clipped up by step-down cummax).
    p = [0.01, 0.04, 0.03, 0.005]
    result = holm_bonferroni(p, alpha=0.05)
    assert result.m == 4
    # Sorted ascending: idx3(0.005), idx0(0.01), idx2(0.03), idx1(0.04)
    assert result.p_holm[3] == pytest.approx(0.02)
    assert result.p_holm[0] == pytest.approx(0.03)
    assert result.p_holm[2] == pytest.approx(0.06)
    assert result.p_holm[1] == pytest.approx(0.06)
    # reject iff p_holm <= alpha=0.05: idx0=0.03 (reject), idx1=0.06 (no),
    # idx2=0.06 (no), idx3=0.02 (reject).
    assert result.reject == [True, False, False, True]


def test_holm_bonferroni_empty_family():
    result = holm_bonferroni([], alpha=0.05)
    assert result.m == 0
    assert result.p_holm == []
    assert result.reject == []


def test_holm_bonferroni_never_less_conservative_than_bonferroni_for_smallest_p():
    p = [0.001, 0.02, 0.03, 0.04, 0.5]
    m = len(p)
    result = holm_bonferroni(p, alpha=0.05)
    # The smallest raw p always gets the full m-factor, same as plain Bonferroni.
    assert result.p_holm[0] == pytest.approx(min(p[0] * m, 1.0))


def test_median_iqr_curve_basic():
    values = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])  # one x-point, 5 seeds
    median, q25, q75 = median_iqr_curve(values, axis=1)
    assert median[0] == pytest.approx(3.0)
    assert q25[0] < median[0] < q75[0]


def test_enumerate_planned_comparisons_current_matrix_is_ten():
    plan = enumerate_planned_comparisons(
        arms=["direct", "nca", "rule_based", "rule_based_monotone"],
        strategies=["ea", "random_search"],
        tasks=["Walker-v0"],
    )
    assert len(plan) == 10
    assert sum(c.kind == "cross_arm" for c in plan) == 6
    assert sum(c.kind == "arm_vs_own_random_search" for c in plan) == 4


def test_enumerate_planned_comparisons_generalizes_to_a_second_task():
    plan = enumerate_planned_comparisons(
        arms=["direct", "nca", "rule_based", "rule_based_monotone"],
        strategies=["ea", "random_search"],
        tasks=["Walker-v0", "UpStepper-v0"],
    )
    assert len(plan) == 20  # 10 per task, doubled, no code change needed


def test_enumerate_planned_comparisons_excludes_cross_arm_cross_strategy_pairs():
    plan = enumerate_planned_comparisons(
        arms=["a", "b"],
        strategies=["ea", "random_search"],
        tasks=["t"],
    )
    pairs = {(c.cell_a, c.cell_b) for c in plan}
    forbidden = (Cell("a", "ea", "t"), Cell("b", "random_search", "t"))
    assert forbidden not in pairs
    assert tuple(reversed(forbidden)) not in pairs


def test_run_protocol_end_to_end_on_synthetic_cells():
    fitness_by_cell = {
        ("a", "ea", "t"): [10.0, 11.0, 12.0, 9.0, 10.5],
        ("b", "ea", "t"): [1.0, 2.0, 3.0, 0.5, 1.5],
        ("a", "random_search", "t"): [5.0, 5.5, 6.0, 4.5, 5.2],
        ("b", "random_search", "t"): [1.0, 1.2, 0.8, 1.1, 0.9],
    }
    result = run_protocol(fitness_by_cell, arms=["a", "b"], strategies=["ea", "random_search"], tasks=["t"])
    assert len(result.comparisons) == 3  # C(2,2)=1 cross-arm + 2 arm-vs-own-random_search
    assert result.holm.m == len(result.comparisons)
    cross_arm = [c for c in result.comparisons if c.kind == "cross_arm"][0]
    assert cross_arm.a12 == pytest.approx(1.0)  # a's values fully exceed b's


def test_enumerate_planned_comparisons_distinguishes_ablation_within_arm_from_cross_arm():
    """rule_based vs rule_based_monotone (same base encoder) is an ablation
    question, not a representation-choice one -- it must get a distinct
    `kind`, derived generically from arm_base_encoder, not hardcoded by name."""
    arm_base_encoder = {
        "direct": "direct",
        "nca": "nca",
        "rule_based": "rule_based",
        "rule_based_monotone": "rule_based",
    }
    plan = enumerate_planned_comparisons(
        arms=["direct", "nca", "rule_based", "rule_based_monotone"],
        strategies=["ea", "random_search"],
        tasks=["Walker-v0"],
        arm_base_encoder=arm_base_encoder,
    )
    assert len(plan) == 10  # unchanged count -- only the label changes
    by_pair = {(c.cell_a.arm, c.cell_b.arm): c.kind for c in plan if c.kind != ARM_VS_OWN_RANDOM_SEARCH}
    assert by_pair[("rule_based", "rule_based_monotone")] == ABLATION_WITHIN_ARM
    assert by_pair[("direct", "nca")] == CROSS_ARM
    assert by_pair[("direct", "rule_based")] == CROSS_ARM
    assert by_pair[("direct", "rule_based_monotone")] == CROSS_ARM
    assert by_pair[("nca", "rule_based")] == CROSS_ARM
    assert by_pair[("nca", "rule_based_monotone")] == CROSS_ARM
    assert sum(c.kind == ABLATION_WITHIN_ARM for c in plan) == 1
    assert sum(c.kind == CROSS_ARM for c in plan) == 5


def test_enumerate_planned_comparisons_generalizes_to_a_future_ablation_variant():
    """Not special-cased to rule_based by name: an unrelated encoder's own
    hypothetical ablation variant is classified the same way, purely from
    arm_base_encoder."""
    arm_base_encoder = {"widget": "widget", "widget_ablated": "widget", "other": "other"}
    plan = enumerate_planned_comparisons(
        arms=["widget", "widget_ablated", "other"],
        strategies=["ea"],
        tasks=["t"],
        arm_base_encoder=arm_base_encoder,
    )
    by_pair = {(c.cell_a.arm, c.cell_b.arm): c.kind for c in plan}
    assert by_pair[("widget", "widget_ablated")] == ABLATION_WITHIN_ARM
    assert by_pair[("widget", "other")] == CROSS_ARM
    assert by_pair[("widget_ablated", "other")] == CROSS_ARM


def test_enumerate_planned_comparisons_defaults_to_all_cross_arm_without_base_encoder_map():
    """Backward compatible: a caller with no arm/encoder metadata gets the
    old behaviour (every pair CROSS_ARM), not a crash."""
    plan = enumerate_planned_comparisons(arms=["a", "b"], strategies=["ea"], tasks=["t"])
    assert all(c.kind == CROSS_ARM for c in plan)


def test_run_protocol_threads_arm_base_encoder_through():
    fitness_by_cell = {
        ("rule_based", "ea", "t"): [10.0, 11.0, 12.0, 9.0, 10.5],
        ("rule_based_monotone", "ea", "t"): [10.1, 10.9, 11.8, 9.2, 10.4],
    }
    result = run_protocol(
        fitness_by_cell,
        arms=["rule_based", "rule_based_monotone"],
        strategies=["ea"],
        tasks=["t"],
        arm_base_encoder={"rule_based": "rule_based", "rule_based_monotone": "rule_based"},
    )
    assert len(result.comparisons) == 1
    assert result.comparisons[0].kind == ABLATION_WITHIN_ARM


def test_run_protocol_holm_correction_is_internally_consistent_with_holm_bonferroni():
    fitness_by_cell = {
        ("a", "ea", "t"): [10.0, 11.0, 12.0, 9.0, 10.5],
        ("b", "ea", "t"): [10.1, 10.9, 11.8, 9.2, 10.4],
        ("a", "random_search", "t"): [10.0, 11.0, 12.0, 9.0, 10.5],
        ("b", "random_search", "t"): [10.1, 10.9, 11.8, 9.2, 10.4],
    }
    result = run_protocol(fitness_by_cell, arms=["a", "b"], strategies=["ea", "random_search"], tasks=["t"])
    expected = holm_bonferroni([c.p_raw for c in result.comparisons])
    assert [c.p_holm for c in result.comparisons] == pytest.approx(expected.p_holm)
    assert [c.significant_holm for c in result.comparisons] == expected.reject
