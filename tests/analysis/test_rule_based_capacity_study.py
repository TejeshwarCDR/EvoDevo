"""Smoke tests for evodevo/analysis/rule_based_capacity_study.py -- fast,
small-N checks of each part's mechanics; the full N=2000 evidence lives in
the committed results/rule_based/CAPACITY_STUDY.md."""
import numpy as np

from evodevo.encoders.rule_based._rules import genome as genome_mod
from evodevo.encoders.rule_based._rules.genome import MAX_RULES, MIN_RULES
from evodevo.analysis.rule_based_capacity_study import (
    _patched_rule_bounds,
    _sample_genome_with_n_rules,
    description_length_report,
    effective_rule_count_distributions,
    min_rules_check,
    render_report,
    rule_based_action_space_size,
    rule_based_condition_space_size,
    sweep_max_rules,
)


def test_condition_and_action_space_sizes_match_hand_count():
    # always(1) + cell_type(5) + neighbor_count(4) + empty_neighbor(4) + position(6) = 20
    assert rule_based_condition_space_size() == 20
    # 5 SET_* + grow_directional(4x4=16) + copy_neighbor(4) = 25
    assert rule_based_action_space_size() == 25


def test_description_length_report_is_internally_consistent():
    dl = description_length_report()
    assert dl["rule_space"] == dl["cond_size"] * dl["action_size"]
    assert dl["rb_bits_max"] > dl["rb_bits_min"]  # more rules -> longer description
    assert dl["nca_current_params"] != dl["nca_h3_params"]  # the stale-792 discrepancy is real
    assert dl["nca_h3_params"] == 792  # matches the task's own historical citation
    assert dl["direct_bits"] > 0


def test_patched_rule_bounds_is_reversible():
    before = (genome_mod.MIN_RULES, genome_mod.MAX_RULES)
    with _patched_rule_bounds(1, 30):
        assert (genome_mod.MIN_RULES, genome_mod.MAX_RULES) == (1, 30)
    assert (genome_mod.MIN_RULES, genome_mod.MAX_RULES) == before


def test_sample_genome_with_n_rules_produces_exact_count():
    for n in (1, 4, 12, 24):
        g = _sample_genome_with_n_rules(n)
        assert len(g.rules) == n
    # bounds restored after use for every n, including n > committed MAX_RULES
    assert (genome_mod.MIN_RULES, genome_mod.MAX_RULES) == (MIN_RULES, MAX_RULES)


def test_effective_rule_count_distributions_shapes(small_n=20):
    erc = effective_rule_count_distributions(n=small_n, seed=1)
    assert len(erc["random_sampler_counts"]) == small_n
    assert len(erc["full_budget_counts"]) == small_n
    assert all(1 <= c <= 7 for c in erc["random_sampler_counts"])
    assert all(1 <= c <= MAX_RULES for c in erc["full_budget_counts"])


def test_sweep_max_rules_covers_all_sweep_points(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = sweep_max_rules(n=10, seed=2)
    assert set(r.keys()) == {4, 8, 12, 24}
    for mr, metrics in r.items():
        assert 0.0 <= metrics["valid_rate"] <= 1.0
        assert 0.0 <= metrics["unique_valid_body_fraction"] <= 1.0
    # bounds restored after the n=24 sweep cell (which needed monkeypatching)
    assert (genome_mod.MIN_RULES, genome_mod.MAX_RULES) == (MIN_RULES, MAX_RULES)


def test_min_rules_check_covers_both_points(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = min_rules_check(n=10, seed=3)
    assert set(r.keys()) == {"n=1", "n=2 (MIN_RULES)"}
    for metrics in r.values():
        assert 0.0 <= metrics["valid_rate"] <= 1.0


def test_render_report_is_nonempty_and_mentions_key_findings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dl = description_length_report()
    erc = effective_rule_count_distributions(n=10, seed=4)
    sweep = sweep_max_rules(n=10, seed=4)
    min_check = min_rules_check(n=10, seed=4)
    report = render_report(dl, erc, sweep, min_check)
    assert "description length" in report
    assert "792" in report  # the historical figure is disclosed, not silently dropped
    assert str(dl["nca_current_params"]) in report  # the current figure is the one used
    assert "MIN_RULES" in report and "MAX_RULES" in report
