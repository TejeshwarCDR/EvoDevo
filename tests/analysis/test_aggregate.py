import pandas as pd
import pytest

from evodevo.analysis.aggregate import per_seed_finals, three_way_table


def _make_df():
    rows = []
    # direct: 2 seeds, fitness climbs to 10 and 12
    for seed, finals in ((0, [1, 5, 10]), (1, [2, 6, 12])):
        for i, fitness in enumerate(finals):
            rows.append(
                dict(
                    encoder="direct", seed=seed, eval_index=i, fitness=fitness, valid=True,
                    body_hash=f"direct-{seed}-{i}", wall_clock_s=0.1, cache_hit=False,
                )
            )
    # nca: 2 seeds, fitness climbs to 20 (the best arm)
    for seed, finals in ((0, [5, 15, 20]), (1, [4, 14, 20])):
        for i, fitness in enumerate(finals):
            rows.append(
                dict(
                    encoder="nca", seed=seed, eval_index=i, fitness=fitness, valid=(fitness > 3),
                    body_hash=f"nca-{seed}-{i}", wall_clock_s=0.2, cache_hit=False,
                )
            )
    # rule_based: variable-length genome arm, but the log never carries a
    # genome (only body-level fields) -- verifies aggregate.py doesn't
    # need any special-casing for it.
    for seed, finals in ((0, [3, 8, 16]), (1, [1, 9, 15])):
        for i, fitness in enumerate(finals):
            rows.append(
                dict(
                    encoder="rule_based", seed=seed, eval_index=i, fitness=fitness, valid=True,
                    body_hash=f"rb-{seed}-{i}", wall_clock_s=0.3, cache_hit=False,
                )
            )
    return pd.DataFrame(rows)


def test_per_seed_finals():
    df = _make_df()
    finals = per_seed_finals(df)
    assert set(finals) == {"direct", "nca", "rule_based"}
    assert sorted(finals["direct"]) == [10.0, 12.0]
    assert sorted(finals["nca"]) == [20.0, 20.0]


def test_three_way_table_populated_for_all_three_encoders():
    df = _make_df()
    table = three_way_table(df)
    assert set(table["encoder"]) == {"direct", "nca", "rule_based"}
    assert len(table) == 3

    for _, row in table.iterrows():
        assert row["n_seeds"] == 2
        assert 0.0 <= row["validity_rate"] <= 1.0
        assert row["unique_bodies_explored"] > 0
        assert row["mean_wall_clock_s_per_eval"] > 0

    nca_row = table.loc[table["encoder"] == "nca"].iloc[0]
    assert nca_row["final_best_fitness_mean"] == pytest.approx(20.0)
    # nca is the best arm, so it should reach 50%/90% of its own final
    # fitness quickly (its final fitness IS the "best arm" reference).
    assert nca_row["evaluations_to_50pct_of_best_arm_mean"] <= 3


def test_three_way_table_handles_rule_based_variable_length_genome_arm_without_crashing():
    # The point: run_log.jsonl records never contain a raw genome (only
    # body-level fields), so the Rule-Based arm's variable-length,
    # non-numeric genome never reaches this loader -- this test proves
    # three_way_table() processes its rows exactly like any other arm's.
    df = _make_df()
    table = three_way_table(df)
    rb_row = table.loc[table["encoder"] == "rule_based"].iloc[0]
    assert rb_row["final_best_fitness_mean"] == pytest.approx(15.5)


def test_three_way_table_empty_df_returns_empty():
    assert three_way_table(pd.DataFrame()).empty
    assert per_seed_finals(pd.DataFrame()) == {}
