"""Unit tests for evodevo/analysis/random_search_decomposition.py -- the
EA-minus-random prior/search decomposition and its guard clause."""
import pandas as pd
import pytest

from evodevo.analysis.random_search_decomposition import (
    compute_decomposition,
    generate_decomposition_report,
)


def _make(encoder, strategy, seed, n, spent_step, fitness_fn):
    rows = []
    spent, valid, best, vi = 0, 0, float("-inf"), 0
    while valid < n:
        spent += 1
        is_valid = (spent % spent_step) == 0
        if is_valid:
            fitness = fitness_fn(vi)
            vi += 1
            valid += 1
        else:
            fitness = -10.0
        best = max(best, fitness)
        rows.append({
            "encoder": encoder, "strategy": strategy, "seed": seed, "eval_index": spent - 1,
            "evals_spent": spent, "evals_valid_cumulative": valid,
            "fitness": fitness, "best_fitness_so_far": best,
        })
    return rows


def test_missing_strategy_column_returns_empty():
    df = pd.DataFrame({"encoder": ["direct"], "seed": [0], "fitness": [1.0]})
    assert compute_decomposition(df) == {}


def test_encoder_without_both_strategies_is_excluded():
    rows = _make("direct", "ea", 0, 10, 1, lambda i: float(i))
    df = pd.DataFrame(rows)
    assert compute_decomposition(df) == {}


def test_clear_ea_advantage_is_interpretable():
    rows = []
    rows += _make("direct", "ea", 0, 20, 1, lambda i: i * 1.0)
    rows += _make("direct", "ea", 1, 20, 1, lambda i: i * 1.0)
    rows += _make("direct", "random_search", 0, 20, 1, lambda i: i * 0.1)
    rows += _make("direct", "random_search", 1, 20, 1, lambda i: i * 0.1)
    df = pd.DataFrame(rows)

    decomposition = compute_decomposition(df)
    assert "direct" in decomposition
    for counter in ("evals_spent", "evals_valid"):
        stats = decomposition["direct"][counter]
        assert stats["advantage"] > 0
        assert stats["interpretable"] is True


def test_noisy_small_advantage_is_not_interpretable():
    rows = []
    rows += _make("nca", "ea", 0, 20, 1, lambda i: i * 1.0)
    rows += _make("nca", "ea", 1, 20, 1, lambda i: i * 0.01)  # huge seed-to-seed variance
    rows += _make("nca", "random_search", 0, 20, 1, lambda i: i * 0.9)
    rows += _make("nca", "random_search", 1, 20, 1, lambda i: i * 0.88)
    df = pd.DataFrame(rows)

    decomposition = compute_decomposition(df)
    for counter in ("evals_spent", "evals_valid"):
        assert decomposition["nca"][counter]["interpretable"] is False


def test_report_flags_non_interpretable_results_in_bold():
    rows = []
    rows += _make("nca", "ea", 0, 20, 1, lambda i: i * 1.0)
    rows += _make("nca", "ea", 1, 20, 1, lambda i: i * 0.01)
    rows += _make("nca", "random_search", 0, 20, 1, lambda i: i * 0.9)
    rows += _make("nca", "random_search", 1, 20, 1, lambda i: i * 0.88)
    df = pd.DataFrame(rows)

    text = generate_decomposition_report(df)
    assert "**NO**" in text
    assert "NOT interpretable" in text


def test_report_distinguishes_prior_from_search_claims():
    rows = []
    rows += _make("direct", "ea", 0, 20, 1, lambda i: i * 1.0)
    rows += _make("direct", "ea", 1, 20, 1, lambda i: i * 1.0)
    rows += _make("direct", "random_search", 0, 20, 1, lambda i: i * 0.1)
    rows += _make("direct", "random_search", 1, 20, 1, lambda i: i * 0.1)
    df = pd.DataFrame(rows)

    text = generate_decomposition_report(df)
    assert "PRIOR" in text
    assert "SEARCH" in text
    assert "different claims" in text


def test_empty_dataframe_returns_placeholder_message():
    text = generate_decomposition_report(pd.DataFrame())
    assert "no random_search control data" in text
