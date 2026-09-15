"""Unit tests for evodevo/analysis/budget_curves.py's two-panel figure and
its automatically generated interpretation paragraph."""
import numpy as np
import pandas as pd
import pytest

from evodevo.analysis.budget_curves import (
    generate_interpretation,
    plot_two_panel_comparison,
    plot_two_panel_comparison_with_interpretation,
)


def _synthetic_run(encoder, seed, n, spent_step_per_valid, fitness_fn):
    """Builds a synthetic per-eval record set for one (encoder, seed): every
    other evaluation is invalid (spent_step_per_valid controls how many
    evals_spent it takes per real/valid attempt)."""
    rows = []
    evals_spent = 0
    evals_valid = 0
    best = float("-inf")
    valid_index = 0
    while evals_valid < n:
        evals_spent += 1
        is_valid = (evals_spent % spent_step_per_valid) == 0
        if is_valid:
            evals_valid += 1
            fitness = fitness_fn(valid_index)
            valid_index += 1
        else:
            fitness = -10.0
        best = max(best, fitness)
        rows.append({
            "encoder": encoder,
            "seed": seed,
            "eval_index": evals_spent - 1,
            "evals_spent": evals_spent,
            "evals_valid_cumulative": evals_valid,
            "fitness": fitness,
            "best_fitness_so_far": best,
        })
    return rows


def test_prior_driven_advantage_detected_when_panels_converge():
    # "fast" wastes almost no evaluations (spent_step_per_valid=1); "slow"
    # wastes 4 out of 5 (spent_step_per_valid=5) but reaches the SAME
    # fitness values once only real attempts are counted -- this is exactly
    # the "advantage attributable to the prior" case.
    rows = []
    rows += _synthetic_run("fast", seed=0, n=20, spent_step_per_valid=1, fitness_fn=lambda i: i * 0.1)
    rows += _synthetic_run("slow", seed=0, n=20, spent_step_per_valid=5, fitness_fn=lambda i: i * 0.1)
    df = pd.DataFrame(rows)

    text = generate_interpretation(df)
    assert "attributable to its representational prior" in text
    assert "fast" in text


def test_real_search_advantage_survives_when_panels_agree():
    # "better" both wastes fewer evals AND reaches higher fitness per real
    # attempt -- leads on both panels, advantage should NOT be explained
    # away as prior-driven.
    rows = []
    rows += _synthetic_run("better", seed=0, n=20, spent_step_per_valid=1, fitness_fn=lambda i: i * 0.5)
    rows += _synthetic_run("worse", seed=0, n=20, spent_step_per_valid=1, fitness_fn=lambda i: i * 0.1)
    df = pd.DataFrame(rows)

    text = generate_interpretation(df)
    assert "SURVIVES correction" in text
    assert "better" in text


def test_searches_better_but_wastes_budget_case():
    # "wasteful" needs 5x the evals_spent per real attempt but, PER REAL
    # ATTEMPT, reaches much higher fitness than "frugal" -- so it leads on
    # evals_valid but not necessarily on evals_spent.
    rows = []
    rows += _synthetic_run("frugal", seed=0, n=20, spent_step_per_valid=1, fitness_fn=lambda i: i * 0.05)
    rows += _synthetic_run("wasteful", seed=0, n=20, spent_step_per_valid=5, fitness_fn=lambda i: i * 1.0)
    df = pd.DataFrame(rows)

    text = generate_interpretation(df)
    # wasteful's final valid-indexed best is 19*1.0=19.0, frugal's is 19*0.05=0.95:
    # wasteful leads on evals_valid by a lot -- confirm it isn't described as
    # "converged" or as the prior-driven case.
    assert "wasteful" in text
    assert "searches better" in text or "SURVIVES correction" in text


def test_empty_dataframe_returns_placeholder_axes():
    fig, axes = plot_two_panel_comparison(pd.DataFrame())
    assert len(axes) == 2


def test_two_panel_figure_has_two_axes_with_correct_titles():
    rows = []
    rows += _synthetic_run("a", seed=0, n=10, spent_step_per_valid=2, fitness_fn=lambda i: float(i))
    rows += _synthetic_run("b", seed=0, n=10, spent_step_per_valid=2, fitness_fn=lambda i: float(i))
    df = pd.DataFrame(rows)

    fig, axes = plot_two_panel_comparison(df)
    assert len(axes) == 2
    assert "spent" in axes[0].get_title().lower()
    assert "valid" in axes[1].get_title().lower()


def test_single_arm_returns_no_comparison_message():
    rows = _synthetic_run("only_one", seed=0, n=10, spent_step_per_valid=1, fitness_fn=lambda i: float(i))
    df = pd.DataFrame(rows)
    text = generate_interpretation(df)
    assert "Only one arm" in text


def test_wrapper_returns_figure_axes_and_text_together():
    rows = []
    rows += _synthetic_run("x", seed=0, n=10, spent_step_per_valid=1, fitness_fn=lambda i: float(i))
    rows += _synthetic_run("y", seed=0, n=10, spent_step_per_valid=1, fitness_fn=lambda i: float(i) * 2)
    df = pd.DataFrame(rows)
    fig, axes, text = plot_two_panel_comparison_with_interpretation(df)
    assert len(axes) == 2
    assert isinstance(text, str) and len(text) > 0
