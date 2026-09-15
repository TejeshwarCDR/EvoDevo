"""Generic strategy-conformance suite: runs the SAME checks against every
registered search strategy (evodevo.evolution.ea.run_ea,
evodevo.evolution.random_search.run_random_search), mirroring how
tests/encoders/test_contract.py checks every encoder once instead of
duplicating checks per encoder file.

random_search must pass every one of these checks UNMODIFIED (no
strategy-specific special-casing) -- that is the acceptance criterion for
adding it as a control: it must be a first-class citizen of exactly the
same interface/budget/policy contract as the EA, not a bespoke shortcut."""
from __future__ import annotations

import numpy as np
import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea
from evodevo.evolution.random_search import run_random_search

STRATEGIES = {"ea": run_ea, "random_search": run_random_search}
ENCODER_CLASSES = {"direct": DirectEncoder, "nca": NCAEncoder, "rule_based": RuleBasedEncoder}


def _factors(**overrides):
    params = dict(dev_steps=5, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


def _make_config(**overrides):
    defaults = dict(
        seed=0,
        encoder="direct",
        task="dummy-task",
        population_size=4,
        offspring_size=4,
        eval_budget=20,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _run(strategy_name, encoder_name, **config_overrides):
    strategy = STRATEGIES[strategy_name]
    encoder = ENCODER_CLASSES[encoder_name](factors=_factors())
    config = _make_config(encoder=encoder_name, **config_overrides)
    return strategy(encoder, evaluate_body_surrogate, config), encoder, config


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_stops_exactly_at_budget(strategy_name, encoder_name):
    result, _encoder, config = _run(strategy_name, encoder_name, eval_budget=17)
    assert result.evaluations_used == 17
    assert len(result.records) == 17


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_evals_valid_never_exceeds_evals_spent(strategy_name, encoder_name):
    result, _encoder, _config = _run(strategy_name, encoder_name)
    for record in result.records:
        assert record["evals_valid_cumulative"] <= record["evals_spent"]
        assert record["evals_spent"] == record["evals_valid_cumulative"] + record["evals_invalid_cumulative"]


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_counters_are_monotonic(strategy_name, encoder_name):
    result, _encoder, _config = _run(strategy_name, encoder_name)
    for field in ("evals_spent", "evals_valid_cumulative", "evals_invalid_cumulative",
                  "evals_cache_hit_cumulative", "best_fitness_so_far"):
        values = [r[field] for r in result.records]
        assert values == sorted(values)


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_best_fitness_so_far_matches_running_max(strategy_name, encoder_name):
    result, _encoder, _config = _run(strategy_name, encoder_name)
    running_max = float("-inf")
    for record in result.records:
        running_max = max(running_max, record["fitness"])
        assert record["best_fitness_so_far"] == pytest.approx(running_max)


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_resolved_policy_present_and_well_formed(strategy_name, encoder_name):
    result, _encoder, _config = _run(strategy_name, encoder_name)
    assert set(result.resolved_policy) == {
        "invalid_genome_policy", "cache_charging_policy", "budget_counter_semantics",
    }


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
def test_resolved_policy_identical_across_strategies(encoder_name):
    result_ea, _e1, _c1 = _run("ea", encoder_name)
    result_rs, _e2, _c2 = _run("random_search", encoder_name)
    assert result_ea.resolved_policy == result_rs.resolved_policy


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_generation_records_present(strategy_name, encoder_name):
    result, _encoder, _config = _run(strategy_name, encoder_name)
    assert len(result.generation_records) >= 1
    for gen_record in result.generation_records:
        assert set(gen_record) >= {
            "generation", "evals_spent", "evals_valid", "evals_invalid",
            "evals_invalid_reason_counts", "evals_cache_hit", "best_fitness_so_far",
        }


@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_deterministic_given_same_seed(strategy_name, encoder_name):
    result_a, _e1, _c1 = _run(strategy_name, encoder_name, eval_budget=12)
    result_b, _e2, _c2 = _run(strategy_name, encoder_name, eval_budget=12)
    fitnesses_a = [r["fitness"] for r in result_a.records]
    fitnesses_b = [r["fitness"] for r in result_b.records]
    assert fitnesses_a == fitnesses_b


def test_random_search_rejects_use_crossover():
    encoder = DirectEncoder(factors=_factors())
    config = _make_config(use_crossover=True)
    with pytest.raises(ValueError):
        run_random_search(encoder, evaluate_body_surrogate, config)


def test_random_search_has_no_selection_every_genome_is_a_fresh_sample():
    """random_search's population should show no parent/survivor reuse --
    every individual's parent_id is None (fresh i.i.d. samples only)."""
    result, _encoder, _config = _run("random_search", "direct", eval_budget=15)
    assert all(record["parent_id"] is None for record in result.records)


def test_ea_does_have_selection_for_offspring_beyond_the_initial_population():
    result, _encoder, config = _run("ea", "direct", eval_budget=15, population_size=4, offspring_size=4)
    offspring_records = [r for r in result.records if r["generation"] > 0]
    assert offspring_records  # sanity: budget was enough to reach offspring
    assert any(r["parent_id"] is not None for r in offspring_records)
