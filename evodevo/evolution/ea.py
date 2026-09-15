"""Encoding-agnostic (mu + lambda) evolutionary loop.

This module may only call methods on the Encoder interface (evodevo.encoders.base.Encoder)
and the evaluator callable it is given. It must never import the physics simulator package
or the RL training library, nor any specific encoder class, so that it can be exercised
end-to-end with the direct encoder and the surrogate evaluator alone.

Per-evaluation bookkeeping (budget/invalid-genome/cache-charging policy, the
P6 dual counters) lives in evodevo.evolution.eval_bookkeeping and is shared
byte-for-byte with evodevo.evolution.random_search -- this module is
responsible only for WHICH genome to evaluate next (selection + mutation/
crossover), never for HOW an evaluation is scored/counted/logged.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from evodevo.encoders.base import Encoder
from evodevo.evaluation.cache import FitnessCache
from evodevo.evolution.eval_bookkeeping import (
    BUDGET_COUNTER_SEMANTICS,
    CACHE_CHARGING_POLICY,
    EvaluationTracker,
    Individual,
    RunResult,
    assert_resolved_policy_unchanged,
    resolve_cache,
    resolved_policy_for,
)
from evodevo.evolution.logging import RunLogger

__all__ = [
    "run_ea",
    "RunResult",
    "Individual",
    "resolved_policy_for",
    "CACHE_CHARGING_POLICY",
    "BUDGET_COUNTER_SEMANTICS",
]


def run_ea(
    encoder: Encoder,
    evaluator: Callable,
    config,
    logger: Optional[RunLogger] = None,
    cache: Optional[FitnessCache] = None,
) -> RunResult:
    rng = np.random.default_rng(config.seed)
    mu = config.population_size
    lam = config.offspring_size
    budget = config.eval_budget

    # Run-start assertion: the invalid-genome policy, cache-charging policy,
    # and budget-counter semantics must be IDENTICAL for every arm (and,
    # since P7, every strategy). See eval_bookkeeping.py for why this is a
    # self-consistency guard, not the primary enforcement mechanism (that's
    # evodevo.experiments.fairness.assert_comparable()/assert_identical_policy()).
    resolved_policy = resolved_policy_for(config)
    assert_resolved_policy_unchanged(resolved_policy)

    cache = resolve_cache(config, cache)

    # The primary three-way comparison is mutation-only for every arm: Direct
    # and NCA have no crossover operator, so an asymmetric reproduction
    # operator (crossover for one arm only) would confound the representation
    # comparison. use_crossover defaults to False; when an encoding-specific
    # ablation deliberately wants crossover (e.g. a labelled "rule_based_xover"
    # arm), config.use_crossover=True switches every offspring's reproduction
    # to encoder.crossover() instead of encoder.mutate() -- checked once here,
    # at startup, rather than letting a config/encoder mismatch fail silently
    # partway through a run.
    use_crossover = bool(getattr(config, "use_crossover", False))
    if use_crossover and not encoder.supports_crossover():
        raise ValueError(
            f"config.use_crossover=True but encoder '{encoder.name}' does not support "
            "crossover (supports_crossover() returned False)"
        )

    tracker = EvaluationTracker(encoder, evaluator, config, cache, logger)

    population: List[Individual] = []
    generation = 0
    for _ in range(mu):
        if tracker.evaluations_used >= budget:
            break
        genome = encoder.random_genome(rng)
        population.append(tracker.evaluate(genome, parent_id=None, generation=generation))
    tracker.record_generation_summary(generation)

    while tracker.evaluations_used < budget:
        generation += 1
        survivors = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:mu]

        offspring: List[Individual] = []
        for _ in range(lam):
            if tracker.evaluations_used >= budget:
                break
            if use_crossover:
                parent_a = survivors[rng.integers(0, len(survivors))]
                parent_b = survivors[rng.integers(0, len(survivors))]
                child_genome = encoder.crossover(parent_a.genome, parent_b.genome, rng)
                parent_id = parent_a.genome_id
            else:
                parent_a = survivors[rng.integers(0, len(survivors))]
                child_genome = encoder.mutate(parent_a.genome, rng)
                parent_id = parent_a.genome_id
            offspring.append(tracker.evaluate(child_genome, parent_id=parent_id, generation=generation))

        population = survivors + offspring
        tracker.record_generation_summary(generation)

    return RunResult(
        population=population,
        evaluations_used=tracker.evaluations_used,
        best_fitness=tracker.best_fitness,
        records=tracker.records,
        cache_hit_rate=cache.hit_rate if cache is not None else None,
        resolved_policy=resolved_policy,
        generation_records=tracker.generation_records,
    )
