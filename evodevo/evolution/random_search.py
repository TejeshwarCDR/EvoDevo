"""Random search: the control strategy that separates the encoding's
representational PRIOR from the evolutionary algorithm's SEARCH
contribution.

Conforms to exactly the same interface as evodevo.evolution.ea.run_ea --
same signature, same RunResult shape, same budget semantics, same
invalid-genome policy, same cache-charging policy, same P6 dual counters
(evals_spent vs. evals_valid), same evaluation pipeline (both strategies
share evodevo.evolution.eval_bookkeeping.EvaluationTracker byte-for-byte,
so there is no way for the two strategies' accounting to drift apart).

The only difference from run_ea: no selection, no mutation, no crossover,
no population. Every genome is a FRESH i.i.d. sample from the encoder's own
random_genome() distribution; the best fitness found so far is tracked
exactly like the EA does. This module may only call methods on the Encoder
interface and the evaluator callable it is given, exactly like ea.py.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from evodevo.encoders.base import Encoder
from evodevo.evaluation.cache import FitnessCache
from evodevo.evolution.eval_bookkeeping import (
    EvaluationTracker,
    Individual,
    RunResult,
    assert_resolved_policy_unchanged,
    resolve_cache,
    resolved_policy_for,
)
from evodevo.evolution.logging import RunLogger

__all__ = ["run_random_search"]

# Purely for generation_log.jsonl granularity (readability/comparability
# with the EA's per-generation cadence); has NO effect on search behaviour,
# budget accounting, or the evals_spent/evals_valid counters -- random
# search has no actual notion of "generations".
_LOG_BATCH_SIZE_FALLBACK = 10


def run_random_search(
    encoder: Encoder,
    evaluator: Callable,
    config,
    logger: Optional[RunLogger] = None,
    cache: Optional[FitnessCache] = None,
) -> RunResult:
    rng = np.random.default_rng(config.seed)
    budget = config.eval_budget

    # Identical run-start assertion to run_ea() -- see eval_bookkeeping.py.
    resolved_policy = resolved_policy_for(config)
    assert_resolved_policy_unchanged(resolved_policy)

    cache = resolve_cache(config, cache)

    tracker = EvaluationTracker(encoder, evaluator, config, cache, logger)

    # use_crossover/population_size/offspring_size are meaningless for a
    # strategy with no selection or reproduction -- config.use_crossover=True
    # would be a caller error (there is nothing here that could honour it),
    # checked explicitly rather than silently ignored.
    if bool(getattr(config, "use_crossover", False)):
        raise ValueError("random_search does not support use_crossover=True (no selection/reproduction exists)")

    log_batch_size = int(getattr(config, "population_size", _LOG_BATCH_SIZE_FALLBACK)) or _LOG_BATCH_SIZE_FALLBACK

    population: List[Individual] = []
    generation = 0
    while tracker.evaluations_used < budget:
        genome = encoder.random_genome(rng)
        population.append(tracker.evaluate(genome, parent_id=None, generation=generation))

        if tracker.evaluations_used % log_batch_size == 0 or tracker.evaluations_used >= budget:
            tracker.record_generation_summary(generation)
            generation += 1

    return RunResult(
        population=population,
        evaluations_used=tracker.evaluations_used,
        best_fitness=tracker.best_fitness,
        records=tracker.records,
        cache_hit_rate=cache.hit_rate if cache is not None else None,
        resolved_policy=resolved_policy,
        generation_records=tracker.generation_records,
    )
