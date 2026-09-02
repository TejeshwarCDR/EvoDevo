"""Encoding-agnostic (mu + lambda) evolutionary loop.

This module may only call methods on the Encoder interface (evodevo.encoders.base.Encoder)
and the evaluator callable it is given. It must never import the physics simulator package
or the RL training library, nor any specific encoder class, so that it can be exercised
end-to-end with the direct encoder and the surrogate evaluator alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from evodevo.encoders.base import Encoder
from evodevo.evaluation.cache import FitnessCache
from evodevo.evolution.logging import RunLogger
from evodevo.morphology.metrics import body_hash as compute_body_hash
from evodevo.morphology.metrics import body_metrics


@dataclass
class Individual:
    genome_id: int
    parent_id: Optional[int]
    genome: np.ndarray
    fitness: float
    valid: bool


@dataclass
class RunResult:
    population: List[Individual]
    evaluations_used: int
    best_fitness: float
    records: list = field(default_factory=list)
    cache_hit_rate: Optional[float] = None


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

    # A cache hit MUST still consume one evaluation from the budget and MUST be
    # logged (cache_hit=True). If a hit were free, whichever encoding happens to
    # revisit bodies more often would gain a silent budget advantage over the
    # others -- exactly the search-efficiency confound this project exists to
    # rule out. cache_enabled defaults to False for the primary comparison.
    cache_enabled = bool(getattr(config, "cache_enabled", False))
    if cache_enabled and cache is None:
        cache = FitnessCache()
    elif not cache_enabled:
        cache = None

    evaluations_used = 0
    next_genome_id = 0
    records: list = []
    best_fitness = float("-inf")

    def evaluate_genome(genome: np.ndarray, parent_id: Optional[int], generation: int) -> Individual:
        nonlocal evaluations_used, next_genome_id, best_fitness

        genome_id = next_genome_id
        next_genome_id += 1

        body = encoder.decode(genome)

        cache_hit = False
        result = None
        if cache is not None:
            h = compute_body_hash(body)
            cached_result = cache.get(h)
            if cached_result is not None:
                result = cached_result
                cache_hit = True

        if result is None:
            result = evaluator(
                body,
                task=config.task,
                ppo_timesteps=config.ppo_timesteps,
                eval_episodes=config.ppo_eval_episodes,
                seed=config.seed,
                sentinel_fitness=getattr(config, "sentinel_fitness", -10.0),
            )
            if cache is not None:
                cache.put(result.body_hash, result)

        evaluations_used += 1
        best_fitness = max(best_fitness, result.fitness)

        record = {
            "eval_index": evaluations_used - 1,
            "generation": generation,
            "genome_id": genome_id,
            "parent_id": parent_id,
            "fitness": result.fitness,
            "valid": result.valid,
            "reason": result.reason,
            "body_hash": result.body_hash,
            "wall_clock_s": result.wall_clock_s,
            "cache_hit": cache_hit,
            **body_metrics(body),
        }
        records.append(record)
        if logger is not None:
            logger.log(record)

        return Individual(
            genome_id=genome_id,
            parent_id=parent_id,
            genome=genome,
            fitness=result.fitness,
            valid=result.valid,
        )

    population: List[Individual] = []
    generation = 0
    for _ in range(mu):
        if evaluations_used >= budget:
            break
        genome = encoder.random_genome(rng)
        population.append(evaluate_genome(genome, parent_id=None, generation=generation))

    while evaluations_used < budget:
        generation += 1
        survivors = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:mu]

        offspring: List[Individual] = []
        for _ in range(lam):
            if evaluations_used >= budget:
                break
            parent = survivors[rng.integers(0, len(survivors))]
            child_genome = encoder.mutate(parent.genome, rng)
            offspring.append(evaluate_genome(child_genome, parent_id=parent.genome_id, generation=generation))

        population = survivors + offspring

    return RunResult(
        population=population,
        evaluations_used=evaluations_used,
        best_fitness=best_fitness,
        records=records,
        cache_hit_rate=cache.hit_rate if cache is not None else None,
    )
