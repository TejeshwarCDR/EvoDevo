"""Encoding-agnostic (mu + lambda) evolutionary search."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Callable, List, Optional

import numpy as np

from evodevo.encoders.base import Encoder
from evodevo.evaluation.cache import FitnessCache
from evodevo.evolution.eval_bookkeeping import (
    BUDGET_COUNTER_SEMANTICS, CACHE_CHARGING_POLICY, EvaluationTracker,
    Individual, RunResult, assert_resolved_policy_unchanged, resolve_cache,
    resolved_policy_for,
)
from evodevo.evolution.logging import RunLogger
from evodevo.morphology.metrics import body_hash as compute_body_hash

__all__ = ["Candidate", "EvolutionRun", "run_ea", "RunResult", "Individual",
           "resolved_policy_for", "CACHE_CHARGING_POLICY", "BUDGET_COUNTER_SEMANTICS"]


@dataclass
class Candidate:
    """A fully specified evaluation slot. ``body`` makes worker dispatch stateless."""
    genome: object
    genome_id: int
    parent_id: Optional[int]
    generation: int
    eval_index: int
    body: np.ndarray
    cache_hit: bool = False
    cached_result: object = None
    cache_source_eval_index: Optional[int] = None


class EvolutionRun:
    """Resumable proposal/tell state machine with no evaluator dependency."""
    def __init__(self, encoder: Encoder, config, logger: Optional[RunLogger] = None,
                 cache: Optional[FitnessCache] = None):
        self.encoder, self.config = encoder, config
        self.rng = np.random.default_rng(config.seed)
        self.mu, self.lam, self.budget = config.population_size, config.offspring_size, config.eval_budget
        self.resolved_policy = resolved_policy_for(config)
        assert_resolved_policy_unchanged(self.resolved_policy)
        self.cache = resolve_cache(config, cache)
        self.use_crossover = bool(getattr(config, "use_crossover", False))
        if self.use_crossover and not encoder.supports_crossover():
            raise ValueError(f"config.use_crossover=True but encoder '{encoder.name}' does not support crossover (supports_crossover() returned False)")
        # record_result() is EvaluationTracker's non-evaluating half.
        self.tracker = EvaluationTracker(encoder, None, config, self.cache, logger)
        self.population: List[Individual] = []
        self.generation = 0
        self._initial_population, self._pending = True, []
        self._done = self.budget <= 0

    @property
    def done(self) -> bool:
        return self._done

    def propose(self) -> list[Candidate]:
        if self.done:
            return []
        if self._pending:
            raise RuntimeError("tell() must be called before propose() again")
        remaining = self.budget - self.tracker.evaluations_used
        batch_size = min(self.mu if self._initial_population else self.lam, remaining)
        if self._initial_population:
            survivors = []
        else:
            self.generation += 1
            survivors = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:self.mu]
        proposed = []
        # This mirrors an interleaved cache exactly without evaluating yet:
        # a repeated body later in this batch is the hit that it would have
        # seen after its earlier slot had populated the cache.
        pending_by_hash = {}
        for offset in range(batch_size):
            if self._initial_population:
                genome, parent_id = self.encoder.random_genome(self.rng), None
            elif self.use_crossover:
                parent_a = survivors[self.rng.integers(0, len(survivors))]
                parent_b = survivors[self.rng.integers(0, len(survivors))]
                genome = self.encoder.crossover(parent_a.genome, parent_b.genome, self.rng)
                parent_id = parent_a.genome_id
            else:
                parent_a = survivors[self.rng.integers(0, len(survivors))]
                genome = self.encoder.mutate(parent_a.genome, self.rng)
                parent_id = parent_a.genome_id
            body = self.encoder.decode(genome)
            cached_result = self.cache.get(compute_body_hash(body)) if self.cache is not None else None
            cache_source_eval_index = None
            if self.cache is not None and cached_result is None:
                source = pending_by_hash.get(compute_body_hash(body))
                if source is not None:
                    self.cache.hits += 1
                    cache_source_eval_index = source.eval_index
            proposed.append(Candidate(
                genome, self.tracker.next_genome_id + offset, parent_id, self.generation,
                self.tracker.evaluations_used + offset, body,
                cached_result is not None or cache_source_eval_index is not None, cached_result,
                cache_source_eval_index,
            ))
            if self.cache is not None and cached_result is None and cache_source_eval_index is None:
                pending_by_hash[compute_body_hash(body)] = proposed[-1]
        self._pending = proposed
        return list(proposed)

    def tell(self, results: list) -> None:
        if not self._pending:
            raise RuntimeError("tell() called without a proposed batch")
        if len(results) != len(self._pending):
            raise ValueError("tell() requires exactly one result per proposed candidate")
        # EvalResult has no index field today; (eval_index, EvalResult) is the
        # explicit order-independent protocol, while positional results retain
        # the convenient run_ea API.
        indexed = []
        fallback = iter(self._pending)
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                index, result = item
            elif hasattr(item, "eval_index"):
                index, result = item.eval_index, item
            else:
                index, result = next(fallback).eval_index, item
            indexed.append((index, result))
        result_by_index = dict(indexed)
        expected = {candidate.eval_index for candidate in self._pending}
        if set(result_by_index) != expected or len(result_by_index) != len(self._pending):
            raise ValueError("results must identify every pending eval_index exactly once")
        evaluated = []
        for candidate in sorted(self._pending, key=lambda c: c.eval_index):
            result = result_by_index[candidate.eval_index]
            if self.cache is not None and not candidate.cache_hit:
                self.cache.put(result.body_hash, result)
            individual = self.tracker.record_result(candidate.genome, candidate.parent_id,
                candidate.generation, result, body=candidate.body, cache_hit=candidate.cache_hit)
            if individual.genome_id != candidate.genome_id:
                raise AssertionError("candidate genome_id drifted from tracker order")
            evaluated.append(individual)
        if self._initial_population:
            self.population, self._initial_population = evaluated, False
        else:
            survivors = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:self.mu]
            self.population = survivors + evaluated
        self.tracker.record_generation_summary(self.generation)
        self._pending = []
        self._done = self.tracker.evaluations_used >= self.budget

    def result(self) -> RunResult:
        if not self.done:
            raise RuntimeError("result() is only available after the run is done")
        return RunResult(self.population, self.tracker.evaluations_used, self.tracker.best_fitness,
            self.tracker.records, self.cache.hit_rate if self.cache is not None else None,
            self.resolved_policy, self.tracker.generation_records)

    def get_state(self) -> dict:
        if self._pending:
            raise RuntimeError("checkpoints may only be taken between generations")
        return {
            "population": [{"genome_id": i.genome_id, "parent_id": i.parent_id,
                "genome": self.encoder.serialize_genome(i.genome), "fitness": i.fitness,
                "valid": i.valid} for i in self.population],
            "counters": {"evaluations_used": self.tracker.evaluations_used,
                "evals_valid_cumulative": self.tracker.evals_valid_cumulative,
                "evals_invalid_cumulative": self.tracker.evals_invalid_cumulative,
                "evals_cache_hit_cumulative": self.tracker.evals_cache_hit_cumulative,
                "invalid_reason_counts": copy.deepcopy(self.tracker.invalid_reason_counts),
                "best_fitness": self.tracker.best_fitness},
            "next_genome_id": self.tracker.next_genome_id, "generation": self.generation,
            "initial_population": self._initial_population, "done": self.done,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "records": copy.deepcopy(self.tracker.records),
            "generation_records": copy.deepcopy(self.tracker.generation_records),
            "cache": None if self.cache is None else {
                "entries": {key: vars(value).copy() for key, value in self.cache._store.items()},
                "hits": self.cache.hits,
                "misses": self.cache.misses,
            },
        }

    @classmethod
    def from_state(cls, encoder: Encoder, config, state: dict) -> "EvolutionRun":
        run = cls(encoder, config)
        run.population = [Individual(i["genome_id"], i["parent_id"],
            encoder.deserialize_genome(i["genome"]), i["fitness"], i["valid"])
            for i in state["population"]]
        for name, value in state["counters"].items():
            setattr(run.tracker, name, value)
        run.tracker.next_genome_id = state["next_genome_id"]
        run.generation, run._initial_population, run._done = state["generation"], state["initial_population"], state["done"]
        run.rng.bit_generator.state = state["rng_state"]
        run.tracker.records = state.get("records", [])
        run.tracker.generation_records = state.get("generation_records", [])
        cache_state = state.get("cache")
        if cache_state is not None:
            from evodevo.evaluation.ppo_runner import EvalResult
            run.cache._store = {key: EvalResult(**value) for key, value in cache_state["entries"].items()}
            run.cache.hits, run.cache.misses = cache_state["hits"], cache_state["misses"]
        return run


def run_ea(encoder: Encoder, evaluator: Callable, config, logger: Optional[RunLogger] = None,
           cache: Optional[FitnessCache] = None) -> RunResult:
    """Evaluate each complete proposed generation, then report it to the run."""
    run = EvolutionRun(encoder, config, logger=logger, cache=cache)
    while not run.done:
        candidates = run.propose()
        results = []
        by_index = {}
        for c in candidates:
            if c.cached_result is not None:
                result = c.cached_result
            elif c.cache_source_eval_index is not None:
                result = by_index[c.cache_source_eval_index]
            else:
                result = evaluator(c.body, task=config.task,
                    ppo_timesteps=config.ppo_timesteps, eval_episodes=config.ppo_eval_episodes,
                    seed=config.seed, sentinel_fitness=getattr(config, "sentinel_fitness", -10.0),
                    ppo_seeds=getattr(config, "ppo_seeds", 1),
                    ppo_seed_aggregation=getattr(config, "ppo_seed_aggregation", "max"),
                    output_dir=logger.output_dir if logger is not None else None,
                    arm=getattr(config, "encoder", ""), generation=c.generation,
                    genome_id=c.genome_id, eval_index=c.eval_index)
            # A cache hit can share the exact stored EvalResult with an
            # earlier slot; copy before attaching dispatch metadata so each
            # slot keeps its own eval_index without mutating cache contents.
            dispatched = copy.copy(result)
            dispatched.eval_index = c.eval_index
            by_index[c.eval_index] = result
            results.append(dispatched)
        run.tell(results)
    return run.result()
