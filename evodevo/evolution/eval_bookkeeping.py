"""Shared per-evaluation bookkeeping used IDENTICALLY by every search
strategy (evodevo.evolution.ea.run_ea, evodevo.evolution.random_search.
run_random_search, and any future strategy) so budget semantics, the
invalid-genome policy, the cache-charging policy, and the P6 dual counters
(evals_spent vs. evals_valid) can never drift between strategies -- this is
the single implementation both call, not two independently written copies
that happen to agree today.

This module may only call methods on the Encoder interface and the
evaluator callable it is given, exactly like evodevo.evolution.ea itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from evodevo.evaluation.cache import FitnessCache
from evodevo.evolution.logging import RunLogger
from evodevo.morphology.metrics import body_hash as compute_body_hash
from evodevo.morphology.metrics import body_metrics
from evodevo.morphology.validate import categorize_reason

# Fixed constants of the shared evaluation pipeline, never parameterised per
# encoder/config/strategy -- every (arm, strategy) cell gets exactly this
# policy, by construction. Recorded into every run's run_meta.json (see
# resolved_policy_for()) so the claim "identical across all arms and
# strategies" is auditable after the fact, not just asserted once.
CACHE_CHARGING_POLICY = "cache_hit_still_consumes_one_evaluation_slot"
BUDGET_COUNTER_SEMANTICS = (
    "evals_spent (== eval_index+1) is the sole budget-governing counter; "
    "evals_valid_cumulative tracks real (decoded-and-validated) search "
    "attempts separately and is NEVER used to gate the eval_budget loop"
)


def resolved_policy_for(config) -> dict:
    """The invalid-genome policy, cache-charging policy, and budget-counter
    semantics this run actually used. `sentinel_fitness`/`cache_enabled` are
    the only config-dependent pieces, and both are SHARED_BASE_KEYS fields
    already enforced identical across every arm by
    evodevo.experiments.fairness.assert_comparable() -- this dict is the
    second, explicit, human/test-auditable record of that same invariant,
    not a new source of truth. Identical regardless of which strategy
    (evolutionary search or random_search) consumes it."""
    sentinel_fitness = getattr(config, "sentinel_fitness", -10.0)
    return {
        "invalid_genome_policy": (
            f"an invalid genome consumes one evaluation slot and is scored "
            f"sentinel_fitness={sentinel_fitness} -- never resampled, never free"
        ),
        "cache_charging_policy": CACHE_CHARGING_POLICY,
        "budget_counter_semantics": BUDGET_COUNTER_SEMANTICS,
    }


def assert_resolved_policy_unchanged(resolved_policy: dict) -> None:
    """Run-start self-consistency guard, called by every strategy: fails
    loudly if a future edit ever makes the cache-charging policy or
    budget-counter semantics configurable per arm/strategy (they must
    remain fixed constants of this shared module)."""
    assert resolved_policy["cache_charging_policy"] == CACHE_CHARGING_POLICY
    assert resolved_policy["budget_counter_semantics"] == BUDGET_COUNTER_SEMANTICS


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
    resolved_policy: dict = field(default_factory=dict)
    generation_records: list = field(default_factory=list)


class EvaluationTracker:
    """Owns the counters and the single evaluate() entry point every
    strategy calls -- see module docstring. A strategy (ea.py,
    random_search.py) is responsible only for deciding WHICH genome to
    evaluate next (selection/mutation vs. fresh random sampling); this
    class owns HOW an evaluation is scored, cached, counted, and logged,
    identically regardless of who's calling."""

    def __init__(self, encoder, evaluator: Callable, config, cache: Optional[FitnessCache],
                 logger: Optional[RunLogger]):
        self.encoder = encoder
        self.evaluator = evaluator
        self.config = config
        self.cache = cache
        self.logger = logger

        self.evaluations_used = 0  # == evals_spent, the sole budget-governing counter
        self.evals_valid_cumulative = 0
        self.evals_invalid_cumulative = 0
        self.evals_cache_hit_cumulative = 0
        self.invalid_reason_counts: dict = {}
        self.next_genome_id = 0
        self.records: list = []
        self.generation_records: list = []
        self.best_fitness = float("-inf")

    def evaluate(self, genome: np.ndarray, parent_id: Optional[int], generation: int) -> Individual:
        genome_id = self.next_genome_id
        self.next_genome_id += 1

        body = self.encoder.decode(genome)

        cache_hit = False
        result = None
        if self.cache is not None:
            h = compute_body_hash(body)
            cached_result = self.cache.get(h)
            if cached_result is not None:
                result = cached_result
                cache_hit = True

        if result is None:
            result = self.evaluator(
                body,
                task=self.config.task,
                ppo_timesteps=self.config.ppo_timesteps,
                eval_episodes=self.config.ppo_eval_episodes,
                seed=self.config.seed,
                sentinel_fitness=getattr(self.config, "sentinel_fitness", -10.0),
            )
            if self.cache is not None:
                self.cache.put(result.body_hash, result)

        # evals_spent (evaluations_used) is incremented unconditionally --
        # this is THE budget-governing counter, identical semantics for
        # every arm/strategy: an invalid genome and a cache hit both still
        # consume a slot. The other counters are purely observational and
        # never gate any strategy's loop.
        self.evaluations_used += 1
        if result.valid:
            self.evals_valid_cumulative += 1
            reason_category = "none"
        else:
            self.evals_invalid_cumulative += 1
            reason_category = categorize_reason(result.reason)
            self.invalid_reason_counts[reason_category] = self.invalid_reason_counts.get(reason_category, 0) + 1
        if cache_hit:
            self.evals_cache_hit_cumulative += 1
        self.best_fitness = max(self.best_fitness, result.fitness)

        record = {
            "eval_index": self.evaluations_used - 1,
            "generation": generation,
            "genome_id": genome_id,
            "parent_id": parent_id,
            "fitness": result.fitness,
            "valid": result.valid,
            "reason": result.reason,
            "reason_category": reason_category,
            "body_hash": result.body_hash,
            "wall_clock_s": result.wall_clock_s,
            "cache_hit": cache_hit,
            # Budget-vs-real-attempt bookkeeping (P6/P7's whole point):
            # evals_spent is THE budget-governing counter; evals_valid_cumulative
            # is how many of those spent slots reached a real (decode-and-
            # validate-passing) candidate. best_fitness_so_far is the SAME
            # running value either way -- only the x-axis it gets plotted
            # against differs (evodevo.analysis.budget_curves).
            "evals_spent": self.evaluations_used,
            "evals_valid_cumulative": self.evals_valid_cumulative,
            "evals_invalid_cumulative": self.evals_invalid_cumulative,
            "evals_cache_hit_cumulative": self.evals_cache_hit_cumulative,
            "best_fitness_so_far": self.best_fitness,
            **body_metrics(body),
        }
        self.records.append(record)
        if self.logger is not None:
            self.logger.log(record)

        return Individual(
            genome_id=genome_id,
            parent_id=parent_id,
            genome=genome,
            fitness=result.fitness,
            valid=result.valid,
        )

    def record_generation_summary(self, generation: int) -> None:
        summary = {
            "generation": generation,
            "evals_spent": self.evaluations_used,
            "evals_valid": self.evals_valid_cumulative,
            "evals_invalid": self.evals_invalid_cumulative,
            "evals_invalid_reason_counts": dict(self.invalid_reason_counts),
            "evals_cache_hit": self.evals_cache_hit_cumulative,
            "best_fitness_so_far": self.best_fitness,
        }
        self.generation_records.append(summary)
        if self.logger is not None:
            self.logger.log_generation(summary)


def resolve_cache(config, cache: Optional[FitnessCache]) -> Optional[FitnessCache]:
    """Module-level (not the accidental method above) shared cache_enabled/
    cache-object resolution used by every strategy. A cache hit MUST still
    consume one evaluation from the budget and MUST be logged (cache_hit=
    True) -- if a hit were free, whichever arm/strategy happens to revisit
    bodies more often would gain a silent budget advantage. cache_enabled
    defaults to False for the primary comparison."""
    cache_enabled = bool(getattr(config, "cache_enabled", False))
    if cache_enabled and cache is None:
        return FitnessCache()
    if not cache_enabled:
        return None
    return cache
