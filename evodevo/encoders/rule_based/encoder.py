"""RuleBasedEncoder: the Encoder-contract adapter around the ported V2
developmental rule language (_rules/types.py / _rules/genome.py /
_rules/interpreter.py -- internal submodules of this package, ported
verbatim from a standalone reference implementation; see their module
headers and docs/decisions.md for provenance).

This file is NOT itself a port -- it is new code, written to satisfy
evodevo.encoders.base.Encoder using the ported modules unchanged. The
ported modules' behavioral equivalence to their original reference
implementation (its checkout has since been permanently removed -- see
docs/decisions.md) is
permanently proven by tests/encoders/fixtures/rule_based_port_equivalence.json.
"""
from __future__ import annotations

import random as _global_random
from typing import Any, List

import numpy as np

from evodevo.encoders.base import Encoder
from evodevo.encoders.rule_based._rules.interpreter import DEFAULT_MAX_ITERATIONS, develop_rules
from evodevo.encoders.rule_based._rules.genome import (
    MAX_RULES,
    MIN_RULES,
    DevelopmentalRuleGenome,
    crossover_developmental_rules,
    mutate_developmental_rules,
    sample_developmental_rules,
)
from evodevo.encoders.rule_based._rules.types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)

# Maps each rule-parameter key to the Enum type its value belongs to, for
# serialize_genome()/deserialize_genome(). "threshold" is a plain int and
# has no entry (left as-is).
_PARAM_ENUM_TYPES = {
    "state": VoxelState,
    "direction": Direction,
    "position": PositionType,
    "voxel_type": VoxelState,
}


def _params_from_dict(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        enum_type = _PARAM_ENUM_TYPES.get(key)
        out[key] = enum_type(value) if enum_type is not None else value
    return out


def _rule_from_dict(d: dict) -> DevelopmentalRule:
    return DevelopmentalRule(
        condition_type=ConditionType(d["condition_type"]),
        condition_params=_params_from_dict(d["condition_params"]),
        action_type=ActionType(d["action_type"]),
        action_params=_params_from_dict(d["action_params"]),
    )


def _seed_global_random(rng: np.random.Generator) -> None:
    """The ported V2 rule operators (sample/mutate/crossover) use Python's
    global `random` module internally, unchanged from the original
    implementation (see rule_encoding.py) -- porting them to consume an
    `np.random.Generator` instead would be a rewrite of the algorithm, not
    a port. To still give this adapter the same reproducibility contract
    as the other encoders (same rng seed -> same output, different seed ->
    different output), a seed is deterministically derived from the
    caller-supplied Generator and used to seed the global `random` module
    before each call. This mutates process-wide random state; the EA loop
    is single-threaded and sequential, so this is safe for this project's
    actual usage, but would not be safe under concurrent callers."""
    seed = int(rng.integers(0, 2**31 - 1))
    _global_random.seed(seed)


class RuleBasedEncoder(Encoder):
    """Symbolic developmental-rules encoding (the "Rule-Based" arm).

    Genome: a DevelopmentalRuleGenome -- a variable-length (MIN_RULES..
    MAX_RULES), order-meaningful list of DevelopmentalRule objects. Unlike
    DirectEncoder/NCAEncoder, this genome is not a fixed-shape numeric
    array (see genome_dtype).
    """

    def __init__(
        self,
        grid_shape=(5, 5),
        developmental_iterations: int = DEFAULT_MAX_ITERATIONS,
        mutation_rate: float = 0.2,
    ):
        self.grid_shape = tuple(grid_shape)
        self.developmental_iterations = developmental_iterations
        self.mutation_rate = mutation_rate

    @property
    def name(self) -> str:
        return "rule_based"

    @property
    def genome_dtype(self) -> Any:
        return object

    def genome_length(self) -> int:
        """Not meaningful for this variable-length genome (MIN_RULES..
        MAX_RULES rules); returns the maximum possible rule count as a
        nominal upper bound. Nothing in the shared pipeline (ea.py,
        ppo_runner.py) actually calls this for allocation purposes."""
        return MAX_RULES

    def random_genome(self, rng: np.random.Generator) -> DevelopmentalRuleGenome:
        _seed_global_random(rng)
        return sample_developmental_rules()

    def decode(self, genome: DevelopmentalRuleGenome) -> np.ndarray:
        """Runs the ported interpreter directly (not
        generate_robot_from_developmental_rules, which additionally
        returns None on an invalid body) -- decode() must always return a
        body and must never signal validity itself; that is the
        pipeline's job (evodevo/morphology/validate.py via
        evodevo/evaluation/ppo_runner.py)."""
        body = develop_rules(genome, shape=self.grid_shape, iterations=self.developmental_iterations)
        return np.asarray(body, dtype=int)

    def mutate(self, genome: DevelopmentalRuleGenome, rng: np.random.Generator) -> DevelopmentalRuleGenome:
        _seed_global_random(rng)
        return mutate_developmental_rules(genome, mutation_rate=self.mutation_rate)

    def develop_trajectory(self, genome: DevelopmentalRuleGenome):
        """develop_rules() is not incremental -- it always restarts from the
        single-SOFT-seed-at-centre initial grid and re-applies the genome's
        rules for exactly `iterations` steps (see interpreter.py). The
        trajectory is therefore obtained by calling it once per iteration
        count t=0..developmental_iterations, which is cheap at this grid
        size/iteration cap and keeps the interpreter itself untouched."""
        return [
            np.asarray(develop_rules(genome, shape=self.grid_shape, iterations=t), dtype=int)
            for t in range(self.developmental_iterations + 1)
        ]

    def supports_crossover(self) -> bool:
        return True

    def crossover(
        self, parent_a: DevelopmentalRuleGenome, parent_b: DevelopmentalRuleGenome, rng: np.random.Generator
    ) -> DevelopmentalRuleGenome:
        _seed_global_random(rng)
        return crossover_developmental_rules(parent_a, parent_b)

    def serialize_genome(self, genome: DevelopmentalRuleGenome) -> List[dict]:
        return [rule.to_dict() for rule in genome.rules]

    def deserialize_genome(self, obj: List[dict]) -> DevelopmentalRuleGenome:
        rules = [_rule_from_dict(d) for d in obj]
        return DevelopmentalRuleGenome(rules)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "grid_shape": list(self.grid_shape),
            "developmental_iterations": self.developmental_iterations,
            "mutation_rate": self.mutation_rate,
            "min_rules": MIN_RULES,
            "max_rules": MAX_RULES,
            "condition_types": [c.value for c in ConditionType],
            "action_types": [a.value for a in ActionType],
        }
