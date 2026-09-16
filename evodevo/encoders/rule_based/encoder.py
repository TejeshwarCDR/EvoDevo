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

import dataclasses

import random as _global_random
from typing import Any, List

import numpy as np

from evodevo.analysis.windowed_readout import windowed_body_at
from evodevo.core.factors import ExperimentFactors
from evodevo.core.initial_conditions import get_initial_condition
from evodevo.encoders.base import Encoder
from evodevo.encoders.rule_based._rules.interpreter import (
    DEFAULT_CONFLICT_RESOLUTION,
    VALID_ACTION_MONOTONICITY,
    VALID_BOUNDARIES,
    VALID_CONFLICT_RESOLUTIONS,
    VALID_COPY_NEIGHBOR_READS,
    VALID_NEIGHBOURHOODS,
    develop_rules,
)
from evodevo.encoders.rule_based._rules.genome import (
    MAX_RULES,
    MIN_RULES,
    VALID_POSITION_PREDICATES,
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
        factors: ExperimentFactors,
        grid_shape=(5, 5),
        mutation_rate: float = 0.2,
        readout_mode: str = "fixed_t",
        readout_window: int = 5,
        conflict_resolution: str = DEFAULT_CONFLICT_RESOLUTION,
        copy_neighbor_reads: str = "pre_iteration",
        position_predicate: str = "disabled",
        boundary: str = "no_op",
        neighbourhood: str = "von_neumann",
        action_monotonicity: str = "unrestricted",
    ):
        self.grid_shape = tuple(grid_shape)
        self.factors = factors
        self.developmental_iterations = factors.dev_steps
        self._initial_condition = get_initial_condition(factors.init_condition)
        # factors.mutation_scale multiplies this arm's own mutation
        # probability -- it does not change the mutation operator itself
        # (still mutate_developmental_rules' per-rule Bernoulli flip).
        self.mutation_rate = min(1.0, mutation_rate * factors.mutation_scale)
        # factors.neighbourhood is NCA-specific (it crosses NCAEncoder's
        # perception axis as an experimental factor) and has no meaning for
        # this arm -- rule_based's OWN neighbourhood axis (below) is a
        # constructor parameter, not driven by `factors.neighbourhood`, for
        # the same reason readout_mode/conflict_resolution are constructor
        # parameters rather than factors: it is a per-arm architecture
        # choice, not a shared cross-arm experimental factor.
        # factors.hidden_channels remains NCA-specific and has no meaning
        # here (explicit no-op).
        if readout_mode not in ("fixed_t", "mode_of_window"):
            raise ValueError(f"unsupported readout_mode '{readout_mode}'; must be one of ('fixed_t', 'mode_of_window')")
        if readout_window < 1:
            raise ValueError(f"readout_window must be >= 1, got {readout_window}")
        # Class default stays "fixed_t" -- unlike NCAEncoder, whose class
        # default was flipped to mode_of_window (its own T_primary is
        # comparable to its resolved W=50, so shallow-T callers were never
        # a risk the same way). rule_based's own resolved (T_primary, W) is
        # threaded explicitly by whichever caller constructs "the primary
        # rule_based arm" -- see results/T_PRIMARY_UNIFIED.md and
        # evodevo/encoders/nca/resolved_config.py's sibling wiring pattern.
        self.readout_mode = readout_mode
        self.readout_window = readout_window
        if conflict_resolution not in VALID_CONFLICT_RESOLUTIONS:
            raise ValueError(f"unsupported conflict_resolution '{conflict_resolution}'; must be one of {VALID_CONFLICT_RESOLUTIONS}")
        if copy_neighbor_reads not in VALID_COPY_NEIGHBOR_READS:
            raise ValueError(f"unsupported copy_neighbor_reads '{copy_neighbor_reads}'; must be one of {VALID_COPY_NEIGHBOR_READS}")
        # Class default is the interpreter's own DEFAULT_CONFLICT_RESOLUTION
        # -- chosen on evidence, not in advance, see
        # results/rule_based/CONFLICT_POLICY.md. "row_major_last_wins" (the
        # historical, order-DEPENDENT behaviour) remains fully selectable
        # and is what the permanently-frozen port-equivalence fixture test
        # explicitly pins itself to (tests/encoders/test_rule_based.py),
        # decoupling that permanent proof from any future default change.
        self.conflict_resolution = conflict_resolution
        self.copy_neighbor_reads = copy_neighbor_reads

        if position_predicate not in VALID_POSITION_PREDICATES:
            raise ValueError(f"unsupported position_predicate '{position_predicate}'; must be one of {VALID_POSITION_PREDICATES}")
        if boundary not in VALID_BOUNDARIES:
            raise ValueError(f"unsupported boundary '{boundary}'; must be one of {VALID_BOUNDARIES}")
        if neighbourhood not in VALID_NEIGHBOURHOODS:
            raise ValueError(f"unsupported neighbourhood '{neighbourhood}'; must be one of {VALID_NEIGHBOURHOODS}")
        # Class defaults restore Proposition R3 (translation-equivariance)
        # and bring this arm to parity with NCAEncoder's own boundary/
        # neighbourhood axes -- see docs/rule-language-spec.md and
        # results/rule_based/LOCALITY_STUDY.md. `position_predicate=
        # "disabled"` means the sampler/mutator never GENERATE a new
        # POSITION condition (a genome that already has one, e.g.
        # deserialized from an old result, still decodes correctly --
        # POSITION remains fully evaluable by the interpreter regardless of
        # this flag, only its GENERATION is gated). `boundary`/
        # `neighbourhood` default to the interpreter's entire historical
        # behaviour ("no_op"/"von_neumann").
        self.position_predicate = position_predicate
        self.boundary = boundary
        self.neighbourhood = neighbourhood

        if action_monotonicity not in VALID_ACTION_MONOTONICITY:
            raise ValueError(f"unsupported action_monotonicity '{action_monotonicity}'; must be one of {VALID_ACTION_MONOTONICITY}")
        # Class default "unrestricted" -- "monotone" is an ABLATION (the
        # rule-based counterpart of NCAEncoder's spectral_normalise
        # certified ablation), NOT the primary, per
        # results/rule_based/MONOTONE_ABLATION.md's measured expressivity
        # cost. Selected explicitly by whichever caller wants the
        # termination guarantee (Proposition R4).
        self.action_monotonicity = action_monotonicity

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
        return sample_developmental_rules(position_predicate=self.position_predicate)

    def decode(self, genome: DevelopmentalRuleGenome) -> np.ndarray:
        """Runs the ported interpreter directly (not
        generate_robot_from_developmental_rules, which additionally
        returns None on an invalid body) -- decode() must always return a
        body and must never signal validity itself; that is the
        pipeline's job (evodevo/morphology/validate.py via
        evodevo/evaluation/ppo_runner.py).

        Two readout policies (`self.readout_mode`), both selectable, the
        SAME semantics as NCAEncoder's own (see its decode() docstring and
        results/rule_based/SETTLING_AUDIT.md / results/T_PRIMARY_UNIFIED.md):
          "fixed_t" (default): body = develop_rules() at iterations=T. For
              rule_based's confirmed-genuine cycling population (~11.55%,
              exact rule-cycles, mostly period-2), the returned body depends
              on the phase T happens to land on -- an arbitrary,
              T-shift-sensitive answer for that subpopulation.
          "mode_of_window": body = the most frequent WHOLE-GRID configuration
              among the last `self.readout_window` (=W) steps of the
              trajectory (t = T-W+1 .. T; fewer if T < W), via the SAME
              shared `evodevo.analysis.windowed_readout.mode_of_window`
              utility NCA's encoder uses (deterministic lowest-material-ID
              tie-break on exact ties, no RNG). For genomes that settle
              cleanly, every snapshot in the window is identical and this
              returns exactly the same body as fixed_t (a strict
              generalisation, verified in
              tests/encoders/test_rule_based_readout_policy.py).
        """
        if self.readout_mode == "fixed_t":
            initial_grid = self._initial_condition.as_discrete_grid(self.grid_shape)
            body = develop_rules(
                genome,
                shape=self.grid_shape,
                iterations=self.developmental_iterations,
                initial_grid=initial_grid,
                conflict_resolution=self.conflict_resolution,
                copy_neighbor_reads=self.copy_neighbor_reads,
                boundary=self.boundary,
                neighbourhood=self.neighbourhood,
                action_monotonicity=self.action_monotonicity,
            )
            return np.asarray(body, dtype=int)
        elif self.readout_mode == "mode_of_window":
            trajectory = self.develop_trajectory(genome)
            return windowed_body_at(
                trajectory, self.developmental_iterations, self.readout_window,
                validity_aware=True, grid_shape=self.grid_shape,
            )
        raise ValueError(f"unsupported readout_mode '{self.readout_mode}'")  # pragma: no cover -- validated in __init__

    def mutate(self, genome: DevelopmentalRuleGenome, rng: np.random.Generator) -> DevelopmentalRuleGenome:
        _seed_global_random(rng)
        return mutate_developmental_rules(
            genome, mutation_rate=self.mutation_rate, position_predicate=self.position_predicate,
        )

    def develop_trajectory(self, genome: DevelopmentalRuleGenome):
        """develop_rules() is not incremental -- it always restarts from the
        single-SOFT-seed-at-centre initial grid and re-applies the genome's
        rules for exactly `iterations` steps (see interpreter.py). The
        trajectory is therefore obtained by calling it once per iteration
        count t=0..developmental_iterations, which is cheap at this grid
        size/iteration cap and keeps the interpreter itself untouched."""
        initial_grid = self._initial_condition.as_discrete_grid(self.grid_shape)
        return [
            np.asarray(
                develop_rules(
                    genome, shape=self.grid_shape, iterations=t, initial_grid=initial_grid,
                    conflict_resolution=self.conflict_resolution, copy_neighbor_reads=self.copy_neighbor_reads,
                    boundary=self.boundary, neighbourhood=self.neighbourhood,
                    action_monotonicity=self.action_monotonicity,
                ),
                dtype=int,
            )
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
            "factors": dataclasses.asdict(self.factors),
            "grid_shape": list(self.grid_shape),
            "developmental_iterations": self.developmental_iterations,
            "init_condition": self.factors.init_condition,
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.factors.mutation_scale,
            "min_rules": MIN_RULES,
            "max_rules": MAX_RULES,
            "condition_types": [c.value for c in ConditionType],
            "action_types": [a.value for a in ActionType],
            "readout_mode": self.readout_mode,
            "readout_window": self.readout_window,
            "conflict_resolution": self.conflict_resolution,
            "copy_neighbor_reads": self.copy_neighbor_reads,
            "position_predicate": self.position_predicate,
            "boundary": self.boundary,
            "neighbourhood": self.neighbourhood,
            "action_monotonicity": self.action_monotonicity,
        }
