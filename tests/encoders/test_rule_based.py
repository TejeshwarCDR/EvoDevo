"""RuleBasedEncoder-SPECIFIC tests: port-equivalence against a frozen
fixture (see fixtures/rule_based_port_equivalence.json), numeric-vocabulary
preservation, and adapter-level behaviour (serialization, crossover, EA
integration).

Generic Encoder-contract properties are covered once for every registered
encoder by tests/encoders/test_contract.py and are not repeated here.

PORT-EQUIVALENCE PROVENANCE: this project originally ported the rule-based
developmental encoder from a standalone reference implementation, checked
out (with its own nested .git) at external/rule-based-reference/. That
checkout has since been permanently removed (see docs/decisions.md); its
behavioural equivalence to this port was proven ONE TIME, while both
implementations were still on disk, by running the ORIGINAL and PORTED
interpreters on the same 20 sampled genomes (plus 5 mutated genomes) and
asserting identical output bodies -- modulo one deliberate, documented
exception: SOFT/HARD are corrected to match EvoGym's real material
vocabulary (see interpreter.py's module-level comment). That comparison's
inputs and outputs were frozen verbatim into
fixtures/rule_based_port_equivalence.json, which is now the PERMANENT proof
of port equivalence: the tests below replay the ported interpreter against
those frozen values, not against a live external checkout, and can no
longer be regenerated or re-verified against the original source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.encoders.rule_based._rules.interpreter import (
    STATE_TO_VOXEL as PORTED_STATE_TO_VOXEL,
    develop_rules as ported_develop_rules,
)
from evodevo.encoders.rule_based._rules.genome import (
    MAX_RULES,
    MIN_RULES,
    DevelopmentalRuleGenome,
)
from evodevo.encoders.rule_based._rules.types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "rule_based_port_equivalence.json"


def _load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


_FIXTURE = _load_fixture()


def _factors(**overrides):
    """Reproduces the rule-based encoder's pre-factors historical default
    (5 iterations, single-SOFT-seed-at-centre) unless overridden."""
    params = dict(dev_steps=5, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


# ---------------------------------------------------------------------------
# The single most important test in this file: proves the ported
# interpreter, run TODAY, still reproduces the exact bodies frozen at the
# time the original reference checkout was removed -- i.e. that the port's
# behaviour has not drifted since equivalence was proven against the
# original. See the module docstring for the full provenance chain.
# ---------------------------------------------------------------------------


def test_ported_interpreter_matches_frozen_port_equivalence_fixture():
    # Pinned explicitly to conflict_resolution="row_major_last_wins" -- the
    # fixture was frozen against the ORIGINAL reference implementation's
    # growth-conflict behaviour (order-dependent, last-write-wins), which is
    # no longer RuleBasedEncoder's own default (see
    # evodevo/encoders/rule_based/_rules/interpreter.py::
    # DEFAULT_CONFLICT_RESOLUTION and results/rule_based/CONFLICT_POLICY.md)
    # -- this permanent proof must stay decoupled from any future default
    # change, exactly like every other frozen-fixture/pinned-setting test in
    # this project.
    encoder = RuleBasedEncoder(factors=_factors(), conflict_resolution="row_major_last_wins")
    for entry in _FIXTURE["genomes_20"]:
        genome = encoder.deserialize_genome(entry["genome"])
        body = encoder.decode(genome)
        np.testing.assert_array_equal(
            np.asarray(entry["expected_body"]),
            np.asarray(body),
            err_msg=(
                f"seed {entry['seed']}: ported interpreter's output has drifted from "
                "the frozen port-equivalence fixture (permanent proof of equivalence "
                "with the removed external/rule-based-reference checkout)"
            ),
        )


def test_ported_mutation_matches_frozen_port_equivalence_fixture():
    # Pinned to row_major_last_wins -- see the sibling test above.
    encoder = RuleBasedEncoder(factors=_factors(), conflict_resolution="row_major_last_wins")
    for entry in _FIXTURE["mutation_5"]:
        child = encoder.deserialize_genome(entry["child_genome"])
        body = encoder.decode(child)
        np.testing.assert_array_equal(
            np.asarray(entry["expected_body"]),
            np.asarray(body),
            err_msg=f"seed {entry['seed']}: mutated-genome output has drifted from the frozen fixture",
        )


# ---------------------------------------------------------------------------
# Numeric-vocabulary preservation: pinned explicitly, since the original
# commit's values (documented in the port's header comments and this
# fixture's frozen header) are known and no longer re-derivable from a live
# checkout.
# ---------------------------------------------------------------------------


def test_state_to_voxel_matches_evogym_and_core_materials():
    # Original developmental_interpreter.py (formerly
    # external/rule-based-reference, commit 703e711 -- see
    # fixtures/rule_based_port_equivalence.json's header) defined EMPTY=0,
    # SOFT=1, HARD=2, HORIZONTAL=3, VERTICAL=4 -- which did NOT match
    # EvoGym's real material vocabulary (evogym.utils.VOXEL_TYPES: EMPTY=0,
    # RIGID=1, SOFT=2, H_ACT=3, V_ACT=4) and was a pre-existing bug (see
    # interpreter.py's module comment and docs/decisions.md). This port
    # deliberately corrects SOFT/HARD to match EvoGym/core.materials;
    # EMPTY/HORIZONTAL/VERTICAL are unchanged.
    assert PORTED_STATE_TO_VOXEL[VoxelState.EMPTY] == 0
    assert PORTED_STATE_TO_VOXEL[VoxelState.SOFT] == 2
    assert PORTED_STATE_TO_VOXEL[VoxelState.HARD] == 1
    assert PORTED_STATE_TO_VOXEL[VoxelState.ACTUATOR_HORIZONTAL] == 3
    assert PORTED_STATE_TO_VOXEL[VoxelState.ACTUATOR_VERTICAL] == 4


# ---------------------------------------------------------------------------
# Serialization round-trip covering every ConditionType and ActionType.
# ---------------------------------------------------------------------------


def _one_rule_per_type() -> DevelopmentalRuleGenome:
    rules = [
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
        DevelopmentalRule(
            ConditionType.CELL_TYPE, {"state": VoxelState.SOFT}, ActionType.SET_SOFT, {}
        ),
        DevelopmentalRule(
            ConditionType.NEIGHBOR_COUNT, {"threshold": 2}, ActionType.SET_HARD, {}
        ),
        DevelopmentalRule(
            ConditionType.EMPTY_NEIGHBOR,
            {"direction": Direction.UP},
            ActionType.SET_ACTUATOR_HORIZONTAL,
            {},
        ),
        DevelopmentalRule(
            ConditionType.POSITION,
            {"position": PositionType.BOUNDARY},
            ActionType.SET_ACTUATOR_VERTICAL,
            {},
        ),
        DevelopmentalRule(
            ConditionType.ALWAYS,
            {},
            ActionType.GROW_DIRECTIONAL,
            {"direction": Direction.RIGHT, "voxel_type": VoxelState.HARD},
        ),
        DevelopmentalRule(
            ConditionType.ALWAYS,
            {},
            ActionType.COPY_NEIGHBOR,
            {"direction": Direction.LEFT},
        ),
    ]
    assert len(rules) <= MAX_RULES
    assert len(rules) >= MIN_RULES
    return DevelopmentalRuleGenome(rules)


def test_serialize_deserialize_roundtrip_covers_every_rule_and_action_type():
    encoder = RuleBasedEncoder(factors=_factors())
    genome = _one_rule_per_type()

    # every ConditionType and ActionType member is exercised at least once.
    used_conditions = {rule.condition_type for rule in genome.rules}
    used_actions = {rule.action_type for rule in genome.rules}
    assert used_conditions == set(ConditionType)
    assert used_actions == set(ActionType)

    serialized = encoder.serialize_genome(genome)
    import json

    json.dumps(serialized)  # must be JSON-serialisable
    roundtripped = encoder.deserialize_genome(serialized)
    assert roundtripped == genome

    # decode() must still produce a body after a round trip.
    body = encoder.decode(roundtripped)
    assert body.shape == (5, 5)
    assert np.issubdtype(body.dtype, np.integer)


# ---------------------------------------------------------------------------
# Adapter-level behaviour: crossover, registry, decode-never-signals-validity,
# and EA integration.
# ---------------------------------------------------------------------------


def test_supports_crossover_and_registered_as_rule_based():
    encoder = get_encoder("rule_based", factors=_factors())
    assert isinstance(encoder, RuleBasedEncoder)
    assert encoder.name == "rule_based"
    assert encoder.supports_crossover() is True


def test_crossover_produces_a_valid_length_child_and_does_not_mutate_parents():
    encoder = RuleBasedEncoder(factors=_factors())
    rng_a, rng_b, rng_c = (np.random.default_rng(s) for s in (1, 2, 3))
    parent_a = encoder.random_genome(rng_a)
    parent_b = encoder.random_genome(rng_b)
    parent_a_before = [r.copy() for r in parent_a.rules]
    parent_b_before = [r.copy() for r in parent_b.rules]

    child = encoder.crossover(parent_a, parent_b, rng_c)

    assert MIN_RULES <= len(child.rules) <= MAX_RULES
    assert [r for r in parent_a.rules] == parent_a_before
    assert [r for r in parent_b.rules] == parent_b_before


def test_crossover_without_supports_crossover_check_would_be_a_pipeline_error():
    # supports_crossover() is the flag ea.py's use_crossover gate checks;
    # crossover() itself works regardless (see the base class default for
    # encoders that don't support it, which raises instead).
    encoder = RuleBasedEncoder(factors=_factors())
    assert encoder.supports_crossover() is True
    rng = np.random.default_rng(0)
    genome = encoder.random_genome(rng)
    child = encoder.crossover(genome, genome, np.random.default_rng(1))
    assert MIN_RULES <= len(child.rules) <= MAX_RULES


def test_decode_never_returns_none_even_for_an_invalid_body():
    # A genome of all-SET_EMPTY rules develops into (almost) nothing but the
    # single SOFT seed voxel -- invalid (too few actuators / disconnected
    # once seed is isolated -- in this case just the seed cell survives) but
    # decode() must still return an array, never None, unlike
    # generate_robot_from_developmental_rules().
    encoder = RuleBasedEncoder(factors=_factors())
    genome = DevelopmentalRuleGenome(
        [
            DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
            DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_EMPTY, {}),
        ]
    )
    body = encoder.decode(genome)
    assert isinstance(body, np.ndarray)
    assert body.shape == (5, 5)


def test_ea_runs_rule_based_encoder_in_surrogate_mode_to_exact_budget():
    encoder = RuleBasedEncoder(factors=_factors())
    config = ExperimentConfig(
        seed=0,
        encoder="rule_based",
        task="dummy-task",
        population_size=4,
        offspring_size=4,
        eval_budget=15,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 15
    assert len(result.records) == 15
