"""RuleBasedEncoder-SPECIFIC tests: port-equivalence against the original
(unported) reference checkout, numeric-vocabulary preservation, and
adapter-level behaviour (serialization, crossover, EA integration).

Generic Encoder-contract properties are covered once for every registered
encoder by tests/encoders/test_contract.py and are not repeated here.
"""
from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.encoders import get_encoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.encoders.rule_based.interpreter import (
    DEFAULT_MAX_ITERATIONS as PORTED_DEFAULT_ITERATIONS,
    STATE_TO_VOXEL as PORTED_STATE_TO_VOXEL,
    develop_rules as ported_develop_rules,
)
from evodevo.encoders.rule_based.rule_encoding import (
    MAX_RULES,
    MIN_RULES,
    DevelopmentalRuleGenome,
    crossover_developmental_rules,
    mutate_developmental_rules,
    sample_developmental_rules as ported_sample_developmental_rules,
)
from evodevo.encoders.rule_based.rule_types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    PositionType,
    VoxelState,
)
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea

REFERENCE_EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "external" / "rule-based-reference" / "examples"
)


def _load_original_modules():
    """Import the ORIGINAL (unported) V2 developmental-rule modules
    directly from external/rule-based-reference by file path, so the
    port-equivalence test can compare against genuinely unmodified
    reference code (not a copy that could itself have drifted).

    Returns (rule_types, interpreter, rule_encoding) module objects, or
    None if the reference checkout is missing or fails to import (e.g. its
    own dependency chain is broken) -- in which case the equivalence test
    is skipped with a loud, explicit reason rather than silently omitted.
    """
    if not REFERENCE_EXAMPLES_ROOT.is_dir():
        return None

    already_on_path = str(REFERENCE_EXAMPLES_ROOT) in sys.path
    if not already_on_path:
        sys.path.insert(0, str(REFERENCE_EXAMPLES_ROOT))
    try:
        rule_types = importlib.import_module("utils.developmental_rule_types")
        interpreter = importlib.import_module("utils.developmental_interpreter")
        rule_encoding = importlib.import_module("utils.developmental_rule_encoding")
        return rule_types, interpreter, rule_encoding
    except Exception:
        return None
    finally:
        if not already_on_path and str(REFERENCE_EXAMPLES_ROOT) in sys.path:
            sys.path.remove(str(REFERENCE_EXAMPLES_ROOT))


_ORIGINAL = _load_original_modules()

_skip_if_no_reference = pytest.mark.skipif(
    _ORIGINAL is None,
    reason=(
        "external/rule-based-reference/examples/utils/developmental_*.py could not be "
        "imported (reference checkout missing, or its import chain failed in this "
        "environment). SKIPPING the port-equivalence test means the port is UNVERIFIED "
        "against the original implementation for this test run -- this is a loud, "
        "explicit skip, not a silent pass."
    ),
)


# ---------------------------------------------------------------------------
# The single most important test in this file: proves the port preserves
# behaviour by running the PORTED interpreter and the ORIGINAL interpreter on
# the same 20 seeded genomes and asserting identical bodies -- modulo one
# deliberate, documented exception: SOFT/HARD are corrected to match
# EvoGym's real material vocabulary (see interpreter.py's module-level
# comment and docs/decisions.md). _correct_original_soft_hard_swap() undoes
# exactly that one change before comparing, so this test proves the SOFT/HARD
# fix is the *only* behavioural difference -- everything else (rule
# matching, growth propagation, synchronous update order) is unchanged.
# ---------------------------------------------------------------------------


def _correct_original_soft_hard_swap(body: np.ndarray) -> np.ndarray:
    """Remap an ORIGINAL-interpreter body's SOFT(1)/HARD(2) codes to this
    port's corrected RIGID(1)/SOFT(2) convention, leaving EMPTY/H_ACT/V_ACT
    untouched. Used only to prove equivalence in these tests."""
    corrected = np.asarray(body).copy()
    original = np.asarray(body)
    corrected[original == 1] = 2  # original SOFT -> corrected SOFT (core.materials.SOFT == 2)
    corrected[original == 2] = 1  # original HARD -> corrected HARD/RIGID (core.materials.RIGID == 1)
    return corrected


@_skip_if_no_reference
def test_ported_interpreter_matches_original_across_20_seeded_genomes():
    _orig_rule_types, orig_interpreter, orig_rule_encoding = _ORIGINAL

    for seed in range(20):
        # Both the ported and original sample_developmental_rules() consume
        # Python's global `random` module (a single process-wide singleton,
        # shared by both modules) in exactly the same call sequence, since
        # the port is verbatim. Reseeding identically before each call makes
        # the two genomes structurally equivalent even though they are
        # instances of two independent (ported vs. original) class
        # hierarchies.
        random.seed(seed)
        original_genome = orig_rule_encoding.sample_developmental_rules()
        random.seed(seed)
        ported_genome = ported_sample_developmental_rules()

        assert len(original_genome.rules) == len(ported_genome.rules), (
            f"seed {seed}: genome rule counts diverged between original and ported sampling"
        )

        original_body = orig_interpreter.develop_rules(
            original_genome, shape=(5, 5), iterations=orig_interpreter.DEFAULT_MAX_ITERATIONS
        )
        ported_body = ported_develop_rules(
            ported_genome, shape=(5, 5), iterations=PORTED_DEFAULT_ITERATIONS
        )

        np.testing.assert_array_equal(
            _correct_original_soft_hard_swap(original_body),
            np.asarray(ported_body),
            err_msg=(
                f"seed {seed}: ported interpreter's output diverged from the original "
                "by more than the documented SOFT/HARD fix"
            ),
        )


@_skip_if_no_reference
def test_ported_mutation_matches_original_across_5_seeded_genomes():
    _orig_rule_types, orig_interpreter, orig_rule_encoding = _ORIGINAL

    for seed in range(5):
        random.seed(seed)
        original_parent = orig_rule_encoding.sample_developmental_rules()
        random.seed(seed)
        ported_parent = ported_sample_developmental_rules()

        random.seed(seed + 1000)
        original_child = orig_rule_encoding.mutate_developmental_rules(original_parent)
        random.seed(seed + 1000)
        ported_child = mutate_developmental_rules(ported_parent)

        original_body = orig_interpreter.develop_rules(original_child, shape=(5, 5))
        ported_body = ported_develop_rules(ported_child, shape=(5, 5))
        np.testing.assert_array_equal(
            _correct_original_soft_hard_swap(original_body), np.asarray(ported_body)
        )


# ---------------------------------------------------------------------------
# Numeric-vocabulary preservation: does NOT require the reference checkout,
# since the original commit's values (documented in the port's header
# comments) are known and pinned here explicitly.
# ---------------------------------------------------------------------------


def test_state_to_voxel_matches_evogym_and_core_materials():
    # Original developmental_interpreter.py (external/rule-based-reference,
    # commit 703e711) defined EMPTY=0, SOFT=1, HARD=2, HORIZONTAL=3,
    # VERTICAL=4 -- which did NOT match EvoGym's real material vocabulary
    # (evogym.utils.VOXEL_TYPES: EMPTY=0, RIGID=1, SOFT=2, H_ACT=3, V_ACT=4)
    # and was a pre-existing bug (see interpreter.py's module comment and
    # docs/decisions.md). This port deliberately corrects SOFT/HARD to
    # match EvoGym/core.materials; EMPTY/HORIZONTAL/VERTICAL are unchanged.
    assert PORTED_STATE_TO_VOXEL[VoxelState.EMPTY] == 0
    assert PORTED_STATE_TO_VOXEL[VoxelState.SOFT] == 2
    assert PORTED_STATE_TO_VOXEL[VoxelState.HARD] == 1
    assert PORTED_STATE_TO_VOXEL[VoxelState.ACTUATOR_HORIZONTAL] == 3
    assert PORTED_STATE_TO_VOXEL[VoxelState.ACTUATOR_VERTICAL] == 4


@_skip_if_no_reference
def test_state_to_voxel_deliberately_fixes_soft_hard_swap_from_original():
    _orig_rule_types, orig_interpreter, _orig_rule_encoding = _ORIGINAL
    original_mapping = {state.value: code for state, code in orig_interpreter.STATE_TO_VOXEL.items()}
    ported_mapping = {state.value: code for state, code in PORTED_STATE_TO_VOXEL.items()}

    # EMPTY/HORIZONTAL/VERTICAL are unchanged from the original.
    for key in ("empty", "actuator_horizontal", "actuator_vertical"):
        assert original_mapping[key] == ported_mapping[key], key

    # SOFT and HARD are DELIBERATELY swapped relative to the original, to
    # match EvoGym's actual material vocabulary instead of preserving the
    # original's own inverted convention.
    assert original_mapping["soft"] == 1 and ported_mapping["soft"] == 2
    assert original_mapping["hard"] == 2 and ported_mapping["hard"] == 1


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
    encoder = RuleBasedEncoder()
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
    encoder = get_encoder("rule_based")
    assert isinstance(encoder, RuleBasedEncoder)
    assert encoder.name == "rule_based"
    assert encoder.supports_crossover() is True


def test_crossover_produces_a_valid_length_child_and_does_not_mutate_parents():
    encoder = RuleBasedEncoder()
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
    encoder = RuleBasedEncoder()
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
    encoder = RuleBasedEncoder()
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
    encoder = RuleBasedEncoder()
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
