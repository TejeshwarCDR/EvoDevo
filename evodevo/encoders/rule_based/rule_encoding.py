# Ported from external/rule-based-reference/examples/utils/developmental_rule_encoding.py
# (commit 703e711, "Initial project implementation"). Original author: the
# project's Rule-Based teammate. This is a port, not a rewrite: the only
# changes from the original are import paths (utils.* -> this project's
# evodevo.encoders.rule_based.*). The genome representation, sampling,
# mutation operators (rate, structural operator set), crossover, and the
# MIN_RULES/MAX_RULES/MIN_ACTUATORS constants are all unchanged.
"""
V2 developmental genome representation, mutation, crossover, and
EvoGym integration (Step 13).

    DevelopmentalRule       = condition (+ params) -> action (+ params)  (rule_types.py)
    DevelopmentalRuleGenome = ordered collection of DevelopmentalRules (this module)
    develop_rules()         = deterministic, bounded-iteration synchronous
                               interpreter (interpreter.py)

This module is fully independent of V1 (not ported here, since V1 is a
superseded earlier iteration).
"""

import random
from typing import List

from evodevo.morphology.connectivity import get_full_connectivity
from evodevo.morphology.validate import has_actuator, is_connected

from evodevo.encoders.rule_based.rule_types import (
    ActionType,
    ConditionType,
    Direction,
    DevelopmentalRule,
    GROWABLE_STATES,
    MIN_NEIGHBOR_THRESHOLD,
    MAX_NEIGHBOR_THRESHOLD,
    PositionType,
    VoxelState,
)
from evodevo.encoders.rule_based.interpreter import develop_rules, DEFAULT_MAX_ITERATIONS, HORIZONTAL, VERTICAL

# Bounds on genome size.
MIN_RULES = 2
MAX_RULES = 12

# Project constraint (same as V1): a valid robot must have at least this
# many actuators.
MIN_ACTUATORS = 2


class DevelopmentalRuleGenome:
    """
    A V2 genome: an ordered list of DevelopmentalRule objects. Rule order
    is meaningful (first-match semantics, see interpreter.py). Independent
    from V1's RuleGenome — not a subclass, no shared state.
    """

    def __init__(self, rules: List[DevelopmentalRule]):
        if not (MIN_RULES <= len(rules) <= MAX_RULES):
            raise ValueError(
                f"DevelopmentalRuleGenome must contain between {MIN_RULES} and "
                f"{MAX_RULES} rules, got {len(rules)}"
            )
        for rule in rules:
            if not isinstance(rule, DevelopmentalRule):
                raise ValueError(f"All genome entries must be DevelopmentalRule instances, got {rule!r}")

        self.rules: List[DevelopmentalRule] = list(rules)

    def copy(self) -> "DevelopmentalRuleGenome":
        return DevelopmentalRuleGenome([rule.copy() for rule in self.rules])

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DevelopmentalRuleGenome):
            return NotImplemented
        return self.rules == other.rules

    def __repr__(self) -> str:
        rule_lines = "\n".join(f"    {rule!r}" for rule in self.rules)
        return f"DevelopmentalRuleGenome(\n{rule_lines}\n)"


# --------------------------------------------------------------------------
# Random construction helpers (used by both sample_developmental_rules and
# the mutation operators below).
# --------------------------------------------------------------------------

def _random_condition(condition_type=None):
    if condition_type is None:
        condition_type = random.choice(list(ConditionType))

    if condition_type == ConditionType.ALWAYS:
        params = {}
    elif condition_type == ConditionType.CELL_TYPE:
        params = {"state": random.choice(list(VoxelState))}
    elif condition_type == ConditionType.NEIGHBOR_COUNT:
        params = {"threshold": random.randint(MIN_NEIGHBOR_THRESHOLD, MAX_NEIGHBOR_THRESHOLD)}
    elif condition_type == ConditionType.EMPTY_NEIGHBOR:
        params = {"direction": random.choice(list(Direction))}
    elif condition_type == ConditionType.POSITION:
        params = {"position": random.choice(list(PositionType))}
    else:
        raise ValueError(f"Unsupported condition_type: {condition_type}")

    return condition_type, params


def _random_action(action_type=None, growth_biased=False):
    if action_type is None:
        if growth_biased:
            # Bias toward growth-capable actions so sampled genomes are
            # meaningfully developmental rather than static SET_* only
            # (Part 15's explicit requirement).
            action_type = random.choices(
                list(ActionType),
                weights=[1, 1, 1, 1, 1, 4, 2],  # index order matches ActionType enum declaration
                k=1,
            )[0]
        else:
            action_type = random.choice(list(ActionType))

    if action_type in (
        ActionType.SET_EMPTY, ActionType.SET_SOFT, ActionType.SET_HARD,
        ActionType.SET_ACTUATOR_HORIZONTAL, ActionType.SET_ACTUATOR_VERTICAL,
    ):
        params = {}
    elif action_type == ActionType.GROW_DIRECTIONAL:
        params = {"direction": random.choice(list(Direction)), "voxel_type": random.choice(GROWABLE_STATES)}
    elif action_type == ActionType.COPY_NEIGHBOR:
        params = {"direction": random.choice(list(Direction))}
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    return action_type, params


def _random_rule(growth_biased=False) -> DevelopmentalRule:
    condition_type, condition_params = _random_condition()
    action_type, action_params = _random_action(growth_biased=growth_biased)
    return DevelopmentalRule(condition_type, condition_params, action_type, action_params)


# Rule-count range for sample_developmental_rules() — modest, well inside
# [MIN_RULES, MAX_RULES].
_SAMPLE_MIN_RULES = 3
_SAMPLE_MAX_RULES = 7


def sample_developmental_rules() -> DevelopmentalRuleGenome:
    """
    Generate a random, valid initial DevelopmentalRuleGenome, biased
    toward growth-capable rules (GROW_DIRECTIONAL/COPY_NEIGHBOR) rather
    than only static SET_* rules, per Part 15.

    Construction: one guaranteed CELL_TYPE(SOFT)->GROW_DIRECTIONAL rule
    (so the seed cell can always attempt to grow at least once), one
    guaranteed EMPTY_NEIGHBOR->GROW_DIRECTIONAL rule (so growth can
    propagate along already-grown material), then further randomly
    generated rules (condition type fully random; action type weighted
    4:2:1 toward GROW_DIRECTIONAL : COPY_NEIGHBOR : each SET_* action) up
    to a randomly chosen total length in [_SAMPLE_MIN_RULES, _SAMPLE_MAX_RULES].
    The full list is shuffled before being returned so rule order also
    varies (first-match semantics make order meaningful).
    """

    rule_count = random.randint(_SAMPLE_MIN_RULES, _SAMPLE_MAX_RULES)

    seed_growth_direction = random.choice(list(Direction))
    seed_growth_voxel = random.choice(GROWABLE_STATES)
    propagate_direction = random.choice(list(Direction))
    propagate_voxel = random.choice(GROWABLE_STATES)

    rules = [
        DevelopmentalRule(
            ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
            ActionType.GROW_DIRECTIONAL, {"direction": seed_growth_direction, "voxel_type": seed_growth_voxel},
        ),
        DevelopmentalRule(
            ConditionType.EMPTY_NEIGHBOR, {"direction": propagate_direction},
            ActionType.GROW_DIRECTIONAL, {"direction": propagate_direction, "voxel_type": propagate_voxel},
        ),
    ]

    while len(rules) < rule_count:
        rules.append(_random_rule(growth_biased=True))

    random.shuffle(rules)

    return DevelopmentalRuleGenome(rules)


def generate_robot_from_developmental_rules(
    rules: DevelopmentalRuleGenome,
    shape=(5, 5),
    iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """
    Deterministically develop a DevelopmentalRuleGenome into a robot body.

        DevelopmentalRuleGenome -> develop_rules() -> body -> EvoGym validation -> (body, connectivity)

    develop_rules() is run exactly once (deterministic, no retry loop).
    If the final grid is invalid (disconnected, or fewer than
    MIN_ACTUATORS actuators), returns None — never randomly repaired.
    """

    body = develop_rules(rules, shape, iterations)

    actuator_count = int((body == HORIZONTAL).sum() + (body == VERTICAL).sum())
    if actuator_count < MIN_ACTUATORS:
        return None

    if not (is_connected(body) and has_actuator(body)):
        return None

    return (body, get_full_connectivity(body))


# --------------------------------------------------------------------------
# Structural mutation operators — same philosophy as V1's mutate_rules().
# --------------------------------------------------------------------------

def _mutate_condition_type(genome: DevelopmentalRuleGenome) -> None:
    index = random.randrange(len(genome.rules))
    old_rule = genome.rules[index]
    choices = [c for c in ConditionType if c != old_rule.condition_type]
    new_condition_type, new_params = _random_condition(random.choice(choices))
    genome.rules[index] = DevelopmentalRule(new_condition_type, new_params, old_rule.action_type, old_rule.action_params)


def _mutate_condition_params(genome: DevelopmentalRuleGenome) -> None:
    index = random.randrange(len(genome.rules))
    old_rule = genome.rules[index]
    _, new_params = _random_condition(old_rule.condition_type)
    genome.rules[index] = DevelopmentalRule(old_rule.condition_type, new_params, old_rule.action_type, old_rule.action_params)


def _mutate_action_type(genome: DevelopmentalRuleGenome) -> None:
    index = random.randrange(len(genome.rules))
    old_rule = genome.rules[index]
    choices = [a for a in ActionType if a != old_rule.action_type]
    new_action_type, new_params = _random_action(random.choice(choices))
    genome.rules[index] = DevelopmentalRule(old_rule.condition_type, old_rule.condition_params, new_action_type, new_params)


def _mutate_action_params(genome: DevelopmentalRuleGenome) -> None:
    index = random.randrange(len(genome.rules))
    old_rule = genome.rules[index]
    _, new_params = _random_action(old_rule.action_type)
    genome.rules[index] = DevelopmentalRule(old_rule.condition_type, old_rule.condition_params, old_rule.action_type, new_params)


def _add_rule(genome: DevelopmentalRuleGenome) -> None:
    if len(genome.rules) >= MAX_RULES:
        return
    index = random.randint(0, len(genome.rules))
    genome.rules.insert(index, _random_rule(growth_biased=True))


def _remove_rule(genome: DevelopmentalRuleGenome) -> None:
    if len(genome.rules) <= MIN_RULES:
        return
    index = random.randrange(len(genome.rules))
    del genome.rules[index]


def _reorder_rules(genome: DevelopmentalRuleGenome) -> None:
    if len(genome.rules) < 2:
        return
    i, j = random.sample(range(len(genome.rules)), 2)
    genome.rules[i], genome.rules[j] = genome.rules[j], genome.rules[i]


_MUTATION_OPERATORS = (
    _mutate_condition_type,
    _mutate_condition_params,
    _mutate_action_type,
    _mutate_action_params,
    _add_rule,
    _remove_rule,
    _reorder_rules,
)


def mutate_developmental_rules(
    rules: DevelopmentalRuleGenome,
    mutation_rate: float = 0.2,
) -> DevelopmentalRuleGenome:
    """
    Produce a mutated copy of a DevelopmentalRuleGenome. The parent (and
    its rule objects) is never modified. Each structural operator (mutate
    condition type/params, action type/params, add/remove/reorder rule)
    is applied independently with probability mutation_rate.
    """

    child = rules.copy()
    for operator in _MUTATION_OPERATORS:
        if random.random() < mutation_rate:
            operator(child)
    return child


# --------------------------------------------------------------------------
# Crossover — rule-level one-point, same philosophy as V1's crossover_rules.
# --------------------------------------------------------------------------

def crossover_developmental_rules_at(
    parent1: DevelopmentalRuleGenome, parent2: DevelopmentalRuleGenome, cut1: int, cut2: int
) -> DevelopmentalRuleGenome:
    """Deterministic one-point crossover with explicit cut points:
    child.rules = parent1.rules[:cut1] + parent2.rules[cut2:], with the
    same length-repair strategy as V1's crossover_rules_at()."""

    if not (0 <= cut1 <= len(parent1.rules)):
        raise ValueError(f"cut1 must be in [0, {len(parent1.rules)}], got {cut1}")
    if not (0 <= cut2 <= len(parent2.rules)):
        raise ValueError(f"cut2 must be in [0, {len(parent2.rules)}], got {cut2}")

    child_rules = (
        [rule.copy() for rule in parent1.rules[:cut1]]
        + [rule.copy() for rule in parent2.rules[cut2:]]
    )

    if len(child_rules) > MAX_RULES:
        child_rules = child_rules[:MAX_RULES]

    if len(child_rules) < MIN_RULES:
        fillers = [rule.copy() for rule in parent1.rules] + [rule.copy() for rule in parent2.rules]
        i = 0
        while len(child_rules) < MIN_RULES and i < len(fillers):
            child_rules.append(fillers[i])
            i += 1

    return DevelopmentalRuleGenome(child_rules)


def crossover_developmental_rules(
    parent1: DevelopmentalRuleGenome, parent2: DevelopmentalRuleGenome
) -> DevelopmentalRuleGenome:
    """Randomized one-point crossover; parents are never modified."""
    cut1 = random.randint(0, len(parent1.rules))
    cut2 = random.randint(0, len(parent2.rules))
    return crossover_developmental_rules_at(parent1, parent2, cut1, cut2)
