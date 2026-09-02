# Ported from a standalone reference implementation's developmental_rule_types.py
# (commit 703e711, "Initial project implementation"; that checkout has since
# been permanently removed -- see docs/decisions.md). Original author: the
# project's Rule-Based teammate. This is a port, not a rewrite: only the
# module docstring below and this header were changed; the rule language
# itself (vocabulary, enums, dataclass, validation) is unmodified.
"""
Vocabulary and validated Rule representation for the V2 DEVELOPMENTAL rule
language (Step 13).

This is a separate, independent representation from the V1 static rule
language (rule_types.py / rule_encoding.py / rule_interpreter.py in the
original checkout, not ported here since V1 is a superseded earlier
iteration). V2 does not reuse V1's Rule/RuleGenome classes — it is its own
small, data-only rule language, interpreted by
evodevo/encoders/rule_based/interpreter.py.

A DevelopmentalRule is:

    condition_type (+ condition_params)  ->  action_type (+ action_params)

exactly like V1's `condition -> outcome`, except the outcome is now an
ACTION rather than a direct voxel assignment: most actions still just set
the current cell (SET_EMPTY/SET_SOFT/SET_HARD/SET_ACTUATOR_*), but two
actions (GROW_DIRECTIONAL, COPY_NEIGHBOR) read/write neighboring cells,
which is what gives V2 spatial/directional growth capability that V1's
per-cell-independent language does not have.

Rules are plain data (dataclasses + enums) — never executable Python —
exactly like V1.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class VoxelState(Enum):
    """The developmental grid's cell states, which double as the final
    EvoGym voxel types once development finishes (see
    evodevo/encoders/rule_based/interpreter.py for the exact numeric
    mapping used -- sourced from evodevo.core.materials, the project's
    single authoritative material vocabulary)."""

    EMPTY = "empty"
    SOFT = "soft"
    HARD = "hard"
    ACTUATOR_HORIZONTAL = "actuator_horizontal"
    ACTUATOR_VERTICAL = "actuator_vertical"


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class PositionType(Enum):
    BOUNDARY = "boundary"
    INTERIOR = "interior"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class ConditionType(Enum):
    """The kinds of conditions a DevelopmentalRule can be triggered by."""

    ALWAYS = "always"
    CELL_TYPE = "cell_type"
    NEIGHBOR_COUNT = "neighbor_count"
    EMPTY_NEIGHBOR = "empty_neighbor"
    POSITION = "position"


class ActionType(Enum):
    """The actions a DevelopmentalRule can trigger for a matching cell."""

    SET_EMPTY = "set_empty"
    SET_SOFT = "set_soft"
    SET_HARD = "set_hard"
    SET_ACTUATOR_HORIZONTAL = "set_actuator_horizontal"
    SET_ACTUATOR_VERTICAL = "set_actuator_vertical"
    GROW_DIRECTIONAL = "grow_directional"
    COPY_NEIGHBOR = "copy_neighbor"


# Growable voxel states for GROW_DIRECTIONAL — growing EMPTY would be a
# no-op by definition, so it is excluded from the valid vocabulary.
GROWABLE_STATES = (
    VoxelState.SOFT,
    VoxelState.HARD,
    VoxelState.ACTUATOR_HORIZONTAL,
    VoxelState.ACTUATOR_VERTICAL,
)

# NEIGHBOR_COUNT threshold bounds. Unlike V1's row-major, already-assigned
# -only interpreter (rule_interpreter.py), V2's synchronous interpreter
# evaluates every rule against a fully-formed snapshot of the CURRENT grid
# each iteration (see interpreter.py), so all 4 neighbors are always
# legitimately observable — thresholds 1..4 are all reachable, unlike V1's
# 1..2 cap.
MIN_NEIGHBOR_THRESHOLD = 1
MAX_NEIGHBOR_THRESHOLD = 4

# condition_type -> required condition_params keys.
_REQUIRED_CONDITION_PARAMS = {
    ConditionType.ALWAYS: (),
    ConditionType.CELL_TYPE: ("state",),
    ConditionType.NEIGHBOR_COUNT: ("threshold",),
    ConditionType.EMPTY_NEIGHBOR: ("direction",),
    ConditionType.POSITION: ("position",),
}

# action_type -> required action_params keys.
_REQUIRED_ACTION_PARAMS = {
    ActionType.SET_EMPTY: (),
    ActionType.SET_SOFT: (),
    ActionType.SET_HARD: (),
    ActionType.SET_ACTUATOR_HORIZONTAL: (),
    ActionType.SET_ACTUATOR_VERTICAL: (),
    ActionType.GROW_DIRECTIONAL: ("direction", "voxel_type"),
    ActionType.COPY_NEIGHBOR: ("direction",),
}


def _validate_condition_params(condition_type: ConditionType, params: Dict[str, Any]) -> None:
    if not isinstance(params, dict):
        raise ValueError("condition_params must be a dict")

    required = _REQUIRED_CONDITION_PARAMS[condition_type]
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"condition_type {condition_type} is missing condition_params: {missing}")
    extra = [p for p in params if p not in required]
    if extra:
        raise ValueError(f"condition_type {condition_type} does not accept condition_params: {extra}")

    if condition_type == ConditionType.CELL_TYPE:
        if not isinstance(params["state"], VoxelState):
            raise ValueError(f"CELL_TYPE 'state' must be a VoxelState, got {params['state']!r}")

    if condition_type == ConditionType.NEIGHBOR_COUNT:
        threshold = params["threshold"]
        if (
            not isinstance(threshold, int)
            or isinstance(threshold, bool)
            or not (MIN_NEIGHBOR_THRESHOLD <= threshold <= MAX_NEIGHBOR_THRESHOLD)
        ):
            raise ValueError(
                f"NEIGHBOR_COUNT threshold must be an int in "
                f"[{MIN_NEIGHBOR_THRESHOLD}, {MAX_NEIGHBOR_THRESHOLD}], got {threshold!r}"
            )

    if condition_type == ConditionType.EMPTY_NEIGHBOR:
        if not isinstance(params["direction"], Direction):
            raise ValueError(f"EMPTY_NEIGHBOR 'direction' must be a Direction, got {params['direction']!r}")

    if condition_type == ConditionType.POSITION:
        if not isinstance(params["position"], PositionType):
            raise ValueError(f"POSITION 'position' must be a PositionType, got {params['position']!r}")


def _validate_action_params(action_type: ActionType, params: Dict[str, Any]) -> None:
    if not isinstance(params, dict):
        raise ValueError("action_params must be a dict")

    required = _REQUIRED_ACTION_PARAMS[action_type]
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"action_type {action_type} is missing action_params: {missing}")
    extra = [p for p in params if p not in required]
    if extra:
        raise ValueError(f"action_type {action_type} does not accept action_params: {extra}")

    if action_type == ActionType.GROW_DIRECTIONAL:
        if not isinstance(params["direction"], Direction):
            raise ValueError(f"GROW_DIRECTIONAL 'direction' must be a Direction, got {params['direction']!r}")
        voxel_type = params["voxel_type"]
        if not isinstance(voxel_type, VoxelState) or voxel_type not in GROWABLE_STATES:
            raise ValueError(
                f"GROW_DIRECTIONAL 'voxel_type' must be one of {GROWABLE_STATES}, got {voxel_type!r}"
            )

    if action_type == ActionType.COPY_NEIGHBOR:
        if not isinstance(params["direction"], Direction):
            raise ValueError(f"COPY_NEIGHBOR 'direction' must be a Direction, got {params['direction']!r}")


@dataclass
class DevelopmentalRule:
    """
    A single rule in the V2 developmental rule language:

        condition_type (+ condition_params) -> action_type (+ action_params)

    Examples (conceptually):
        DevelopmentalRule(ConditionType.ALWAYS, {}, ActionType.SET_SOFT, {})
        DevelopmentalRule(ConditionType.CELL_TYPE, {"state": VoxelState.SOFT},
                           ActionType.GROW_DIRECTIONAL, {"direction": Direction.RIGHT, "voxel_type": VoxelState.SOFT})
    """

    condition_type: ConditionType
    condition_params: Dict[str, Any] = field(default_factory=dict)
    action_type: ActionType = ActionType.SET_EMPTY
    action_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.condition_type, ConditionType):
            raise ValueError(f"condition_type must be a ConditionType, got {self.condition_type!r}")
        if not isinstance(self.action_type, ActionType):
            raise ValueError(f"action_type must be an ActionType, got {self.action_type!r}")

        _validate_condition_params(self.condition_type, self.condition_params)
        _validate_action_params(self.action_type, self.action_params)

    def copy(self) -> "DevelopmentalRule":
        return DevelopmentalRule(
            condition_type=self.condition_type,
            condition_params=deepcopy(self.condition_params),
            action_type=self.action_type,
            action_params=deepcopy(self.action_params),
        )

    def to_dict(self) -> Dict[str, Any]:
        def _params_to_dict(params):
            out = {}
            for k, v in params.items():
                out[k] = v.value if isinstance(v, Enum) else v
            return out

        return {
            "condition_type": self.condition_type.value,
            "condition_params": _params_to_dict(self.condition_params),
            "action_type": self.action_type.value,
            "action_params": _params_to_dict(self.action_params),
        }

    def __repr__(self) -> str:
        return (
            f"DevelopmentalRule({self.condition_type.value}"
            f"{self.condition_params if self.condition_params else ''} "
            f"-> {self.action_type.value}"
            f"{self.action_params if self.action_params else ''})"
        )
