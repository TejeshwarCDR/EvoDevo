"""Shared morphology validation. Used identically by every encoding."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evodevo.core.materials import ACTUATORS, EMPTY, MATERIALS
from evodevo.morphology.connectivity import largest_component_size

ACTUATOR_IDS = ACTUATORS


def is_connected(body) -> bool:
    """Native equivalent of ``evogym.is_connected``.

    Returns whether every non-empty voxel of ``body`` lies in a single
    4-connected component. An all-empty body is not connected (matches
    evogym: no start voxel found -> False).
    """
    body_arr = np.asarray(body)
    occupied = body_arr != EMPTY
    occupied_count = int(occupied.sum())
    if occupied_count == 0:
        return False
    return largest_component_size(occupied) == occupied_count


def has_actuator(body) -> bool:
    """Native equivalent of ``evogym.has_actuator``: whether any voxel is
    an actuator material (H_ACT or V_ACT, see evodevo.core.materials)."""
    body_arr = np.asarray(body)
    return bool(np.isin(body_arr, list(ACTUATOR_IDS)).any())


# Substrings of a validate_body() failure reason (or the "; "-joined reason
# string an evaluator attaches to an invalid body) -> a stable category
# label. Order matters: checked top-to-bottom, first match wins. Single
# authoritative home for this categorization -- used by the EA's per-eval
# invalid-reason breakdown (evodevo/evolution/ea.py) and by the PPO-free
# characterization tools (evodevo/experiments/inspect_encoder.py,
# evodevo/analysis/structural.py) that historically each defined their own
# copy of the same substring list.
_REASON_CATEGORIES = (
    ("min_voxels", "too_few_voxels"),
    ("connected component", "disconnected"),
    ("no actuator", "no_actuator"),
    ("0..4", "invalid_material_values"),
    ("shape", "wrong_shape"),
    ("dtype", "non_integer_dtype"),
)


def categorize_reason(reason) -> str:
    """Maps a validate_body()/EvalResult failure reason string to a stable
    category label. `reason` may be None/empty (valid body -- returns
    "none")."""
    if not reason:
        return "none"
    for substr, category in _REASON_CATEGORIES:
        if substr in reason:
            return category
    return "other"


@dataclass
class ValidationResult:
    valid: bool
    reasons: list = field(default_factory=list)


def validate_body(body, grid_shape=(5, 5), min_voxels=3, min_actuators=1) -> ValidationResult:
    reasons = []

    body_arr = np.asarray(body)
    if body_arr.shape != tuple(grid_shape):
        reasons.append(f"wrong shape: expected {tuple(grid_shape)}, got {body_arr.shape}")
        return ValidationResult(valid=False, reasons=reasons)

    if not np.issubdtype(body_arr.dtype, np.integer):
        reasons.append(f"non-integer dtype: {body_arr.dtype}")
        return ValidationResult(valid=False, reasons=reasons)

    valid_ids = set(MATERIALS.values())
    if not set(np.unique(body_arr).tolist()) <= valid_ids:
        reasons.append("values outside 0..4")
        return ValidationResult(valid=False, reasons=reasons)

    occupied = body_arr != EMPTY
    occupied_count = int(occupied.sum())
    if occupied_count < min_voxels:
        reasons.append(f"fewer than min_voxels ({min_voxels}) non-empty voxels: {occupied_count}")

    largest = largest_component_size(occupied)
    if occupied_count == 0 or largest != occupied_count:
        reasons.append("not a single connected component under cardinal adjacency")

    actuator_count = int(np.isin(body_arr, list(ACTUATOR_IDS)).sum())
    if actuator_count < min_actuators:
        if actuator_count == 0:
            # Exact wording preserved for callers (e.g. nca_sanity.py's reason
            # categorization) that pattern-match on "no actuator".
            reasons.append("no actuator (material 3 or 4) present")
        else:
            reasons.append(
                f"fewer than min_actuators ({min_actuators}) actuator (material 3 or 4) voxels present: {actuator_count}"
            )

    return ValidationResult(valid=len(reasons) == 0, reasons=reasons)
