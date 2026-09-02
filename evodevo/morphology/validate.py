"""Shared morphology validation. Used identically by every encoding."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evodevo.core.materials import ACTUATORS, EMPTY, MATERIALS
from evodevo.morphology.connectivity import largest_component_size

ACTUATOR_IDS = ACTUATORS


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
