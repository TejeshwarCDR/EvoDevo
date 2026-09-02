"""Shared morphology validation. Used identically by every encoding."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Deliberately excludes FIXED=5, which EvoGym reserves for terrain, not robots.
MATERIALS = {"EMPTY": 0, "RIGID": 1, "SOFT": 2, "H_ACT": 3, "V_ACT": 4}
ACTUATOR_IDS = {MATERIALS["H_ACT"], MATERIALS["V_ACT"]}


@dataclass
class ValidationResult:
    valid: bool
    reasons: list = field(default_factory=list)


def _largest_component_size(occupied: np.ndarray) -> int:
    """Largest 4-connected component among True cells in a boolean grid."""
    visited = np.zeros_like(occupied, dtype=bool)
    best = 0
    rows, cols = occupied.shape
    for r in range(rows):
        for c in range(cols):
            if not occupied[r, c] or visited[r, c]:
                continue
            size = 0
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                size += 1
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and occupied[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, size)
    return best


def validate_body(body, grid_shape=(5, 5), min_voxels=3) -> ValidationResult:
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

    occupied = body_arr != MATERIALS["EMPTY"]
    occupied_count = int(occupied.sum())
    if occupied_count < min_voxels:
        reasons.append(f"fewer than min_voxels ({min_voxels}) non-empty voxels: {occupied_count}")

    largest = _largest_component_size(occupied)
    if occupied_count == 0 or largest != occupied_count:
        reasons.append("not a single connected component under cardinal adjacency")

    has_actuator = bool(np.isin(body_arr, list(ACTUATOR_IDS)).any())
    if not has_actuator:
        reasons.append("no actuator (material 3 or 4) present")

    return ValidationResult(valid=len(reasons) == 0, reasons=reasons)
