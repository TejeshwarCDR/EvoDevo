"""Shared descriptive morphology metrics. Used identically by every encoding."""
from __future__ import annotations

import hashlib

import numpy as np

from evodevo.morphology.validate import MATERIALS


def _largest_component_size(occupied: np.ndarray) -> int:
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


def body_metrics(body) -> dict:
    body_arr = np.asarray(body)
    total = body_arr.size

    empty_count = int(np.sum(body_arr == MATERIALS["EMPTY"]))
    rigid_count = int(np.sum(body_arr == MATERIALS["RIGID"]))
    soft_count = int(np.sum(body_arr == MATERIALS["SOFT"]))
    h_act_count = int(np.sum(body_arr == MATERIALS["H_ACT"]))
    v_act_count = int(np.sum(body_arr == MATERIALS["V_ACT"]))
    occupied_count = total - empty_count
    actuator_count = h_act_count + v_act_count
    actuator_fraction = actuator_count / total if total > 0 else 0.0

    mirrored_h = body_arr[:, ::-1]
    horizontal_symmetry_score = float(np.mean(body_arr == mirrored_h))
    mirrored_v = body_arr[::-1, :]
    vertical_symmetry_score = float(np.mean(body_arr == mirrored_v))

    occupied = body_arr != MATERIALS["EMPTY"]
    largest_component_size = _largest_component_size(occupied)

    return {
        "occupied_count": occupied_count,
        "empty_count": empty_count,
        "rigid_count": rigid_count,
        "soft_count": soft_count,
        "h_act_count": h_act_count,
        "v_act_count": v_act_count,
        "actuator_fraction": actuator_fraction,
        "horizontal_symmetry_score": horizontal_symmetry_score,
        "vertical_symmetry_score": vertical_symmetry_score,
        "largest_component_size": largest_component_size,
    }


def body_hash(body) -> str:
    body_arr = np.asarray(body)
    payload = str(body_arr.shape).encode() + b"|" + str(body_arr.dtype).encode() + b"|" + body_arr.tobytes()
    return hashlib.sha256(payload).hexdigest()
