"""Shared descriptive morphology metrics. Used identically by every encoding."""
from __future__ import annotations

import hashlib

import numpy as np

from evodevo.core.materials import EMPTY, H_ACT, RIGID, SOFT, V_ACT
from evodevo.morphology.connectivity import largest_component_size


def body_metrics(body) -> dict:
    body_arr = np.asarray(body)
    total = body_arr.size

    empty_count = int(np.sum(body_arr == EMPTY))
    rigid_count = int(np.sum(body_arr == RIGID))
    soft_count = int(np.sum(body_arr == SOFT))
    h_act_count = int(np.sum(body_arr == H_ACT))
    v_act_count = int(np.sum(body_arr == V_ACT))
    occupied_count = total - empty_count
    actuator_count = h_act_count + v_act_count
    actuator_fraction = actuator_count / total if total > 0 else 0.0

    mirrored_h = body_arr[:, ::-1]
    horizontal_symmetry_score = float(np.mean(body_arr == mirrored_h))
    mirrored_v = body_arr[::-1, :]
    vertical_symmetry_score = float(np.mean(body_arr == mirrored_v))

    occupied = body_arr != EMPTY
    largest_component = largest_component_size(occupied)

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
        "largest_component_size": largest_component,
    }


def body_hash(body) -> str:
    body_arr = np.asarray(body)
    payload = str(body_arr.shape).encode() + b"|" + str(body_arr.dtype).encode() + b"|" + body_arr.tobytes()
    return hashlib.sha256(payload).hexdigest()
