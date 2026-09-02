"""Equivalence tests: evodevo.morphology's native is_connected / has_actuator /
get_full_connectivity must behave identically to evogym's originals.

This is the verification evidence for moving rule_based's evogym imports
(is_connected, has_actuator, get_full_connectivity) into evodevo.morphology
(see docs/decisions.md). Test code is allowed to import evogym directly --
the encoder-isolation contract applies to encoder modules, not tests.
"""
import random

import numpy as np
import pytest

evogym = pytest.importorskip("evogym")

from evodevo.morphology.connectivity import get_full_connectivity as native_get_full_connectivity
from evodevo.morphology.validate import has_actuator as native_has_actuator
from evodevo.morphology.validate import is_connected as native_is_connected

SHAPE = (5, 5)


def _random_body(rng: random.Random, connected: bool, with_actuator: bool) -> np.ndarray:
    body = np.zeros(SHAPE, dtype=int)
    if connected:
        # Random connected blob via a self-avoiding-ish random walk / flood growth.
        r, c = rng.randrange(SHAPE[0]), rng.randrange(SHAPE[1])
        cells = {(r, c)}
        target = rng.randint(1, SHAPE[0] * SHAPE[1])
        frontier = [(r, c)]
        while len(cells) < target and frontier:
            fr, fc = frontier[rng.randrange(len(frontier))]
            neighbors = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
            rng.shuffle(neighbors)
            grown = False
            for nr, nc in neighbors:
                if 0 <= nr < SHAPE[0] and 0 <= nc < SHAPE[1] and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    frontier.append((nr, nc))
                    grown = True
                    break
            if not grown:
                frontier.remove((fr, fc))
        materials = [1, 2] + ([3, 4] if with_actuator else [1, 2])
        for i, (rr, cc) in enumerate(cells):
            if with_actuator and i == 0:
                body[rr, cc] = rng.choice([3, 4])
            else:
                body[rr, cc] = rng.choice(materials if with_actuator else [1, 2])
    else:
        # Two disjoint blobs in opposite corners -> guaranteed disconnected
        # (as long as both are non-empty).
        body[0, 0] = rng.choice([1, 2])
        body[0, 1] = 3 if with_actuator else rng.choice([1, 2])
        body[4, 4] = rng.choice([1, 2])
        body[4, 3] = 4 if with_actuator else rng.choice([1, 2])
        if not with_actuator:
            # ensure no actuator anywhere
            body[body == 3] = 1
            body[body == 4] = 1
    return body


def _generate_bodies():
    bodies = []

    # Hand-constructed fixed cases.
    empty = np.zeros(SHAPE, dtype=int)
    bodies.append(("all_empty", empty))

    single_no_act = np.zeros(SHAPE, dtype=int)
    single_no_act[2, 2] = 1
    bodies.append(("single_voxel_no_actuator", single_no_act))

    single_act = np.zeros(SHAPE, dtype=int)
    single_act[2, 2] = 3
    bodies.append(("single_voxel_actuator", single_act))

    plus_shape = np.zeros(SHAPE, dtype=int)
    plus_shape[2, :] = 1
    plus_shape[:, 2] = 2
    plus_shape[2, 2] = 4
    bodies.append(("plus_shape_connected_with_actuator", plus_shape))

    full_grid = np.ones(SHAPE, dtype=int)
    full_grid[0, 0] = 3
    full_grid[4, 4] = 4
    bodies.append(("full_grid_connected_with_actuators", full_grid))

    diag_only = np.zeros(SHAPE, dtype=int)
    for i in range(5):
        diag_only[i, i] = 1
    diag_only[0, 0] = 3
    bodies.append(("diagonal_only_disconnected_under_4conn", diag_only))

    two_blobs_no_act = np.zeros(SHAPE, dtype=int)
    two_blobs_no_act[0, 0] = 1
    two_blobs_no_act[0, 1] = 1
    two_blobs_no_act[4, 4] = 2
    bodies.append(("two_blobs_no_actuator", two_blobs_no_act))

    l_shape = np.zeros(SHAPE, dtype=int)
    l_shape[1, 1] = 1
    l_shape[2, 1] = 1
    l_shape[3, 1] = 1
    l_shape[3, 2] = 3
    l_shape[3, 3] = 4
    bodies.append(("l_shape_connected_with_actuators", l_shape))

    all_soft_no_actuator = np.full(SHAPE, 2, dtype=int)
    bodies.append(("all_soft_full_grid_no_actuator", all_soft_no_actuator))

    single_row = np.zeros(SHAPE, dtype=int)
    single_row[0, :] = [1, 2, 1, 3, 4]
    bodies.append(("single_row_connected_with_actuators", single_row))

    # Randomly generated bodies covering the connected/disconnected x
    # with/without actuator matrix, well beyond the required 20 total.
    rng = random.Random(20240921)
    for i in range(4):
        for connected in (True, False):
            for with_actuator in (True, False):
                body = _random_body(rng, connected=connected, with_actuator=with_actuator)
                bodies.append((f"random_{i}_connected={connected}_actuator={with_actuator}", body))

    return bodies


BODIES = _generate_bodies()


def test_at_least_twenty_bodies_generated():
    assert len(BODIES) >= 20


@pytest.mark.parametrize("name,body", BODIES, ids=[b[0] for b in BODIES])
def test_is_connected_matches_evogym(name, body):
    assert native_is_connected(body) == evogym.is_connected(body), name


@pytest.mark.parametrize("name,body", BODIES, ids=[b[0] for b in BODIES])
def test_has_actuator_matches_evogym(name, body):
    assert native_has_actuator(body) == evogym.has_actuator(body), name


@pytest.mark.parametrize("name,body", BODIES, ids=[b[0] for b in BODIES])
def test_get_full_connectivity_matches_evogym(name, body):
    native = native_get_full_connectivity(body)
    reference = evogym.get_full_connectivity(body)
    assert native.shape == reference.shape, name
    np.testing.assert_array_equal(native, reference, err_msg=name)
