"""Shared windowed ("mode-of-window") readout utility, used by both NCA and
rule_based's encoders.

body = the most frequent WHOLE-GRID (not per-cell) configuration among the
last W steps of a decoded-body trajectory, with ties between equally-
frequent configurations broken deterministically by the lexicographically
smallest flattened (row-major) material-ID sequence ("lowest material ID
first" at the first cell where the tied configurations differ) --
deterministic, no RNG. This is the EXACT convention N13 established for NCA
(`evodevo/encoders/nca/encoder.py::_mode_of_window`, originally embedded
there); this module is the generalised, shared version both encoders now
import, so there is exactly one definition of "mode of window" in the
codebase, not two independently-maintained copies.
"""
from __future__ import annotations

import numpy as np


def mode_of_window(window: list, validity_aware: bool = False, grid_shape=(5, 5)) -> np.ndarray:
    """The most frequent WHOLE-GRID configuration among `window` (a list of
    (rows, cols) int body arrays). Deterministic, no RNG. `window` must be
    non-empty.

    Tie-break, when >1 configuration shares the max count:
      `validity_aware=False` (default, NCA's usage -- UNCHANGED behaviour,
          preserves every previously-verified NCA number in
          results/nca/READOUT_POLICY_SELECTION.md): lexicographically
          smallest flattened (row-major) material-ID sequence among the
          tied configurations.
      `validity_aware=True` (rule_based's usage as of the 2026-09-16
          tie-break fix -- see results/rule_based/TIEBREAK_VALIDITY_CHECK.md):
          among the tied configurations, prefer whichever one is VALID
          (`evodevo.morphology.validate.validate_body`), if EXACTLY ONE of
          them is. This directly addresses a measured, non-trivial harm
          case for rule_based's dominant period-2 cycling population (the
          plain lexicographic tie-break discarded a legitimately-available
          valid body 30.5% of the time, TIEBREAK_VALIDITY_CHECK.md) -- it
          is still fully deterministic and depends only on the tied
          configurations themselves (not on t), so it remains T-shift
          stable by construction for any exact cycle, exactly like the
          plain tie-break it augments. If zero or more than one of the
          tied configurations is valid, falls back to the SAME
          lexicographically-smallest tie-break as the default case.
    """
    if not window:
        raise ValueError("mode_of_window requires a non-empty window")
    shape = window[0].shape
    keyed = [tuple(int(v) for v in body.flatten()) for body in window]
    counts: dict = {}
    for k in keyed:
        counts[k] = counts.get(k, 0) + 1
    max_count = max(counts.values())
    candidates = sorted(k for k, v in counts.items() if v == max_count)

    if validity_aware and len(candidates) > 1:
        from evodevo.morphology.validate import validate_body  # local import: avoids a hard
        # dependency from this shared utility onto evodevo.morphology for the (more common,
        # NCA) validity_aware=False path.

        valid_candidates = [
            c for c in candidates if validate_body(np.array(c, dtype=int).reshape(shape), grid_shape=grid_shape).valid
        ]
        if len(valid_candidates) == 1:
            return np.array(valid_candidates[0], dtype=int).reshape(shape)

    winner = candidates[0]
    return np.array(winner, dtype=int).reshape(shape)


def windowed_body_at(trajectory: list, t: int, w: int, validity_aware: bool = False, grid_shape=(5, 5)) -> np.ndarray:
    """The mode-of-window body using the W steps ending at (clipped to)
    step t of `trajectory` (a list of decoded/readout bodies, index 0..T).
    Clips to whatever prefix of the trajectory is available if t < w - 1 or
    t exceeds the trajectory's length, matching decode()'s own clipping
    behaviour for a `dev_steps` shorter than the readout window. See
    `mode_of_window`'s own docstring for `validity_aware`."""
    t = max(0, min(t, len(trajectory) - 1))
    start = max(0, t - w + 1)
    return mode_of_window(trajectory[start : t + 1], validity_aware=validity_aware, grid_shape=grid_shape)
