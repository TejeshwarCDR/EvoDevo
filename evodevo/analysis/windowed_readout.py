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

import warnings

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
    `mode_of_window`'s own docstring for `validity_aware`.

    Both clips below are WARNED, not silent -- a silent version of the
    second one (returning `min(w, available history)` steps rather than a
    genuine trailing W-step window) previously caused a real bug: a
    (T_primary, W) pair with W > T looked like a valid W-step-window
    measurement but was actually reading the full available trajectory
    instead, and this was not caught until the numbers it produced were
    scrutinised directly (see docs/decisions.md's "NCA T_primary/W
    consistency fix" row). Callers that intentionally sweep across W > T to
    characterise this exact behaviour (e.g. evodevo.analysis
    .readout_policy_selection's own local, duplicate clamp, or
    rule_based_readout_policy.py's W-vs-T sweep, both of which already mark
    or filter these cells explicitly) still get a real, disclosed clip --
    the window-shrink warning below uses one fixed message so Python's
    default per-location warning dedup collapses a legitimate sweep's many
    hits into a single notice per process, rather than spamming."""
    clamped_t = max(0, min(t, len(trajectory) - 1))
    if clamped_t != t:
        warnings.warn(
            f"windowed_body_at: requested t={t} is outside the available trajectory "
            f"(0..{len(trajectory) - 1}) -- clipped to t={clamped_t}. This is a real clip and "
            "must not pass silently; check the caller's T/shift bookkeeping.",
            stacklevel=2,
        )
    start = max(0, clamped_t - w + 1)
    if clamped_t - w + 1 < 0:
        warnings.warn(
            "windowed_body_at: requested window width w exceeds the available trajectory at t "
            "(w > t + 1) -- returning the full available prefix instead of a genuine w-step "
            "trailing window. This is documented, intentional degradation (matching decode()'s "
            "own short-trajectory behaviour) but has caused a real bug before when silently "
            "mistaken for a genuine window (docs/decisions.md's \"NCA T_primary/W consistency "
            "fix\" row) -- if unexpected, check that the readout window does not exceed dev_steps.",
            stacklevel=2,
        )
    return mode_of_window(trajectory[start : clamped_t + 1], validity_aware=validity_aware, grid_shape=grid_shape)
