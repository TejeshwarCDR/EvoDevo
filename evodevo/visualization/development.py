"""Developmental-trajectory figures: t=0..T body snapshots for any encoder
that exposes a develop_trajectory() hook (see evodevo.encoders.base.Encoder).

This is the single most communicative visual this project can produce for a
review panel -- the same rule set producing different structure over
developmental time -- and it exists for two of the three arms (NCA and
Rule-Based); Direct has no developmental process and this module handles
that gracefully rather than crashing.
"""
from __future__ import annotations

import matplotlib

# Must be set before `matplotlib.pyplot` is imported anywhere in the process.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evodevo.core.materials import N_MATERIALS  # noqa: E402
from evodevo.visualization.morphology import MATERIAL_COLORS  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

_CMAP = ListedColormap(MATERIAL_COLORS)


def plot_development_trajectory(encoder, genome, ax_row=None, title: str = None):
    """Render t=0..T for `encoder.develop_trajectory(genome)`.

    Returns the trajectory's matplotlib Axes row, or None if the encoder
    has no trajectory to show (develop_trajectory() returned None, e.g.
    DirectEncoder) -- callers must handle a None return, not treat it as
    an error.
    """
    trajectory = encoder.develop_trajectory(genome)
    if trajectory is None:
        return None

    n_cols = len(trajectory)
    if ax_row is None:
        fig, ax_row = plt.subplots(1, n_cols, figsize=(1.3 * n_cols, 1.6), squeeze=False)
        ax_row = ax_row[0]
        owns_figure = True
    else:
        fig = ax_row[0].figure
        owns_figure = False

    for col, body in enumerate(trajectory):
        body = np.asarray(body)
        ax = ax_row[col]
        ax.imshow(body, cmap=_CMAP, vmin=0, vmax=N_MATERIALS - 1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={col}", fontsize=8)

    if title and owns_figure:
        fig.suptitle(title)
    if owns_figure:
        fig.tight_layout()

    return ax_row
