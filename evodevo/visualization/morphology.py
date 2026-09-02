"""Static voxel-grid figures: a single body, or a side-by-side grid of
several bodies (e.g. one per encoding arm's best evolved robot). Matplotlib
only, no display side effects.
"""
from __future__ import annotations

import matplotlib

# Must be set before `matplotlib.pyplot` is imported anywhere in the process.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from evodevo.core.materials import MATERIAL_NAMES, N_MATERIALS  # noqa: E402

# One fixed, stable colour per material id (0..4), reused everywhere a body
# is rendered so the same material always reads as the same colour across
# every figure in the project.
MATERIAL_COLORS = (
    "#f2f2f2",  # EMPTY
    "#4c72b0",  # RIGID
    "#55a868",  # SOFT
    "#c44e52",  # H_ACT
    "#8172b2",  # V_ACT
)

_CMAP = ListedColormap(MATERIAL_COLORS)


def _legend_handles():
    return [Patch(facecolor=MATERIAL_COLORS[i], edgecolor="black", label=MATERIAL_NAMES[i]) for i in range(N_MATERIALS)]


def plot_body(body: np.ndarray, ax=None, title: str = None):
    """Render a single (5, 5) (or any HxW) int body as a static voxel-grid
    figure, with a material-name legend."""
    body = np.asarray(body)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(body, cmap=_CMAP, vmin=0, vmax=N_MATERIALS - 1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    if title:
        ax.set_title(title)
    ax.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=7, frameon=False)

    return ax


def plot_body_grid(bodies, labels=None, axes=None):
    """Render N bodies side by side (e.g. one best-evolved body per
    encoding arm), sharing one legend. Handles N=1 and N>1 identically."""
    bodies = list(bodies)
    n = len(bodies)
    if labels is None:
        labels = [None] * n
    if len(labels) != n:
        raise ValueError(f"labels must have the same length as bodies ({n}), got {len(labels)}")

    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
        axes = axes[0]
    elif n == 1 and not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, body, label in zip(axes, bodies, labels):
        body = np.asarray(body)
        ax.imshow(body, cmap=_CMAP, vmin=0, vmax=N_MATERIALS - 1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if label:
            ax.set_title(label)

    fig = axes[0].figure if hasattr(axes[0], "figure") else None
    if fig is not None:
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=N_MATERIALS, fontsize=8, frameon=False)

    return axes
