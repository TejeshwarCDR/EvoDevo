import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from evodevo.visualization.morphology import plot_body, plot_body_grid

GOOD_WALKER = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

OTHER_BODY = np.array(
    [
        [4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    dtype=int,
)


def test_plot_body_produces_a_non_empty_png():
    ax = plot_body(GOOD_WALKER, title="test body")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "body.png")
        ax.figure.savefig(path)
        plt.close(ax.figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_body_grid_handles_a_single_body():
    axes = plot_body_grid([GOOD_WALKER], labels=["direct"])
    assert len(axes) == 1
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "grid.png")
        axes[0].figure.savefig(path)
        plt.close(axes[0].figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_body_grid_handles_n_bodies():
    axes = plot_body_grid(
        [GOOD_WALKER, OTHER_BODY, GOOD_WALKER],
        labels=["direct", "nca", "rule_based"],
    )
    assert len(axes) == 3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "grid3.png")
        axes[0].figure.savefig(path)
        plt.close(axes[0].figure)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_body_grid_rejects_mismatched_labels_length():
    import pytest

    with pytest.raises(ValueError):
        plot_body_grid([GOOD_WALKER, OTHER_BODY], labels=["only_one"])
