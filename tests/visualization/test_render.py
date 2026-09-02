import importlib.util
import os
import tempfile

import numpy as np
import pytest

from evodevo.visualization.render import render_body_rollout

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

DISCONNECTED_BODY = np.array(
    [
        [1, 0, 0, 0, 3],
        [4, 1, 2, 0, 4],
        [3, 0, 3, 0, 2],
        [0, 3, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ],
    dtype=int,
)

HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None


def test_invalid_body_is_refused_before_touching_the_simulator():
    """CRITICAL safety property: an invalid body must never reach the
    EvoGym C++ simulator, since that can segfault the whole process
    (unrecoverable by any Python try/except) rather than raise. This must
    raise a normal Python ValueError instead, regardless of whether the
    simulator is even built."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="invalid body"):
            render_body_rollout(
                DISCONNECTED_BODY,
                controller_path=None,
                task="Walker-v0",
                n_steps=5,
                output_path=os.path.join(tmp, "rollout"),
            )


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_random_action_rollout_produces_a_real_gif_and_mp4():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "rollout")
        result = render_body_rollout(
            GOOD_WALKER,
            controller_path=None,
            task="Walker-v0",
            n_steps=5,
            output_path=output_path,
            fps=10,
        )

        assert result.used_controller is False
        assert "RANDOM ACTIONS" in result.label

        assert os.path.exists(result.gif_path)
        assert os.path.getsize(result.gif_path) > 0
        assert os.path.exists(result.mp4_path)
        assert os.path.getsize(result.mp4_path) > 0

        import imageio

        gif_reader = imageio.get_reader(result.gif_path)
        gif_frame0 = gif_reader.get_data(0)
        assert gif_frame0.ndim == 3
        gif_reader.close()

        mp4_reader = imageio.get_reader(result.mp4_path)
        mp4_frame0 = mp4_reader.get_data(0)
        assert mp4_frame0.ndim == 3
        mp4_reader.close()
