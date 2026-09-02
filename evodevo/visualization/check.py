"""Rendering self-test: `python -m evodevo.visualization.check`.

Verifies, in order:
  1. the compiled evogym simulator is importable;
  2. an EvoGym env can be created with render_mode="img" (never "human");
  3. a frame can actually be captured from it;
  4. imageio can write both a gif and an mp4.

Each step that fails prints a precise, actionable diagnosis (referencing
the exact fixes documented in external/evogym/README.md) instead of a bare
traceback, and the script keeps checking the remaining steps rather than
stopping at the first failure, so a single run tells you everything that's
wrong at once.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

_GOOD_WALKER = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)


class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.ok = False
        self.diagnosis = None

    def passed(self):
        self.ok = True
        return self

    def failed(self, diagnosis: str):
        self.ok = False
        self.diagnosis = diagnosis
        return self


def check_simulator_importable() -> CheckResult:
    result = CheckResult("evogym simulator importable")
    if importlib.util.find_spec("evogym") is None:
        return result.failed(
            "the `evogym` package itself is not importable -- it isn't installed in this "
            "environment. Fix: `pip install -e external/evogym` (or wherever your evogym "
            "checkout lives), or `pip install evogym` if using a released wheel."
        )
    if importlib.util.find_spec("evogym.simulator_cpp") is None:
        return result.failed(
            "evogym is importable but its COMPILED C++ simulator (evogym.simulator_cpp) is "
            "not built. Fix: reinstall from source with the compiled extension, e.g. "
            "`pip install -e external/evogym` from a directory with CMake and a C++ "
            "toolchain available (see external/evogym/README.md's 'From Source' section). "
            "On Linux you also need: `sudo apt-get install xorg-dev libglu1-mesa-dev`."
        )
    try:
        import evogym  # noqa: F401
        import evogym.envs  # noqa: F401
    except Exception:
        return result.failed(
            "evogym.simulator_cpp exists but importing evogym/evogym.envs still raised -- "
            "see the traceback above. If the error mentions `libGL` or `MESA-LOADER` "
            "(common on Linux + Conda), the documented fix is: "
            "`conda install -c conda-forge libstdcxx-ng` (external/evogym/README.md, "
            "'Known Issues')."
        )
    return result.passed()


def check_env_creation_and_frame_capture() -> CheckResult:
    result = CheckResult('env creation with render_mode="img" + frame capture')
    try:
        import evogym.envs  # noqa: F401
        import gymnasium as gym
        from evogym import get_full_connectivity

        connections = get_full_connectivity(_GOOD_WALKER)
        env = gym.make("Walker-v0", body=_GOOD_WALKER, connections=connections, render_mode="img")
        env.reset(seed=0)
        frame = env.render()
        env.close()
    except Exception:
        tb = traceback.format_exc()
        diagnosis = (
            "creating a render_mode=\"img\" env or capturing a frame raised an exception. "
            "This step needs an (offscreen) OpenGL context even in a headless/no-display "
            "environment -- unlike render_mode=None, which external/evogym/README.md's "
            "'Headless Mode' section documents as needing NO rendering libraries at all. "
            "If the traceback below mentions `libGL`, `MESA-LOADER`, `iris_dri.so`, or GLEW: "
            "on Linux install `xorg-dev libglu1-mesa-dev` (`sudo apt-get install xorg-dev "
            "libglu1-mesa-dev`), and on Linux+Conda specifically also try "
            "`conda install -c conda-forge libstdcxx-ng` (external/evogym/README.md, "
            "'Known Issues'). If it mentions a missing display/X server (e.g. "
            "`cannot connect to X server`, `DISPLAY` errors) on a remote/CI machine, run "
            "under a virtual framebuffer, e.g. `xvfb-run -a python -m "
            "evodevo.visualization.check`.\n\n"
            f"Original traceback:\n{tb}"
        )
        return result.failed(diagnosis)

    if frame is None:
        return result.failed(
            'env.render() returned None instead of an image array -- render_mode was not '
            'honoured. Double check the env was constructed with render_mode="img" '
            '(never "human", which opens a debug window and returns nothing).'
        )
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return result.failed(
            f"env.render() returned something other than an (H, W, 3) RGB array: "
            f"{type(frame)} shape={getattr(frame, 'shape', None)}."
        )
    return result.passed()


def check_imageio_gif_and_mp4() -> CheckResult:
    result = CheckResult("imageio can write gif and mp4")
    if importlib.util.find_spec("imageio") is None:
        return result.failed(
            "the `imageio` package is not installed. Fix: `pip install imageio "
            "imageio-ffmpeg` (both are listed in requirements.txt)."
        )

    try:
        import imageio

        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
        frames[1][:] = 255

        with tempfile.TemporaryDirectory() as tmp:
            gif_path = os.path.join(tmp, "check.gif")
            mp4_path = os.path.join(tmp, "check.mp4")

            imageio.mimsave(gif_path, frames, fps=5)
            if not (os.path.exists(gif_path) and os.path.getsize(gif_path) > 0):
                return result.failed(f"imageio.mimsave wrote no data to {gif_path}.")

            try:
                imageio.mimwrite(mp4_path, frames, fps=5, macro_block_size=None)
            except Exception:
                tb = traceback.format_exc()
                ffmpeg_present = importlib.util.find_spec("imageio_ffmpeg") is not None
                hint = (
                    "the `imageio-ffmpeg` package (which bundles a portable ffmpeg binary) "
                    "does not appear to be installed -- fix: `pip install imageio-ffmpeg`."
                    if not ffmpeg_present
                    else "imageio-ffmpeg is installed, but writing the mp4 still failed -- "
                    "see the traceback below; this can indicate a corrupted/incompatible "
                    "bundled ffmpeg binary (try `pip install --force-reinstall imageio-ffmpeg`) "
                    "or a system-level codec issue."
                )
                return result.failed(f"{hint}\n\nOriginal traceback:\n{tb}")

            if not (os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0):
                return result.failed(f"imageio.mimwrite wrote no data to {mp4_path}.")
    except Exception:
        return result.failed(f"unexpected failure writing test gif/mp4:\n{traceback.format_exc()}")

    return result.passed()


def check_gifsicle_optional() -> CheckResult:
    """Not a hard requirement -- reported for visibility only, never a
    failure. See evodevo/visualization/render.py::_maybe_optimize_gif."""
    result = CheckResult("gifsicle (optional gif optimization)")
    if shutil.which("gifsicle") is not None and importlib.util.find_spec("pygifsicle") is not None:
        result.passed()
        result.diagnosis = "available -- gifs will be optimized"
    else:
        result.passed()
        result.diagnosis = "not available -- gifs will be written unoptimized (this is fine, not an error)"
    return result


def run_all() -> list:
    return [
        check_simulator_importable(),
        check_env_creation_and_frame_capture(),
        check_imageio_gif_and_mp4(),
        check_gifsicle_optional(),
    ]


def main(argv=None) -> None:
    results = run_all()

    print("evodevo rendering self-test")
    print("=" * 60)
    all_ok = True
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}")
        if r.diagnosis:
            print(f"      {r.diagnosis}")
        if not r.ok:
            all_ok = False

    print("=" * 60)
    # gifsicle is optional/informational and never affects the overall verdict.
    required = [r for r in results if r.name != "gifsicle (optional gif optimization)"]
    all_required_ok = all(r.ok for r in required)
    if all_required_ok:
        print("ALL REQUIRED CHECKS PASSED.")
    else:
        print("SOME CHECKS FAILED -- see the diagnosis for each [FAIL] line above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
