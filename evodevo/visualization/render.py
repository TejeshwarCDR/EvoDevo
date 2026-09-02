"""Animated rollout rendering: instantiate the real EvoGym environment with
render_mode="img" (never "human" -- that opens a debug window and is not
usable in an automated/headless path), step a trained controller (or
random actions, if none is supplied) for n_steps, collect frames, and write
both an mp4 and a gif.

This is the one place besides evodevo/evaluation/ppo_runner.py that may
import evogym/stable_baselines3 -- it must never be imported from
evodevo/evolution/ea.py or any encoder module, and rendering must never be
allowed to crash a benchmark run (see evodevo/visualization/__init__.py's
callers, which always wrap these functions in try/except).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from evodevo.morphology.validate import validate_body


@dataclass
class RolloutResult:
    mp4_path: Optional[str]
    gif_path: Optional[str]
    n_frames: int
    used_controller: bool
    label: str


def _make_env(body: np.ndarray, task: str, seed: int = 0):
    # CRITICAL: an invalid (e.g. disconnected) body handed to EvoGym's
    # compiled C++ simulator can SEGFAULT the whole process rather than
    # raising a catchable Python exception -- no try/except in this
    # process can recover from that. Refusing here, before the simulator
    # is ever touched, is the only way "rendering must never crash a
    # benchmark run" can actually hold; a benchmark's "best" body CAN be
    # invalid (e.g. every individual in a tiny/unlucky run was invalid,
    # so the max-sentinel-fitness individual is still an invalid body).
    validation = validate_body(body)
    if not validation.valid:
        raise ValueError(
            f"refusing to render an invalid body (would risk crashing EvoGym's simulator): "
            f"{'; '.join(validation.reasons)}"
        )

    import evogym.envs  # noqa: F401  registers evogym gym environments
    import gymnasium as gym
    from evogym import get_full_connectivity

    connections = get_full_connectivity(body)
    env = gym.make(task, body=body, connections=connections, render_mode="img")
    env.reset(seed=seed)
    return env


def _load_controller(controller_path: str):
    from stable_baselines3 import PPO

    return PPO.load(controller_path)


def render_body_rollout(
    body: np.ndarray,
    controller_path: Optional[str],
    task: str,
    n_steps: int,
    output_path: str,
    fps: int = 30,
    seed: int = 0,
    optimize_gif: bool = True,
) -> RolloutResult:
    """Roll out `body` in `task` for n_steps, rendering every step, and
    write two files: f"{output_path}.mp4" and f"{output_path}.gif".

    If controller_path is given, loads a trained stable_baselines3 PPO
    model and acts deterministically; otherwise falls back to random
    actions, and the returned RolloutResult.label makes that explicit (the
    caller should burn this label into the output, e.g. as a figure/file
    caption) so a random-action rollout is never mistaken for a trained one.
    """
    import imageio

    # Resolve to an absolute path BEFORE constructing the env: on some
    # platforms, creating an EvoGym env (which initializes an offscreen
    # OpenGL/GLFW context even for render_mode="img") has the side effect
    # of changing the process's current working directory (observed on
    # macOS). A relative output_path resolved after that point would
    # silently land somewhere unrelated to the caller's cwd.
    output_path = os.path.abspath(output_path)
    if controller_path is not None:
        controller_path = os.path.abspath(controller_path)

    env = _make_env(body, task, seed=seed)
    used_controller = controller_path is not None
    model = _load_controller(controller_path) if used_controller else None
    label = f"trained controller ({os.path.basename(controller_path)})" if used_controller else "RANDOM ACTIONS (no controller supplied)"

    frames = []
    try:
        obs, _info = env.reset(seed=seed)
        frames.append(env.render())

        for _ in range(n_steps):
            if model is not None:
                action, _state = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
    finally:
        env.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mp4_path = f"{output_path}.mp4"
    gif_path = f"{output_path}.gif"

    imageio.mimwrite(mp4_path, frames, fps=fps, macro_block_size=None)
    # The gif (pillow) plugin deprecated `fps` in favour of `duration` (ms
    # per frame); the ffmpeg-backed mp4 writer above still takes `fps`.
    imageio.mimsave(gif_path, frames, duration=1000.0 / fps)

    if optimize_gif:
        _maybe_optimize_gif(gif_path)

    return RolloutResult(
        mp4_path=mp4_path,
        gif_path=gif_path,
        n_frames=len(frames),
        used_controller=used_controller,
        label=label,
    )


def _maybe_optimize_gif(gif_path: str) -> bool:
    """Best-effort gif size optimization via the external `gifsicle`
    binary, if present on PATH. pygifsicle is deliberately NOT a project
    dependency (it shells out to a binary that isn't guaranteed to exist on
    every machine) -- this degrades gracefully to a no-op when gifsicle
    isn't installed, rather than failing the whole render."""
    import shutil

    if shutil.which("gifsicle") is None:
        return False

    try:
        from pygifsicle import optimize

        optimize(gif_path)
        return True
    except ImportError:
        return False
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"gif optimization failed for {gif_path}, keeping the unoptimized file: {e}")
        return False
