"""The single shared fitness evaluator.

This is the only module in the project allowed to import stable_baselines3
or evogym environments. Both encoding arms (direct, NCA, and any future
encoding) must be evaluated exclusively through evaluate_body(), so that
nothing about the PPO protocol can differ between arms.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from evodevo.morphology.metrics import body_hash as compute_body_hash
from evodevo.morphology.metrics import body_metrics
from evodevo.morphology.validate import validate_body

DEFAULT_SENTINEL_FITNESS = -10.0


@dataclass
class EvalResult:
    fitness: float
    valid: bool
    reason: Optional[str]
    wall_clock_s: float
    body_hash: str


def evaluate_body(
    body: np.ndarray,
    task: str,
    ppo_timesteps: int,
    eval_episodes: int,
    seed: int,
    output_dir: Optional[str] = None,
    sentinel_fitness: float = DEFAULT_SENTINEL_FITNESS,
) -> EvalResult:
    start = time.time()
    h = compute_body_hash(body)

    validation = validate_body(body)
    if not validation.valid:
        return EvalResult(
            fitness=sentinel_fitness,
            valid=False,
            reason="; ".join(validation.reasons),
            wall_clock_s=time.time() - start,
            body_hash=h,
        )

    import evogym.envs  # noqa: F401  registers evogym gym environments
    from evogym import get_full_connectivity
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    connections = get_full_connectivity(body)

    # A single (encoder, seed) evaluation must never abort the whole 11-hour
    # multi-seed benchmark: an unstable simulation (e.g. NaN/inf physics
    # state, a solver blow-up, or any other exception raised by evogym/
    # stable_baselines3 while training or rolling out this one body) is
    # recorded as an invalid evaluation with the standard sentinel fitness --
    # exactly like a structurally-invalid body -- and evaluations_used still
    # advances by one in ea.py, so no encoder can gain a silent budget
    # advantage by crashing more or less often than another. This is
    # deliberately scoped to only this one evaluation's PPO train+eval call;
    # it must not be widened to catch anything outside this block.
    vec_env = None
    eval_env = None
    try:
        env_kwargs = {"body": body, "connections": connections}
        vec_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs=env_kwargs)

        model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
        model.learn(total_timesteps=ppo_timesteps)

        eval_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs=env_kwargs)
        episode_returns = []
        for _ in range(eval_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, _infos = eval_env.step(action)
                total_reward += float(reward[0])
                done = bool(dones[0])
            episode_returns.append(total_reward)

        fitness = float(np.mean(episode_returns))

        return EvalResult(
            fitness=fitness,
            valid=True,
            reason=None,
            wall_clock_s=time.time() - start,
            body_hash=h,
        )
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any
        # simulation/training instability for THIS body must degrade to a
        # recorded failure, not kill the run. See comment above.
        return EvalResult(
            fitness=sentinel_fitness,
            valid=False,
            reason=f"simulation_instability: {type(exc).__name__}: {exc}",
            wall_clock_s=time.time() - start,
            body_hash=h,
        )
    finally:
        if vec_env is not None:
            vec_env.close()
        if eval_env is not None:
            eval_env.close()


def evaluate_body_surrogate(
    body: np.ndarray,
    task: str = "",
    ppo_timesteps: int = 0,
    eval_episodes: int = 0,
    seed: int = 0,
    output_dir: Optional[str] = None,
    sentinel_fitness: float = DEFAULT_SENTINEL_FITNESS,
) -> EvalResult:
    """FAST, deterministic, physics-free stand-in for evaluate_body.

    NOT a scientific result. Used only to test the EA loop and pipeline
    plumbing without paying for PPO training. Fitness is a cheap function of
    body_metrics (rewards actuator count and horizontal symmetry) and has no
    connection to actual locomotion performance.
    """
    start = time.time()
    h = compute_body_hash(body)

    validation = validate_body(body)
    if not validation.valid:
        return EvalResult(
            fitness=sentinel_fitness,
            valid=False,
            reason="; ".join(validation.reasons),
            wall_clock_s=time.time() - start,
            body_hash=h,
        )

    metrics = body_metrics(body)
    fitness = (
        metrics["h_act_count"] + metrics["v_act_count"]
    ) + metrics["horizontal_symmetry_score"]

    return EvalResult(
        fitness=float(fitness),
        valid=True,
        reason=None,
        wall_clock_s=time.time() - start,
        body_hash=h,
    )


def make_evaluator(mode: str) -> Callable[..., EvalResult]:
    if mode == "real":
        return evaluate_body
    if mode == "surrogate":
        return evaluate_body_surrogate
    raise ValueError(f"unknown evaluator mode '{mode}'; expected 'real' or 'surrogate'")
