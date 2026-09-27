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
    # Assigned by the proposal dispatcher before EvolutionRun.tell(); the
    # evaluator itself deliberately remains unaware of EA ordering.
    eval_index: Optional[int] = None


@dataclass
class CurveResult:
    curve: list  # [(timesteps: int, fitness: float), ...], empty if invalid/unstable
    fitness: float  # the final point's fitness (curve[-1][1] when valid)
    valid: bool
    reason: Optional[str]
    wall_clock_s: float
    body_hash: str


def _rollout_fitness(model, eval_env, eval_episodes: int) -> float:
    """The exact rollout methodology evaluate_body()'s own final evaluation
    uses -- deterministic policy, full episodes, mean return across
    eval_episodes. Factored out so evaluate_body_with_curve's periodic
    checkpoints use IDENTICAL methodology to a normal evaluate_body() call,
    not a cheaper approximation that would make curve points incomparable
    to every other measurement in this project."""
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
    return float(np.mean(episode_returns))


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
        fitness = _rollout_fitness(model, eval_env, eval_episodes)

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


def evaluate_body_with_curve(
    body: np.ndarray,
    task: str,
    max_timesteps: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
    sentinel_fitness: float = DEFAULT_SENTINEL_FITNESS,
) -> CurveResult:
    """Trains PPO to max_timesteps ONCE, periodically evaluating the
    current policy every eval_interval timesteps via an SB3 callback --
    never retrains from scratch per checkpoint (which would multiply cost
    by the number of checkpoints for no additional information; see
    evodevo/preflight/stages/b4_learning_curves.py, the only caller). Uses
    _rollout_fitness -- the exact same rollout methodology evaluate_body()
    itself uses -- for every checkpoint, so a curve's final point matches
    what evaluate_body() would report for the same body/task/seed/
    max_timesteps, and every checkpoint is directly comparable to every
    other measurement in this project.

    Same invalid-genome and simulation-instability handling as
    evaluate_body(): an invalid body returns immediately with an empty
    curve and the sentinel fitness; a training/simulation exception
    degrades to a recorded failure (empty curve), never propagates.
    """
    start = time.time()
    h = compute_body_hash(body)

    validation = validate_body(body)
    if not validation.valid:
        return CurveResult(
            curve=[],
            fitness=sentinel_fitness,
            valid=False,
            reason="; ".join(validation.reasons),
            wall_clock_s=time.time() - start,
            body_hash=h,
        )

    import evogym.envs  # noqa: F401  registers evogym gym environments
    from evogym import get_full_connectivity
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    connections = get_full_connectivity(body)

    class _PeriodicEvalCallback(BaseCallback):
        def __init__(self, eval_env, eval_episodes, eval_interval):
            super().__init__()
            self._eval_env = eval_env
            self._eval_episodes = eval_episodes
            self._eval_interval = eval_interval
            self._next_checkpoint = eval_interval
            self.curve: list = []

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_checkpoint:
                fitness = _rollout_fitness(self.model, self._eval_env, self._eval_episodes)
                self.curve.append((int(self.num_timesteps), fitness))
                self._next_checkpoint += self._eval_interval
            return True

    vec_env = None
    eval_env = None
    try:
        env_kwargs = {"body": body, "connections": connections}
        vec_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs=env_kwargs)
        eval_env = make_vec_env(task, n_envs=1, seed=seed, env_kwargs=env_kwargs)

        model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
        callback = _PeriodicEvalCallback(eval_env, eval_episodes, eval_interval)
        model.learn(total_timesteps=max_timesteps, callback=callback)

        curve = list(callback.curve)
        # PPO's total_timesteps is a MINIMUM, not a hard cap: SB3 collects
        # full rollout-buffer batches (n_steps=2048 by default here, since
        # evaluate_body()'s own PPO(...) call never overrides it, and this
        # function must not either -- see module docstring), so training
        # can genuinely finish a bit past max_timesteps. The callback's own
        # last recorded point already reflects the fully-trained policy at
        # its REAL final step count; the only gap to fill is when training
        # finishes before even the FIRST periodic checkpoint could fire
        # (e.g. max_timesteps < eval_interval) -- there, and only there,
        # record one point at the model's actual final step count, never a
        # fabricated "max_timesteps" label that may not match what really
        # happened.
        if not curve:
            curve.append((int(model.num_timesteps), _rollout_fitness(model, eval_env, eval_episodes)))

        return CurveResult(
            curve=curve,
            fitness=curve[-1][1],
            valid=True,
            reason=None,
            wall_clock_s=time.time() - start,
            body_hash=h,
        )
    except Exception as exc:  # noqa: BLE001 -- same discipline as evaluate_body(): a
        # simulation/training instability for THIS body must degrade to a
        # recorded failure, not kill the run.
        return CurveResult(
            curve=[],
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
