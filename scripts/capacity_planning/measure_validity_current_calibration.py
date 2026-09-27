"""Round 3: offspring validity-by-generation under the CURRENT calibrated encoder/mutation
config, measured with the REAL evodevo.evolution.ea.run_ea loop and NO PPO / NO EvoGym.

Encoder construction reuses the live benchmark path (benchmark._resolved_arm_kwargs +
extract_factors + get_encoder), so calibrated mutation strengths, T_primary=100, NCA/rule_based
resolved dynamics all come from the live config, not a copy.

Fitness proxy: evodevo.morphology.validate.validate_body (the SAME first stage evaluate_body
runs) -> invalid => sentinel_fitness (-10, ranked last, exactly as in reality);
valid => proxy fitness (random U[0,1) by default, or constant 1.0 for the 'fixed' variant).
Justification: validity of a genome is decided pre-PPO and task-free (see report), and the only
selection effect on validity in a (mu+lambda) loop is valid-outranks-invalid. LIMITATION: among
valid parents, real fitness would favour some over others; if fitness correlates with mutational
robustness (validity of offspring), real curves could differ. Round 3 tests the proxy against
the historical direct run (real fitness) under matched (historical) mutation settings."""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.getcwd())
import evodevo.encoders  # noqa: F401
from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders.registry import get_encoder
from evodevo.evaluation.ppo_runner import EvalResult
from evodevo.experiments import benchmark as bm
from evodevo.evolution.ea import run_ea
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body

RUN = "results/preflight/20260922T122730Z_4be8402b"
OUT = f"{RUN}/report/capacity_planning"
POP, GENS, N_SEEDS = 15, 20, 20
ARMS = ("direct", "nca", "rule_based")


def make_encoder(arm, seed, override=None):
    bconfig = bm.BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
    spec = next(a for a in bconfig.resolved_arms() if a.arm_id == arm)
    kwargs = bm._resolved_arm_kwargs(bconfig, spec)
    if override:
        kwargs.update(override)
    factors, remaining = extract_factors(kwargs, seed=seed)
    return get_encoder(spec.encoder, factors=factors, grid_shape=(5, 5), **remaining), kwargs


def run_one(args):
    arm, seed, proxy, override = args
    enc, _ = make_encoder(arm, seed, override)
    prng = np.random.default_rng(10_000_000 + seed)

    def evaluator(body, task, ppo_timesteps, eval_episodes, seed, sentinel_fitness):
        v = validate_body(body)
        h = body_hash(body)
        if not v.valid:
            return EvalResult(fitness=sentinel_fitness, valid=False, reason="; ".join(v.reasons), wall_clock_s=0.0, body_hash=h)
        f = float(prng.random()) if proxy == "random" else 1.0
        return EvalResult(fitness=f, valid=True, reason=None, wall_clock_s=0.0, body_hash=h)

    cfg = ExperimentConfig(seed=seed, encoder=arm, task="Walker-v0", population_size=POP, offspring_size=POP,
                           eval_budget=POP * GENS, ppo_timesteps=1, ppo_eval_episodes=1, output_dir="unused")
    res = run_ea(enc, evaluator, cfg)
    v = np.zeros(GENS)
    for r in res.records:
        v[r["generation"]] += r["valid"]
    return arm, seed, proxy, bool(override), v.tolist()


def main():
    t0 = time.time()
    tasks = [(a, s, "random", None) for a in ARMS for s in range(N_SEEDS)]
    tasks += [(a, s, "fixed", None) for a in ARMS for s in range(N_SEEDS)]
    # proxy validation: direct under the HISTORICAL mutation setting (0.1) with the random proxy
    tasks += [("direct", s, "random", {"mutation_rate": 0.1}) for s in range(N_SEEDS)]
    with Pool(min(24, os.cpu_count())) as p:
        results = p.map(run_one, tasks, chunksize=1)
    out = {"n_seeds": N_SEEDS, "pop": POP, "gens": GENS, "live_kwargs": {}, "counts": {}}
    for a in ARMS:
        _, kw = make_encoder(a, 0)
        out["live_kwargs"][a] = kw
    for arm, seed, proxy, hist, v in results:
        key = f"{arm}|{proxy}|{'histmut' if hist else 'current'}"
        out["counts"].setdefault(key, {})[seed] = v      # valid count per generation (of 15) per seed
    out["wall_s"] = time.time() - t0
    json.dump(out, open(f"{OUT}/validity_current_calibration.json", "w"), indent=1)
    print("done", out["wall_s"])


if __name__ == "__main__":
    main()
