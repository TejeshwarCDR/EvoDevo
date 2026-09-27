"""Step 2: per-generation orchestration overhead of the REAL evodevo.evolution.ea.run_ea
code path (selection + mutation + decode + bookkeeping + JSONL logging), with an
instant-return mock evaluator (no PPO, no EvoGym). Read-only w.r.t. all results.

Per-generation orchestration = wall-clock of a generation's window minus the time
spent inside the mock evaluator. Window g = [return of last evaluator call of gen g-1,
return of last evaluator call of gen g]; the final generation's summary lands in a tail
window. Nothing in ea.py is reimplemented or patched.
"""
import json, os, sys, tempfile, time, statistics as st
import numpy as np

sys.path.insert(0, os.getcwd())
import evodevo.encoders  # noqa: F401  (registers encoders)
from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.registry import get_encoder
from evodevo.evaluation.ppo_runner import EvalResult
from evodevo.evolution.ea import run_ea
from evodevo.evolution.logging import RunLogger

POP, GENS, REPS = 15, 20, 20
ARMS = {  # kwargs mirror evaluation_cost.json observation meta
    "direct": dict(hidden_channels=3, kw={}),
    "nca": dict(hidden_channels=0, kw={}),
    "rule_based": dict(hidden_channels=3, kw=dict(readout_mode="mode_of_window", readout_window=6)),
}


def one_run(arm, rep, tmp):
    spec = ARMS[arm]
    factors = ExperimentFactors(dev_steps=100, init_condition="center_seed", seed=rep,
                                hidden_channels=spec["hidden_channels"])
    enc = get_encoder(arm, factors=factors, **spec["kw"])
    cfg = ExperimentConfig(seed=rep, encoder=arm, task="Walker-v0", population_size=POP,
                           offspring_size=POP, eval_budget=POP * GENS, ppo_timesteps=1,
                           ppo_eval_episodes=1, output_dir=tmp)
    stamps = []  # (t_enter, t_exit) per evaluator call
    rng = np.random.default_rng(rep)

    # decode() runs in-process inside EvaluationTracker.evaluate, so its time is timed
    # and reported SEPARATELY (it is not selection/mutation/logging). For rule_based the
    # real decode costs ~0.24 s/genome (evaluation_cost.json t_develop), which would blow
    # the one-minute budget, so a body decoded once by the REAL decoder is returned
    # instead; rule_based decode cost is taken from t_develop, not measured here.
    real_decode = enc.decode
    dec_time = [0.0]
    cached_body = [None]

    def timed_decode(genome):
        t0 = time.perf_counter()
        if arm == "rule_based":
            if cached_body[0] is None:
                cached_body[0] = real_decode(genome)
            b = cached_body[0]
            dec_time[0] += 0.0  # stubbed: excluded from measurement entirely
        else:
            b = real_decode(genome)
            dec_time[0] += time.perf_counter() - t0
        return b
    if arm == "rule_based":  # decode the cached body BEFORE the timed region starts
        cached_body[0] = real_decode(enc.random_genome(np.random.default_rng(10_000 + rep)))
    enc.decode = timed_decode

    def mock_eval(body, task, ppo_timesteps, eval_episodes, seed, sentinel_fitness):
        t0 = time.perf_counter()
        r = EvalResult(fitness=float(rng.random()), valid=True, reason=None, wall_clock_s=0.0,
                       body_hash=str(len(stamps)))
        stamps.append((t0, time.perf_counter()))
        return r

    logger = RunLogger(os.path.join(tmp, f"{arm}_{rep}"), encoder=arm, seed=rep)
    t_start = time.perf_counter()
    res = run_ea(enc, mock_eval, cfg, logger=logger)
    t_end = time.perf_counter()
    logger.close()
    assert res.evaluations_used == POP * GENS and len(stamps) == POP * GENS
    per_gen = []
    dec_per_gen = []
    prev_exit = t_start
    for g in range(GENS):
        calls = stamps[g * POP:(g + 1) * POP]
        last_exit = calls[-1][1]
        mock_t = sum(b - a for a, b in calls)
        per_gen.append((last_exit - prev_exit) - mock_t)
        dec_per_gen.append(0.0)
        prev_exit = last_exit
    # the final generation's summary/logging tail is folded into the last generation
    per_gen[-1] += (t_end - prev_exit)
    # decode time is summed per run (not attributed to a window) and subtracted evenly
    dec_each = dec_time[0] / GENS
    return [x - dec_each for x in per_gen], dec_each


def main():
    out = {"population_size": POP, "generations": GENS, "repetitions": REPS, "arms": {}}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        t0 = time.perf_counter()
        for arm in ARMS:
            run_means, all_gen, dec_all = [], [], []
            for rep in range(REPS):
                pg, dec_each = one_run(arm, rep, tmp)
                run_means.append(st.mean(pg)); all_gen += pg; dec_all.append(dec_each)
            out["arms"][arm] = {
                "n_generation_samples": len(all_gen),
                "decode_in_launcher_process_per_generation_s_measured": st.mean(dec_all) if arm != "rule_based" else None,
                "decode_note": "rule_based decode stubbed (see source); use evaluation_cost.json t_develop x pop" if arm == "rule_based" else "real decode, timed and EXCLUDED from per_generation_s",
                "per_generation_s": {"mean": st.mean(all_gen), "sd": st.stdev(all_gen),
                                     "min": min(all_gen), "max": max(all_gen)},
                "per_run_mean_s": {"mean": st.mean(run_means), "sd": st.stdev(run_means),
                                   "min": min(run_means), "max": max(run_means)},
            }
        out["total_measurement_wall_s"] = time.perf_counter() - t0
    out["headline_direct_arm_per_generation_s"] = out["arms"]["direct"]["per_generation_s"]
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
