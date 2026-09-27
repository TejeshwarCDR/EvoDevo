"""P1 Step 4a-4c for rule_based_monotone (plus rule_based under identical conditions).
No PPO. Reuses the project's own measurement code unchanged:
  4a  evodevo.experiments.inspect_encoder.inspect_encoder (seed 0, n 2000), fed the
      campaign-resolved kwargs from the live benchmark path.
  4b  scripts/capacity_planning/measure_validity_current_calibration.run_one (round-3
      no-PPO proxy: real run_ea, pop 15, 20 generations, 20 seeds, random + fixed proxy).
  4c  evodevo.preflight.stages.b3_cost._measure_develop/_measure_sentinel, n=100 each,
      seeded exactly as B3 would seed arm index 3 (workload_seed 0), single process pinned
      to P-core 0.
"""
import json, math, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/capacity_planning")

OUT = "results/campaign_prep/monotone_cost"
ARMS = ("rule_based_monotone", "rule_based")


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def timings():
    os.sched_setaffinity(0, {0})
    from evodevo.core.factors import extract_factors
    from evodevo.encoders import get_encoder
    from evodevo.preflight.stages.b2_sweep import _resolved_dev_steps_and_dynamics
    from evodevo.preflight.stages import b3_cost as b3
    dev_steps, init, dyn = _resolved_dev_steps_and_dynamics()
    out = {}
    for arm_id, arm_index in (("rule_based_monotone", 3), ("rule_based", 2)):
        enc_name, kw = dyn[arm_id]
        base = 0 * 1_000_000_000 + arm_index * b3.ARM_SEED_STRIDE
        factors, ekw = extract_factors({"dev_steps": dev_steps, "init_condition": init, **kw}, seed=base)
        enc = get_encoder(enc_name, factors=factors, **ekw)
        dev = b3._measure_develop(enc, 100, base + b3._DEVELOP_OFFSET)
        sen, draws = b3._measure_sentinel(enc, 100, b3.DEFAULT_COST_MAX_DRAWS, base + b3._SENTINEL_OFFSET)
        out[arm_id] = {"seed_base": base, "encoder_kwargs": ekw,
                       "t_develop": b3.describe_values(dev), "t_develop_p99": b3._percentile(dev, 99),
                       "t_sentinel": b3.describe_values(sen), "t_sentinel_p99": b3._percentile(sen, 99),
                       "t_sentinel_draw_attempts": draws, "t_develop_raw": dev, "t_sentinel_raw": sen}
    return out


def inspect(arm_id):
    from measure_validity_current_calibration import make_encoder
    from evodevo.experiments.inspect_encoder import inspect_encoder
    _, kw = make_encoder(arm_id, 0)
    t0 = time.time()
    st = inspect_encoder("rule_based", seed=0, n=2000, output_dir=f"{OUT}/sanity", encoder_kwargs=dict(kw),
                         report_prefix=arm_id)
    b = st["rule_based"]
    k = round(b["valid_rate"] * 2000)
    return arm_id, {"kwargs": kw, "n": 2000, "k_valid": k, "valid_rate": b["valid_rate"], "wilson95": wilson(k, 2000),
                    "invalid_reason_counts": b["invalid_reason_counts"], "degenerate_fraction": b["degenerate_fraction"],
                    "never_settled_fraction": b.get("never_settled_fraction"), "config": st["rule_based_config"],
                    "wall_s": time.time() - t0}


def round3(args):
    from measure_validity_current_calibration import run_one
    return run_one(args)


def main():
    t0 = time.time()
    # FIX (2026-09-27): CPU affinity is inherited by child processes. The original version
    # called timings() -- which pins ITS process to CPU 0 -- in the parent, so the Pool created
    # below inherited the single-CPU mask and all 26 workers ran time-sliced on CPU 0 (7,299 s
    # instead of ~10 min). The pinned timing measurement now runs in its own child process.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1) as ex:
        res = {"timings": ex.submit(timings).result()}
    res["timings_wall_s"] = time.time() - t0
    json.dump(res, open(f"{OUT}/cpu_measurements.json", "w"), indent=1)
    print("timings done", res["timings_wall_s"], flush=True)
    jobs = [(a, s, p, None) for a in ARMS for p in ("random", "fixed") for s in range(20)]
    with Pool(26) as pool:
        ins = pool.map_async(inspect, ARMS)
        r3 = pool.map(round3, jobs, chunksize=1)
        ins = ins.get()
    res["inspection"] = dict(ins)
    res["round3"] = {"pop": 15, "gens": 20, "n_seeds": 20, "counts": {}}
    for arm, seed, proxy, hist, v in r3:
        res["round3"]["counts"].setdefault(f"{arm}|{proxy}|current", {})[seed] = v
    res["total_wall_s"] = time.time() - t0
    json.dump(res, open(f"{OUT}/cpu_measurements.json", "w"), indent=1)
    print("done", res["total_wall_s"], flush=True)


if __name__ == "__main__":
    main()
