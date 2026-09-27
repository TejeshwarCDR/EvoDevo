"""P1 Step 4d/4e: t_PPO for 3 valid rule_based_monotone genomes per task plus ONE rule_based
anchor genome per task, all 8 trainings concurrently in one EvaluationPool pinned to the 8
P-core primaries (evodevo.parallel.topology.cpu_slots(topology, 8)), mode "real",
eval_episodes = CAMPAIGN_EVAL_EPISODES, per-task budgets from docs/decisions.md's D6
application row (Walker-v0 1,000,000; UpStepper-v0 1,200,000).

Genome selection, fixed before running:
  monotone: the first 3 valid draws from B3's own valid-draw seed range for arm index 3
            (workload_seed 0): 300,000,000 + 3,000,000 + i.
  anchor:   rule_based genome_seed 203000001 -- B3's own rule_based genome (also B4's
            "middle" UpStepper body), so its t_PPO can be compared with B3's measurement of
            the SAME genome at 1M timesteps; the body hash is checked against B3's record.
Task seed = genome_seed, as in B3. Also samples P-core clock frequency every 15 s, because
B3 ran only 3 trainings at a time and this batch runs 8.
"""
import json, os, sys, threading, time
sys.path.insert(0, os.getcwd())
# FIX (2026-09-27): only stdlib at module level. EvaluationPool uses the "spawn" start method,
# which re-imports this script in every worker BEFORE _worker_initializer runs; a top-level
# numpy import there trips configure_threading(strict=True)'s ThreadingConfiguredTooLateError.

OUT = "results/campaign_prep/monotone_cost"
BUDGET = {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
B3_PATH = ("results/preflight/20260922T040932Z_4be8402b/stages/evaluation_cost.json")


def build(arm_id, arm_index):
    from evodevo.core.factors import extract_factors
    from evodevo.encoders import get_encoder
    from evodevo.preflight.stages.b2_sweep import _resolved_dev_steps_and_dynamics
    from evodevo.preflight.stages import b3_cost as b3
    dev_steps, init, dyn = _resolved_dev_steps_and_dynamics()
    enc_name, kw = dyn[arm_id]
    base = arm_index * b3.ARM_SEED_STRIDE
    factors, ekw = extract_factors({"dev_steps": dev_steps, "init_condition": init, **kw}, seed=base)
    enc = get_encoder(enc_name, factors=factors, **ekw)
    fd = {k: v for k, v in factors.__dict__.items() if k != "seed"}
    return enc_name, enc, {"factors": fd, "encoder_kwargs": ekw}, base


def main():
    import numpy as np
    from evodevo.morphology.metrics import body_hash
    from evodevo.parallel.topology import detect_topology, cpu_slots
    from evodevo.parallel.worker_pool import EvaluationPool, EvaluationTask
    from evodevo.preflight.stages.b2_sweep import CAMPAIGN_EVAL_EPISODES
    from evodevo.preflight.stages import b3_cost as b3
    B3 = json.load(open(B3_PATH))
    # No parent pinning (FIX 2026-09-27): affinity is inherited by children; each pool worker
    # pins itself in evodevo.parallel.worker_pool._worker_initializer, the parent stays unpinned.
    genomes = []
    name, enc, meta, base = build("rule_based_monotone", 3)
    valid, draws = b3._draw_valid_genomes(enc, 3, b3.DEFAULT_COST_MAX_DRAWS, base + b3._VALID_DRAW_OFFSET)
    for gs, g in valid:
        genomes.append(("rule_based_monotone", name, enc, meta, gs, g))
    rname, renc, rmeta, _ = build("rule_based", 2)
    g = renc.random_genome(np.random.default_rng(203000001))
    h = body_hash(renc.decode(g))
    b3_hash = {o["genome_seed"]: o["body_hash"] for o in B3["arms"]["rule_based"]["tasks"]["Walker-v0"]["observations"]}[203000001]
    assert h == b3_hash, (h, b3_hash)
    genomes.append(("rule_based", rname, renc, rmeta, 203000001, g))

    tasks, info = [], {}
    for task in BUDGET:
        for arm_id, ename, e, m, gs, gen in genomes:
            tid = f"{arm_id}|{task}|{gs}"
            info[tid] = dict(arm_id=arm_id, task=task, genome_seed=gs, ppo_timesteps=BUDGET[task],
                             body_hash=body_hash(e.decode(gen)))
            tasks.append(EvaluationTask(task_id=tid, arm=ename, strategy="campaign_prep_p1", task=task, seed=gs,
                                        genome=e.serialize_genome(gen), ppo_timesteps=BUDGET[task],
                                        eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=m))
    plan = cpu_slots(detect_topology(), 8)
    freq, stop = [], threading.Event()

    def sample():
        cpus = [s[0] for s in plan.to_dict()["slots"]]
        while not stop.is_set():
            row = {"t": time.time()}
            for c in cpus:
                try:
                    row[c] = int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read()) / 1000
                except OSError:
                    pass
            freq.append(row); stop.wait(15)
    th = threading.Thread(target=sample, daemon=True); th.start()
    t0 = time.time()
    with EvaluationPool(n_workers=8, plan=plan, mode="real", timeout_s=4500.0, max_retries=1) as pool:
        results = []
        with open(f"{OUT}/tppo_results.jsonl", "w") as jf:          # partial results survive a stop
            for r in pool.map_tasks(tasks):
                results.append(r)
                d = dict(info[r.task_id]); d.update(wall_clock_s=r.wall_clock_s, fitness=r.fitness, cpu_ids=list(r.cpu_ids),
                                                     timed_out=r.timed_out, error=r.error)
                jf.write(json.dumps(d) + "\n"); jf.flush()
        stats = pool.stats()
    stop.set(); wall = time.time() - t0
    rows = []
    for r in results:
        d = dict(info[r.task_id]); d.update(fitness=r.fitness, valid=r.valid, reason=r.reason, wall_clock_s=r.wall_clock_s,
                                             worker_body_hash=r.body_hash, cpu_ids=list(r.cpu_ids), peak_rss_mb=r.peak_rss_mb,
                                             timed_out=r.timed_out, error=r.error, affinity_result=r.affinity_result)
        rows.append(d)
    json.dump({"plan": plan.to_dict(), "eval_episodes": CAMPAIGN_EVAL_EPISODES, "budgets": BUDGET,
               "monotone_valid_draw_attempts": draws, "batch_wall_s": wall, "pool_stats": stats,
               "results": rows, "freq_mhz_samples": freq}, open(f"{OUT}/tppo_measurements.json", "w"), indent=1, default=str)
    print("done", wall)


if __name__ == "__main__":
    main()
