"""Worked trace of FIFO vs lowest-generation-first scheduling in the pooled INTERLEAVED
Walker phase (15 runs = 3 arms x 5 seeds, 20 workers), using the historical validity curves.
Analysis only; changes no launcher code."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(__file__))
import simulate_campaign as sc

inp = sc.load_inputs(); vg = json.load(open(f"{sc.OUT}/validity_by_generation.json"))["arms"]
sim = json.load(open(f"{sc.OUT}/simulation_results.json"))
curves = sim["historical_validity_curves_used"]
f20 = sc.speed_factors(sc.worker_speeds(20), inp["ratio"])
H = 3600.0
res = {}
for label, kw in (("BASE_flat_validity", {}), ("HIST_curve_validity", dict(validity_curves=curves))):
    model = sc.Model(inp, **kw)
    out = {}
    for pol in ("fifo", "lgf"):
        fin = {a: [] for a in sc.ARMS}; mk = []; prog = {a: [] for a in sc.ARMS}; spread = []
        for trial in range(60):
            rng = np.random.default_rng(1000 + trial)          # same draws for both policies
            keys = [(a, s) for a in sc.ARMS for s in range(sc.SEEDS)]
            draws = [model.draw_run(rng, a, "Walker-v0") for a, s in keys]
            tr = []
            ms, fn = sc.run_interleaved(draws, [model.overhead(a) for a, _ in keys], f20, policy=pol, trace=tr)
            mk.append(ms)
            for (a, s), f in zip(keys, fn): fin[a].append(f)
            # progress snapshot at 25% of the phase: generations completed per run
            t25 = 0.25 * ms
            done = np.zeros(len(keys))
            for r, g, t in tr:
                if t <= t25: done[r] += 1
            for i, (a, s) in enumerate(keys): prog[a].append(done[i])
            if trial == 0 and pol in ("fifo", "lgf") and label == "HIST_curve_validity":
                # worked example: first direct run, seed index 0 -> per-generation time in pool vs its own longest task
                r0 = 0; ts = sorted((g, t) for r, g, t in tr if r == r0)
                prev = 0.0; rows = []
                for g, t in ts:
                    d = draws[r0][g]
                    rows.append(dict(gen=g, n_valid_ppo_tasks=int((d > 100).sum()), longest_task_s_Pcore=float(d.max()),
                                     gen_time_in_pool_s=float(t - prev - model.overhead("direct")), done_at_h=t / H))
                    prev = t
                out.setdefault("example_direct_run", {})[pol] = rows
        out[pol] = dict(mean_finish_h={a: float(np.mean(fin[a])) / H for a in sc.ARMS}, phase_makespan_h=float(np.mean(mk)) / H,
                        gens_done_at_25pct_of_phase={a: float(np.mean(prog[a])) for a in sc.ARMS})
    res[label] = out
json.dump(res, open(f"{sc.OUT}/scheduling_trace.json", "w"), indent=2)
for label, out in res.items():
    print(label)
    for pol in ("fifo", "lgf"):
        o = out[pol]; print(" ", pol, "makespan %.1fh" % o["phase_makespan_h"], "finish(h):", {a: round(v, 1) for a, v in o["mean_finish_h"].items()}, "gens@25%:", {a: round(v, 1) for a, v in o["gens_done_at_25pct_of_phase"].items()})
ex = res["HIST_curve_validity"]["example_direct_run"]
for pol in ("fifo", "lgf"):
    print(pol, "example direct run (trial 0, seed 0):")
    for r in ex[pol]: print("   gen %2d valid-PPO tasks %d  longest task %8.1fs  time-in-pool %9.1fs  done at %.2fh" % (r["gen"], r["n_valid_ppo_tasks"], r["longest_task_s_Pcore"], r["gen_time_in_pool_s"], r["done_at_h"]))


# =====================  ROUND 3: solo-run figures and the inline-sentinel variant  =====================
# Third scheduling variant "inline": invalid genomes (sentinel evaluations, decode+validate only) are
# resolved in the launcher process and take NO pool slot; their measured sentinel time is charged
# serially to the run. Same draws for every policy within a trial (common random numbers).
cur_curves = sim["current_validity_curves_used"]
SCEN = {"flat": {}, "historical": dict(validity_curves=curves), "current": dict(validity_curves=cur_curves)}


def q(x):
    x = np.asarray(x) / H
    return dict(mean=float(x.mean()), p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))


r3 = {}
for scen, kw in SCEN.items():
    model = sc.Model(inp, **kw)
    o = {}
    # (a) SOLO: one direct run (and the 5-seed direct row) alone on the 20-worker pool
    solo1 = {"fifo": [], "inline": []}; solo5 = {"fifo": [], "inline": []}
    for t_ in range(500):
        rng = np.random.default_rng(5000 + t_)
        dm = [model.draw_run_with_mask(rng, "direct", "Walker-v0") for _ in range(5)]
        for pol in ("fifo", "inline"):
            mk = None if pol == "fifo" else [dm[0][1]]
            solo1[pol].append(sc.run_interleaved([dm[0][0]], [model.overhead("direct")], f20, inline_masks=mk)[0])
            mk5 = None if pol == "fifo" else [x[1] for x in dm]
            solo5[pol].append(sc.run_interleaved([x[0] for x in dm], [model.overhead("direct")] * 5, f20, inline_masks=mk5)[0])
    o["solo_single_direct_run_h"] = {p: q(v) for p, v in solo1.items()}
    o["solo_5seed_direct_row_h"] = {p: q(v) for p, v in solo5.items()}
    # (b) pooled Walker phase (15 runs) and (c) full campaign, FIFO vs inline
    phase = {"fifo": [], "inline": []}; dfin = {"fifo": [], "inline": []}; camp = {"fifo": [], "inline": []}
    ofin = {"fifo": [], "inline": []}
    for t_ in range(100):
        rng = np.random.default_rng(9000 + t_)
        tot = {"fifo": 0.0, "inline": 0.0}
        for task in sc.TASKS:
            keys = [(a, s) for a in sc.ARMS for s in range(sc.SEEDS)]
            dm = [model.draw_run_with_mask(rng, a, task) for a, s in keys]
            for pol in ("fifo", "inline"):
                mk = None if pol == "fifo" else [x[1] for x in dm]
                ms, fn = sc.run_interleaved([x[0] for x in dm], [model.overhead(a) for a, _ in keys], f20, inline_masks=mk)
                tot[pol] += ms
                if task == "Walker-v0":
                    phase[pol].append(ms)
                    dfin[pol].append(np.mean([f for (a, _), f in zip(keys, fn) if a == "direct"]))
                    ofin[pol].append(np.mean([f for (a, _), f in zip(keys, fn) if a != "direct"]))
        for pol in tot: camp[pol].append(tot[pol])
    o["walker_phase_makespan_h"] = {p: q(v) for p, v in phase.items()}
    o["walker_phase_mean_direct_run_finish_h"] = {p: q(v) for p, v in dfin.items()}
    o["walker_phase_mean_nca_rule_run_finish_h"] = {p: q(v) for p, v in ofin.items()}
    o["full_campaign_interleaved_h"] = {p: q(v) for p, v in camp.items()}
    r3[scen] = o
json.dump(r3, open(f"{sc.OUT}/round3_solo_and_inline.json", "w"), indent=2)
print("\nROUND 3")
for scen, o in r3.items():
    f = lambda d: "%.2f [%.2f,%.2f]" % (d["mean"], d["p5"], d["p95"])
    print(scen)
    print("  solo single direct run  fifo", f(o["solo_single_direct_run_h"]["fifo"]), " inline", f(o["solo_single_direct_run_h"]["inline"]))
    print("  solo 5-seed direct row  fifo", f(o["solo_5seed_direct_row_h"]["fifo"]), " inline", f(o["solo_5seed_direct_row_h"]["inline"]))
    print("  Walker phase makespan   fifo", f(o["walker_phase_makespan_h"]["fifo"]), " inline", f(o["walker_phase_makespan_h"]["inline"]))
    print("  pooled direct-run finish fifo", f(o["walker_phase_mean_direct_run_finish_h"]["fifo"]), " inline", f(o["walker_phase_mean_direct_run_finish_h"]["inline"]))
    print("  pooled nca/rule finish   fifo", f(o["walker_phase_mean_nca_rule_run_finish_h"]["fifo"]), " inline", f(o["walker_phase_mean_nca_rule_run_finish_h"]["inline"]))
    print("  FULL CAMPAIGN interleaved fifo", f(o["full_campaign_interleaved_h"]["fifo"]), " inline", f(o["full_campaign_interleaved_h"]["inline"]))
