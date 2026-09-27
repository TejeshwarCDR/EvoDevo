"""Final campaign wall-clock estimate (Prompt 2 Step 4): the simulate_campaign.py discrete-event
model, extended to the ACCEPTED final design (docs/decisions.md, 2026-09-27).

Design simulated:
  - 4 arms: direct, nca, rule_based, rule_based_monotone; 5 seeds; pop 15, offspring 15,
    eval_budget 300 genomes = 20 generations (gen 0 + 19 offspring generations).
  - D7 strict phases: all Walker-v0 runs complete before any UpStepper-v0 run starts.
  - D6 budgets: Walker-v0 1,000,000 timesteps, UpStepper-v0 1,200,000.
  - D16: Walker-v0 K=1; UpStepper-v0 K=2, each PPO seed of a valid genome is a SEPARATE pool
    task (a generation completes when all its tasks complete).
  - Validity resolved in the scheduler: an invalid genome takes NO worker slot; its measured
    sentinel time is charged serially to its own run (simulate_campaign's inline_masks mode).
  - 20 workers, P/E model (8 P-core primaries at speed 1, 12 E-cores at 1/2.3172, production-scale
    ratio), lowest-generation-first dispatch (policy "lgf"), 500 Monte Carlo trials.

t_PPO inputs, fixed BEFORE the P1 t_PPO batch results were read:
  Everything is expressed at B3's measurement conditions (3 concurrent trainings on P-core
  primaries) -- the footing of every earlier simulation and of the 2.3172 P/E ratio.
  c = anchor(rule_based genome 203000001, Walker-v0, 1M, P1 batch of 8) / B3(same genome, task,
      budget): a pure measurement-condition factor (8-way vs 3-way concurrency).
  s = [anchor(same genome, UpStepper-v0, 1.2M) / B3(same genome, UpStepper-v0, 1M)] / c:
      the MEASURED t_PPO scaling for 1M -> 1.2M timesteps.
  direct/nca/rule_based: B3 mean/sd (n=3) on Walker-v0 (budget = B3's 1M, no scaling);
      on UpStepper-v0, B3 mean/sd x s.
  rule_based_monotone: its own P1-measured mean/sd (n=3 per task, at the task's budget) / c.
  Sensitivities: s = 1.2 (pure linear); c = 1 (no condition correction); resampled per-seed
  validity trajectories instead of the pooled per-generation curve.

Validity: current-calibration round-3 curves (validity_round3_analysis.json) for the three arms,
exactly as simulate_campaign's CUR_curve_validity; rule_based_monotone's own round-3 curve from
results/campaign_prep/monotone_cost/cpu_measurements.json (same proxy, same 20 seeds).
t_sentinel/t_develop: B3 for three arms; P1 Step 4c for monotone. Orchestration overhead:
monotone uses rule_based's measured value (same encoder class; ~1 ms/generation either way).

Budget guard (docs/decisions.md, 2026-09-27), applied mechanically on the p95 total:
  > 432 h -> (a) rule_based_monotone on Walker-v0 only; still over -> (b) drop rule_based_monotone
  entirely; still over -> (c) UpStepper-v0 K=1 (noise disclosed as a limitation).
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(__file__))
import simulate_campaign as sc

RUN = sc.RUN
B3_RUN = "results/preflight/20260922T040932Z_4be8402b"
P1 = "results/campaign_prep/monotone_cost"
OUT = "results/campaign_prep/final_estimate"
ARMS = ("direct", "nca", "rule_based", "rule_based_monotone")
TASKS = ("Walker-v0", "UpStepper-v0")
BUDGET = {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
K = {"Walker-v0": 1, "UpStepper-v0": 2}
GUARD_H = 432.0
ANCHOR_SEED = 203000001
N_TRIALS = 500
SEED = 20260927


def load_measured():
    ec = json.load(open(f"{B3_RUN}/stages/evaluation_cost.json"))
    tp = json.load(open(f"{P1}/tppo_measurements.json"))
    cpu = json.load(open(f"{P1}/cpu_measurements.json"))
    b3_obs = {t: {o["genome_seed"]: o["wall_clock_s"] for o in ec["arms"]["rule_based"]["tasks"][t]["observations"]} for t in TASKS}
    rows = [r for r in tp["results"] if r["error"] is None and not r["timed_out"]]
    anchor = {r["task"]: r["wall_clock_s"] for r in rows if r["arm_id"] == "rule_based" and r["genome_seed"] == ANCHOR_SEED}
    mono = {t: [r["wall_clock_s"] for r in rows if r["arm_id"] == "rule_based_monotone" and r["task"] == t] for t in TASKS}
    c = anchor["Walker-v0"] / b3_obs["Walker-v0"][ANCHOR_SEED]
    raw_up = anchor["UpStepper-v0"] / b3_obs["UpStepper-v0"][ANCHOR_SEED]
    s = raw_up / c
    return dict(ec=ec, cpu=cpu, b3_obs=b3_obs, anchor=anchor, mono=mono, c=c, raw_up_ratio=raw_up, s=s)


def build_inputs(m, s, c):
    ec, cpu = m["ec"], m["cpu"]
    oh = json.load(open(f"{sc.OUT}/orchestration_overhead.json"))
    ppo, sent, tdev, orch = {}, {}, {}, {}
    for a in ARMS[:3]:
        for t in TASKS:
            st = ec["arms"][a]["tasks"][t]["t_ppo"]
            f = s if t == "UpStepper-v0" else 1.0
            ppo[(a, t)] = (st["mean"] * f, st["sd"] * f)
        sent[a] = (ec["arms"][a]["t_sentinel"]["mean"], ec["arms"][a]["t_sentinel"]["sd"])
        tdev[a] = ec["arms"][a]["t_develop"]["mean"]
        orch[a] = oh["arms"][a]["per_generation_s"]["mean"]
    a = "rule_based_monotone"
    for t in TASKS:
        x = np.asarray(m["mono"][t], float)
        ppo[(a, t)] = (float(x.mean()) / c, float(x.std(ddof=1)) / c)
    mt = cpu["timings"][a]
    sent[a] = (mt["t_sentinel"]["mean"], mt["t_sentinel"]["sd"])
    tdev[a] = mt["t_develop"]["mean"]
    orch[a] = orch["rule_based"]
    ratio = json.load(open(f"{RUN}/pe_core_ratio_measurement_production_scale.json"))["ratio_e_over_p"]
    return dict(ppo=ppo, sent=sent, tdev=tdev, orch=orch, ratio=ratio, valid={x: 0.0 for x in ARMS})


def validity_inputs(cpu):
    cg = json.load(open(f"{sc.OUT}/validity_round3_analysis.json"))["arms"]
    curves, trajs = {}, {}
    for a, r in cg.items():
        if a == "direct":
            b = r["logit_slope"]
            b0 = math.log(b["fitted_gen0"] / (1 - b["fitted_gen0"]))
            curves[a] = [1 / (1 + math.exp(-(b0 + b["slope"] * g))) for g in range(sc.GENS)]
        else:
            curves[a] = r["rate_by_generation"]
        trajs[a] = r["per_seed_rate"]
    d = cpu["round3"]["counts"]["rule_based_monotone|random|current"]
    M = np.array([d[k] for k in sorted(d, key=int)], float) / sc.POP
    curves["rule_based_monotone"] = M.mean(0).tolist()
    trajs["rule_based_monotone"] = M.tolist()
    return curves, trajs


def draw_run(rng, inp, arm, task, k, vprob):
    """-> (durs (GENS, POP*k), mask (GENS, POP*k)): slot i*k+j is PPO seed j of genome i.
    Valid genome: k independent single-training t_PPO draws, each its own pool task.
    Invalid genome: one sentinel draw in its first slot, charged inline (mask False); other slots 0."""
    is_valid = rng.random((sc.GENS, sc.POP)) < vprob[:, None]
    mu, sg = sc.lognormal_params(*inp["ppo"][(arm, task)])
    smu, ssg = sc.lognormal_params(*inp["sent"][arm])
    d_ppo = rng.lognormal(mu, sg, (sc.GENS, sc.POP, k))
    d_sen = rng.lognormal(smu, ssg, (sc.GENS, sc.POP))
    durs = np.where(is_valid[:, :, None], d_ppo, 0.0)
    durs[:, :, 0] = np.where(is_valid, d_ppo[:, :, 0], d_sen)
    mask = np.repeat(is_valid[:, :, None], k, axis=2)
    return durs.reshape(sc.GENS, sc.POP * k), mask.reshape(sc.GENS, sc.POP * k)


def overhead(inp, arm):
    return inp["orch"][arm] + sc.POP * inp["tdev"][arm]     # launcher decodes every genome to validate it


def trial(rng, inp, curves, trajs, factors, design, resample):
    phases = {}
    for task in TASKS:
        keys = [(a, s_) for a in design["arms"][task] for s_ in range(5)]
        durs, masks = [], []
        for a, _ in keys:
            if resample:
                tr = np.asarray(trajs[a], float); vprob = tr[rng.integers(len(tr))]
            else:
                vprob = np.asarray(curves[a], float)
            d, mk = draw_run(rng, inp, a, task, design["K"][task], vprob)
            durs.append(d); masks.append(mk)
        ms, _ = sc.run_interleaved(durs, [overhead(inp, a) for a, _ in keys], factors, policy="lgf", inline_masks=masks)
        phases[task] = ms / 3600.0
    return phases


def simulate(inp, curves, trajs, factors, design, resample=False, n=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = [trial(rng, inp, curves, trajs, factors, design, resample) for _ in range(n)]
    W = np.array([r["Walker-v0"] for r in rows]); U = np.array([r["UpStepper-v0"] for r in rows]); T = W + U

    def st(x):
        return dict(mean=float(x.mean()), p5=float(np.percentile(x, 5)), median=float(np.median(x)),
                    p95=float(np.percentile(x, 95)), sd=float(x.std(ddof=1)), n_trials=len(x))
    return dict(walker_phase_h=st(W), upstepper_phase_h=st(U), total_h=st(T),
                walker_milestone_pct_of_total=dict(of_means=float(100 * W.mean() / T.mean()),
                                                   per_trial=st(100 * W / T)))


def design(monotone_tasks=TASKS, k_up=2):
    return dict(arms={t: [a for a in ARMS if a != "rule_based_monotone" or t in monotone_tasks] for t in TASKS},
                K={"Walker-v0": 1, "UpStepper-v0": k_up})


GUARD_STEPS = [
    ("none", design()),
    ("a_monotone_walker_only", design(monotone_tasks=("Walker-v0",))),
    ("b_drop_monotone", design(monotone_tasks=())),
    ("c_drop_monotone_and_upstepper_k1", design(monotone_tasks=(), k_up=1)),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    m = load_measured()
    cpu = m["cpu"]
    curves, trajs = validity_inputs(cpu)
    factors = sc.speed_factors(sc.worker_speeds(20), json.load(open(f"{RUN}/pe_core_ratio_measurement_production_scale.json"))["ratio_e_over_p"])
    inp = build_inputs(m, m["s"], m["c"])
    out = {"inputs": {"c_condition_factor": m["c"], "raw_upstepper_anchor_ratio": m["raw_up_ratio"],
                      "s_measured_scaling_1M_to_1_2M": m["s"], "linear_scaling_assumption": 1.2,
                      "anchor_wall_s": m["anchor"], "b3_anchor_genome_wall_s": {t: m["b3_obs"][t][ANCHOR_SEED] for t in TASKS},
                      "monotone_raw_wall_s": m["mono"],
                      "t_ppo_mean_sd_at_b3_footing": {f"{a}|{t}": v for (a, t), v in inp["ppo"].items()},
                      "t_sentinel_mean_sd": inp["sent"], "t_develop_mean": inp["tdev"], "orch_per_gen_s": inp["orch"],
                      "validity_curves": curves, "pe_ratio": inp["ratio"],
                      "workers": {"n": 20, "n_P": sum(1 for f in factors if f == 1.0), "n_E": sum(1 for f in factors if f != 1.0)}},
           "guard_threshold_h": GUARD_H, "guard_sequence": [], "n_trials": N_TRIALS, "mc_seed": SEED}
    applied = None
    for name, dz in GUARD_STEPS:
        r = simulate(inp, curves, trajs, factors, dz)
        out["guard_sequence"].append(dict(step=name, design=dz, result=r, p95_total_h=r["total_h"]["p95"],
                                          over_threshold=r["total_h"]["p95"] > GUARD_H))
        print(f"{name}: total mean {r['total_h']['mean']:.1f} h [p5 {r['total_h']['p5']:.1f}, p95 {r['total_h']['p95']:.1f}]  "
              f"Walker {r['walker_phase_h']['mean']:.1f} h  UpStepper {r['upstepper_phase_h']['mean']:.1f} h  "
              f"milestone {r['walker_milestone_pct_of_total']['of_means']:.1f}%", flush=True)
        if r["total_h"]["p95"] <= GUARD_H:
            applied = name
            break
    out["guard_outcome"] = applied or "all_steps_exhausted_still_over"
    final = out["guard_sequence"][-1]
    out["final_configuration"] = dict(step=final["step"], design=final["design"])
    # sensitivities on the final configuration
    fz = final["design"]
    sens = {}
    sens["linear_scaling_s_1_2"] = simulate(build_inputs(m, 1.2, m["c"]), curves, trajs, factors, fz)
    sens["no_condition_correction_c_1"] = simulate(build_inputs(m, m["raw_up_ratio"], 1.0), curves, trajs, factors, fz)
    sens["resampled_validity_trajectories"] = simulate(inp, curves, trajs, factors, fz, resample=True)
    out["sensitivities"] = sens
    for k_, r in sens.items():
        print(f"  sens {k_}: total mean {r['total_h']['mean']:.1f} h [p5 {r['total_h']['p5']:.1f}, p95 {r['total_h']['p95']:.1f}]")
    out["sim_wall_s"] = time.time() - t0
    json.dump(out, open(f"{OUT}/final_estimate.json", "w"), indent=1, default=float)
    print("guard outcome:", out["guard_outcome"], "| wall", round(out["sim_wall_s"]), "s")


if __name__ == "__main__":
    main()
