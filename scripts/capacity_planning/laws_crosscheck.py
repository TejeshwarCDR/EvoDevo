"""Step 4: cross-validate the Monte Carlo simulator against Amdahl, Gunther's USL and
Gustafson. Read-only over preflight artifacts; writes laws_results.json next to the
simulation output. Re-uses simulate_campaign's model/scheduler (no re-run of any
preflight stage)."""
import json, os, sys, math
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(__file__))
import simulate_campaign as sc

RUN, OUT = sc.RUN, sc.OUT
inp = sc.load_inputs()
ratio = inp["ratio"]
f20 = sc.speed_factors(sc.worker_speeds(20), ratio)
f15 = sc.speed_factors(sc.worker_speeds(15), ratio)
model = sc.Model(inp)
sim = json.load(open(f"{OUT}/simulation_results.json"))["variants"]["BASE"]
H = 3600.0
res = {}

# ---------------- Amdahl ----------------
n_batches = len(sc.ARMS) * len(sc.TASKS) * sc.SEEDS * sc.GENS       # 600 generation-batches
o = float(np.mean([model.overhead(a) for a in sc.ARMS]))              # launcher-side overhead / generation (decode included: conservative)
o_orch_only = float(np.mean([inp["orch"][a] for a in sc.ARMS]))
work_s = sum(sc.SEEDS * sc.GENS * sc.POP * model.expected_cost(a, t) for a in sc.ARMS for t in sc.TASKS)   # P-core-seconds
T1 = work_s + n_batches * o
G15 = float(np.mean([r["mean_serial_batch_eval_makespan_s"] for r in sim["rows"].values()]))
sim_int_s = sim["interleaved_total_h"]["mean"] * H
sim_ser_s = sim["serial_total_h"]["mean"] * H
G20 = sim_int_s / n_batches       # pool-level wall-clock per generation-batch worth of work


def amdahl(N, G, T1, n_eff=None):
    s = o / (o + G)
    P = 1 - s
    S = 1.0 / (s + P / N)
    S_h = 1.0 / (s + P / n_eff) if n_eff else None
    return dict(N=N, mean_gen_eval_time_s=G, serial_fraction=s, S=S, T_pred_h=T1 / S / H,
                n_eff_P_equiv=n_eff, S_hetero=S_h, T_pred_hetero_h=(T1 / S_h / H) if S_h else None)


neff15 = sum(1 / f for f in f15)
neff20 = sum(1 / f for f in f20)
a15 = amdahl(15, G15, T1, neff15)
a20 = amdahl(20, G20, T1, neff20)
a15["sim_mean_h"] = sim_ser_s / H
a20["sim_mean_h"] = sim_int_s / H
for a in (a15, a20):
    a["pct_diff_amdahl_vs_sim"] = 100 * (a["T_pred_h"] - a["sim_mean_h"]) / a["sim_mean_h"]
    a["pct_diff_hetero_amdahl_vs_sim"] = 100 * (a["T_pred_hetero_h"] - a["sim_mean_h"]) / a["sim_mean_h"]
res["amdahl"] = dict(T1_single_Pcore_h=T1 / H, overhead_per_gen_s_with_decode=o, overhead_per_gen_s_orch_only=o_orch_only,
                     serial_per_run_N15=a15, interleaved_N20=a20)

# ---------------- USL ----------------
cs = json.load(open(f"{RUN}/stages/concurrency_sweep.json"))
pts = sorted((l["level"], l["genomes_per_minute"]) for l in cs["levels"])
Ns = np.array([p[0] for p in pts], float); X = np.array([p[1] for p in pts], float)


def usl(N, lam, al, be):
    return lam * N / (1 + al * (N - 1) + be * N * (N - 1))


popt, pcov = curve_fit(usl, Ns, X, p0=[4, 0.05, 1e-4], bounds=([0, 0, 0], [np.inf, 5, 1]), maxfev=100000)
pred = usl(Ns, *popt)
ss_res = float(((X - pred) ** 2).sum()); ss_tot = float(((X - X.mean()) ** 2).sum())
perr = np.sqrt(np.diag(pcov))
# sim-implied capacity at width N: sum(1/f_i) P-core-equivalents; throughput ratio vs N=8 (P-cores only)
def cap(n):
    return sum(1 / f for f in sc.speed_factors(sc.worker_speeds(n), ratio))
sim_ratio = {n: cap(n) / cap(8) for n in (8, 12, 15, 16, 20)}
usl_ratio = {n: float(usl(n, *popt) / usl(8, *popt)) for n in (8, 12, 15, 16, 20)}
meas = dict(zip(Ns.astype(int).tolist(), X.tolist()))
meas_ratio = {n: meas[n] / meas[8] for n in meas}
# realised campaign throughput (evals / P-core-normalised second) in the simulator
tot_evals = n_batches * sc.POP
res["usl"] = dict(source=f"{RUN}/stages/concurrency_sweep.json (ppo_timesteps=20000 probe, NOT campaign budget)",
                  points=pts, lambda_=popt[0], alpha=popt[1], beta=popt[2], std_err=perr.tolist(),
                  r_squared=1 - ss_res / ss_tot, fitted=pred.tolist(),
                  pred_X15=float(usl(15, *popt)), pred_X20=float(usl(20, *popt)),
                  usl_ratio_vs_8=usl_ratio, measured_ratio_vs_8=meas_ratio,
                  sim_capacity_ratio_vs_8=sim_ratio,
                  sim_realised_interleaved_evals_per_h=tot_evals / (sim_int_s / H),
                  sim_realised_serial_evals_per_h=tot_evals / (sim_ser_s / H),
                  sim_capacity_evals_per_h_N20=cap(20) / (work_s / (tot_evals)) * H,
                  sim_capacity_evals_per_h_N15=cap(15) / (work_s / (tot_evals)) * H,
                  peak_N_usl=math.sqrt((1 - popt[1]) / popt[2]) if popt[2] > 0 else None)
res["usl"]["sim_realised_interleaved_fraction_of_capacity_N20"] = res["usl"]["sim_realised_interleaved_evals_per_h"] / res["usl"]["sim_capacity_evals_per_h_N20"]
res["usl"]["sim_realised_serial_fraction_of_capacity_N15"] = res["usl"]["sim_realised_serial_evals_per_h"] / res["usl"]["sim_capacity_evals_per_h_N15"]

# ---------------- Gustafson + D14 options ----------------
def inter_alone(arm, task, seeds, rng, oh_arm=None):
    d = [model.draw_run(rng, arm, task) for _ in range(seeds)]
    ms, _ = sc.run_interleaved(d, [model.overhead(oh_arm or arm)] * seeds, f20)
    return ms


def stats(x): return dict(mean=float(np.mean(x)) / H, p5=float(np.percentile(x, 5)) / H, p95=float(np.percentile(x, 95)) / H)


def serial_1worker_h(arm, task, seeds):     # one P-core worker, no barrier waste: total expected work + overhead
    return (seeds * sc.GENS * sc.POP * model.expected_cost(arm, task) + seeds * sc.GENS * model.overhead(arm)) / H


rng = np.random.default_rng(77)
NT = 400
ref = ("direct", "Walker-v0")


def seed_scan_for(mdl, seeds_list, n):
    global model
    keep, model = model, mdl
    try:
        return {s_: [inter_alone(*ref, s_, rng) for _ in range(n)] for s_ in seeds_list}
    finally:
        model = keep


scan_raw = seed_scan_for(model, (5, 10, 15, 20, 30, 40, 60), NT)
seed_scan = {s_: stats(v) for s_, v in scan_raw.items()}
T_ref = scan_raw[5]
T_budget = float(np.mean(T_ref))
TOL = 1.25   # a scaled workload is "absorbed in the same wall-clock" if it finishes within +25% of T
fits_N20 = max(s_ for s_, v in seed_scan.items() if v["mean"] * H <= T_budget * TOL)
seeds_N1 = T_budget / H / serial_1worker_h(*ref, 1)
bound_model = sc.Model(inp, later_gen_valid=1.0)
bound_raw = seed_scan_for(bound_model, (5, 10, 20), 100)
g = dict(reference_cell="pop15 gen20 seeds5 direct Walker-v0, INTERLEAVED, N=20", T_budget_h=T_budget / H,
         T_budget_pct_p5_p95=[float(np.percentile(T_ref, 5)) / H, float(np.percentile(T_ref, 95)) / H],
         seed_scan_interleaved_N20=seed_scan, max_seeds_within_1p25T_N20_from_scan=fits_N20, tolerance_over_T=TOL,
         bound_all_valid_after_gen0_seed_scan={k: stats(v) for k, v in bound_raw.items()},
         seeds_absorbable_in_T_N1=seeds_N1,
         classic_gustafson=dict(alpha=o / (o + G20), N=20, S_scaled=20 - o / (o + G20) * 19,
                                S_scaled_hetero=neff20 - o / (o + G20) * (neff20 - 1), neff20=neff20))

def option_hours(kind, rng, n=100):
    """(N=20 interleaved mean wall-clock h, N=1 serial P-core work h) for each D14 option; campaign options as DELTAS."""
    out = {}
    base_c = []
    def campaign(extra=(), seeds=sc.SEEDS, dup=1):
        tot = 0.0
        for t in sc.TASKS:
            runs = [(a, t) for a in sc.ARMS for _ in range(seeds * dup)] + [(a, t) for a, _ in extra for _ in range(sc.SEEDS)]
            draws = [model.draw_run(rng, a, t) for a, t in runs]
            ms, _ = sc.run_interleaved(draws, [model.overhead(a) for a, _ in runs], f20)
            tot += ms
        return tot
    def work1(extra=(), seeds=sc.SEEDS, dup=1):
        w = sum(serial_1worker_h(a, t, seeds * dup) for a in sc.ARMS for t in sc.TASKS)
        w += sum(serial_1worker_h(a, t, sc.SEEDS) for a, _ in extra for t in sc.TASKS)
        return w
    base = [campaign() for _ in range(n)]
    variants = {
        "more_seeds_5_to_10 (whole campaign)": dict(seeds=10),
        "second_grid_size (2x cross product, B6 model)": dict(dup=2),
        "direct_repaired_arm (direct proxy)": dict(extra=(("direct", None),)),
        "nca_compact_arm (nca proxy)": dict(extra=(("nca", None),)),
    }
    b = float(np.mean(base))
    res_ = {"baseline_campaign": dict(N20_h=b / H, N1_h=work1())}
    for name, kw in variants.items():
        x = [campaign(**kw) for _ in range(n)]
        res_[name] = dict(N20_total_h=float(np.mean(x)) / H, N20_delta_h=(float(np.mean(x)) - b) / H,
                          N1_total_h=work1(**kw), N1_delta_h=work1(**kw) - work1())
    return res_

g["d14_options"] = option_hours(None, rng)
# single-cell view of the reference cell (what projection.md's D14 table uses)
g["ref_cell_more_seeds_5_to_10"] = dict(N20_h=seed_scan[10]["mean"], N20_delta_h=seed_scan[10]["mean"] - seed_scan[5]["mean"],
                                        N1_h=serial_1worker_h(*ref, 10), N1_delta_h=serial_1worker_h(*ref, 5))
res["gustafson"] = g
json.dump(res, open(f"{OUT}/laws_results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=float))
