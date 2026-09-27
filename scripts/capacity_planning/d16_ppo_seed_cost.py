"""D16 costing: apply N PPO seeds per genome to UpStepper-v0 only (asymmetric) vs to both
tasks (uniform), under INTERLEAVED scheduling and the current-calibration validity curves
(the best-available estimate; baseline 165.6h). A "PPO seed" here multiplies the PPO cost of
every VALID evaluation by ppo_seeds (repeat the training run ppo_seeds times and average);
invalid genomes still resolve via the single sentinel check (multi-seeding a structurally
invalid body is meaningless -- decode+validate is deterministic given the genome, so re-running
it would just repeat the identical sentinel result eval_bookkeeping.py already produces once).
This is a Monte Carlo re-run of the existing simulator, not a launcher and not real PPO."""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(__file__))
import simulate_campaign as sc

inp = sc.load_inputs()
f20 = sc.speed_factors(sc.worker_speeds(20), inp["ratio"])
H = 3600.0
sim = json.load(open(f"{sc.OUT}/simulation_results.json"))
cur_curves = sim["current_validity_curves_used"]
model = sc.Model(inp, validity_curves=cur_curves)


def draw_run_seeded(rng, arm, task, ppo_seeds):
    """Like Model.draw_run, but a valid evaluation's PPO cost is the MEAN of ppo_seeds
    independent lognormal draws (repeat-and-average), not a single draw; invalid genomes
    still cost one t_sentinel draw (unaffected -- see module docstring)."""
    v = np.full((sc.GENS, sc.POP), model.inp["valid"][arm])
    if arm in model.validity_curves:
        v = np.repeat(np.asarray(model.validity_curves[arm], float)[:, None], sc.POP, axis=1)
    is_valid = rng.random((sc.GENS, sc.POP)) < v
    mu, sg = model.ppo_ln[(arm, task)]
    smu, ssg = model.sent_ln[arm]
    d_ppo = rng.lognormal(mu, sg, (sc.GENS, sc.POP, ppo_seeds)).sum(axis=2)  # cost multiplies (cost_multiplier_N_seeds in evaluation_noise.json), not averages
    d_sen = rng.lognormal(smu, ssg, (sc.GENS, sc.POP))
    return np.where(is_valid, d_ppo, d_sen)


def campaign_h(rng, seeds_by_task):
    tot = 0.0
    for t in sc.TASKS:
        ns = seeds_by_task[t]
        keys = [(a, s) for a in sc.ARMS for s in range(sc.SEEDS)]
        draws = [draw_run_seeded(rng, a, t, ns) for a, _ in keys]
        ms, _ = sc.run_interleaved(draws, [model.overhead(a) for a, _ in keys], f20)
        tot += ms
    return tot / H


def scenario(n, name, seeds_by_task, trials=500):
    rng = np.random.default_rng(hash(name) % (2**31))
    x = np.array([campaign_h(rng, seeds_by_task) for _ in range(trials)])
    return dict(mean=float(x.mean()), p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
               median=float(np.median(x)))


def main():
    trials = 500
    baseline = scenario(1, "baseline", {"Walker-v0": 1, "UpStepper-v0": 1}, trials)
    out = {"baseline_single_eval_both_tasks_h": baseline}
    options = {
        "2seed_asymmetric_UpStepper_only": {"Walker-v0": 1, "UpStepper-v0": 2},
        "3seed_asymmetric_UpStepper_only": {"Walker-v0": 1, "UpStepper-v0": 3},
        "2seed_uniform_both_tasks": {"Walker-v0": 2, "UpStepper-v0": 2},
        "3seed_uniform_both_tasks": {"Walker-v0": 3, "UpStepper-v0": 3},
    }
    for name, sbt in options.items():
        r = scenario(1, name, sbt, trials)
        r["delta_h_mean"] = r["mean"] - baseline["mean"]
        r["delta_pct_mean"] = 100 * r["delta_h_mean"] / baseline["mean"]
        out[name] = r
    out["n_trials"] = trials
    json.dump(out, open(f"{sc.OUT}/d16_ppo_seed_cost.json", "w"), indent=2)
    print("baseline", baseline)
    for name, sbt in options.items():
        r = out[name]
        print(name, "mean %.2fh [%.2f,%.2f] delta +%.2fh (+%.1f%%)" % (r["mean"], r["p5"], r["p95"], r["delta_h_mean"], r["delta_pct_mean"]))


if __name__ == "__main__":
    main()
