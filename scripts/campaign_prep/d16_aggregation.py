"""P1 Step 5: D16 aggregation check from B5's raw fitness values (analysis only, no PPO).
Within-body sd of each estimator (K=1: the 5 single values; K=2 mean / K=2 max: the 10
unordered pairs of the 5 values), pooled across bodies with B5's own _pooled_sd, divided by
B5's between-body sd (sd of B3's 9 pooled per-task fitness observations) -- exactly B5's
within_over_between_fraction definition, applied to each estimator."""
import itertools, json, statistics, sys, os
sys.path.insert(0, os.getcwd())
from evodevo.preflight.stages.b5_noise import _pooled_sd

SRC = "results/preflight/20260922T040932Z_4be8402b/stages/evaluation_noise.json"
d = json.load(open(SRC))
out = {"source": SRC, "tasks": {}}
for task, t in d["tasks"].items():
    between = t["between_body_spread_from_b3"]["sd"]
    est = {"single": [], "mean_of_2": [], "max_of_2": []}
    bodies = []
    for b in t["bodies"]:
        v = b["fitness_values"]; pairs = list(itertools.combinations(v, 2))
        vals = {"single": v, "mean_of_2": [(a + c) / 2 for a, c in pairs], "max_of_2": [max(a, c) for a, c in pairs]}
        med = statistics.median(v)
        meaningful = abs(med) > 1e-6
        collapses = [x for x in v if x < 0.5 * med] if meaningful and med > 0 else None
        row = {"label": b["label"], "arm": b["arm"], "genome_seed": b["genome_seed"], "values": v, "median": med,
               "collapse_threshold": 0.5 * med if meaningful and med > 0 else None,
               "collapse_events": None if collapses is None else len(collapses), "collapse_values": collapses}
        for k, xs in vals.items():
            sd = statistics.stdev(xs)
            est[k].append((sd, len(xs)))
            row[f"sd_{k}"] = sd; row[f"mean_{k}"] = statistics.mean(xs)
        bodies.append(row)
    res = {k: {"pooled_within_sd": _pooled_sd(p), "within_over_between": _pooled_sd(p) / between} for k, p in est.items()}
    out["tasks"][task] = {"between_body_sd_b3": between, "b5_reported_fraction": t["within_over_between_fraction"],
                          "estimators": res, "bodies": bodies}
json.dump(out, open("results/campaign_prep/d16/d16_aggregation.json", "w"), indent=1)
for task, r in out["tasks"].items():
    print(task, "between", round(r["between_body_sd_b3"], 4), "B5", round(r["b5_reported_fraction"], 4))
    for k, e in r["estimators"].items():
        print(f"   {k:10} pooled_sd={e['pooled_within_sd']:.4f} frac={e['within_over_between']:.4f}")
    for b in r["bodies"]:
        print(f"   {b['label']:6} {b['arm']:10} median={b['median']:.4g} collapses={b['collapse_events']} {b['collapse_values']} sd1={b['sd_single']:.4f} sdmean2={b['sd_mean_of_2']:.4f} sdmax2={b['sd_max_of_2']:.4f}")
