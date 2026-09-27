"""P1 Step 4a/4b analysis: Wilson 95% CIs (same function as scripts/capacity_planning/
analyze_validity_round3.py), per-generation offspring validity for rule_based_monotone, compared
with the stored round-3 curves of the other three arms."""
import json, math, numpy as np
OUT = "results/campaign_prep/monotone_cost"
R3 = "results/preflight/20260922T122730Z_4be8402b/report/capacity_planning"


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


cur = json.load(open(f"{OUT}/cpu_measurements.json"))
stored = json.load(open(f"{R3}/validity_current_calibration.json"))["counts"]
stored_an = json.load(open(f"{R3}/validity_round3_analysis.json"))["arms"]
res = {"reproduces_stored_rule_based": {}, "arms": {}}
for proxy in ("random", "fixed"):
    a = cur["round3"]["counts"][f"rule_based|{proxy}|current"]; b = stored[f"rule_based|{proxy}|current"]
    res["reproduces_stored_rule_based"][proxy] = all(a[str(s)] == b[str(s)] for s in range(20)) if isinstance(next(iter(a)), str) else all(a[s] == b[str(s)] for s in range(20))
for arm in ("rule_based_monotone", "rule_based"):
    d = cur["round3"]["counts"][f"{arm}|random|current"]
    M = np.array([d[k] for k in sorted(d, key=int)]); k = M.sum(0); n = np.full(20, 15 * M.shape[0])
    ci = [wilson(int(x), int(y)) for x, y in zip(k, n)]
    fx = cur["round3"]["counts"][f"{arm}|fixed|current"]; F = np.array([fx[x] for x in sorted(fx, key=int)])
    res["arms"][arm] = dict(n_seeds=int(M.shape[0]), n_per_generation=int(n[0]), rate_by_generation=(k / n).tolist(),
                            wilson95=ci, gen0_rate=float(k[0] / n[0]),
                            offspring_gens1_19=dict(k=int(k[1:].sum()), n=int(n[1:].sum()), rate=float(k[1:].sum() / n[1:].sum()),
                                                    wilson95=wilson(int(k[1:].sum()), int(n[1:].sum()))),
                            last5=dict(rate=float(k[15:].sum() / n[15:].sum()), wilson95=wilson(int(k[15:].sum()), int(n[15:].sum()))),
                            fixed_proxy_rate_by_generation=(F.sum(0) / (15 * F.shape[0])).tolist())
for arm in ("direct", "nca"):
    r = stored_an[arm]; rate = np.array(r["rate_by_generation"])
    res["arms"][arm] = dict(source=f"{R3}/validity_round3_analysis.json", rate_by_generation=rate.tolist(),
                            wilson95=list(zip(r["wilson95_lo"], r["wilson95_hi"])), gen0_rate=float(rate[0]),
                            offspring_gens1_19=dict(rate=float(rate[1:].mean())), last5=dict(rate=float(rate[15:].mean())))
json.dump(res, open(f"{OUT}/validity_analysis.json", "w"), indent=1)
print("reproduces stored rule_based:", res["reproduces_stored_rule_based"])
for a, r in res["arms"].items():
    print(f"{a:20} gen0 {r['gen0_rate']:.3f}  gens1-19 {r['offspring_gens1_19']['rate']:.3f}  last5 {r['last5']['rate']:.3f}")
    print("     ", np.round(r["rate_by_generation"], 2).tolist())
m = res["arms"]["rule_based_monotone"]; print("monotone Wilson per gen", [[round(x, 3) for x in c] for c in m["wilson95"]])
print("monotone gens1-19", m["offspring_gens1_19"], "fixed proxy", np.round(m["fixed_proxy_rate_by_generation"], 2).tolist())
for a, v in cur["inspection"].items(): print(a, {k: v[k] for k in ("n", "k_valid", "valid_rate", "wilson95", "invalid_reason_counts", "degenerate_fraction")})
for a, v in cur["timings"].items(): print(a, "dev", v["t_develop"], "p99", v["t_develop_p99"], "\n   sent", v["t_sentinel"], "p99", v["t_sentinel_p99"], "draws", v["t_sentinel_draw_attempts"])
