"""Round 3 analysis: binomial (Wilson 95%) CIs per generation for the current-calibration
measurement, comparison against the historical run, gen-0 statistics for all three arms, plot."""
import json, math, os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from measure_validity_current_calibration import OUT, ARMS, POP, GENS


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


cur = json.load(open(f"{OUT}/validity_current_calibration.json"))
hist = json.load(open(f"{OUT}/validity_by_generation.json"))["arms"]


def pooled(key):
    d = cur["counts"][key]; M = np.array([d[s] for s in sorted(d, key=int)])   # (seeds, GENS) valid counts of 15
    return M, M.sum(0), np.full(GENS, 15 * M.shape[0])


res = {"arms": {}}
prior = {a: json.load(open(f"results/inspection/{a}/{a}_sanity.json"))[a] for a in ARMS}
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, a in zip(axs, ARMS):
    M, k, n = pooled(f"{a}|random|current")
    lo, hi = zip(*[wilson(int(x), int(y)) for x, y in zip(k, n)])
    rate = k / n
    hk = np.array(hist[a]["rate_by_generation_pooled_5_seeds"]) * 75
    hlo, hhi = zip(*[wilson(round(x), 75) for x in hk])
    # per-generation overlap of the two Wilson intervals
    overlap = [not (l1 > h2 or l2 > h1) for l1, h1, l2, h2 in zip(lo, hi, hlo, hhi)]
    # pooled comparison, gens 1-19 and last 5 gens: two-proportion z test (Fisher exact for robustness)
    def cmp(gs):
        k1, n1 = int(k[gs].sum()), int(n[gs].sum()); k2, n2 = round(hk[gs].sum()), 75 * len(np.arange(GENS)[gs])
        p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]
        return dict(current=k1 / n1, current_n=n1, hist=k2 / n2, hist_n=n2, fisher_p=p)
    gsl = {"gens1_19": slice(1, 20), "last5": slice(15, 20)}
    # gen-0 statistics vs random-genome prior (n=2000)
    pr, pn = prior[a]["valid_rate"], prior[a]["n"]
    plo, phi = wilson(round(pr * pn), pn)
    g0 = {"prior_rate": pr, "prior_n": pn, "prior_wilson95": [plo, phi]}
    for name, kk, nn in (("historical", round(hist[a]["gen0_rate"] * 75), 75), ("current_measured", int(k[0]), int(n[0]))):
        l, h = wilson(kk, nn)
        g0[name] = dict(k=kk, n=nn, rate=kk / nn, wilson95=[l, h], prior_inside_this_ci=bool(l <= pr <= h),
                        this_rate_inside_prior_ci=bool(plo <= kk / nn <= phi),
                        binom_two_sided_p_vs_prior=float(stats.binomtest(kk, nn, pr).pvalue))
    # trend: logit slope
    sys.path.insert(0, os.path.dirname(__file__))
    from validity_by_generation import fit_logit
    b0, b1, cov = fit_logit(np.arange(GENS, dtype=float), k.astype(float), n.astype(float))
    res["arms"][a] = dict(
        n_seeds=int(M.shape[0]), n_per_generation=int(n[0]), rate_by_generation=rate.tolist(),
        wilson95_lo=list(lo), wilson95_hi=list(hi), per_seed_rate=(M / POP).tolist(),
        hist_wilson95_lo=list(hlo), hist_wilson95_hi=list(hhi), ci_overlap_by_generation=overlap,
        n_generations_nonoverlapping=int(GENS - sum(overlap)),
        nonoverlapping_generations=[g for g, o in enumerate(overlap) if not o],
        pooled_tests={n_: cmp(s) for n_, s in gsl.items()},
        gen0=g0, logit_slope=dict(slope=b1, se=float(math.sqrt(cov[1, 1])), fitted_gen0=1 / (1 + math.exp(-b0)), fitted_gen19=1 / (1 + math.exp(-(b0 + b1 * 19)))),
        fixed_proxy_rate_by_generation=(pooled(f"{a}|fixed|current")[1] / pooled(f"{a}|fixed|current")[2]).tolist())
    ax.fill_between(range(GENS), lo, hi, color="C0", alpha=0.25)
    ax.plot(range(GENS), rate, "o-", color="C0", ms=3, label=f"current calibration (n={int(n[0])}/gen)")
    ax.fill_between(range(GENS), hlo, hhi, color="C1", alpha=0.2)
    ax.plot(range(GENS), np.array(hist[a]["rate_by_generation_pooled_5_seeds"]), "s--", color="C1", ms=3, label="historical (n=75/gen)")
    ax.axhline(pr, ls=":", color="C3", label="random-genome prior")
    ax.set_title(a); ax.set_xlabel("generation"); ax.set_ylim(-0.02, 1.02)
axs[0].set_ylabel("offspring validity"); axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Offspring validity by generation: current calibration (real ea.py loop, no PPO) vs historical run; bands = Wilson 95%", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/validity_current_vs_historical.png", dpi=130)

# proxy validation: direct under historical mutation (0.1), random proxy, vs historical real-fitness direct curve
M, k, n = pooled("direct|random|histmut")
hk = np.array(hist["direct"]["rate_by_generation_pooled_5_seeds"]) * 75
hlo, hhi = zip(*[wilson(round(x), 75) for x in hk]); lo, hi = zip(*[wilson(int(x), int(y)) for x, y in zip(k, n)])
ov = [not (l1 > h2 or l2 > h1) for l1, h1, l2, h2 in zip(lo, hi, hlo, hhi)]
res["proxy_validation_direct_historical_mutation"] = dict(
    n_seeds=int(M.shape[0]), rate_by_generation=(k / n).tolist(), n_gens_ci_nonoverlap=int(GENS - sum(ov)),
    mean_gens1_19_proxy=float(k[1:].sum() / n[1:].sum()), mean_gens1_19_historical=float(hk[1:].sum() / (75 * 19)),
    last5_proxy=float(k[15:].sum() / n[15:].sum()), last5_historical=float(hk[15:].sum() / 375))
json.dump(res, open(f"{OUT}/validity_round3_analysis.json", "w"), indent=1, default=float)
for a, r in res["arms"].items():
    print(a, "seeds", r["n_seeds"], "gen0 cur %.3f" % r["rate_by_generation"][0], "nonoverlap gens", r["nonoverlapping_generations"], "slope %.3f±%.3f" % (r["logit_slope"]["slope"], r["logit_slope"]["se"]), "fit19 %.2f" % r["logit_slope"]["fitted_gen19"])
    print("   rates", np.round(r["rate_by_generation"], 2).tolist())
    print("   tests", {k_: {a_: round(b_, 3) for a_, b_ in v.items()} for k_, v in r["pooled_tests"].items()})
    print("   gen0", json.dumps(r["gen0"], default=float))
    print("   fixed proxy", np.round(r["fixed_proxy_rate_by_generation"], 2).tolist())
print("proxy val", json.dumps(res["proxy_validation_direct_historical_mutation"], default=float))
