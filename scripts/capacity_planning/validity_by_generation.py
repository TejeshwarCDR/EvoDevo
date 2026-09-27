"""Validity-rate-by-generation from the historical 15-run primary comparison
(results/runs/{direct,nca,rule_based}/Walker-v0/seed_{0..4}/run_log.jsonl). READ-ONLY.
Every record in run_log.jsonl is one evaluated OFFSPRING (survivors are carried forward
without re-evaluation), so grouping by `generation` gives per-generation offspring validity;
generation 0 is the random initial population."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
RUN = "results/preflight/20260922T122730Z_4be8402b"
OUT = f"{RUN}/report/capacity_planning"
ARMS = ("direct", "nca", "rule_based")
GENS = 20


def load(root="results/runs"):
    d = {}
    for a in ARMS:
        d[a] = {}
        for s in range(5):
            recs = [json.loads(l) for l in open(f"{root}/{a}/Walker-v0/seed_{s}/run_log.jsonl")]
            v = np.zeros((GENS, 15)); 
            cnt = np.zeros(GENS)
            for r in recs:
                g = r["generation"]; v[g, int(cnt[g])] = r["valid"]; cnt[g] += 1
            assert (cnt == 15).all()
            d[a][s] = v
    return d


def fit_logit(g, k, n):
    """Binomial GLM, logit(p) = b0 + b1*g, Newton/IRLS. Returns (b0, b1, cov)."""
    X = np.stack([np.ones_like(g), g], 1); b = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b)); W = n * p * (1 - p) + 1e-12
        H = X.T @ (X * W[:, None]); grad = X.T @ (k - n * p)
        step = np.linalg.solve(H, grad); b = b + step
        if abs(step).max() < 1e-10: break
    return b[0], b[1], np.linalg.inv(H)


def main():
    data = load()
    prior = {a: json.load(open(f"results/inspection/{a}/{a}_sanity.json"))[a] for a in ARMS}
    out = {"source": "results/runs/{arm}/Walker-v0/seed_0..4/run_log.jsonl (15 runs, pop 15, offspring 15, eval_budget 300 = 20 generations, ppo_timesteps=5000)", "arms": {}}
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    g = np.arange(GENS, dtype=float)
    for ax, a in zip(axs, ARMS):
        V = np.stack([data[a][s] for s in range(5)])           # (5, GENS, 15)
        k = V.sum(axis=(0, 2)); n = np.full(GENS, 75.0); rate = k / n
        b0, b1, cov = fit_logit(g, k, n)
        fit = 1 / (1 + np.exp(-(b0 + b1 * g)))
        se1 = np.sqrt(cov[1, 1])
        # Wilson interval for gen 0 vs inspection prior
        p0 = rate[0]; z = 1.96
        den = 1 + z**2 / 75; cen = (p0 + z**2 / 150) / den; half = z * np.sqrt(p0 * (1 - p0) / 75 + z**2 / (4 * 75**2)) / den
        per_seed = V.mean(axis=2)                               # (5, GENS)
        # trend test on gens 1..19 only (offspring, excluding random gen 0)
        b0o, b1o, covo = fit_logit(g[1:], k[1:], n[1:])
        out["arms"][a] = {
            "rate_by_generation_pooled_5_seeds": rate.tolist(),
            "per_seed_rate_by_generation": per_seed.tolist(),
            "gen0_rate": p0, "gen0_wilson95": [cen - half, cen + half],
            "inspection_random_genome_rate": prior[a]["valid_rate"], "inspection_n": prior[a]["n"],
            "gen0_agrees_with_inspection": bool(cen - half <= prior[a]["valid_rate"] <= cen + half),
            "logit_fit_gens0_19": {"intercept": b0, "slope_per_gen": b1, "slope_se": se1, "slope_z": b1 / se1,
                                   "fitted_gen0": fit[0], "fitted_gen19": fit[-1]},
            "logit_fit_gens1_19": {"intercept": b0o, "slope_per_gen": b1o, "slope_se": float(np.sqrt(covo[1, 1])),
                                   "fitted_gen1": 1 / (1 + np.exp(-(b0o + b1o))), "fitted_gen19": 1 / (1 + np.exp(-(b0o + b1o * 19)))},
            "mean_rate_gens1_19": float(k[1:].sum() / n[1:].sum()),
            "mean_rate_gens15_19": float(k[15:].sum() / n[15:].sum()),
            "run_mean_validity": V.mean(axis=(1, 2)).tolist(),
        }
        for s in range(5): ax.plot(g, per_seed[s], color="0.75", lw=1)
        ax.plot(g, rate, "o", color="C0", ms=4, label="pooled 5 seeds")
        ax.plot(g, fit, "-", color="C1", label="logit fit (gens 0-19)")
        ax.axhline(prior[a]["valid_rate"], ls="--", color="C3", label="random-genome prior (N=2000)")
        ax.set_title(a); ax.set_xlabel("generation"); ax.set_ylim(-0.02, 1.02)
    axs[0].set_ylabel("offspring validity rate"); axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Historical primary run (results/runs, Walker-v0, 5 seeds/arm): offspring validity by generation", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/validity_by_generation.png", dpi=130)
    json.dump(out, open(f"{OUT}/validity_by_generation.json", "w"), indent=2)
    for a, r in out["arms"].items():
        print(a, "gen0 %.3f wilson [%.3f,%.3f] prior %.4f agree=%s" % (r["gen0_rate"], *r["gen0_wilson95"], r["inspection_random_genome_rate"], r["gen0_agrees_with_inspection"]))
        print("  pooled:", np.round(r["rate_by_generation_pooled_5_seeds"], 2).tolist())
        f = r["logit_fit_gens0_19"]; print("  fit slope %.3f (se %.3f, z %.1f) gen0 %.3f -> gen19 %.3f" % (f["slope_per_gen"], f["slope_se"], f["slope_z"], f["fitted_gen0"], f["fitted_gen19"]))
        f = r["logit_fit_gens1_19"]; print("  fit(1-19) slope %.3f (se %.3f) gen19 %.3f; mean(1-19) %.3f mean(15-19) %.3f" % (f["slope_per_gen"], f["slope_se"], f["fitted_gen19"], r["mean_rate_gens1_19"], r["mean_rate_gens15_19"]))
        print("  per-run mean validity:", np.round(r["run_mean_validity"], 3).tolist())
        print("  per-seed final-5-gen:", np.round(np.array(r["per_seed_rate_by_generation"])[:, 15:].mean(axis=1), 2).tolist())


if __name__ == "__main__":
    main()
