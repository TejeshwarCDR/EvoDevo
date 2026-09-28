"""Paper-drafting package for a completed phase or labelled partial snapshot."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from evodevo.analysis.aggregate import load_runs, per_seed_finals, three_way_table
from evodevo.visualization.morphology import plot_body
from evodevo.morphology.validate import validate_body

REQUIRED = ("figures/fitness_curves.png", "figures/fitness_curves.pdf",
            "stats/omnibus.csv", "stats/pairwise.csv", "stats/tables.md",
            "stats/tables.tex", "stats/rule_based_vs_monotone.csv", "MILESTONE_REPORT.md")


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _a12(a, b):
    x, y = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float((np.sum(x > y) + 0.5 * np.sum(x == y)) / (x.size * y.size))


def _a12_ci(a, b, seed=0, n_boot=4000):
    rng = np.random.default_rng(seed)
    samples = [_a12(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))


def _chi2_sf(statistic, degrees):
    """Upper regularized gamma Q(df/2, statistic/2), no SciPy dependency."""
    a, x = degrees / 2.0, statistic / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = 1.0 / a
        series = term
        for n in range(1, 1000):
            term *= x / (a + n)
            series += term
            if abs(term) < abs(series) * 1e-14:
                break
        return max(0.0, 1.0 - series * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1 - a
    c = 1e30
    d = 1 / b
    fraction = d
    for n in range(1, 1000):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1) < 1e-14:
            break
    return min(1.0, max(0.0, fraction * math.exp(-x + a * math.log(x) - math.lgamma(a))))


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_sizes = []
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for index in order[i:j]:
            ranks[index] = rank
        tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def _kruskal(groups):
    lengths = [len(values) for values in groups]
    pooled = [float(v) for values in groups for v in values]
    n = len(pooled)
    ranks, ties = _average_ranks(pooled)
    h = 0.0
    offset = 0
    for size in lengths:
        h += sum(ranks[offset:offset + size]) ** 2 / size
        offset += size
    h = 12 * h / (n * (n + 1)) - 3 * (n + 1)
    correction = 1 - sum(t ** 3 - t for t in ties) / (n ** 3 - n)
    h = h / correction if correction else 0.0
    return h, _chi2_sf(h, len(groups) - 1)


def _mann_whitney(a, b):
    pooled = [float(v) for v in a] + [float(v) for v in b]
    na, nb = len(a), len(b)
    ranks, _ = _average_ranks(pooled)
    u = sum(ranks[:na]) - na * (na + 1) / 2
    mean = na * nb / 2
    if math.comb(na + nb, na) <= 100000:
        total = extreme = 0
        for choice in itertools.combinations(range(na + nb), na):
            candidate = sum(ranks[i] for i in choice) - na * (na + 1) / 2
            extreme += abs(candidate - mean) >= abs(u - mean) - 1e-12
            total += 1
        p = extreme / total
    else:
        sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
        p = math.erfc(abs(u - mean) / (sigma * math.sqrt(2))) if sigma else 1.0
    return u, p


def _stats(finals):
    arms = sorted(finals)
    if len(arms) >= 2 and all(len(finals[a]) for a in arms):
        if len({float(value) for arm in arms for value in finals[arm]}) == 1:
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = _kruskal([finals[a] for a in arms])
        omnibus_row = {"test": "Kruskal-Wallis", "statistic": statistic,
                       "p": pvalue, "n_arms": len(arms)}
    else:
        omnibus_row = {"test": "Kruskal-Wallis", "statistic": float("nan"),
                       "p": float("nan"), "n_arms": len(arms)}
    rows = []
    for a, b in itertools.combinations(arms, 2):
        u, p = _mann_whitney(finals[a], finals[b])
        lo, hi = _a12_ci(finals[a], finals[b])
        rows.append({"arm_a": a, "arm_b": b, "u": float(u), "p_raw": float(p),
                     "a12": _a12(finals[a], finals[b]), "a12_ci_low": lo,
                     "a12_ci_high": hi})
    # Holm step-down adjustment, monotone across sorted p-values.
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, rows[index]["p_raw"] * (len(rows) - rank)))
        rows[index]["p_holm"] = previous
    return omnibus_row, rows


def _run_dirs(root, task):
    return sorted(path.parent for path in (root / "runs").glob(f"*/{task}/seed_*/run_log.jsonl"))


def build(campaign_root, task, *, snapshot=False, timestamp=None):
    # EvoGym's rendering path can change cwd as a side effect; every package
    # path must be absolute before the visualisation gate is exercised.
    root = Path(campaign_root).resolve()
    runs = _run_dirs(root, task)
    if not runs:
        raise RuntimeError(f"no run logs for {task} under {root}")
    if snapshot:
        label = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output = root / "milestones" / "snapshots" / label
    else:
        output = root / "milestones" / task
    output.mkdir(parents=True, exist_ok=True)
    # The existing analysis reader expects run_meta.json; scheduler records
    # are sufficient for this package even if metadata has not yet landed.
    frames = []
    import pandas as pd
    for directory in runs:
        records = [json.loads(line) for line in (directory / "run_log.jsonl").read_text().splitlines() if line]
        if not records:
            continue
        frame = pd.DataFrame(records)
        frame["encoder"] = directory.parents[1].name
        frame["task"] = task
        frame["seed"] = int(directory.name.removeprefix("seed_"))
        frames.append(frame)
    if not frames:
        raise RuntimeError("run logs contain no evaluations")
    data = pd.concat(frames, ignore_index=True)
    finals = per_seed_finals(data)
    omnibus, pairwise = _stats(finals)
    _csv(output / "stats/omnibus.csv", [omnibus], list(omnibus))
    fields = ["arm_a", "arm_b", "u", "p_raw", "p_holm", "a12", "a12_ci_low", "a12_ci_high"]
    _csv(output / "stats/pairwise.csv", pairwise, fields)
    contrast = [r for r in pairwise if {r["arm_a"], r["arm_b"]} == {"rule_based", "rule_based_monotone"}]
    _csv(output / "stats/rule_based_vs_monotone.csv", contrast, fields)
    summary = three_way_table(data)
    summary.to_csv(output / "stats/summary.csv", index=False)
    try:
        markdown = summary.to_markdown(index=False)
    except ImportError:
        markdown = summary.to_csv(index=False)
    _atomic_text(output / "stats/tables.md", markdown + "\n")
    _atomic_text(output / "stats/tables.tex", summary.to_latex(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, group in data.groupby("encoder"):
        curves = []
        for _, run in group.groupby("seed"):
            run = run.sort_values("eval_index")
            curves.append(run["fitness"].cummax().to_numpy())
        n = min(map(len, curves))
        ax.plot(np.arange(1, n + 1), np.mean([v[:n] for v in curves], axis=0), label=arm)
    ax.set(xlabel="Genomes evaluated", ylabel="Mean best fitness", title=f"{task}: evolutionary progress")
    ax.legend()
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures/fitness_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / "figures/fitness_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    best_files = []
    video_notes = []
    visualization_green = None
    for arm, group in data.groupby("encoder"):
        best = group.loc[group["fitness"].idxmax()]
        body = np.asarray(best["body"], dtype=int)
        best_dir = output / "best_bodies"
        best_dir.mkdir(exist_ok=True)
        npz = best_dir / f"{arm}.npz"
        np.savez(npz, body=body, fitness=float(best["fitness"]), seed=int(best["seed"]))
        axes = plot_body(body, title=f"{arm}, {task}, fitness {best['fitness']:.3f}")
        image = best_dir / f"{arm}.png"
        axes.figure.savefig(image, dpi=160, bbox_inches="tight")
        plt.close(axes.figure)
        best_files.extend((str(npz.relative_to(output)), str(image.relative_to(output))))
        # Only a *trained* controller produces paper-facing video. The
        # repository's visualisation self-test gates rollout rendering; a
        # failed gate or renderer never invalidates the statistics package.
        policies = best.get("policy_paths")
        if not snapshot and validate_body(body).valid and isinstance(policies, list) and policies:
            policy = str(Path(str(policies[0])).resolve())
            if visualization_green is None:
                try:
                    from evodevo.visualization.check import run_all
                    visualization_green = all(check.ok for check in run_all())
                except Exception as exc:
                    visualization_green = False
                    video_notes.append(f"Visualisation self-test raised {type(exc).__name__}: {exc}.")
            if visualization_green:
                try:
                    from evodevo.visualization.render import render_body_rollout
                    if not os.path.isfile(policy):
                        raise FileNotFoundError(policy)
                    rollout = render_body_rollout(body, policy, task, 200,
                        str(best_dir / f"{arm}_rollout"), seed=int(best["seed"]))
                    best_files.extend((str(Path(rollout.mp4_path).relative_to(output.resolve())),
                                       str(Path(rollout.gif_path).relative_to(output.resolve()))))
                    video_notes.append(f"{arm}: trained-controller MP4 and GIF saved.")
                except Exception as exc:
                    video_notes.append(f"{arm}: video unavailable ({type(exc).__name__}: {exc}).")
            else:
                video_notes.append(f"{arm}: video omitted because the visualisation self-test failed.")
    sensitivity = None
    if task == "UpStepper-v0" and "per_seed_fitness" in data:
        changed = data.copy()
        changed["fitness"] = changed.apply(
            lambda r: float(np.mean([v for v in (r["per_seed_fitness"] or []) if v is not None]))
            if isinstance(r["per_seed_fitness"], list) and any(v is not None for v in r["per_seed_fitness"])
            else r["fitness"], axis=1)
        sensitivity = three_way_table(changed)
        sensitivity.to_csv(output / "stats/mean_of_2_sensitivity.csv", index=False)
        mean_omnibus, mean_pairwise = _stats(per_seed_finals(changed))
        _csv(output / "stats/mean_of_2_omnibus.csv", [mean_omnibus], list(mean_omnibus))
        _csv(output / "stats/mean_of_2_pairwise.csv", mean_pairwise, fields)
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    planned = sum(int(r["config"]["eval_budget"]) for p in manifest.get("plan", {}).get("phases", [])
                  if p["task"] == task for r in p["runs"])
    fraction = len(data) / planned if planned else None
    report = (f"# {task} milestone report\n\n"
              f"Completion: {len(data)}/{planned or '?'} evaluations"
              f" ({fraction:.1%} of plan).\n\n" if fraction is not None else
              f"# {task} milestone report\n\nCompletion: {len(data)} evaluations; plan unavailable.\n\n")
    report += ("The table contains final best fitness per independent run. "
               "Kruskal-Wallis tests the arm distributions; pairwise Mann-Whitney U p-values "
               "use Holm correction. A12 is the probability that an observation from the first "
               "arm exceeds one from the second, with ties counted as half; bootstrap 95% "
               "intervals are descriptive at five seeds.\n\n" + markdown + "\n\n"
               f"Omnibus H={omnibus['statistic']:.4g}, p={omnibus['p']:.4g}.\n\n"
               "The rule_based versus rule_based_monotone contrast is in "
               "stats/rule_based_vs_monotone.csv.\n\n")
    if sensitivity is not None:
        report += ("UpStepper mean-of-2 sensitivity: stats/mean_of_2_sensitivity.csv, "
                   "stats/mean_of_2_omnibus.csv and stats/mean_of_2_pairwise.csv.\n\n")
    report += "Figures and best-body files are listed in package_manifest.json. Videos require a green visualisation check.\n"
    if video_notes:
        report += "\n" + "\n".join(video_notes) + "\n"
    _atomic_text(output / "MILESTONE_REPORT.md", report)
    files = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    _atomic_text(output / "package_manifest.json", json.dumps({"task": task, "snapshot": snapshot,
                  "completion_fraction": fraction, "files": files}, indent=2, sort_keys=True) + "\n")
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    missing += [name for name in best_files if not (output / name).is_file()]
    if task == "UpStepper-v0":
        missing += [name for name in ("stats/mean_of_2_sensitivity.csv", "stats/mean_of_2_omnibus.csv",
                                      "stats/mean_of_2_pairwise.csv") if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete milestone package: {missing}")
    return {"output": str(output), "files": files, "completion_fraction": fraction}


def run_subprocess(campaign_root, task, alert_manager=None, *, snapshot=False):
    argv = [sys.executable, "-m", "evodevo.campaign.milestone", "--campaign-root", str(campaign_root), "--task", task]
    if snapshot:
        argv.append("--snapshot")
    try:
        completed = subprocess.run(argv, capture_output=True, text=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if alert_manager:
            alert_manager.milestone_failure(task, ["subprocess launch"], error)
        return {"ok": False, "missing": ["subprocess launch"], "error": error}
    output = Path(campaign_root) / "milestones" / task
    if snapshot and completed.returncode == 0:
        try:
            output = Path(json.loads(completed.stdout)["output"])
        except (ValueError, KeyError, TypeError):
            pass
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if not (output / "package_manifest.json").is_file():
        missing.append("package_manifest.json")
    if not list((output / "best_bodies").glob("*.npz")):
        missing.append("best_bodies/*.npz")
    if not list((output / "best_bodies").glob("*.png")):
        missing.append("best_bodies/*.png")
    if task == "UpStepper-v0":
        missing += [name for name in ("stats/mean_of_2_sensitivity.csv", "stats/mean_of_2_omnibus.csv",
                                      "stats/mean_of_2_pairwise.csv") if not (output / name).is_file()]
    if completed.returncode or missing:
        if alert_manager:
            alert_manager.milestone_failure(task, missing, completed.stderr[-2000:])
        return {"ok": False, "missing": missing, "error": completed.stderr}
    return {"ok": True, "output": completed.stdout.strip()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign")
    parser.add_argument("--campaign-root")
    parser.add_argument("--output-root", default="results/campaign")
    parser.add_argument("--task", required=True)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root or os.path.join(args.output_root, args.campaign)
    print(json.dumps(build(root, args.task, snapshot=args.snapshot), sort_keys=True))


if __name__ == "__main__":
    main()
