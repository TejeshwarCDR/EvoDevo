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

from evodevo.analysis.aggregate import per_seed_finals, three_way_table
from evodevo.analysis.budget_curves import plot_two_panel_comparison
from evodevo.morphology.metrics import body_metrics
from evodevo.visualization.morphology import plot_body
from evodevo.morphology.validate import validate_body

REQUIRED = ("figures/fitness_curves.png", "figures/fitness_curves.pdf",
            "figures/budget_curves.png", "figures/budget_curves.pdf",
            "figures/validity_diversity.png", "figures/validity_diversity.pdf",
            "stats/omnibus.csv", "stats/pairwise.csv", "stats/tables.md",
            "stats/tables.tex", "stats/final_fitness_summary.csv",
            "stats/validity_diversity_by_generation.csv", "stats/current_rates.csv",
            "stats/final_best_morphology.csv", "stats/evaluations_to_90pct_arm_median.csv",
            "stats/rule_based_vs_monotone.csv", "MILESTONE_REPORT.md")


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


def _primary_summary(finals, status, fraction):
    rows = []
    for arm, values in sorted(finals.items()):
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append({"arm": arm, "n_seeds": len(values), "median": float(median),
                     "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
                     "status": status, "completion_fraction": fraction})
    return rows


def _secondary_tables(data, output, finals, status, fraction):
    generation_rows = []
    current_run_rows = []
    morphology_rows = []
    threshold_rows = []
    for (arm, seed), group in data.groupby(["encoder", "seed"]):
        group = group.sort_values("eval_index")
        for generation, records in group.groupby("generation"):
            valid = records[records["valid"] == True]
            distinct = valid["body_hash"].dropna().nunique() if "body_hash" in valid else 0
            generation_rows.append({"arm": arm, "seed": seed, "generation": generation,
                "evaluations": len(records), "valid": len(valid), "validity_rate": len(valid) / len(records),
                "unique_valid_bodies": distinct, "diversity_per_valid": distinct / len(valid) if len(valid) else 0,
                "status": status, "completion_fraction": fraction})
        valid = group[group["valid"] == True]
        distinct = valid["body_hash"].dropna().nunique() if "body_hash" in valid else 0
        current_run_rows.append({"arm": arm, "seed": seed, "evaluations": len(group),
            "validity_rate": len(valid) / len(group), "unique_valid_bodies": distinct,
            "diversity_per_valid": distinct / len(valid) if len(valid) else 0,
            "status": status, "completion_fraction": fraction})
        best = group.loc[group["fitness"].idxmax()]
        body = np.asarray(best["body"], dtype=int)
        metrics = body_metrics(body)
        morphology_rows.append({"arm": arm, "seed": seed, "best_fitness": float(best["fitness"]),
            "occupied_voxels": metrics["occupied_count"], "actuator_fraction": metrics["actuator_fraction"],
            "horizontal_symmetry": metrics["horizontal_symmetry_score"],
            "vertical_symmetry": metrics["vertical_symmetry_score"],
            "distinct_materials": len(set(body.flat) - {0}), "status": status,
            "completion_fraction": fraction})
        threshold = 0.9 * float(np.median(finals[arm]))
        reached = group[group["fitness"].cummax() >= threshold]
        threshold_rows.append({"arm": arm, "seed": seed, "arm_final_median": float(np.median(finals[arm])),
            "threshold": threshold, "reached": not reached.empty,
            "evaluations_to_threshold": int(reached.iloc[0]["eval_index"]) + 1 if not reached.empty else "",
            "status": status, "completion_fraction": fraction})
    current_rows = []
    for arm, group in data.groupby("encoder"):
        valid = group[group["valid"] == True]
        distinct = valid["body_hash"].dropna().nunique() if "body_hash" in valid else 0
        current_rows.append({"arm": arm, "n_seeds": group["seed"].nunique(),
            "evaluations": len(group), "valid": len(valid),
            "validity_rate": len(valid) / len(group), "unique_valid_bodies": distinct,
            "diversity_per_valid": distinct / len(valid) if len(valid) else 0,
            "status": status, "completion_fraction": fraction})
    tables = (
        ("validity_diversity_by_generation.csv", generation_rows),
        ("current_rates.csv", current_rows),
        ("current_run_rates.csv", current_run_rows),
        ("final_best_morphology.csv", morphology_rows),
        ("evaluations_to_90pct_arm_median.csv", threshold_rows),
    )
    for name, rows in tables:
        _csv(output / "stats" / name, rows, list(rows[0]))
    return generation_rows


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
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    planned = sum(int(r["config"]["eval_budget"]) for p in manifest.get("plan", {}).get("phases", [])
                  if p["task"] == task for r in p["runs"])
    fraction = len(data) / planned if planned else None
    status = "PARTIAL" if snapshot or fraction is None or fraction < 1 else "COMPLETE"
    for (_, _), indices in data.groupby(["encoder", "seed"]).groups.items():
        ordered = data.loc[indices].sort_values("eval_index")
        data.loc[ordered.index, "evals_spent"] = ordered["eval_index"].to_numpy() + 1
        data.loc[ordered.index, "evals_valid_cumulative"] = ordered["valid"].astype(int).cumsum().to_numpy()
        data.loc[ordered.index, "best_fitness_so_far"] = ordered["fitness"].cummax().to_numpy()
    finals = per_seed_finals(data)
    omnibus, pairwise = _stats(finals)
    if status == "PARTIAL":
        omnibus.update(status=status, completion_fraction=fraction,
                       n_per_arm=json.dumps({arm: len(values) for arm, values in finals.items()}, sort_keys=True))
        for row in pairwise:
            row.update(status=status, completion_fraction=fraction,
                       n_a=len(finals[row["arm_a"]]), n_b=len(finals[row["arm_b"]]))
    _csv(output / "stats/omnibus.csv", [omnibus], list(omnibus))
    fields = ["arm_a", "arm_b", "u", "p_raw", "p_holm", "a12", "a12_ci_low", "a12_ci_high"]
    if status == "PARTIAL":
        fields += ["status", "completion_fraction", "n_a", "n_b"]
    _csv(output / "stats/pairwise.csv", pairwise, fields)
    contrast = [r for r in pairwise if {r["arm_a"], r["arm_b"]} == {"rule_based", "rule_based_monotone"}]
    _csv(output / "stats/rule_based_vs_monotone.csv", contrast, fields)
    summary = three_way_table(data)
    summary.to_csv(output / "stats/summary.csv", index=False)
    primary = pd.DataFrame(_primary_summary(finals, status, fraction))
    primary.to_csv(output / "stats/final_fitness_summary.csv", index=False)
    try:
        markdown = primary.to_markdown(index=False)
    except ImportError:
        markdown = primary.to_csv(index=False)
    _atomic_text(output / "stats/tables.md", markdown + "\n")
    _atomic_text(output / "stats/tables.tex", primary.to_latex(index=False))
    generation_rows = _secondary_tables(data, output, finals, status, fraction)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, group in data.groupby("encoder"):
        curves = [run.sort_values("eval_index")["fitness"].cummax().to_numpy()
                  for _, run in group.groupby("seed")]
        n = min(map(len, curves))
        matrix = np.stack([curve[:n] for curve in curves])
        x = np.arange(1, n + 1)
        ax.plot(x, np.median(matrix, axis=0), label=arm)
        ax.fill_between(x, np.quantile(matrix, .25, axis=0), np.quantile(matrix, .75, axis=0), alpha=.2)
    ax.set(xlabel="Genomes evaluated", ylabel="Median best fitness (IQR band)",
           title=f"{task}: evolutionary progress" + (f" — PARTIAL {fraction:.1%}" if status == "PARTIAL" and fraction is not None else ""))
    ax.legend()
    (output / "figures").mkdir(exist_ok=True)
    fig.savefig(output / "figures/fitness_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / "figures/fitness_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, _ = plot_two_panel_comparison(data)
    fig.suptitle(f"{task}: spent versus valid evaluations" +
                 (f" — PARTIAL {fraction:.1%}" if status == "PARTIAL" and fraction is not None else ""))
    fig.savefig(output / "figures/budget_curves.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / "figures/budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    rates = pd.DataFrame(generation_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, group in rates.groupby("arm"):
        by_gen = group.groupby("generation")
        for ax, field in zip(axes, ("validity_rate", "diversity_per_valid")):
            values = by_gen[field]
            x = np.asarray(sorted(group["generation"].unique()))
            ax.plot(x, values.median().reindex(x), label=arm)
            ax.fill_between(x, values.quantile(.25).reindex(x), values.quantile(.75).reindex(x), alpha=.2)
    axes[0].set(xlabel="Generation", ylabel="Valid evaluations / evaluations")
    axes[1].set(xlabel="Generation", ylabel="Unique valid bodies / valid evaluations")
    axes[0].legend()
    fig.suptitle(f"{task}: validity and diversity" +
                 (f" — PARTIAL {fraction:.1%}" if status == "PARTIAL" and fraction is not None else ""))
    fig.savefig(output / "figures/validity_diversity.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / "figures/validity_diversity.pdf", bbox_inches="tight")
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
        np.savez(npz, body=body, fitness=float(best["fitness"]), seed=int(best["seed"]),
                 status=status, completion_fraction=fraction if fraction is not None else np.nan)
        axes = plot_body(body, title=f"{arm}, {task}, fitness {best['fitness']:.3f}" +
                         (f", PARTIAL {fraction:.1%}" if status == "PARTIAL" and fraction is not None else ""))
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
        if status == "PARTIAL":
            sensitivity["status"] = status
            sensitivity["completion_fraction"] = fraction
            sensitivity["n_seeds"] = sensitivity["encoder"].map(
                {arm: len(values) for arm, values in per_seed_finals(changed).items()})
        sensitivity.to_csv(output / "stats/mean_of_2_sensitivity.csv", index=False)
        mean_omnibus, mean_pairwise = _stats(per_seed_finals(changed))
        if status == "PARTIAL":
            mean_finals = per_seed_finals(changed)
            mean_omnibus.update(status=status, completion_fraction=fraction,
                                n_per_arm=json.dumps({arm: len(values) for arm, values in mean_finals.items()},
                                                     sort_keys=True))
            for row in mean_pairwise:
                row.update(status=status, completion_fraction=fraction,
                           n_a=len(mean_finals[row["arm_a"]]), n_b=len(mean_finals[row["arm_b"]]))
        _csv(output / "stats/mean_of_2_omnibus.csv", [mean_omnibus], list(mean_omnibus))
        _csv(output / "stats/mean_of_2_pairwise.csv", mean_pairwise, fields)
        pd.DataFrame(_primary_summary(per_seed_finals(changed), status, fraction)).to_csv(
            output / "stats/mean_of_2_final_fitness_summary.csv", index=False)
    report = (f"# {task} milestone report\n\n"
              f"Status: {status}. Completion: {len(data)}/{planned or '?'} evaluations"
              f" ({fraction:.1%} of plan).\n\n" if fraction is not None else
              f"# {task} milestone report\n\nStatus: {status}. Completion: {len(data)} evaluations; plan unavailable.\n\n")
    if status == "PARTIAL":
        report += ("PARTIAL — monitoring and drafting only; do not cite without a completeness check. "
                   "Every table has n per arm or run and a completion-fraction column.\n\n")
    report += ("The primary table contains median and IQR of final best-so-far fitness "
               "per independent run; its n_seeds column reports the available replicates. "
               "Kruskal-Wallis tests the arm distributions; pairwise Mann-Whitney U p-values "
               "use Holm correction. A12 is the probability that an observation from the first "
               "arm exceeds one from the second, with ties counted as half; bootstrap 95% "
               "intervals are descriptive at five seeds. The progress plot is the across-seed "
               "median with an IQR band. The two-panel budget plot separates evaluations spent "
               "from valid search attempts.\n\n" + markdown + "\n\n"
               f"Omnibus H={omnibus['statistic']:.4g}, p={omnibus['p']:.4g}.\n\n"
               "The rule_based versus rule_based_monotone contrast is in "
               "stats/rule_based_vs_monotone.csv.\n\n")
    report += ("Secondary endpoints: stats/validity_diversity_by_generation.csv gives realised "
               "validity and unique-valid-body diversity by generation; stats/current_rates.csv "
               "gives per-run rates. stats/final_best_morphology.csv gives occupied voxels, "
               "actuator fraction, two axis-specific symmetry scores, and distinct nonempty "
               "materials for each run's best body. stats/evaluations_to_90pct_arm_median.csv "
               "uses 90% of each arm's current final median; unreached runs are explicit.\n\n")
    if sensitivity is not None:
        report += ("UpStepper mean-of-2 sensitivity: stats/mean_of_2_sensitivity.csv, "
                   "stats/mean_of_2_final_fitness_summary.csv, stats/mean_of_2_omnibus.csv "
                   "and stats/mean_of_2_pairwise.csv.\n\n")
    report += ("Pre-registered limitations: NCA development does not settle; the validity "
               "proxy assumes random fitness; planned n=5 seeds per arm per task.\n\n")
    report += "Figures and best-body files are listed in package_manifest.json. Videos require a green visualisation check.\n"
    if video_notes:
        report += "\n" + "\n".join(video_notes) + "\n"
    _atomic_text(output / "MILESTONE_REPORT.md", report)
    files = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    _atomic_text(output / "package_manifest.json", json.dumps({"task": task, "snapshot": snapshot,
                  "completion_fraction": fraction, "generated_at": time.time(),
                  "files": files}, indent=2, sort_keys=True) + "\n")
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    missing += [name for name in best_files if not (output / name).is_file()]
    if task == "UpStepper-v0":
        missing += [name for name in ("stats/mean_of_2_sensitivity.csv", "stats/mean_of_2_omnibus.csv",
                                      "stats/mean_of_2_pairwise.csv", "stats/mean_of_2_final_fitness_summary.csv")
                    if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete milestone package: {missing}")
    return {"output": str(output), "files": files, "completion_fraction": fraction}


def run_subprocess(campaign_root, task, alert_manager=None, *, snapshot=False):
    argv = [sys.executable, "-m", "evodevo.campaign.milestone", "--campaign-root", str(campaign_root), "--task", task]
    if snapshot:
        argv.append("--snapshot")
    started_at = time.time()
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
    package_path = output / "package_manifest.json"
    package = None
    if not package_path.is_file():
        missing.append("package_manifest.json")
    else:
        try:
            package = json.loads(package_path.read_text())
            if package.get("generated_at", 0) < started_at:
                missing.append("fresh package_manifest.json for this invocation")
            missing += [name for name in package.get("files", []) if not (output / name).is_file()]
        except (OSError, ValueError, TypeError):
            missing.append("valid package_manifest.json")
    for arm in {directory.parents[1].name for directory in _run_dirs(Path(campaign_root), task)}:
        missing += [name for name in (f"best_bodies/{arm}.npz", f"best_bodies/{arm}.png")
                    if not (output / name).is_file()]
    if task == "UpStepper-v0":
        missing += [name for name in ("stats/mean_of_2_sensitivity.csv", "stats/mean_of_2_omnibus.csv",
                                      "stats/mean_of_2_pairwise.csv", "stats/mean_of_2_final_fitness_summary.csv")
                    if not (output / name).is_file()]
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
