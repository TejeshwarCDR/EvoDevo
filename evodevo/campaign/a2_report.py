"""Amendment 2 milestone packages: merged phase provenance and the re-training tables.

* ``phase_sources`` resolves which run directories a phase's package merges
  (this campaign's phases, and the original campaign's seeds 0-4 read-only)
  and labels every run with where it came from.
* ``build`` writes the re-evaluation package (secondary endpoint) after a
  ``reeval`` stage, the robustness package after a ``robustness`` stage, and
  the final task-budget-versus-2x package ("robustness:all").

The unit of analysis is the champion (one per EA run): its re-trainings are
averaged first, so the arm comparison keeps n = number of runs per arm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import units as unit_keys
from .amend2 import OUTPUT_FILES, latest_rows, original_manifest, planned_trainings, resolve_repo_path, settled

REEVAL_FILES = ("stats/reeval_trainings.csv", "stats/reeval_per_champion.csv", "stats/reeval_per_arm.csv",
                "stats/reeval_omnibus.csv", "stats/reeval_pairwise.csv", "stats/reeval_vs_recorded.csv",
                "figures/reeval_by_arm.png", "figures/reeval_by_arm.pdf", "MILESTONE_REPORT.md")
ROBUSTNESS_FILES = ("stats/robustness_trainings.csv", "stats/robustness_per_champion.csv",
                    "stats/robustness_rankings.csv", "stats/robustness_a12.csv",
                    "figures/robustness_by_arm.png", "figures/robustness_by_arm.pdf", "MILESTONE_REPORT.md")


def required_files(key: str) -> tuple:
    return REEVAL_FILES if key.startswith("reeval:") else ROBUSTNESS_FILES


# --- evolutionary phase packages ------------------------------------------------

def _spec_phase(plan: dict, key: str) -> dict:
    return next(p for p in plan["amendment2"]["phases"] if p["id"] == key)


def phase_sources(root, plan: dict, key: str) -> dict:
    """Run directories merged into the package of phase ``key``."""
    root = Path(root)
    spec = plan["amendment2"]
    item = _spec_phase(plan, key)
    package = item["package"]
    phases = {unit_keys.phase_key(p): p for p in plan["phases"]}
    campaign_id = json.loads((root / "manifest.json").read_text()).get("campaign_id", root.name)
    original = spec["original_campaign"]
    original_root = resolve_repo_path(original["path"])
    original_budgets = {}
    runs, planned, seeds_by_origin = [], 0, []
    contrast = []
    for entry in package["merge"]:
        if "phase" in entry:
            phase = phases[entry["phase"]]
            label = f"campaign {campaign_id}, phase {entry['phase']}"
            for run in phase["runs"]:
                directory = root / "runs" / run["arm"] / phase["task"] / f"seed_{run['seed']}"
                planned += int(run["config"]["eval_budget"])
                if (directory / "run_log.jsonl").is_file():
                    runs.append((directory, {"label": label, "origin": "campaign", "phase": entry["phase"]}))
            seeds_by_origin.append(f"seeds {_span(sorted({r['seed'] for r in phase['runs']}))} of {entry['phase']}")
        else:
            label = f"original campaign {original['id']} (commit {original['git_commit'][:7]}), read-only"
            if not original_budgets:
                original_budgets = {(r["arm"], ph["task"], r["seed"]): int(r["config"]["eval_budget"])
                                    for ph in original_manifest(original)["plan"]["phases"] for r in ph["runs"]}
            for arm in entry["arms"]:
                for seed in entry["seeds"]:
                    directory = original_root / "runs" / arm / item["task"] / f"seed_{seed}"
                    runs.append((directory, {"label": label, "origin": "original", "phase": "original"}))
                    planned += original_budgets[(arm, item["task"], seed)]
            seeds_by_origin.append(f"original seeds {_span(entry['seeds'])}")
            if package.get("monotone_contrast"):
                for arm in ("rule_based", "rule_based_monotone"):
                    for seed in entry["seeds"]:
                        contrast.append((original_root / "runs" / arm / item["task"] / f"seed_{seed}",
                                         {"label": label, "origin": "original", "phase": "original"}))
    total_seeds = sorted({int(d.name.removeprefix("seed_")) for d, _ in runs})
    return {"runs": runs, "contrast_runs": contrast, "planned": planned, "label": package["label"],
            "seed_summary": f"{len(total_seeds)} seeds per arm: " + "; ".join(seeds_by_origin),
            "origin_labels": sorted({p["label"] for _, p in runs})}


def _span(seeds) -> str:
    seeds = sorted(seeds)
    return f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else ",".join(map(str, seeds))


def write_provenance_tables(data, output: Path, sources: dict) -> None:
    import pandas as pd
    rows = []
    for (arm, seed), group in data.groupby(["encoder", "seed"]):
        group = group.sort_values("eval_index")
        best = group.loc[group["fitness"].idxmax()]
        rows.append({"arm": arm, "seed": int(seed), "final_best_fitness": float(group["fitness"].cummax().iloc[-1]),
                     "champion_eval_index": int(best["eval_index"]), "provenance": group["provenance"].iloc[0]})
    pd.DataFrame(rows).to_csv(output / "stats/per_run_finals.csv", index=False)
    morphology = output / "stats/final_best_morphology.csv"
    if morphology.is_file():
        frame = pd.read_csv(morphology)
        label = {(r["arm"], r["seed"]): r["provenance"] for r in rows}
        frame["provenance"] = [label[(a, int(s))] for a, s in zip(frame["arm"], frame["seed"])]
        frame.to_csv(morphology, index=False)


def provenance_paragraph(sources: dict, status: str) -> str:
    text = ("Provenance: every run carries a provenance column (stats/per_run_finals.csv, "
            "stats/final_best_morphology.csv). Merged sources: " + "; ".join(sources["origin_labels"]) + ". "
            f"{sources['seed_summary']}.\n\n")
    if status == "INTERIM":
        text += ("INTERIM: this package covers the first seed block only. It is a preview, committed "
                 "unconditionally, and cannot stop or change any later phase.\n\n")
    if sources["contrast_runs"]:
        text += ("The rule_based versus rule_based_monotone contrast (stats/rule_based_vs_monotone.csv) uses "
                 "the original campaign's seeds 0-4 only; the monotone arm never ran at seeds 5-9.\n\n")
    return text


# --- re-training stages -------------------------------------------------------------

def _champion_rows(rows: list) -> dict:
    """{(arm, task, seed): [usable secondary-endpoint rows]}.

    Successful PPO trainings contribute their measured fitness.  A champion
    from an all-invalid evolutionary run contributes its preregistered
    sentinel once per planned training seed: excluding it would bias the
    comparison toward encodings that more often find a valid body.
    """
    out = {}
    for row in rows:
        invalid_champion = str(row.get("reason") or "").startswith("invalid_champion:")
        if row.get("fitness") is not None and (row.get("completed") or invalid_champion):
            out.setdefault((row["arm"], row["task"], row["seed"]), []).append(row)
    return out


def per_champion(rows: list) -> list:
    table = []
    for (arm, task, seed), items in sorted(_champion_rows(rows).items()):
        values = [float(r["fitness"]) for r in items]
        table.append({"task": task, "arm": arm, "seed": seed, "provenance": _origin(items[0]),
                      "champion_eval_index": items[0]["champion_eval_index"],
                      "champion_recorded_fitness": items[0]["champion_fitness"],
                      "n_trainings": len(values), "mean": float(np.mean(values)),
                      "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                      "median": float(np.median(values)), "min": min(values), "max": max(values)})
    return table


def _origin(row: dict) -> str:
    return f"{row['source_kind']}: {row['source_campaign']} (commit {str(row['source_commit'])[:7]})"


def per_arm(champions: list) -> list:
    table = []
    for arm in sorted({c["arm"] for c in champions}):
        mine = [c for c in champions if c["arm"] == arm]
        means = [c["mean"] for c in mine]
        within = [c["sd"] for c in mine if c["sd"] == c["sd"]]
        table.append({"arm": arm, "n_champions": len(mine), "mean_of_champion_means": float(np.mean(means)),
                      "sd_between_champions": float(np.std(means, ddof=1)) if len(means) > 1 else float("nan"),
                      "median_of_champion_means": float(np.median(means)),
                      "mean_within_champion_sd": float(np.mean(within)) if within else float("nan"),
                      "mean_recorded_champion_fitness": float(np.mean([c["champion_recorded_fitness"] for c in mine])),
                      "mean_shrinkage_vs_recorded": float(np.mean([c["champion_recorded_fitness"] - c["mean"]
                                                                   for c in mine]))})
    return table


def _stats_of(champions: list):
    from .milestone import _stats
    by_arm = {}
    for c in champions:
        by_arm.setdefault(c["arm"], []).append(c["mean"])
    return _stats(by_arm)


def _savefig(fig, output: Path, name: str) -> None:
    (output / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"figures/{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"figures/{name}.pdf", bbox_inches="tight")


def _plot_by_arm(champions: list, rows: list, title: str, output: Path, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({c["arm"] for c in champions})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, arm in enumerate(arms):
        points = [float(r["fitness"]) for r in rows if r["arm"] == arm and r.get("completed")
                  and r.get("fitness") is not None]
        means = [c["mean"] for c in champions if c["arm"] == arm]
        ax.scatter(np.full(len(points), index) + np.linspace(-0.18, 0.18, len(points) or 1)[:len(points)],
                   points, s=8, alpha=0.35, label="single trainings" if index == 0 else None)
        ax.boxplot([means], positions=[index], widths=0.35, showfliers=False)
    ax.set_xticks(range(len(arms)), arms)
    ax.set(ylabel="Fitness of re-trained champion", title=title)
    ax.legend()
    _savefig(fig, output, name)
    plt.close(fig)


def _csv(path: Path, rows: list, fields=None) -> None:
    from .milestone import _csv as write_csv
    rows = rows or [{}]
    write_csv(path, rows, fields or list(rows[0]))


def _mark(manifest: dict) -> str:
    stamp = manifest.get("artifact_stamp")
    return f" — {stamp}" if stamp else ""


def _stage_rows(root: Path, kind: str, stage_id: str | None = None, task: str | None = None) -> list:
    rows = [row for key, row in latest_rows(root / OUTPUT_FILES[kind]).items()
            if (stage_id is None or key[0] == stage_id) and (task is None or row["task"] == task)]
    return [row for row in rows if settled(row)]


def _is_interim(plan: dict, stage: dict) -> bool:
    by_id = {p["id"]: p for p in plan["amendment2"]["phases"]}
    sources = [s["phase"] for s in stage["sources"] if "phase" in s]
    return bool(sources) and len(sources) == len(stage["sources"]) and all(
        by_id[s]["package"]["label"] == "INTERIM" for s in sources)


def build(root, key: str, manifest: dict, *, snapshot=False, timestamp=None) -> dict:
    root = Path(root)
    plan = manifest["plan"]
    stamp = manifest.get("artifact_stamp")
    if key == "robustness:all":
        return _build_robustness(root, key, plan, manifest, tasks=None, stages=None)
    unit = unit_keys.unit_by_key(plan, key)
    if unit is None or unit["kind"] not in unit_keys.STAGE_KINDS:
        raise RuntimeError(f"{key!r} is not a re-training stage of this campaign")
    if unit["kind"] == unit_keys.REEVAL:
        return _build_reeval(root, key, unit["stage"], plan, manifest)
    return _build_robustness(root, key, plan, manifest, tasks=[unit["task"]], stages=[unit["stage"]])


def _finish(output: Path, key: str, status: str, fraction, files_note: str, manifest: dict, report: str) -> dict:
    stamp = manifest.get("artifact_stamp")
    from .milestone import _atomic_text
    if stamp:
        _atomic_text(output / "STAMP.txt", stamp + "\n")
        report = f"> **{stamp}**\n\n" + report
    _atomic_text(output / "MILESTONE_REPORT.md", report)
    files = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
    _atomic_text(output / "package_manifest.json", json.dumps(
        {"task": key, "snapshot": False, "artifact_stamp": stamp, "status": status, "completion_fraction": fraction,
         "generated_at": time.time(), "files": files}, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), "files": files, "completion_fraction": fraction}


def _build_reeval(root: Path, key: str, stage: dict, plan: dict, manifest: dict) -> dict:
    output = root / "milestones" / unit_keys.slug(key)
    output.mkdir(parents=True, exist_ok=True)
    rows = _stage_rows(root, "reeval", key)
    planned = planned_trainings(stage, plan)
    fraction = len(rows) / planned if planned else None
    interim = _is_interim(plan, stage)
    status = "PARTIAL" if fraction is None or fraction < 1 else ("INTERIM" if interim else "COMPLETE")
    champions = per_champion(rows)
    arms = per_arm(champions)
    omnibus, pairwise = _stats_of(champions)
    _csv(output / "stats/reeval_trainings.csv", rows)
    _csv(output / "stats/reeval_per_champion.csv", champions)
    _csv(output / "stats/reeval_per_arm.csv", arms)
    _csv(output / "stats/reeval_omnibus.csv", [omnibus])
    _csv(output / "stats/reeval_pairwise.csv", pairwise)
    _csv(output / "stats/reeval_vs_recorded.csv", [
        {"task": c["task"], "arm": c["arm"], "seed": c["seed"], "recorded_champion_fitness":
         c["champion_recorded_fitness"], "reeval_mean": c["mean"],
         "shrinkage": c["champion_recorded_fitness"] - c["mean"]} for c in champions])
    mark = _mark(manifest) + (" — INTERIM" if status == "INTERIM" else "")
    _plot_by_arm(champions, rows, f"{stage['task']}: champion re-training, {stage['n']} trainings each{mark}",
                 output, "reeval_by_arm")
    table = "\n".join(f"| {a['arm']} | {a['n_champions']} | {a['mean_of_champion_means']:.4g} | "
                      f"{a['sd_between_champions']:.4g} | {a['mean_within_champion_sd']:.4g} | "
                      f"{a['mean_shrinkage_vs_recorded']:.4g} |" for a in arms)
    report = (f"# {stage['task']} re-evaluation ({key}) milestone report\n\n"
              f"Status: {status}. Completion: {len(rows)}/{planned} trainings"
              + (f" ({fraction:.1%})" if fraction is not None else "") + ".\n\n"
              f"Secondary endpoint (Amendment 2 (g)): each run's champion (maximum recorded fitness, ties to the "
              f"lowest evaluation index) re-trained {stage['n']} times, each a single training at "
              f"{stage['budget']:,} PPO steps, PPO seeds ppo_seed({stage['seed_base']}, k), k = 0..{stage['n'] - 1}. "
              "The unit of analysis is the champion: its trainings are averaged first.\n\n"
              "| arm | champions | mean of champion means | sd between champions | mean within-champion sd | "
              "mean (recorded - re-trained) |\n|---|---:|---:|---:|---:|---:|\n" + table + "\n\n"
              f"Kruskal-Wallis on champion means: H={omnibus['statistic']:.4g}, p={omnibus['p']:.4g}; pairwise "
              "Mann-Whitney U with Holm correction over the 3 pairs, A12 with bootstrap 95% intervals, in "
              "stats/reeval_pairwise.csv. Per-champion means and spread are in stats/reeval_per_champion.csv; "
              "every single training is in stats/reeval_trainings.csv (and reeval.jsonl in the campaign "
              "directory).\n\n")
    if status == "INTERIM":
        report += ("INTERIM: first seed block only (5 seeds per arm). Committed unconditionally; it cannot stop "
                   "or change later phases.\n\n")
    sources = sorted({_origin(r) for r in rows})
    report += "Champion sources: " + "; ".join(sources) + ".\n"
    return _finish(output, key, status, fraction, "", manifest, report)


def _arm_means(rows: list) -> dict:
    """{(task, arm): {(seed): mean}} over completed trainings."""
    out = {}
    for champion in per_champion(rows):
        out.setdefault((champion["task"], champion["arm"]), {})[champion["seed"]] = champion["mean"]
    return out


def ranking_tables(reeval_rows: list, robust_rows: list, tasks: list) -> tuple:
    from .milestone import _a12, _a12_ci
    rankings, a12_rows, per_champion_rows = [], [], []
    for task in tasks:
        budgets = {"task_budget": [r for r in reeval_rows if r["task"] == task],
                   "2x_budget": [r for r in robust_rows if r["task"] == task]}
        means = {name: _arm_means(rows) for name, rows in budgets.items()}
        for name, rows in budgets.items():
            for champion in per_champion(rows):
                per_champion_rows.append({"task": task, "budget_label": name, **{
                    k: champion[k] for k in ("arm", "seed", "provenance", "champion_eval_index", "n_trainings",
                                             "mean", "sd")}})
        arms = sorted({arm for (t, arm) in set(means["task_budget"]) | set(means["2x_budget"]) if t == task})
        ranks = {}
        for name in budgets:
            summary = {arm: float(np.mean(list(means[name][(task, arm)].values())))
                       for arm in arms if (task, arm) in means[name]}
            ordered = sorted(summary, key=lambda arm: -summary[arm])
            for position, arm in enumerate(ordered, 1):
                ranks[(name, arm)] = position
                rankings.append({"task": task, "budget_label": name, "arm": arm,
                                 "n_champions": len(means[name][(task, arm)]),
                                 "mean_of_champion_means": summary[arm], "rank": position})
        for index, arm_a in enumerate(arms):
            for arm_b in arms[index + 1:]:
                row = {"task": task, "arm_a": arm_a, "arm_b": arm_b}
                for name in budgets:
                    a = list(means[name].get((task, arm_a), {}).values())
                    b = list(means[name].get((task, arm_b), {}).values())
                    if a and b:
                        lo, hi = _a12_ci(a, b)
                        row.update({f"a12_{name}": _a12(a, b), f"a12_{name}_ci_low": lo, f"a12_{name}_ci_high": hi})
                a12_rows.append(row)
    return rankings, a12_rows, per_champion_rows


def _build_robustness(root: Path, key: str, plan: dict, manifest: dict, tasks, stages) -> dict:
    output = root / "milestones" / unit_keys.slug(key)
    output.mkdir(parents=True, exist_ok=True)
    all_stages = [s for s in plan["stages"] if s["kind"] == unit_keys.ROBUSTNESS]
    stages = stages or all_stages
    tasks = tasks or [s["task"] for s in all_stages]
    robust_rows = [r for s in stages for r in _stage_rows(root, "robustness", s["stage_id"])]
    reeval_rows = [r for task in tasks for r in _stage_rows(root, "reeval", None, task)]
    planned = sum(planned_trainings(s, plan) for s in stages)
    fraction = len(robust_rows) / planned if planned else None
    status = "PARTIAL" if fraction is None or fraction < 1 else "COMPLETE"
    rankings, a12_rows, champion_rows = ranking_tables(reeval_rows, robust_rows, tasks)
    _csv(output / "stats/robustness_trainings.csv", robust_rows)
    _csv(output / "stats/robustness_per_champion.csv", champion_rows)
    _csv(output / "stats/robustness_rankings.csv", rankings)
    _csv(output / "stats/robustness_a12.csv", a12_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for arm in sorted({r["arm"] for r in rankings if r["task"] == task}):
            points = [next((r["mean_of_champion_means"] for r in rankings if r["task"] == task and
                            r["arm"] == arm and r["budget_label"] == name), np.nan)
                      for name in ("task_budget", "2x_budget")]
            ax.plot(["task budget", "2x budget"], points, marker="o", label=arm)
        ax.set(title=task, ylabel="Mean of champion means")
        ax.legend()
    fig.suptitle("Arm means at the task budget versus 2x" + _mark(manifest))
    _savefig(fig, output, "robustness_by_arm")
    plt.close(fig)
    lines = []
    for task in tasks:
        for name in ("task_budget", "2x_budget"):
            order = [r for r in rankings if r["task"] == task and r["budget_label"] == name]
            order.sort(key=lambda r: r["rank"])
            lines.append(f"- {task}, {name}: " + " > ".join(f"{r['arm']} ({r['mean_of_champion_means']:.3g})"
                                                              for r in order))
    report = (f"# Robustness at 2x budget ({key}) milestone report\n\n"
              f"Status: {status}. Completion: {len(robust_rows)}/{planned} trainings"
              + (f" ({fraction:.1%})" if fraction is not None else "") + ".\n\n"
              "Amendment 2 (h): every champion re-trained at twice its task's budget; arm rankings and A12 at the "
              "task budget (the re-evaluation trainings) versus at 2x. Descriptive; no pooling across tasks.\n\n"
              + "\n".join(lines) + "\n\nTables: stats/robustness_rankings.csv, stats/robustness_a12.csv, "
              "stats/robustness_per_champion.csv.\n")
    return _finish(output, key, status, fraction, "", manifest, report)
