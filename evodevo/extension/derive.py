"""Apply the pre-registered extension rules and write the launch artifacts.

    python -m evodevo.extension derive --run-id current \\
        --reference-campaign results/campaign/<ORIGINAL_ID>
    python -m evodevo.extension check

`derive` is mechanical: D6-ext (budgets, from curves.jsonl), D16-ext (K,
from noise.jsonl), the cost projection calibrated on the original
campaign's realised phases (with a backtest of the original UpStepper-v0
phase), and GUARD-ext. It writes, and only writes:

  docs/campaign_prep/ext/EXT_DECISIONS.json   decision record (status FINAL)
  docs/campaign_prep/ext/EXT_DECISIONS.md     human-readable report
  docs/campaign_prep/ext/final_estimate_ext.json  v2 estimate (priors, calendar, disk)
  evodevo/experiments/configs/campaign_ext.yaml   generated campaign config

`check` re-derives everything from the same inputs and verifies that the
committed artifacts are byte-for-byte what the rules produce, that the YAML
builds a plan, and that the plan agrees with the decision record.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from . import spec

E_OVER_P = 2.3172


def _rows(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _p_equivalent(row: dict) -> float:
    """Wall time expressed on a P-core (E-core samples divided by the measured
    production-scale E/P ratio)."""
    return row["wall_clock_s"] / (E_OVER_P if row.get("core_kind") == "E" else 1.0)


# --- reference campaign -------------------------------------------------------

def reference_facts(campaign: Path) -> dict:
    """Realised busy time per (arm, task), phase makespans and utilisation."""
    manifest = json.loads((campaign / "manifest.json").read_text())
    metrics = json.loads((campaign / "phase_metrics.json").read_text())
    busy = defaultdict(float)
    count = defaultdict(int)
    for row in _rows(campaign / "journal.jsonl"):
        parts = row.get("run_key", "").split("/")
        if len(parts) < 2:
            continue
        seconds = row.get("result", {}).get("wall_clock_s") or 0.0
        busy[(parts[0], parts[1])] += float(seconds)
        count[(parts[0], parts[1])] += 1
    phases = {}
    for task in spec.REFERENCE_TASKS:
        if task not in metrics or metrics[task].get("status") != "complete":
            raise ValueError(f"{campaign}: phase {task} is not complete in phase_metrics.json")
        phases[task] = {key: metrics[task][key] for key in
                        ("makespan_s", "worker_count", "worker_utilization", "worker_busy_s")}
    status = manifest.get("status")
    if status != "complete":
        raise ValueError(f"{campaign}: campaign status is {status!r}, not 'complete'")
    run_bytes = 0
    for path in (campaign / "runs").rglob("*"):
        if path.is_file():
            run_bytes += path.stat().st_size
    total_trainings = sum(count.values())
    return {"campaign": str(campaign), "campaign_id": manifest.get("campaign_id"),
            "git_commit": manifest.get("git_commit"), "phases": phases,
            "busy_s": {f"{arm}|{task}": value for (arm, task), value in sorted(busy.items())},
            "trainings": {f"{arm}|{task}": value for (arm, task), value in sorted(count.items())},
            "run_bytes": run_bytes, "bytes_per_training": run_bytes / total_trainings if total_trainings else 0.0}


# --- measurements -------------------------------------------------------------

def cost_ratios(noise_rows: list, budgets: dict, reference_timesteps: dict | None = None) -> dict:
    """Per extension task: (sum over bodies of mean P-equivalent training time
    at the task's budget) / (same bodies' Walker-v0 1M reference), plus the
    UpStepper-v0 1.2M ratio used for the backtest."""
    by = defaultdict(list)
    for row in noise_rows:
        if row["completed"]:
            by[(row["stage"], row["task"], row["body"])].append(_p_equivalent(row))
    walker = {body: statistics.mean(v) for (stage, task, body), v in by.items()
              if stage == "reference" and task == "Walker-v0"}
    ratios = {}
    for task in list(spec.TASK_ORDER) + ["UpStepper-v0"]:
        stage = "reference" if task == "UpStepper-v0" else "noise"
        means = {body: statistics.mean(v) for (s, t, body), v in by.items() if s == stage and t == task}
        common = sorted(set(means) & set(walker))
        if len(common) < 3:
            raise ValueError(f"cost ratio for {task}: only {len(common)} bodies have both measurements")
        ratios[task] = {"ratio": sum(means[b] for b in common) / sum(walker[b] for b in common),
                        "bodies": common,
                        "measured_timesteps": ((reference_timesteps or spec.REFERENCE_TASKS)["UpStepper-v0"]
                                               if task == "UpStepper-v0" else budgets[task])}
    return ratios


def timing_priors(noise_rows: list, task: str, timesteps: int, measured_timesteps: int) -> tuple:
    """(mean, sd) P-equivalent seconds at ``timesteps`` (linearly rescaled
    when the guard lowered the budget below the measured one)."""
    values = [_p_equivalent(r) for r in noise_rows
              if r["stage"] == "noise" and r["task"] == task and r["completed"]]
    if len(values) < 2:
        raise ValueError(f"fewer than 2 completed timing samples for {task}")
    scale = timesteps / measured_timesteps
    return statistics.mean(values) * scale, statistics.stdev(values) * scale


def observed_e_over_p(noise_rows: list) -> float | None:
    """Batch-internal E/P check: same (stage, task, body, timesteps) groups
    that ran on both core classes."""
    groups = defaultdict(lambda: {"P": [], "E": []})
    for row in noise_rows:
        if row["completed"] and row.get("core_kind") in ("P", "E"):
            groups[(row["stage"], row["task"], row["body"], row["timesteps"])][row["core_kind"]].append(
                row["wall_clock_s"])
    ratios = [statistics.mean(g["E"]) / statistics.mean(g["P"]) for g in groups.values() if g["E"] and g["P"]]
    return statistics.median(ratios) if ratios else None


# --- derivation ---------------------------------------------------------------

def derive(run_dir: Path, reference: Path, allow_test: bool = False) -> dict:
    from .rules import (BUDGET_CAP, GUARD_THRESHOLD_H, apply_guard, decide_seeds,
                        project_phase_band)
    manifest = json.loads((run_dir / "preflight_manifest.json").read_text())
    if manifest.get("label") != "PRODUCTION" and not allow_test:
        raise ValueError(f"{run_dir} is a {manifest.get('label')} preflight run; refusing to derive "
                         "launch decisions from it (pass --allow-test for a rehearsal)")
    budgets_doc = json.loads((run_dir / "budgets.json").read_text())
    budget_decisions = budgets_doc["decisions"]
    budgets = {task: budget_decisions[task]["ppo_timesteps"] for task in spec.TASK_ORDER}
    noise_rows = _rows(run_dir / "noise.jsonl")
    bodies = json.loads((run_dir / "bodies.json").read_text())

    seeds_decisions = {}
    for task in spec.TASK_ORDER:
        repeats = defaultdict(list)
        for row in noise_rows:
            if row["stage"] == "noise" and row["task"] == task and row["completed"] and row["fitness"] is not None:
                if row["timesteps"] != budgets[task]:
                    raise ValueError(f"{task}: noise training at {row['timesteps']} != budget {budgets[task]}")
                repeats[row["body"]].append(float(row["fitness"]))
        seeds_decisions[task] = dataclasses.asdict(decide_seeds(task, dict(repeats)))

    facts = reference_facts(reference)
    ratios = cost_ratios(noise_rows, budgets, manifest.get("params", {}).get("reference_timesteps"))
    walker = facts["phases"]["Walker-v0"]
    upstepper = facts["phases"]["UpStepper-v0"]
    walker_h = walker["makespan_s"] / 3600.0
    busy_all = sum(v for k, v in facts["busy_s"].items() if k.endswith("|Walker-v0"))
    busy_core = sum(facts["busy_s"].get(f"{arm}|Walker-v0", 0.0) for arm in spec.ARMS)
    share = busy_core / busy_all
    # Backtest: predict the original UpStepper-v0 phase (all its arms, K=2,
    # 1.2M) from its Walker-v0 phase with the same method.
    predicted_up = project_phase_band(walker_h, 1.0, ratios["UpStepper-v0"]["ratio"],
                                      spec.REFERENCE_K["UpStepper-v0"], 0.0)["mean"]
    actual_up = upstepper["makespan_s"] / 3600.0
    backtest_error = (predicted_up - actual_up) / actual_up
    workers = int(walker["worker_count"])
    utilization = float(walker["worker_utilization"])

    def project(task, timesteps, k):
        measured = ratios[task]["measured_timesteps"]
        return project_phase_band(walker_h, share, ratios[task]["ratio"] * timesteps / measured, k,
                                  backtest_error)

    decided = {task: {"ppo_timesteps": budgets[task], "ppo_seeds": seeds_decisions[task]["ppo_seeds"]}
               for task in spec.TASK_ORDER}
    final, steps, total, outcome = apply_guard(list(spec.TASK_ORDER), decided, project)
    phase_h = {task: project(task, final[task]["ppo_timesteps"], final[task]["ppo_seeds"])
               for task in spec.TASK_ORDER}

    t_ppo = {}
    for task in spec.TASK_ORDER:
        prior = timing_priors(noise_rows, task, final[task]["ppo_timesteps"], ratios[task]["measured_timesteps"])
        for arm in spec.ARMS:
            t_ppo[f"{arm}|{task}"] = [prior[0], prior[1]]

    trainings = sum(facts["trainings"].get(f"{arm}|Walker-v0", 0) for arm in spec.ARMS)
    disk_total = sum(trainings * final[task]["ppo_seeds"] * facts["bytes_per_training"] for task in spec.TASK_ORDER)
    preflight_commit = manifest["git_commits"][0]["git_commit"] if manifest.get("git_commits") else None

    record = {
        "status": "FINAL",
        "label": manifest.get("label"),
        "derived_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "docs/campaign_prep/ext/PREREGISTRATION_EXT.md",
        "preflight_run": str(run_dir.relative_to(spec.REPO_ROOT) if run_dir.is_relative_to(spec.REPO_ROOT)
                             else run_dir),
        "preflight_git_commit": preflight_commit,
        "reference_campaign": {k: facts[k] for k in ("campaign", "campaign_id", "git_commit")},
        "task_order": list(spec.TASK_ORDER),
        "tiers": dict(spec.TIER),
        "arms": list(spec.ARMS),
        "seeds": list(spec.SEEDS),
        "shared": dict(spec.SHARED),
        "tasks": {task: {"ppo_timesteps": final[task]["ppo_timesteps"], "ppo_seeds": final[task]["ppo_seeds"],
                         "d6_ext": budget_decisions[task], "d16_ext": seeds_decisions[task],
                         "guard_changed": final[task] != decided[task]}
                  for task in spec.TASK_ORDER},
        "guard": {"threshold_h": GUARD_THRESHOLD_H, "outcome": outcome, "steps": steps,
                  "budget_cap": BUDGET_CAP},
        "bodies": {label: {k: v for k, v in b.items() if k != "genome"} for label, b in bodies["bodies"].items()},
    }
    estimate = {
        "format": "evodevo-estimate-v2",
        "method": "realised-phase scaling (original campaign Walker-v0, same 3 arms) x measured cost ratio x K",
        "workers": workers,
        "design": {"tasks": {task: {"arms": sorted(spec.ARMS), "seeds": list(spec.SEEDS),
                                    "ppo_timesteps": final[task]["ppo_timesteps"],
                                    "ppo_seeds": final[task]["ppo_seeds"],
                                    "population_size": spec.SHARED["population_size"],
                                    "eval_budget": spec.SHARED["eval_budget"]} for task in spec.TASK_ORDER}},
        "inputs": {"t_ppo_mean_sd_at_b3_footing": t_ppo,
                   "t_ppo_footing": "P-core-equivalent seconds from B5x (E-core samples / 2.3172)",
                   "reference": facts, "cost_ratios": ratios,
                   "walker_makespan_h": walker_h, "walker_utilization": utilization,
                   "walker_core_arm_busy_s": busy_core, "walker_all_arm_busy_s": busy_all,
                   "core_arm_busy_share": share, "observed_e_over_p": observed_e_over_p(noise_rows),
                   "assumed_e_over_p": E_OVER_P},
        "backtest": {"predicted_upstepper_h": predicted_up, "actual_upstepper_h": actual_up,
                     "relative_error": backtest_error, "actual_walker_h": walker_h},
        "result": {"task_order": list(spec.TASK_ORDER), "phase_h": phase_h, "total_h": total},
        "guard": {"threshold_h": GUARD_THRESHOLD_H, "outcome": outcome},
        "disk": {"total_bytes": disk_total, "three_x_total_bytes": 3 * disk_total,
                 "bytes_per_training": facts["bytes_per_training"]},
    }
    return {"record": record, "estimate": estimate}


YAML_TEMPLATE = """# GENERATED by `python -m evodevo.extension derive` -- do not edit by hand.
# Campaign extension (docs/campaign_prep/ext/PREREGISTRATION_EXT.md): the three
# core arms on {tiers}, D7 order as listed.
# Values below are the mechanical output of D6-ext, D16-ext and GUARD-ext
# recorded in docs/campaign_prep/ext/EXT_DECISIONS.json; the plan refuses to
# build if they ever disagree with that record.
profile: campaign
encoders: [direct, nca, rule_based]
tasks: [{tasks}]
seeds: [0, 1, 2, 3, 4]
configs_root: evodevo/experiments/configs
output_dir: results/runs
surrogate: false
factors:
  dev_steps: 100
  init_condition: center_seed
estimate_path: docs/campaign_prep/ext/final_estimate_ext.json
decisions_path: docs/campaign_prep/ext/EXT_DECISIONS.json
guard_outcome: "extension: 3 core arms; GUARD-ext {outcome}"
profile_overrides:
  population_size: 15
  offspring_size: 15
  eval_budget: 300
  ppo_seed_aggregation: max
  ppo_timesteps:
{timesteps}
  ppo_seeds:
{seeds}
"""


def render_yaml(record: dict) -> str:
    tasks = record["task_order"]
    return YAML_TEMPLATE.format(
        tiers=", ".join(f"{t} ({record['tiers'][t]})" for t in tasks),
        tasks=", ".join(tasks), outcome=record["guard"]["outcome"],
        timesteps="\n".join(f"    {t}: {record['tasks'][t]['ppo_timesteps']}" for t in tasks),
        seeds="\n".join(f"    {t}: {record['tasks'][t]['ppo_seeds']}" for t in tasks))


def render_report(record: dict, estimate: dict) -> str:
    lines = [f"# Extension decisions ({record['label']})", "",
             f"Derived {record['derived_at']} from preflight run `{record['preflight_run']}` "
             f"(commit `{record['preflight_git_commit']}`), calibrated on campaign "
             f"`{record['reference_campaign']['campaign_id']}`. Rules: "
             f"[{record['preregistration']}]({Path(record['preregistration']).name}).", "",
             "| Task | Tier | ppo_timesteps | D6-ext branch | K | D16-ext branch | ratio single / max-of-2 | phase h (p5 / mean / p95) |",
             "|---|---|---:|---|---:|---|---|---|"]
    for task in record["task_order"]:
        t = record["tasks"][task]
        d16 = t["d16_ext"]["details"]
        ratio = (f"{d16['ratio_single']:.3f} / {d16['ratio_max_of_2']:.3f}"
                 if d16.get("ratio_single") is not None and d16.get("ratio_max_of_2") is not None else "n/a")
        band = estimate["result"]["phase_h"][task]
        lines.append(f"| {task} | {record['tiers'][task]} | {t['ppo_timesteps']:,} | {t['d6_ext']['branch']}"
                     f"{' (disclosed floor)' if t['d6_ext'].get('disclosed_floor') else ''} | {t['ppo_seeds']} | "
                     f"{t['d16_ext']['branch']} | {ratio} | {band['p5']:.1f} / {band['mean']:.1f} / {band['p95']:.1f} |")
    total = estimate["result"]["total_h"]
    bt = estimate["backtest"]
    lines += ["", f"**Projected campaign total:** {total['mean']:.1f} h mean "
              f"({total['p5']:.1f}–{total['p95']:.1f} h, p5–p95) = {total['mean'] / 24:.1f} days. "
              f"GUARD-ext threshold {record['guard']['threshold_h']:.0f} h: **{record['guard']['outcome']}**.",
              "", f"**Backtest:** the same method predicts the original UpStepper-v0 phase at "
              f"{bt['predicted_upstepper_h']:.1f} h against {bt['actual_upstepper_h']:.1f} h realised "
              f"({100 * bt['relative_error']:+.1f}%); the band half-width is max(10%, |backtest error|).",
              "", "## Guard steps", ""]
    for step in record["guard"]["steps"]:
        lines.append(f"- `{step['step']}`: p95 total {step['total_h']['p95']:.1f} h; design {json.dumps(step['design'])}")
    lines += ["", "## Cost ratios (P-core-equivalent, same bodies, vs Walker-v0 at 1M)", ""]
    for task, r in estimate["inputs"]["cost_ratios"].items():
        lines.append(f"- {task} at {r['measured_timesteps']:,}: {r['ratio']:.3f} (bodies: {', '.join(r['bodies'])})")
    observed = estimate["inputs"]["observed_e_over_p"]
    lines += ["", f"Observed E/P training-time ratio in the preflight batch: "
              f"{observed:.3f}" if observed is not None else "", "",
              "## Representative bodies", ""]
    for label, body in record["bodies"].items():
        lines.append(f"- `{label}` hash `{body['body_hash'][:12]}` ({body['kind']})")
    lines += ["", "## Raw rule outputs", "", "```json",
              json.dumps({t: {"d6_ext": record['tasks'][t]['d6_ext'], "d16_ext": record['tasks'][t]['d16_ext']}
                          for t in record["task_order"]}, indent=1, default=str), "```", ""]
    return "\n".join(lines)


def _dump(payload) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n"


def write(result: dict) -> list:
    spec.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {spec.DECISIONS_JSON: _dump(result["record"]),
               spec.ESTIMATE_JSON: _dump(result["estimate"]),
               spec.DECISIONS_MD: render_report(result["record"], result["estimate"]),
               spec.CAMPAIGN_YAML: render_yaml(result["record"])}
    for path, text in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    return [str(p.relative_to(spec.REPO_ROOT)) for p in outputs]


def check(run_dir: Path | None, reference: Path | None, allow_test: bool = False) -> list:
    """Problems found (empty = consistent). Without inputs, checks only the
    committed artifacts' internal consistency and that a plan builds."""
    problems = []
    from evodevo.campaign.plan import CampaignPlan
    from evodevo.experiments.benchmark import BenchmarkConfig
    for path in (spec.DECISIONS_JSON, spec.ESTIMATE_JSON, spec.DECISIONS_MD, spec.CAMPAIGN_YAML):
        if not path.is_file():
            problems.append(f"missing {path.relative_to(spec.REPO_ROOT)}")
    if problems:
        return problems
    record = json.loads(spec.DECISIONS_JSON.read_text())
    if spec.CAMPAIGN_YAML.read_text() != render_yaml(record):
        problems.append("campaign_ext.yaml is not the rendering of EXT_DECISIONS.json")
    try:
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(spec.CAMPAIGN_YAML)))
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        problems.append(f"plan does not build: {type(exc).__name__}: {exc}")
        return problems
    if [p.task for p in plan.phases] != record["task_order"]:
        problems.append("plan phase order != decision record")
    for phase in plan.phases:
        decided = record["tasks"][phase.task]
        if (phase.ppo_timesteps, phase.ppo_seeds) != (decided["ppo_timesteps"], decided["ppo_seeds"]):
            problems.append(f"{phase.task}: plan ({phase.ppo_timesteps}, K={phase.ppo_seeds}) != record")
        if sorted({r.arm for r in phase.runs}) != sorted(spec.ARMS):
            problems.append(f"{phase.task}: arms {sorted({r.arm for r in phase.runs})}")
        if sorted({r.seed for r in phase.runs}) != list(spec.SEEDS):
            problems.append(f"{phase.task}: seeds")
        if not phase.timeout_s or phase.timeout_s <= 0:
            problems.append(f"{phase.task}: no PPO-task timeout")
    if run_dir is not None and reference is not None:
        fresh = derive(run_dir, reference, allow_test=allow_test)
        for key, path in (("record", spec.DECISIONS_JSON), ("estimate", spec.ESTIMATE_JSON)):
            committed = json.loads(path.read_text())
            regenerated = json.loads(_dump(fresh[key]))
            committed.pop("derived_at", None)
            regenerated.pop("derived_at", None)
            if committed != regenerated:
                problems.append(f"{path.name} differs from a fresh derivation of the same inputs")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evodevo.extension", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("derive", "check"))
    parser.add_argument("--run-id", default="current")
    parser.add_argument("--reference-campaign")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    run_dir = spec.PREFLIGHT_ROOT / args.run_id
    reference = Path(args.reference_campaign) if args.reference_campaign else None
    if args.command == "derive":
        if reference is None:
            parser.error("--reference-campaign is required")
        written = write(derive(run_dir, reference, allow_test=args.allow_test))
        print("wrote:\n  " + "\n  ".join(written))
        print(spec.DECISIONS_MD.read_text())
        return 0
    problems = check(run_dir if reference else None, reference, allow_test=args.allow_test)
    if problems:
        print("EXTENSION CHECK FAILED:\n  - " + "\n  - ".join(problems))
        return 1
    print("extension check passed")
    return 0
