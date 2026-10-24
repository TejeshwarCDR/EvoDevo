"""Amendment 2: derive and check the production decision record.

    python -m evodevo.extension amend2 derive --t-obstacle 1500000
    python -m evodevo.extension amend2 check

Amendment 2 (docs/campaign_prep/ext/PREREGISTRATION_EXT.md) replaces the
3-task extension with one campaign: ObstacleTraverser-v1 seeds 0-9, plus
Walker-v0 and UpStepper-v0 seeds 5-9 (pooled with the original seeds 0-4),
each followed by champion re-training stages.

`derive` is deterministic from: the extension preflight run (its measured
cost ratios and timings), T_obstacle (Gate T), the original campaign
(manifest: per-run configs; journal and phase metrics: realised times), and
the sha256 of A2_GATES_REPORT.md and of the preflight record. It writes, and
only writes:

  docs/campaign_prep/ext/A2_DECISIONS.json     decision record (FINAL, PRODUCTION)
  docs/campaign_prep/ext/A2_DECISIONS.md       human-readable report
  docs/campaign_prep/ext/final_estimate_a2.json  v2 estimate (per-unit bands, timeouts)
  evodevo/experiments/configs/campaign_a2.yaml   generated campaign config

`check` verifies, without inputs, that every file is the canonical rendering
of the record (a hand edit is a failure), that the hashes of the gates report
and the preflight record match the files, and that the YAML builds a plan
that agrees with the record (60 evolutionary runs, 15 per phase, 150 + 150 +
300 + 300 re-evaluation and 270 robustness trainings). With --run-id and
--reference-campaign it also compares against a fresh derivation.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import derive as base_derive
from . import spec
from evodevo.campaign import amend2 as runtime

DOCS = spec.DOCS_DIR
DECISIONS_JSON = DOCS / "A2_DECISIONS.json"
DECISIONS_MD = DOCS / "A2_DECISIONS.md"
ESTIMATE_JSON = DOCS / "final_estimate_a2.json"
CAMPAIGN_YAML = spec.REPO_ROOT / "evodevo/experiments/configs/campaign_a2.yaml"
GATES_REPORT = DOCS / "A2_GATES_REPORT.md"
PREFLIGHT_RECORD = spec.DECISIONS_JSON
SUPERSEDED = DOCS / "SUPERSEDED_BY_AMENDMENT_2.json"
DEFAULT_RUN_ID = "ext-20261023"
DEFAULT_REFERENCE = runtime.ORIGINAL_CAMPAIGN_PATH
PLANNED_LAUNCH = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))

TASK_ORDER = ("ObstacleTraverser-v1", "UpStepper-v0", "Walker-v0")
NEW_SEEDS = {"ObstacleTraverser-v1": ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), "UpStepper-v0": ((5, 6, 7, 8, 9),),
             "Walker-v0": ((5, 6, 7, 8, 9),)}
ORIGINAL_SEEDS = (0, 1, 2, 3, 4)
ORIGINAL_BUDGETS = {"UpStepper-v0": (1_200_000, 2), "Walker-v0": (1_000_000, 1)}
BAND_MIN_HALF = 0.10

# Gate evidence transcribed from A2_GATES_REPORT.md (its sha256 is recorded).
GATE_T_FACTS = {
    "rule": "registered D6-ext fallback test: every qualifying body >= 85% of its 2M fitness and rank order kept",
    "1000000": {"fraction_ok": False, "order_ok": False}, "1200000": {"fraction_ok": False, "order_ok": False},
    "1500000": {"fraction_ok": False, "order_ok": False, "note": "registered cap; fails the test as well"},
    "2000000": {"fraction_ok": True, "order_ok": True, "note": "passes trivially (compared with itself)"},
}
GATE_R_FACTS = {
    "outcome": "PASS", "config_diff": "only the seed", "library_versions": "identical, evogym f63c649c",
    "behavioural_code_hunks": 0, "replayed_original_evaluations_bit_identical": "12/12 (including both K=2 trainings)",
}


def _rel(path) -> str:
    path = Path(path)
    return str(path.relative_to(spec.REPO_ROOT)) if path.is_relative_to(spec.REPO_ROOT) else str(path)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(payload) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n"


def gates_first_line(path=GATES_REPORT) -> str:
    return Path(path).read_text(encoding="utf-8").splitlines()[0]


def parse_gates_first_line(line: str) -> dict:
    """'GATE T: 1500000 | GATE R: PASS | GATE S: saving possible yes'"""
    parts = [part.strip() for part in line.split("|")]
    if (len(parts) != 3 or not parts[0].startswith("GATE T: ") or parts[1] != "GATE R: PASS"
            or parts[2] != "GATE S: saving possible yes"):
        raise ValueError(f"the gates report's first line is not the expected pass line: {line!r}")
    return {"t_obstacle": int(parts[0].removeprefix("GATE T: "))}


# --- the campaign specification (the YAML's amendment2 block) -------------------

def seed_label(seeds) -> str:
    return f"s{seeds[0]}" if len(seeds) == 1 else f"s{seeds[0]}-{seeds[-1]}"


def phase_id(task: str, seeds) -> str:
    return f"{task}:{seed_label(seeds)}"


def campaign_spec(t_obstacle: int, label: str = runtime.PRODUCTION, reference: str = DEFAULT_REFERENCE,
                  budgets: dict | None = None, reeval_n: int = runtime.REEVAL_PER_CHAMPION,
                  robust_n: int = runtime.ROBUSTNESS_PER_CHAMPION, robust_factor: int = runtime.ROBUSTNESS_FACTOR,
                  versions: dict | None = None, new_seeds: dict | None = None,
                  original_seeds=ORIGINAL_SEEDS, config_source: dict | None = None,
                  pooled_with_original=tuple(ORIGINAL_BUDGETS)) -> dict:
    """Phases, stages and execution order of Amendment 2, from budgets alone.
    ``pooled_with_original`` names the tasks whose champions (and package) also
    include the original campaign's ``original_seeds``, read-only."""
    new_seeds = new_seeds or NEW_SEEDS
    config_source = config_source or {t: "original" if t in ORIGINAL_BUDGETS else "benchmark" for t in TASK_ORDER}
    budgets = budgets or {"ObstacleTraverser-v1": t_obstacle, "UpStepper-v0": ORIGINAL_BUDGETS["UpStepper-v0"][0],
                          "Walker-v0": ORIGINAL_BUDGETS["Walker-v0"][0]}
    arms = list(runtime.CORE_ARMS)
    phases = []
    first_obstacle = None
    for task in TASK_ORDER:
        for seeds in new_seeds[task]:
            item = {"id": phase_id(task, seeds), "task": task, "seeds": list(seeds), "arms": arms,
                    "config_source": config_source[task]}
            if task == "ObstacleTraverser-v1":
                interim = first_obstacle is None
                merge = [{"phase": item["id"]}] if interim else [{"phase": first_obstacle}, {"phase": item["id"]}]
                first_obstacle = first_obstacle or item["id"]
                item["package"] = {"label": "INTERIM" if interim else "FINAL", "merge": merge,
                                   "monotone_contrast": False}
            else:
                pooled = task in pooled_with_original
                item["package"] = {"label": "FINAL", "monotone_contrast": pooled, "merge": (
                    [{"original": True, "arms": arms, "seeds": list(original_seeds)}] if pooled else []) +
                    [{"phase": item["id"]}]}
            phases.append(item)
    by_task = {}
    for item in phases:
        by_task.setdefault(item["task"], []).append(item)

    def sources(task):
        return [s for item in by_task[task] for s in (
            [{"original": True, "arms": arms, "seeds": list(original_seeds)}] if task in pooled_with_original
            else []) + [{"phase": item["id"]}]]

    def reeval(task, suffix, source_list):
        return {"id": f"reeval:{task}:{suffix}", "kind": "reeval", "task": task, "n": reeval_n,
                "seed_base": runtime.SEED_BASE_REEVAL, "budget": budgets[task], "sources": source_list}

    stages, sequence = [], []
    for task in TASK_ORDER:
        items = by_task[task]
        if task == "ObstacleTraverser-v1":
            for item in items:
                sequence.append(item["id"])
                stage = reeval(task, seed_label(item["seeds"]), [{"phase": item["id"]}])
                stages.append(stage)
                sequence.append(stage["id"])
        else:
            sequence.append(items[0]["id"])
            span = sorted((set(original_seeds) if task in pooled_with_original else set())
                          | {seed for item in items for seed in item["seeds"]})
            stage = reeval(task, seed_label(span), sources(task))
            stages.append(stage)
            sequence.append(stage["id"])
    for task in TASK_ORDER:
        stage = {"id": f"robustness:{task}", "kind": "robustness", "task": task, "n": robust_n,
                 "seed_base": runtime.SEED_BASE_ROBUSTNESS, "budget": robust_factor * budgets[task],
                 "sources": sources(task)}
        stages.append(stage)
        sequence.append(stage["id"])
    return {"label": label, "original_campaign": reference, "phases": phases, "stages": stages,
            "sequence": sequence, "recorded_versions": dict(versions or runtime.RECORDED_VERSIONS)}


# --- derivation ----------------------------------------------------------------------

def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=spec.REPO_ROOT, text=True).strip()


def _dirty() -> list:
    from evodevo.preflight.dirty_check import dirty_paths_outside_results
    return dirty_paths_outside_results(str(spec.REPO_ROOT))


def derive(run_dir: Path, reference: Path, t_obstacle: int, *, allow_test: bool = False,
           gates_report=GATES_REPORT, preflight_record=PREFLIGHT_RECORD) -> dict:
    from evodevo.extension.rules import project_phase_band
    gates = parse_gates_first_line(gates_first_line(gates_report))
    if gates["t_obstacle"] != t_obstacle:
        raise ValueError(f"--t-obstacle {t_obstacle} != Gate T {gates['t_obstacle']} in {gates_report}")
    committed = json.loads(Path(preflight_record).read_text())
    superseded = json.loads(SUPERSEDED.read_text())
    if _sha256(preflight_record) != superseded["sha256"]:
        raise ValueError(f"{preflight_record} no longer matches the sha256 in {SUPERSEDED.name}")
    # The preflight's own derivation, recomputed from the run directory; it
    # must reproduce the committed preflight estimate exactly.
    preflight = base_derive.derive(run_dir, reference, allow_test=allow_test)
    committed_estimate = json.loads(spec.ESTIMATE_JSON.read_text())
    fresh_inputs = json.loads(json.dumps(preflight["estimate"]["inputs"]))
    # The same campaign, however its path was spelt on the command line.
    fresh_inputs["reference"]["campaign"] = committed_estimate["inputs"]["reference"]["campaign"]
    if fresh_inputs != committed_estimate["inputs"]:
        raise ValueError("a fresh derivation from the preflight run differs from final_estimate_ext.json "
                         "(inputs); the preflight record is not reproducible from these inputs")
    if preflight["record"]["label"] != "PRODUCTION" and not allow_test:
        raise ValueError("the preflight run is not PRODUCTION-labelled")
    inputs = preflight["estimate"]["inputs"]
    walker_h = inputs["walker_makespan_h"]
    share = inputs["core_arm_busy_share"]
    ratios = inputs["cost_ratios"]
    backtest = preflight["estimate"]["backtest"]["relative_error"]
    facts = inputs["reference"]
    original_manifest = json.loads((reference / "manifest.json").read_text())
    walker_phase = facts["phases"]["Walker-v0"]
    workers = int(walker_phase["worker_count"])
    utilization = float(walker_phase["worker_utilization"])
    trainings_core = sum(facts["trainings"][f"{arm}|Walker-v0"] for arm in spec.ARMS)
    busy_per_training = inputs["walker_core_arm_busy_s"] / trainings_core

    budgets = {"ObstacleTraverser-v1": t_obstacle, "UpStepper-v0": ORIGINAL_BUDGETS["UpStepper-v0"][0],
               "Walker-v0": ORIGINAL_BUDGETS["Walker-v0"][0]}
    ks = {"ObstacleTraverser-v1": preflight["record"]["tasks"]["ObstacleTraverser-v1"]["ppo_seeds"],
          "UpStepper-v0": ORIGINAL_BUDGETS["UpStepper-v0"][1], "Walker-v0": ORIGINAL_BUDGETS["Walker-v0"][1]}
    camp = campaign_spec(t_obstacle)
    measured = {"ObstacleTraverser-v1": ratios["ObstacleTraverser-v1"]["measured_timesteps"],
                "UpStepper-v0": ratios["UpStepper-v0"]["measured_timesteps"], "Walker-v0": 1_000_000}
    cost_ratio = {"ObstacleTraverser-v1": ratios["ObstacleTraverser-v1"]["ratio"],
                  "UpStepper-v0": ratios["UpStepper-v0"]["ratio"], "Walker-v0": 1.0}
    half = max(BAND_MIN_HALF, abs(backtest))

    def phase_band(task):
        return project_phase_band(walker_h, share, cost_ratio[task] * budgets[task] / measured[task], ks[task],
                                  backtest)

    def stage_band(task, n_trainings, budget):
        tau = busy_per_training * cost_ratio[task] * budget / measured[task]
        mean = (n_trainings * tau / (workers * utilization) + 0.5 * tau) / 3600.0
        return {"p5": mean * (1 - half), "mean": mean, "p95": mean * (1 + half)}

    # Timeouts and p95 E-core durations (seconds): Obstacle from the preflight
    # B5x timings scaled to the budget; Walker/UpStepper from the original
    # campaign's own phase timeouts, scaled linearly in the budget.
    noise_rows = base_derive._rows(run_dir / "noise.jsonl")
    prior_mean, prior_sd = base_derive.timing_priors(noise_rows, "ObstacleTraverser-v1", budgets["ObstacleTraverser-v1"],
                                                     measured["ObstacleTraverser-v1"])
    obstacle_p95_ecore = (prior_mean + 1.645 * prior_sd) * base_derive.E_OVER_P
    original_timeout = {phase["task"]: phase["timeout_s"] for phase in original_manifest["plan"]["phases"]}
    p95_ecore, timeouts = {}, {}
    for task in TASK_ORDER:
        base_p95 = obstacle_p95_ecore if task == "ObstacleTraverser-v1" else original_timeout[task] / 4
        for item in camp["phases"] + camp["stages"]:
            if item["task"] != task:
                continue
            scale = (item["budget"] / budgets[task]) if "budget" in item else 1.0
            p95_ecore[item["id"]] = base_p95 * scale
            timeouts[item["id"]] = (original_timeout[task] if task != "ObstacleTraverser-v1" and scale == 1.0
                                    else 4 * base_p95 * scale)

    champions = 15
    bands, design_units = {}, {}
    for item in camp["phases"]:
        bands[item["id"]] = phase_band(item["task"])
        design_units[item["id"]] = {"arms": sorted(item["arms"]), "seeds": item["seeds"],
                                    "ppo_timesteps": budgets[item["task"]], "ppo_seeds": ks[item["task"]],
                                    "population_size": spec.SHARED["population_size"],
                                    "eval_budget": spec.SHARED["eval_budget"]}
    stage_counts = {}
    for item in camp["stages"]:
        count = item["n"] * sum(champions if "phase" in s else len(s["arms"]) * len(s["seeds"])
                                for s in item["sources"])
        stage_counts[item["id"]] = count
        bands[item["id"]] = stage_band(item["task"], count, item["budget"])
        design_units[item["id"]] = {"n": item["n"], "budget": item["budget"], "trainings": count}
    total = {q: sum(bands[key][q] for key in camp["sequence"]) for q in ("p5", "mean", "p95")}

    run_bytes = facts["bytes_per_training"]
    training_units = trainings_core * (sum(ks[item["task"]] for item in camp["phases"])) + sum(stage_counts.values())
    disk_total = training_units * run_bytes
    reference_manifest = {"path": str(reference.relative_to(spec.REPO_ROOT) if reference.is_absolute() and
                                      reference.is_relative_to(spec.REPO_ROOT) else reference),
                          "id": original_manifest["campaign_id"], "git_commit": original_manifest["git_commit"],
                          "config_hash": original_manifest["config_hash"]}
    t_ppo = {f"{arm}|ObstacleTraverser-v1": [prior_mean, prior_sd] for arm in spec.ARMS}
    estimate = {
        "format": "evodevo-estimate-v2",
        "method": ("Amendment 2: phases scaled from the original campaign's realised Walker-v0 phase (same 3 core "
                   "arms) x measured cost ratio x K; re-training stages from pool throughput (mean pool-busy "
                   "seconds per Walker-v0 1M training x cost ratio x budget / (workers x utilisation) + half a "
                   "training for the tail); band +/- max(10%, |backtest error|)"),
        "workers": workers,
        "design": {"units": design_units},
        "inputs": {"t_ppo_mean_sd_at_b3_footing": t_ppo, "walker_makespan_h": walker_h, "core_arm_busy_share": share,
                   "cost_ratios": {task: ratios[task] for task in ("ObstacleTraverser-v1", "UpStepper-v0")},
                   "busy_s_per_walker_training": busy_per_training, "walker_utilization": utilization,
                   "backtest_relative_error": backtest, "preflight_estimate_inputs_sha256": hashlib.sha256(
                       _dump(committed_estimate["inputs"]).encode()).hexdigest()},
        "a2": {"timeout_s": timeouts, "p95_ecore_s": p95_ecore, "band_half_width": half,
               "timeout_rule": "ObstacleTraverser-v1: 4 x p95 E-core duration from the preflight B5x timings scaled "
                               "linearly to the budget; Walker-v0/UpStepper-v0: the original campaign's phase "
                               "timeouts, scaled linearly to the budget"},
        "result": {"task_order": list(camp["sequence"]), "phase_h": bands, "total_h": total},
        "guard": {"applies": False, "outcome": "no guard (Amendment 2 (k)): the extension's 360 h cost guard does "
                                                "not apply; this projection is informational"},
        "disk": {"total_bytes": disk_total, "three_x_total_bytes": 3 * disk_total, "bytes_per_training": run_bytes},
    }
    record = {
        "format": runtime.RECORD_FORMAT, "status": "FINAL", "label": runtime.PRODUCTION if not allow_test
        else preflight["record"]["label"],
        "amendment": "Amendment 2 (docs/campaign_prep/ext/PREREGISTRATION_EXT.md)",
        "derived_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_head(),
        "inputs": {"preflight_run": str(run_dir.relative_to(spec.REPO_ROOT) if run_dir.is_relative_to(spec.REPO_ROOT)
                                        else run_dir),
                   "preflight_record": "docs/campaign_prep/ext/EXT_DECISIONS.json",
                   "preflight_record_sha256": _sha256(preflight_record),
                   "gates_report": "docs/campaign_prep/ext/A2_GATES_REPORT.md",
                   "gates_report_sha256": _sha256(gates_report), "gates_report_first_line": gates_first_line(gates_report),
                   "reference_campaign": reference_manifest, "t_obstacle": t_obstacle},
        "task_order": list(TASK_ORDER), "arms": list(spec.ARMS), "seeds": list(range(10)),
        "shared": dict(spec.SHARED),
        "tasks": {
            "ObstacleTraverser-v1": {"ppo_timesteps": t_obstacle, "ppo_seeds": ks["ObstacleTraverser-v1"],
                                     "ppo_timesteps_source": "Gate T (A2_GATES_REPORT.md): the registered D6-ext "
                                     "fallback test failed at 1.0M and 1.2M, so the registered cap 1.5M applies",
                                     "ppo_seeds_source": "D16-ext (preflight record): max-of-2 reduced the noise "
                                     "ratio from 3.0 to 1.87, i.e. 38% < 50%, so K = 1",
                                     "d6_ext_knee_timesteps": 600000},
            "UpStepper-v0": {"ppo_timesteps": 1_200_000, "ppo_seeds": 2,
                             "ppo_timesteps_source": "original campaign per-run configs, unchanged (seeds 5-9 "
                                                     "pool with seeds 0-4)",
                             "ppo_seeds_source": "original campaign per-run configs, unchanged"},
            "Walker-v0": {"ppo_timesteps": 1_000_000, "ppo_seeds": 1,
                          "ppo_timesteps_source": "original campaign per-run configs, unchanged (seeds 5-9 pool "
                                                  "with seeds 0-4)",
                          "ppo_seeds_source": "original campaign per-run configs, unchanged"}},
        "reevaluation": {"n_per_champion": runtime.REEVAL_PER_CHAMPION, "seed_base": runtime.SEED_BASE_REEVAL,
                         "budget": "the task's own PPO budget", "trainings": {
                             item["id"]: stage_counts[item["id"]] for item in camp["stages"] if item["kind"] == "reeval"}},
        "robustness": {"n_per_champion": runtime.ROBUSTNESS_PER_CHAMPION, "seed_base": runtime.SEED_BASE_ROBUSTNESS,
                       "factor": runtime.ROBUSTNESS_FACTOR,
                       "budgets": {item["task"]: item["budget"] for item in camp["stages"]
                                   if item["kind"] == "robustness"},
                       "order": [item["task"] for item in camp["stages"] if item["kind"] == "robustness"],
                       "trainings": sum(stage_counts[i["id"]] for i in camp["stages"] if i["kind"] == "robustness")},
        "gates": {"T": {"t_obstacle": t_obstacle, **GATE_T_FACTS}, "R": dict(GATE_R_FACTS),
                  "S": "saving possible yes: the worker already saves each policy with its optimizer state after "
                       "the fitness rollout; no observation normaliser exists; learning rate constant 3e-4"},
        "campaign_spec": camp,
    }
    return {"record": record, "estimate": estimate}


# --- rendering -----------------------------------------------------------------------

YAML_TEMPLATE = """# GENERATED by `python -m evodevo.extension amend2 derive` -- do not edit by hand.
# Amendment 2 (docs/campaign_prep/ext/PREREGISTRATION_EXT.md): ObstacleTraverser-v1 seeds 0-9,
# UpStepper-v0 and Walker-v0 seeds 5-9, three core arms, then champion re-training stages.
# Every value is the mechanical output recorded in docs/campaign_prep/ext/A2_DECISIONS.json;
# the plan refuses to build if they ever disagree with that record.
profile: campaign
encoders: [direct, nca, rule_based]
tasks: [{tasks}]
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
configs_root: evodevo/experiments/configs
output_dir: results/runs
surrogate: false
factors:
  dev_steps: 100
  init_condition: center_seed
estimate_path: docs/campaign_prep/ext/final_estimate_a2.json
decisions_path: docs/campaign_prep/ext/A2_DECISIONS.json
guard_outcome: "Amendment 2: no cost guard"
profile_overrides:
  population_size: 15
  offspring_size: 15
  eval_budget: 300
  ppo_seed_aggregation: max
  ppo_timesteps:
{timesteps}
  ppo_seeds:
{seeds}
amendment2:
{block}
"""


def render_yaml(record: dict) -> str:
    tasks = record["task_order"]
    block = "\n".join("  " + line for line in json.dumps(record["campaign_spec"], indent=2, sort_keys=True).splitlines())
    return YAML_TEMPLATE.format(
        tasks=", ".join(tasks),
        timesteps="\n".join(f"    {t}: {record['tasks'][t]['ppo_timesteps']}" for t in tasks),
        seeds="\n".join(f"    {t}: {record['tasks'][t]['ppo_seeds']}" for t in tasks), block=block)


def _stamp(when: dt.datetime) -> str:
    return when.strftime("%a %Y-%m-%d %H:%M IST")


def render_report(record: dict, estimate: dict) -> str:
    camp = record["campaign_spec"]
    bands = estimate["result"]["phase_h"]
    total = estimate["result"]["total_h"]
    lines = [f"# Amendment 2 decisions ({record['label']})", "",
             f"Derived {record['derived_at']} at commit `{record['git_commit']}` from preflight run "
             f"`{record['inputs']['preflight_run']}`, T_obstacle = {record['inputs']['t_obstacle']:,} "
             f"(Gate T), calibrated on campaign `{record['inputs']['reference_campaign']['id']}`. Gate line: "
             f"`{record['inputs']['gates_report_first_line']}`. Rules: [PREREGISTRATION_EXT.md]"
             "(PREREGISTRATION_EXT.md), Amendment 2.", "",
             "| Task | ppo_timesteps | K | source of ppo_timesteps |", "|---|---:|---:|---|"]
    for task in record["task_order"]:
        t = record["tasks"][task]
        lines.append(f"| {task} | {t['ppo_timesteps']:,} | {t['ppo_seeds']} | {t['ppo_timesteps_source']} |")
    lines += ["", "## Execution order (one campaign, strict order)", "",
              "| # | unit | kind | seeds / sources | PPO steps | trainings | h p5 | h mean | h p95 | mean finish (IST) |",
              "|---:|---|---|---|---:|---:|---:|---:|---:|---|"]
    by_id = {item["id"]: item for item in camp["phases"] + camp["stages"]}
    elapsed = 0.0
    for index, key in enumerate(camp["sequence"], 1):
        item = by_id[key]
        band = bands[key]
        elapsed += band["mean"]
        if "seeds" in item:
            what = f"seeds {item['seeds'][0]}-{item['seeds'][-1]} x {len(item['arms'])} arms"
            steps = record["tasks"][item["task"]]["ppo_timesteps"]
            count = len(item["arms"]) * len(item["seeds"])
            kind = "evolution (runs)"
        else:
            what = " + ".join("original seeds %d-%d" % (s["seeds"][0], s["seeds"][-1]) if "original" in s
                              else s["phase"] for s in item["sources"])
            steps, count, kind = item["budget"], estimate["design"]["units"][key]["trainings"], item["kind"]
        lines.append(f"| {index} | `{key}` | {kind} | {what} | {steps:,} | {count} | {band['p5']:.1f} | "
                     f"{band['mean']:.1f} | {band['p95']:.1f} | "
                     f"{_stamp(PLANNED_LAUNCH + dt.timedelta(hours=elapsed))} |")
    lines += ["", f"**Projected total (informational; no cost guard applies, Amendment 2 (k)):** {total['mean']:.1f} h mean "
              f"({total['p5']:.1f}-{total['p95']:.1f} h, p5-p95) = {total['mean'] / 24:.1f} days from the planned "
              f"launch {_stamp(PLANNED_LAUNCH)}: p5 {_stamp(PLANNED_LAUNCH + dt.timedelta(hours=total['p5']))}, "
              f"mean {_stamp(PLANNED_LAUNCH + dt.timedelta(hours=total['mean']))}, "
              f"p95 {_stamp(PLANNED_LAUNCH + dt.timedelta(hours=total['p95']))}.", "",
              "## Re-training stages", "",
              f"Re-evaluation: {record['reevaluation']['n_per_champion']} trainings per champion, PPO seeds "
              f"`ppo_seed({record['reevaluation']['seed_base']}, k)`, k = 0..9, at each task's own budget. "
              f"Robustness: {record['robustness']['n_per_champion']} trainings per champion, "
              f"`ppo_seed({record['robustness']['seed_base']}, k)`, k = 0..2, at {record['robustness']['factor']}x "
              f"the budget: " + ", ".join(f"{t} {b:,}" for t, b in record["robustness"]["budgets"].items()) +
              f" (in that order; {record['robustness']['trainings']} trainings).", "",
              "## Gates", "",
              f"- Gate T: T_obstacle = {record['gates']['T']['t_obstacle']:,}. {record['gates']['T']['rule']}. "
              "Fails at 1.0M and 1.2M; 1.5M is the registered cap and fails as well; only 2M passes (trivially).",
              f"- Gate R: {record['gates']['R']['outcome']}; config differs only in the seed; library versions "
              f"identical (evogym {record['campaign_spec']['recorded_versions']['evogym_commit'][:8]}); "
              f"{record['gates']['R']['replayed_original_evaluations_bit_identical']} replayed original "
              "evaluations bit-identical.",
              f"- Gate S: {record['gates']['S']}", "",
              "## Inputs", "",
              f"- Gates report `{record['inputs']['gates_report']}` sha256 `{record['inputs']['gates_report_sha256']}`",
              f"- Preflight record `{record['inputs']['preflight_record']}` sha256 "
              f"`{record['inputs']['preflight_record_sha256']}` (superseded; evidence only)",
              f"- Original campaign `{record['inputs']['reference_campaign']['path']}` commit "
              f"`{record['inputs']['reference_campaign']['git_commit']}` config hash "
              f"`{record['inputs']['reference_campaign']['config_hash']}`", "",
              "## Integrity (any edit of the two JSON files changes this report; `amend2 check` refuses it)", "",
              f"- `A2_DECISIONS.json` sha256 `{hashlib.sha256(_dump(record).encode()).hexdigest()}`",
              f"- `final_estimate_a2.json` sha256 `{hashlib.sha256(_dump(estimate).encode()).hexdigest()}`", ""]
    return "\n".join(lines)


# --- the TEST rehearsal ----------------------------------------------------------------

REHEARSAL_YAML = spec.REPO_ROOT / "evodevo/experiments/configs/campaign_a2_rehearsal.yaml"
REHEARSAL_ESTIMATE = spec.REPO_ROOT / "evodevo/experiments/configs/a2_rehearsal_estimate.json"
REHEARSAL_STEPS = 2048
REHEARSAL_K = {"ObstacleTraverser-v1": 1, "UpStepper-v0": 2, "Walker-v0": 1}
REHEARSAL_EVAL_BUDGET = 30


def rehearsal_spec() -> dict:
    """Same phase structure as production at pipeline-validation scale: Obstacle
    blocks {0} and {1}, UpStepper-v0 {5}, Walker-v0 {5}; the original campaign's
    seed-0 champions are read for the UpStepper-v0/Walker-v0 stages; 2 trainings
    per champion; robustness at 2x the tiny budget. Never a result."""
    return campaign_spec(
        REHEARSAL_STEPS, label=runtime.TEST, budgets={task: REHEARSAL_STEPS for task in TASK_ORDER}, reeval_n=2,
        robust_n=2, new_seeds={"ObstacleTraverser-v1": ((0,), (1,)), "UpStepper-v0": ((5,),), "Walker-v0": ((5,),)},
        original_seeds=(0,), config_source={task: "benchmark" for task in TASK_ORDER})


REHEARSAL_TEMPLATE = """# GENERATED by `python -m evodevo.extension amend2 rehearsal` -- do not edit by hand.
# TEST rehearsal of Amendment 2: the production phase structure (ObstacleTraverser-v1 seed blocks {{0}} and {{1}},
# UpStepper-v0 {{5}}, Walker-v0 {{5}}, then re-evaluation and robustness stages) at pipeline-validation scale:
# real PPO at 2,048 steps, 30 genomes per run, 2 trainings per champion, robustness at 2x. Never a result:
# always launch with --stamp "VALIDATION - NOT A RESULT" and --skip-config-equality-check (the tiny budgets
# differ from the original campaign's configs by design; only a TEST-labelled config may skip that check).
profile: campaign
encoders: [direct, nca, rule_based]
tasks: [{tasks}]
seeds: [0, 1, 5]
configs_root: evodevo/experiments/configs
output_dir: results/runs
surrogate: false
factors:
  dev_steps: 100
  init_condition: center_seed
estimate_path: evodevo/experiments/configs/a2_rehearsal_estimate.json
guard_outcome: "TEST - Amendment 2 rehearsal"
profile_overrides:
  population_size: 15
  offspring_size: 15
  eval_budget: {eval_budget}
  ppo_seed_aggregation: max
  ppo_timesteps:
{timesteps}
  ppo_seeds:
{seeds}
amendment2:
{block}
"""


def render_rehearsal_yaml() -> str:
    block = "\n".join("  " + line for line in json.dumps(rehearsal_spec(), indent=2, sort_keys=True).splitlines())
    return REHEARSAL_TEMPLATE.format(
        tasks=", ".join(TASK_ORDER), eval_budget=REHEARSAL_EVAL_BUDGET,
        timesteps="\n".join(f"    {t}: {REHEARSAL_STEPS}" for t in TASK_ORDER),
        seeds="\n".join(f"    {t}: {REHEARSAL_K[t]}" for t in TASK_ORDER), block=block)


def render_rehearsal_estimate() -> str:
    camp = rehearsal_spec()
    band = {"p5": 0.05, "mean": 0.1, "p95": 0.2}
    units, timeouts = {}, {}
    for item in camp["phases"]:
        units[item["id"]] = {"arms": sorted(item["arms"]), "seeds": item["seeds"], "ppo_timesteps": REHEARSAL_STEPS,
                             "ppo_seeds": REHEARSAL_K[item["task"]], "population_size": 15,
                             "eval_budget": REHEARSAL_EVAL_BUDGET}
    for item in camp["stages"]:
        units[item["id"]] = {"n": item["n"], "budget": item["budget"]}
    for key in camp["sequence"]:
        timeouts[key] = 900.0
    return _dump({
        "format": "evodevo-estimate-v2", "label": "TEST - synthetic bands for the Amendment 2 rehearsal; never a "
        "scientific estimate", "method": "synthetic", "workers": 18, "design": {"units": units},
        "disk": {"total_bytes": 1.0e9, "three_x_total_bytes": 3.0e9},
        "guard": {"applies": False, "outcome": "TEST rehearsal: no guard"},
        "inputs": {"t_ppo_mean_sd_at_b3_footing": {f"{arm}|{task}": [60.0, 10.0] for arm in spec.ARMS
                                                    for task in TASK_ORDER}},
        "a2": {"timeout_s": timeouts, "p95_ecore_s": {key: 225.0 for key in camp["sequence"]}},
        "result": {"task_order": list(camp["sequence"]), "phase_h": {key: dict(band) for key in camp["sequence"]},
                   "total_h": {q: band[q] * len(camp["sequence"]) for q in band}}})


def write_rehearsal() -> list:
    REHEARSAL_YAML.write_text(render_rehearsal_yaml(), encoding="utf-8")
    REHEARSAL_ESTIMATE.write_text(render_rehearsal_estimate(), encoding="utf-8")
    return [str(p.relative_to(spec.REPO_ROOT)) for p in (REHEARSAL_YAML, REHEARSAL_ESTIMATE)]


def write(result: dict) -> list:
    outputs = {DECISIONS_JSON: _dump(result["record"]), ESTIMATE_JSON: _dump(result["estimate"]),
               DECISIONS_MD: render_report(result["record"], result["estimate"]),
               CAMPAIGN_YAML: render_yaml(result["record"])}
    for path, text in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    return [str(p.relative_to(spec.REPO_ROOT)) for p in outputs]


# --- check ---------------------------------------------------------------------------------

def expected_plan_counts() -> dict:
    return {"evolutionary_runs": 60, "runs_per_phase": 15, "reeval": [150, 150, 300, 300], "robustness": 270}


def check(run_dir: Path | None, reference: Path | None, t_obstacle: int | None = None,
          *, allow_test: bool = False) -> list:
    problems = []
    from evodevo.campaign.plan import CampaignPlan
    from evodevo.experiments.benchmark import BenchmarkConfig
    for path in (DECISIONS_JSON, ESTIMATE_JSON, DECISIONS_MD, CAMPAIGN_YAML, GATES_REPORT, SUPERSEDED):
        if not path.is_file():
            problems.append(f"missing {_rel(path)}")
    if problems:
        return problems
    texts = {}
    for path in (DECISIONS_JSON, ESTIMATE_JSON):
        texts[path] = path.read_text(encoding="utf-8")
        if texts[path] != _dump(json.loads(texts[path])):
            problems.append(f"{path.name} is not the canonical dump of its content (hand edit?)")
    record = json.loads(texts[DECISIONS_JSON])
    estimate = json.loads(texts[ESTIMATE_JSON])
    if record.get("format") != runtime.RECORD_FORMAT or record.get("status") != "FINAL":
        problems.append("A2_DECISIONS.json is not a FINAL Amendment 2 record")
    if not allow_test and record.get("label") != runtime.PRODUCTION:
        problems.append(f"decision record is labelled {record.get('label')!r}, not 'PRODUCTION'")
    if CAMPAIGN_YAML.read_text(encoding="utf-8") != render_yaml(record):
        problems.append("campaign_a2.yaml is not the rendering of A2_DECISIONS.json")
    if DECISIONS_MD.read_text(encoding="utf-8") != render_report(record, estimate):
        problems.append("A2_DECISIONS.md is not the rendering of A2_DECISIONS.json and final_estimate_a2.json")
    if record["inputs"]["gates_report_sha256"] != _sha256(GATES_REPORT):
        problems.append("A2_GATES_REPORT.md differs from the version the record was derived from")
    if record["inputs"]["preflight_record_sha256"] != _sha256(PREFLIGHT_RECORD):
        problems.append("EXT_DECISIONS.json (the preflight record) differs from the version the record was "
                        "derived from")
    if json.loads(SUPERSEDED.read_text())["sha256"] != _sha256(PREFLIGHT_RECORD):
        problems.append("SUPERSEDED_BY_AMENDMENT_2.json does not hold the sha256 of EXT_DECISIONS.json")
    try:
        if parse_gates_first_line(gates_first_line())["t_obstacle"] != record["inputs"]["t_obstacle"]:
            problems.append("T_obstacle differs from the gates report")
    except ValueError as exc:
        problems.append(str(exc))
    if record["campaign_spec"] != campaign_spec(record["inputs"]["t_obstacle"], record["label"],
                                                record["campaign_spec"]["original_campaign"]):
        problems.append("campaign_spec is not the Amendment 2 design for this T_obstacle")
    plan = None
    try:
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CAMPAIGN_YAML)),
                                           allow_test_record=allow_test)
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        problems.append(f"plan does not build: {type(exc).__name__}: {exc}")
    if plan is not None:
        problems += plan_problems(plan, record)
    if REHEARSAL_YAML.read_text(encoding="utf-8") != render_rehearsal_yaml():
        problems.append("campaign_a2_rehearsal.yaml is not the generated rehearsal config")
    if REHEARSAL_ESTIMATE.read_text(encoding="utf-8") != render_rehearsal_estimate():
        problems.append("a2_rehearsal_estimate.json is not the generated rehearsal estimate")
    if run_dir is not None and reference is not None:
        fresh = derive(run_dir, reference, record["inputs"]["t_obstacle"], allow_test=allow_test)
        fresh["record"]["derived_at"] = record.get("derived_at")
        fresh["record"]["git_commit"] = record.get("git_commit")
        for key, path in (("record", DECISIONS_JSON), ("estimate", ESTIMATE_JSON)):
            if texts[path] != _dump(fresh[key]):
                problems.append(f"{path.name} differs from a fresh derivation of the same inputs")
    return problems


def plan_problems(plan, record) -> list:
    problems = []
    expected = expected_plan_counts()
    runs = [run for phase in plan.phases for run in phase.runs]
    if len(runs) != expected["evolutionary_runs"]:
        problems.append(f"plan has {len(runs)} evolutionary runs, expected {expected['evolutionary_runs']}")
    for phase in plan.phases:
        if len(phase.runs) != expected["runs_per_phase"]:
            problems.append(f"{phase.key}: {len(phase.runs)} runs, expected {expected['runs_per_phase']}")
        decided = record["tasks"][phase.task]
        if (phase.ppo_timesteps, phase.ppo_seeds) != (decided["ppo_timesteps"], decided["ppo_seeds"]):
            problems.append(f"{phase.key}: plan ({phase.ppo_timesteps}, K={phase.ppo_seeds}) != record")
        if sorted({r.arm for r in phase.runs}) != sorted(spec.ARMS):
            problems.append(f"{phase.key}: arms")
        if not phase.timeout_s or phase.timeout_s <= 0:
            problems.append(f"{phase.key}: no PPO-task timeout")
    plan_dict = json.loads(json.dumps(plan.to_dict()))
    counts = {s.stage_id: runtime.planned_trainings(json.loads(json.dumps(dataclasses.asdict(s))),
                                                    plan_dict) for s in plan.stages}
    reeval = [counts[s.stage_id] for s in plan.stages if s.kind == "reeval"]
    robust = sum(counts[s.stage_id] for s in plan.stages if s.kind == "robustness")
    if sorted(reeval) != sorted(expected["reeval"]) or sum(reeval) != sum(expected["reeval"]):
        problems.append(f"re-evaluation trainings {reeval}, expected {expected['reeval']}")
    if robust != expected["robustness"]:
        problems.append(f"robustness trainings {robust}, expected {expected['robustness']}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evodevo.extension amend2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("derive", "check", "rehearsal"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reference-campaign", default=None)
    parser.add_argument("--t-obstacle", type=int)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="derive on a dirty tree (the record's git commit then does not describe the code)")
    args = parser.parse_args(argv)
    run_dir = spec.PREFLIGHT_ROOT / args.run_id
    reference = Path(args.reference_campaign or DEFAULT_REFERENCE)
    if not reference.is_absolute():
        reference = spec.REPO_ROOT / reference
    if args.command == "rehearsal":
        print("wrote:\n  " + "\n  ".join(write_rehearsal()))
        return 0
    if args.command == "derive":
        if args.t_obstacle is None:
            parser.error("--t-obstacle is required (Gate T, from A2_GATES_REPORT.md)")
        dirty = _dirty()
        if dirty and not args.allow_dirty:
            parser.error("refusing to derive on a dirty working tree: the record names the git commit "
                         f"that produced it ({'; '.join(dirty)})")
        written = write(derive(run_dir, reference, args.t_obstacle, allow_test=args.allow_test))
        print("wrote:\n  " + "\n  ".join(written))
        print(DECISIONS_MD.read_text())
        return 0
    fresh_inputs = args.reference_campaign is not None
    problems = check(run_dir if fresh_inputs else None, reference if fresh_inputs else None,
                     args.t_obstacle, allow_test=args.allow_test)
    if problems:
        print("AMENDMENT 2 CHECK FAILED:\n  - " + "\n  - ".join(problems))
        return 1
    print("amendment 2 check passed")
    return 0
