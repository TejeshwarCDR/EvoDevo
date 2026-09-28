"""Compare a completed timed-surrogate campaign with the shared simulator."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "simulate_campaign", ROOT / "scripts/capacity_planning/simulate_campaign.py")
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)
SCHEDULER_COSTS = ROOT / "results/campaign_prep/final_estimate/scheduler_costs.json"


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "sd": float(x.std(ddof=1)),
            "n": int(x.size)}


def phase_inputs(manifest, journal, phase):
    runs = phase["runs"]
    run_index = {run["arm"] + "/" + phase["task"] + f"/seed_{run['seed']}": index
                 for index, run in enumerate(runs)}
    population = int(runs[0]["config"]["population_size"])
    budget = int(runs[0]["config"]["eval_budget"])
    generations = (budget + population - 1) // population
    if generations != SIM.GENS or population != SIM.POP:
        raise ValueError(f"validation shape {generations}x{population} != simulator shape "
                         f"{SIM.GENS}x{SIM.POP}")
    k = int(phase["ppo_seeds"])
    scale = manifest["plan"]["validation_profile"]["time_scale"][phase["task"]]
    durations = [np.zeros((generations, population * k), dtype=float) for _ in runs]
    masks = [np.zeros_like(duration, dtype=bool) for duration in durations]
    count = 0
    for row in journal:
        if row.get("run_key") not in run_index:
            continue
        result = row.get("result", {})
        base = result.get("validation_base_duration_s")
        if base is None:
            raise ValueError("validation journal row lacks validation_base_duration_s")
        run = run_index[row["run_key"]]
        generation = int(row["generation"])
        within_generation = int(row["eval_index"]) - generation * population
        slot = within_generation * k + int(row["ppo_k"])
        if not 0 <= within_generation < population:
            raise ValueError(f"eval_index/generation mismatch in {row['run_key']}")
        durations[run][generation, slot] = float(base) * float(scale)
        masks[run][generation, slot] = True
        count += 1
    return durations, masks, count


def simulate_phase(manifest, phase, durations, masks, n_trials=500):
    measured = json.loads(SCHEDULER_COSTS.read_text())
    costs = {
        "prepare_s": [measured["arms"][run["arm"]]["per_generation"]["prepare_s"]["mean"]
                      for run in phase["runs"]],
        "finish_s": [measured["arms"][run["arm"]]["per_generation"]["finish_s"]["mean"]
                     for run in phase["runs"]],
        "journal_s": measured["journal"]["append_fsync_s"]["mean"],
    }
    e_ids = {core["logical_id"] for core in manifest["topology"]["cores"]
             if core["kind"] == "E"}
    ratio = manifest["plan"]["validation_profile"]["e_over_p_ratio"]
    factors = np.asarray([ratio if slot[0] in e_ids else 1.0
                          for slot in manifest["worker_slots"]["slots"]], dtype=float)
    rng = np.random.default_rng(20260928)
    samples = []
    for _ in range(n_trials):
        permuted = factors[rng.permutation(len(factors))].tolist()
        makespan, _finish, metrics = SIM.run_continuous_dispatch(
            durations, permuted, costs, prefetch=int(manifest.get("prefetch", 4)),
            inline_masks=masks)
        samples.append({"makespan_s": makespan,
                        "worker_utilization": metrics["worker_utilization"],
                        "steady_state_worker_utilization": metrics["steady_state_worker_utilization"],
                        "scheduler_busy_fraction": metrics["scheduler_busy_fraction"]})
    return {key: stats([sample[key] for sample in samples]) for key in samples[0]}


def compare(campaign_dir, n_trials=500):
    campaign_dir = Path(campaign_dir)
    manifest = json.loads((campaign_dir / "manifest.json").read_text())
    if manifest.get("artifact_stamp") != "VALIDATION - NOT A RESULT":
        raise ValueError("campaign is not stamped VALIDATION - NOT A RESULT")
    journal = rows(campaign_dir / "journal.jsonl")
    measured = json.loads((campaign_dir / "phase_metrics.json").read_text())
    phases = {}
    all_pass = True
    for phase in manifest["plan"]["phases"]:
        durations, masks, count = phase_inputs(manifest, journal, phase)
        predicted = simulate_phase(manifest, phase, durations, masks, n_trials=n_trials)
        actual = measured[phase["task"]]
        deltas = {
            "makespan_fraction": abs(actual["makespan_s"] - predicted["makespan_s"]["mean"])
                                 / predicted["makespan_s"]["mean"],
            "worker_utilization_points": abs(actual["worker_utilization"]
                                             - predicted["worker_utilization"]["mean"]),
            "steady_state_utilization_points": abs(actual["steady_state_worker_utilization"]
                                                    - predicted["steady_state_worker_utilization"]["mean"]),
        }
        gate = (deltas["makespan_fraction"] <= 0.05
                and deltas["worker_utilization_points"] <= 0.03
                and deltas["steady_state_utilization_points"] <= 0.03)
        all_pass &= gate
        phases[phase["task"]] = {
            "journal_tasks": count, "actual": actual, "simulator": predicted,
            "deltas": deltas, "gate_passed": gate,
        }
    return {"campaign_dir": str(campaign_dir), "n_trials": n_trials,
            "thresholds": {"makespan_fraction": 0.05, "utilization_points": 0.03},
            "phases": phases, "gate_passed": all_pass}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args(argv)
    result = compare(args.campaign_dir, args.trials)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)
    print(payload, end="")
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
