#!/usr/bin/env python3
"""Repeated SIGKILL/SIGINT resume proof for the production campaign CLI."""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evodevo.campaign.checkpoint import latest_checkpoint
from evodevo.campaign.plan import CampaignPlan
from evodevo.experiments.benchmark import BenchmarkConfig


OPERATIONAL_FIELDS = {
    "wall_clock_s", "policy_paths", "policy_file_sizes",
    "training_log_paths", "training_log_file_sizes",
}


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in sorted(value.items())
                if key not in OPERATIONAL_FIELDS}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count_records(campaign):
    return sum(len(rows(path)) for path in campaign.glob("runs/*/*/seed_*/run_log.jsonl"))


def command(config, output_root, campaign_id, *, resume=False):
    return [
        sys.executable, "-m", "evodevo.campaign",
        "--config", str(config), "--output-root", str(output_root),
        "--workers", "4", "--mode", "surrogate", "--no-mirror",
        "--stamp", "VALIDATION - NOT A RESULT",
        *( ["--resume", campaign_id] if resume else ["--campaign-id", campaign_id] ),
    ]


def wait_for_target(process, campaign, target, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"campaign exited before kill/stop target {target}")
        completed = count_records(campaign)
        if completed >= target:
            return completed
        time.sleep(0.01)
    raise TimeoutError(f"campaign did not reach record target {target}")


def snapshot(campaign):
    def load(name):
        path = campaign / name
        return json.loads(path.read_text()) if path.is_file() else {}
    manifest = load("manifest.json")
    dispatch = load("dispatch_status.json")
    generations = list(dispatch.get("run_generations", {}).values())
    return {
        "phase": manifest.get("phase"),
        "records": count_records(campaign),
        "run_generations": dispatch.get("run_generations", {}),
        "generation_skew": (max(generations) - min(generations)) if generations else 0,
    }


def run_to_completion(config, output_root, campaign_id, env, *, resume=False):
    result = subprocess.run(command(config, output_root, campaign_id, resume=resume),
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise AssertionError((result.stdout + result.stderr)[-6000:])
    manifest = json.loads((output_root / campaign_id / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise AssertionError(f"{campaign_id} ended with {manifest.get('status')!r}")


def assert_matches(baseline, candidate, plan):
    for phase in plan.phases:
        for run in phase.runs:
            relative = Path("runs") / run.arm / run.task / f"seed_{run.seed}"
            left, right = baseline / relative, candidate / relative
            if normalized(rows(left / "run_log.jsonl")) != normalized(rows(right / "run_log.jsonl")):
                raise AssertionError(f"run log mismatch: {run.run_key}")
            if rows(left / "generation_log.jsonl") != rows(right / "generation_log.jsonl"):
                raise AssertionError(f"generation log mismatch: {run.run_key}")
            left_state = latest_checkpoint(str(left / "checkpoint"))
            right_state = latest_checkpoint(str(right / "checkpoint"))
            if normalized(left_state) != normalized(right_state):
                raise AssertionError(f"final checkpoint mismatch: {run.run_key}")
    journal = rows(candidate / "journal.jsonl")
    identities = [(row["run_key"], row["eval_index"], row["ppo_k"]) for row in journal]
    if len(identities) != len(set(identities)):
        raise AssertionError("duplicate completed journal task")


def process_group_survivors(pgid):
    output = subprocess.run(["ps", "-eo", "pgid=,stat=,args="], capture_output=True, text=True).stdout
    return [line.strip() for line in output.splitlines()
            if line.split() and line.split()[0] == str(pgid) and not line.split()[1].startswith("Z")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="evodevo/experiments/configs/campaign_resilience_validation.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--notifier-config", required=True)
    parser.add_argument("--kill-trials", type=int, default=20)
    parser.add_argument("--stop-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if args.kill_trials < 20 or args.stop_trials < 10:
        parser.error("the Step 6 gate requires at least 20 kill trials and 10 graceful-stop trials")

    config = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(Path(args.notifier_config).resolve()))
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(config)))
    total_records = sum(run.config["eval_budget"] for phase in plan.phases for run in phase.runs)
    rng = random.Random(args.seed)

    baseline_id = "uninterrupted"
    if not (output_root / baseline_id / "manifest.json").is_file():
        run_to_completion(config, output_root, baseline_id, env)
    baseline = output_root / baseline_id
    evidence = {"status": "running", "seed": args.seed, "total_records": total_records,
                "kill_trials": [], "graceful_stop_trials": []}

    # Stratified random targets cover both D7 phases while remaining random
    # within each stratum.  The small post-target offset makes kills land in
    # proposal/decode/tell work as well as at persisted row boundaries.
    phase_boundary = total_records // len(plan.phases)
    kill_targets = []
    for index in range(args.kill_trials):
        low, high = ((1, phase_boundary - 1) if index % 2 == 0
                     else (phase_boundary + 1, total_records - 2))
        kill_targets.append(rng.randint(low, high))
    for index, target in enumerate(kill_targets):
        campaign_id = f"kill-{index:02d}"
        process = subprocess.Popen(command(config, output_root, campaign_id), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            reached = wait_for_target(process, output_root / campaign_id, target)
            offset = rng.uniform(0.005, 0.08)
            time.sleep(offset)
            before = snapshot(output_root / campaign_id)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        run_to_completion(config, output_root, campaign_id, env, resume=True)
        assert_matches(baseline, output_root / campaign_id, plan)
        evidence["kill_trials"].append({"trial": index, "target": target, "reached": reached,
                                        "post_target_offset_s": offset, **before,
                                        "returncode": process.returncode})

    stop_targets = [rng.randint(1, total_records - 2) for _ in range(args.stop_trials)]
    for index, target in enumerate(stop_targets):
        campaign_id = f"stop-{index:02d}"
        process = subprocess.Popen(command(config, output_root, campaign_id), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            reached = wait_for_target(process, output_root / campaign_id, target)
            before = snapshot(output_root / campaign_id)
            started = time.monotonic()
            os.killpg(process.pid, signal.SIGINT)
            output, _ = process.communicate(timeout=30)
            elapsed = time.monotonic() - started
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process.returncode != 0 or elapsed >= 30:
            raise AssertionError(f"graceful stop {index} failed in {elapsed:.3f}s: {output[-4000:]}")
        survivors = process_group_survivors(process.pid)
        if survivors:
            raise AssertionError(f"graceful stop {index} left process-group survivors: {survivors}")
        manifest = json.loads((output_root / campaign_id / "manifest.json").read_text())
        if manifest.get("status") != "stopped":
            raise AssertionError(f"graceful stop {index} manifest is {manifest.get('status')!r}")
        run_to_completion(config, output_root, campaign_id, env, resume=True)
        assert_matches(baseline, output_root / campaign_id, plan)
        evidence["graceful_stop_trials"].append({"trial": index, "target": target,
                                                  "reached": reached, "elapsed_s": elapsed,
                                                  "survivors": survivors, **before})

    phases = {row["phase"] for row in evidence["kill_trials"]}
    if phases != {phase.task for phase in plan.phases}:
        raise AssertionError(f"kill trials did not cover every phase: {phases}")
    evidence["status"] = "pass"
    evidence["all_resumed_equal_uninterrupted"] = True
    evidence["duplicate_journal_tasks"] = 0
    destination = output_root / "resilience.json"
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
