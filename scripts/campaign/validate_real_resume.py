#!/usr/bin/env python3
"""Three real-PPO SIGKILL/resume checks, including a live generation-skew kill."""
from __future__ import annotations

import argparse
import json
import os
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


OPERATIONAL_FIELDS = {"wall_clock_s", "policy_paths", "policy_file_sizes",
                      "training_log_paths", "training_log_file_sizes"}


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in sorted(value.items())
                if key not in OPERATIONAL_FIELDS}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def command(config, root, campaign_id, *, resume=False):
    return [sys.executable, "-m", "evodevo.campaign", "--config", str(config),
            "--output-root", str(root), "--workers", "4", "--mode", "real", "--no-mirror",
            "--stamp", "VALIDATION - NOT A RESULT",
            *( ["--resume", campaign_id] if resume else ["--campaign-id", campaign_id] )]


def snapshot(campaign):
    manifest_path, dispatch_path = campaign / "manifest.json", campaign / "dispatch_status.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.is_file() else {}
    generations = list(dispatch.get("run_generations", {}).values())
    return {"phase": manifest.get("phase"), "journal_tasks": len(rows(campaign / "journal.jsonl")),
            "run_generations": dispatch.get("run_generations", {}),
            "generation_skew": max(generations) - min(generations) if generations else 0,
            "running": dispatch.get("tasks_running"), "queued": dispatch.get("tasks_queued")}


def wait_for(process, campaign, predicate, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("campaign exited before requested kill point")
        state = snapshot(campaign)
        if predicate(state):
            return state
        time.sleep(0.05)
    raise TimeoutError("requested real-PPO kill point was not reached")


def complete(config, root, campaign_id, env, *, resume=False):
    result = subprocess.run(command(config, root, campaign_id, resume=resume), cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise AssertionError((result.stdout + result.stderr)[-6000:])
    manifest = json.loads((root / campaign_id / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise AssertionError(f"{campaign_id} status {manifest.get('status')!r}")


def assert_matches(baseline, candidate, plan):
    for phase in plan.phases:
        for run in phase.runs:
            relative = Path("runs") / run.arm / run.task / f"seed_{run.seed}"
            left, right = baseline / relative, candidate / relative
            if normalized(rows(left / "run_log.jsonl")) != normalized(rows(right / "run_log.jsonl")):
                raise AssertionError(f"run log mismatch: {run.run_key}")
            if rows(left / "generation_log.jsonl") != rows(right / "generation_log.jsonl"):
                raise AssertionError(f"generation log mismatch: {run.run_key}")
            if normalized(latest_checkpoint(str(left / "checkpoint"))) != normalized(latest_checkpoint(str(right / "checkpoint"))):
                raise AssertionError(f"checkpoint mismatch: {run.run_key}")
    identities = [(row["run_key"], row["eval_index"], row["ppo_k"])
                  for row in rows(candidate / "journal.jsonl")]
    if len(identities) != len(set(identities)):
        raise AssertionError("duplicate completed journal task")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="evodevo/experiments/configs/campaign_resilience_validation.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--notifier-config", required=True)
    args = parser.parse_args(argv)
    config, root = Path(args.config).resolve(), Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(Path(args.notifier_config).resolve()))
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(config)))
    baseline_id = "uninterrupted"
    if not (root / baseline_id / "manifest.json").is_file():
        complete(config, root, baseline_id, env)
    baseline = root / baseline_id
    predicates = [
        ("live_generation_skew", lambda state: state["journal_tasks"] > 0 and state["generation_skew"] > 0),
        ("mid_walker", lambda state: state["phase"] == "Walker-v0" and state["journal_tasks"] >= 4),
        ("mid_upstepper", lambda state: state["phase"] == "UpStepper-v0" and state["journal_tasks"] >= 8),
    ]
    evidence = {"status": "running", "trials": []}
    for index, (kind, predicate) in enumerate(predicates):
        campaign_id = f"real-kill-{index:02d}"
        process = subprocess.Popen(command(config, root, campaign_id), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            state = wait_for(process, root / campaign_id, predicate)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        complete(config, root, campaign_id, env, resume=True)
        assert_matches(baseline, root / campaign_id, plan)
        evidence["trials"].append({"trial": index, "kind": kind, "returncode": process.returncode, **state})
    if evidence["trials"][0]["generation_skew"] <= 0:
        raise AssertionError("real-PPO skew trial did not observe skew")
    evidence.update(status="pass", all_resumed_equal_uninterrupted=True, duplicate_journal_tasks=0)
    destination = root / "real_resume.json"
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
