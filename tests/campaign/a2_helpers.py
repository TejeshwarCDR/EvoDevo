"""A small, self-contained Amendment 2 world for tests: an 'original campaign'
directory (manifest + run logs), a tiny plan with the production phase
structure, and its dict form. Nothing here reads results/ or needs the lab."""
import json
import os
from pathlib import Path

import numpy as np

from evodevo.campaign.amend2 import ORIGINAL_CAMPAIGN_PATH
from evodevo.campaign.plan import CampaignPlan, PhasePlan, RunPlan, StagePlan
from evodevo.core.config import ExperimentConfig
from evodevo.extension.amend2 import campaign_spec
from evodevo.morphology.metrics import body_hash

BODY = [[3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
BODY_B = [[1, 1, 1, 0, 0], [3, 3, 3, 0, 0], [4, 4, 4, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
ARMS = ("nca", "rule_based")
EVALS = 30
ORIGINAL_EVALS = 6
NEW_SEEDS = {"ObstacleTraverser-v1": ((0,), (1,)), "UpStepper-v0": ((5,),), "Walker-v0": ((5,),)}


def config(arm, task, seed, **overrides):
    values = dict(seed=seed, encoder=arm, task=task, population_size=6, offspring_size=6, eval_budget=EVALS,
                  ppo_timesteps=2000, ppo_eval_episodes=1, output_dir="results/runs")
    values.update(overrides)
    return ExperimentConfig(**values).to_dict()


def write_run_log(directory, fitnesses, bodies=None):
    """A completed run: ``fitnesses[i]`` is evaluation i (None = invalid body)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, fitness in enumerate(fitnesses):
        body = (bodies or {}).get(index, BODY)
        rows.append({"eval_index": index, "generation": index // 3, "fitness": -10.0 if fitness is None else fitness,
                     "valid": fitness is not None, "body": body, "body_hash": body_hash(np.array(body)),
                     "cache_hit": False, "wall_clock_s": 1.0, "per_seed_fitness": [fitness],
                     "genome": [0], "policy_paths": []})
    (directory / "run_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def make_original(root, seeds=(0, 1), tasks=("UpStepper-v0", "Walker-v0"), arms=ARMS + ("rule_based_monotone",)):
    """results/campaign/<id> stand-in: a manifest with per-run configs and run logs."""
    root = Path(root)
    phases = []
    for task in tasks:
        runs = [{"arm": arm, "task": task, "seed": seed, "config": config(arm, task, seed, eval_budget=ORIGINAL_EVALS)}
                for arm in arms for seed in seeds]
        phases.append({"task": task, "runs": runs, "ppo_timesteps": 2000, "ppo_seeds": 1, "aggregation": "max",
                       "timeout_s": 100.0})
        for arm in arms:
            for seed in seeds:
                write_run_log(root / "runs" / arm / task / f"seed_{seed}",
                              [0.1 * (seed + 1), 0.9, 0.5, 0.9, 0.2, 0.3], {1: BODY_B, 3: BODY})
    (root / "manifest.json").write_text(json.dumps({
        "campaign_id": "orig-id", "git_commit": "c0ffee0123456789c0ffee0123456789c0ffee01", "config_hash": "0" * 64,
        "plan": {"phases": phases, "guard_outcome": "all four approved campaign arms"}}))
    return root


def spec_for(original_path):
    spec = campaign_spec(2000, label="TEST", reference=str(original_path), budgets={
        "ObstacleTraverser-v1": 2000, "UpStepper-v0": 2000, "Walker-v0": 2000}, reeval_n=2, robust_n=2,
        new_seeds=NEW_SEEDS, original_seeds=(0,), config_source={t: "benchmark" for t in NEW_SEEDS})
    for item in spec["phases"]:
        item["arms"] = list(ARMS)
    for item in spec["stages"]:
        for source in item["sources"]:
            if "arms" in source:
                source["arms"] = list(ARMS)
    for item in spec["phases"]:
        for source in item["package"]["merge"]:
            if "arms" in source:
                source["arms"] = list(ARMS)
    return spec


def make_plan(original_path, timeout=120.0):
    """The production phase structure at test scale (arms direct/nca)."""
    spec = spec_for(original_path)
    phases = []
    for item in spec["phases"]:
        runs = tuple(RunPlan(arm, item["task"], seed, config(arm, item["task"], seed))
                     for arm in item["arms"] for seed in item["seeds"])
        phases.append(PhasePlan(item["task"], runs, 2000, 1, "max", timeout, item["id"]))
    stages = tuple(StagePlan(s["id"], s["kind"], s["task"], s["n"], s["seed_base"], s["budget"],
                             tuple(s["sources"]), timeout) for s in spec["stages"])
    manifest = json.loads((Path(original_path) / "manifest.json").read_text())
    amendment = {"label": "TEST", "sequence": spec["sequence"], "phases": spec["phases"],
                 "recorded_versions": spec["recorded_versions"],
                 "original_campaign": {"path": str(original_path), "id": manifest["campaign_id"],
                                       "git_commit": manifest["git_commit"], "config_hash": manifest["config_hash"]}}
    return CampaignPlan(tuple(phases), "test", None, stages, amendment)
