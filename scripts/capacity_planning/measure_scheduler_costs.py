"""Measure serial campaign-scheduler costs on the real campaign configuration."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evodevo.campaign.checkpoint import write_checkpoint
from evodevo.campaign.journal import ResultJournal
from evodevo.campaign.plan import CampaignPlan
from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import EvalResult, PPOTask, PPOTaskResult
from evodevo.evolution.ea import EvolutionRun
from evodevo.evolution.logging import RunLogger
from evodevo.experiments.benchmark import BenchmarkConfig
from evodevo.morphology.metrics import body_hash
from evodevo.morphology.validate import validate_body


OUT = ROOT / "results/campaign_prep/final_estimate/scheduler_costs.json"
CONFIG = ROOT / "evodevo/experiments/configs/benchmark.yaml"


def stats(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "p5": float(np.percentile(x, 5)), "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)), "min": float(x.min()),
        "max": float(x.max()), "n": int(x.size),
    }


def make_run(run_plan, directory):
    cfg_dict = dict(run_plan.config)
    factor_kwargs = cfg_dict.pop("campaign_encoder_kwargs", cfg_dict.pop("campaign_factors", {}))
    cfg_dict["grid_shape"] = tuple(cfg_dict["grid_shape"])
    config = ExperimentConfig(**cfg_dict)
    factors, kwargs = extract_factors(factor_kwargs, seed=config.seed)
    encoder_name = "rule_based" if run_plan.arm.startswith("rule_based") else run_plan.arm
    if run_plan.arm == "rule_based_monotone":
        kwargs["action_monotonicity"] = "monotone"
    encoder = get_encoder(encoder_name, factors=factors, grid_shape=config.grid_shape, **kwargs)
    logger = RunLogger(str(directory), encoder=run_plan.arm, seed=run_plan.seed)
    run = EvolutionRun(encoder, config, logger=logger)
    return config, encoder, run, logger


def measure_arm(run_plan, root):
    from evogym import get_full_connectivity

    config, encoder, run, logger = make_run(run_plan, root / run_plan.arm)
    propose_s, validate_s, tell_log_s, checkpoint_s, generation_rows = [], [], [], [], []
    try:
        while not run.done:
            started = time.perf_counter()
            candidates = run.propose()
            proposed = time.perf_counter()
            results = []
            for candidate in candidates:
                validation = validate_body(candidate.body)
                if validation.valid:
                    get_full_connectivity(candidate.body)
                results.append(EvalResult(
                    fitness=0.0 if validation.valid else config.sentinel_fitness,
                    valid=validation.valid,
                    reason=None if validation.valid else "; ".join(validation.reasons),
                    wall_clock_s=0.0, body_hash=body_hash(candidate.body),
                    eval_index=candidate.eval_index, per_seed_fitness=[],
                    per_seed_completed=[], aggregation=config.ppo_seed_aggregation,
                    policy_paths=[], training_log_paths=[],
                ))
            validated = time.perf_counter()
            run.tell(results)
            told = time.perf_counter()
            write_checkpoint(run, str(root / run_plan.arm / "checkpoint"))
            checkpointed = time.perf_counter()
            row = {
                "generation": int(run.generation), "genomes": len(candidates),
                "propose_decode_s": proposed - started,
                "validate_connectivity_s": validated - proposed,
                "tell_log_s": told - validated,
                "checkpoint_s": checkpointed - told,
            }
            row["prepare_s"] = row["propose_decode_s"] + row["validate_connectivity_s"]
            row["finish_s"] = row["tell_log_s"] + row["checkpoint_s"]
            generation_rows.append(row)
            propose_s.append(row["propose_decode_s"])
            validate_s.append(row["validate_connectivity_s"])
            tell_log_s.append(row["tell_log_s"])
            checkpoint_s.append(row["checkpoint_s"])
    finally:
        logger.close()
    return {
        "generations": len(generation_rows), "genomes": sum(r["genomes"] for r in generation_rows),
        "per_generation": {
            "propose_decode_s": stats(propose_s), "validate_connectivity_s": stats(validate_s),
            "prepare_s": stats([r["prepare_s"] for r in generation_rows]),
            "tell_log_s": stats(tell_log_s), "checkpoint_s": stats(checkpoint_s),
            "finish_s": stats([r["finish_s"] for r in generation_rows]),
        },
        "per_genome_prepare_s": stats([
            r["prepare_s"] / r["genomes"] for r in generation_rows]),
        "raw_generations": generation_rows,
        "final_state": {"evaluations_used": run.tracker.evaluations_used,
                        "generation": run.generation, "done": run.done},
        "encoder": encoder.describe(),
    }


def measure_journal(root, n=200):
    journal = ResultJournal(root / "journal.jsonl")
    task = PPOTask("measure/run", "measure", "Walker-v0", 0, 0, 0, 0, 0,
                   None, None, 0, 0, None)
    result = PPOTaskResult(0.0, True, None, 1.0, 1.0, 0, [0], None, None, None, None)
    samples = []
    for index in range(n):
        current = dataclasses.replace(task, genome_id=index, eval_index=index)
        started = time.perf_counter()
        journal.append_task(current, result, f"{index:064x}")
        samples.append(time.perf_counter() - started)
    return {"append_fsync_s": stats(samples), "n": n}


def main():
    benchmark = BenchmarkConfig.from_yaml(str(CONFIG))
    plan = CampaignPlan.from_benchmark(benchmark)
    walker = plan.phases[0]
    selected = []
    for arm in benchmark.resolved_arms():
        selected.append(next(run for run in walker.runs if run.arm == arm.arm_id and run.seed == 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scheduler-costs-", dir=OUT.parent) as temporary:
        temporary = Path(temporary)
        measured = {run.arm: measure_arm(run, temporary) for run in selected}
        journal = measure_journal(temporary / "journal-measurement")
    payload = {
        "schema": 1,
        "description": "Real CampaignScheduler-path serial costs; no PPO and no wall-clock sleeps.",
        "config": str(CONFIG.relative_to(ROOT)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        "measured_at_unix": time.time(),
        "arms": measured,
        "journal": journal,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for arm, row in measured.items():
        print(f"{arm}: prepare {row['per_generation']['prepare_s']['mean']:.4f}s, "
              f"finish {row['per_generation']['finish_s']['mean']:.4f}s, "
              f"per genome {row['per_genome_prepare_s']['mean']:.4f}s")
    print(f"journal append+fsync: {journal['append_fsync_s']['mean']:.6f}s (n={journal['n']})")
    print(OUT)


if __name__ == "__main__":
    main()
