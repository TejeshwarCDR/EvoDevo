#!/usr/bin/env python3
"""One-shot serial/interleaved equivalence evidence for campaign profiles.

This validator deliberately uses the production ``CampaignScheduler`` and
``run_ea`` entry points.  It compares every evaluation record after removing
only operational timing/worker/artifact-path fields, plus the final population
and the complete ``RunResult`` state.  Its output is validation evidence, not a
scientific result.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evodevo.campaign.plan import CampaignPlan
from evodevo.campaign.scheduler import CampaignScheduler
from evodevo.campaign.checkpoint import latest_checkpoint
from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import evaluate_body, evaluate_body_surrogate
from evodevo.evolution.ea import EvolutionRun, run_ea
from evodevo.evolution.logging import RunLogger
from evodevo.experiments.benchmark import BenchmarkConfig


OPERATIONAL_FIELDS = {
    "wall_clock_s",
    "policy_paths",
    "policy_file_sizes",
    "training_log_paths",
    "training_log_file_sizes",
}


def normalized(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            key: normalized(item)
            for key, item in sorted(value.items())
            if key not in OPERATIONAL_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def make_encoder(run_plan, config):
    factor_kwargs = dict(run_plan.config.get("campaign_encoder_kwargs", {}))
    factors, kwargs = extract_factors(factor_kwargs, seed=config.seed)
    encoder_name = "rule_based" if run_plan.arm.startswith("rule_based") else run_plan.arm
    if run_plan.arm == "rule_based_monotone":
        kwargs["action_monotonicity"] = "monotone"
    return get_encoder(encoder_name, grid_shape=config.grid_shape, factors=factors, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="evodevo/experiments/configs/benchmark.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--evaluator", choices=("surrogate", "real"), default="surrogate")
    parser.add_argument("--existing-campaign-dir")
    parser.add_argument("--notifier-config", help="validation-only notifier config; defaults to an isolated nonexistent path")
    args = parser.parse_args(argv)

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    notifier_config = Path(args.notifier_config).resolve() if args.notifier_config else root / ".validation-notifier.yaml"
    os.environ["EVODEVO_NOTIFY_CONFIG"] = str(notifier_config)
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(args.config))
    if args.existing_campaign_dir:
        campaign_dir = Path(args.existing_campaign_dir).resolve()
        interleaved = None
    else:
        if args.evaluator != "surrogate":
            parser.error("real interleaved validation must be launched through the CLI and supplied with --existing-campaign-dir")
        campaign_dir = root / "interleaved"
        interleaved = CampaignScheduler(
            plan,
            "interleaved",
            output_root=root,
            workers=args.workers,
            mode="surrogate",
            no_mirror=True,
        ).run()

    checked = []
    for phase in plan.phases:
        for run_plan in phase.runs:
            config_payload = dict(run_plan.config)
            config_payload.pop("campaign_encoder_kwargs", None)
            config_payload["grid_shape"] = tuple(config_payload["grid_shape"])
            config = ExperimentConfig(**config_payload)
            encoder = make_encoder(run_plan, config)
            relative = Path("runs") / run_plan.arm / run_plan.task / f"seed_{run_plan.seed}"
            serial_dir = root / "serial" / relative
            logger = RunLogger(str(serial_dir), encoder=run_plan.arm, seed=run_plan.seed)
            try:
                evaluator = evaluate_body if args.evaluator == "real" else evaluate_body_surrogate
                serial = run_ea(encoder, evaluator, config, logger=logger)
            finally:
                logger.close()
            parallel = None if interleaved is None else interleaved.get(run_plan.run_key)
            if parallel is None:
                state = latest_checkpoint(str(campaign_dir / relative / "checkpoint"))
                if state is None or not state.get("done"):
                    raise AssertionError(f"completed checkpoint unavailable for {run_plan.run_key}")
                parallel = EvolutionRun.from_state(encoder, config, state).result()
            if normalized(serial.records) != normalized(parallel.records):
                raise AssertionError(f"evaluation records differ for {run_plan.run_key}")
            if normalized(serial.population) != normalized(parallel.population):
                raise AssertionError(f"final population differs for {run_plan.run_key}")
            if normalized(serial) != normalized(parallel):
                raise AssertionError(f"RunResult differs for {run_plan.run_key}")
            checked.append({
                "run_key": run_plan.run_key,
                "records": len(serial.records),
                "final_population": len(serial.population),
                "best_fitness": serial.best_fitness,
            })

    evidence = {
        "status": "pass",
        "config": str(Path(args.config)),
        "evaluator": args.evaluator,
        "interleaved_campaign_dir": str(campaign_dir),
        "notifier_config": str(notifier_config),
        "campaign_config_hash": plan.config_hash,
        "workers": args.workers,
        "runs_checked": len(checked),
        "evaluation_records_checked": sum(row["records"] for row in checked),
        "operational_fields_excluded": sorted(OPERATIONAL_FIELDS),
        "checks": ["all evaluation records", "final population", "complete RunResult"],
        "runs": checked,
    }
    destination = root / "equivalence.json"
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
