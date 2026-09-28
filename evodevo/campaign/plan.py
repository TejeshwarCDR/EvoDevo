"""Declarative, hashable phase plan resolved from the campaign profile."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from evodevo.experiments.benchmark import BenchmarkConfig, resolve_configs


@dataclass(frozen=True)
class RunPlan:
    arm: str
    task: str
    seed: int
    config: dict

    @property
    def run_key(self):
        return f"{self.arm}/{self.task}/seed_{self.seed}"


@dataclass(frozen=True)
class PhasePlan:
    task: str
    runs: tuple[RunPlan, ...]
    ppo_timesteps: int
    ppo_seeds: int
    aggregation: str
    timeout_s: float | None = None


@dataclass(frozen=True)
class CampaignPlan:
    phases: tuple[PhasePlan, ...]
    guard_outcome: str = "all four approved campaign arms"

    @classmethod
    def from_benchmark(cls, benchmark: BenchmarkConfig):
        if benchmark.profile != "campaign":
            raise ValueError("CampaignPlan requires the campaign profile")
        resolved = resolve_configs(benchmark)
        phases = []
        # D7 order is the order in the accepted campaign config, never a set.
        for task in benchmark.tasks:
            runs = []
            for arm in benchmark.resolved_arms():
                arm_id = arm.arm_id
                for seed in benchmark.seeds:
                    cfg = resolved[(arm_id, task)]
                    cfg = dataclasses.replace(cfg, seed=seed)
                    config = cfg.to_dict()
                    config["campaign_factors"] = dict(benchmark.factors)
                    runs.append(RunPlan(arm_id, task, seed, config))
            sample = resolved[(benchmark.resolved_arms()[0].arm_id, task)]
            phases.append(PhasePlan(task, tuple(runs), sample.ppo_timesteps,
                                    sample.ppo_seeds, sample.ppo_seed_aggregation))
        return cls(tuple(phases))

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def config_hash(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
