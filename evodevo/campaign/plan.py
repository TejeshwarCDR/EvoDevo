"""Declarative, hashable phase plan resolved from the campaign profile."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from evodevo.experiments.benchmark import BenchmarkConfig, resolve_configs, _resolved_arm_kwargs


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
    def from_dict(cls, payload):
        return cls(tuple(PhasePlan(p["task"], tuple(RunPlan(**run) for run in p["runs"]),
                                   p["ppo_timesteps"], p["ppo_seeds"], p["aggregation"],
                                   p.get("timeout_s")) for p in payload["phases"]),
                   payload.get("guard_outcome", "all four approved campaign arms"))

    @classmethod
    def from_benchmark(cls, benchmark: BenchmarkConfig):
        if benchmark.profile != "campaign":
            raise ValueError("CampaignPlan requires the campaign profile")
        resolved = resolve_configs(benchmark)
        estimate_path = Path(__file__).resolve().parents[2] / "results/campaign_prep/final_estimate/final_estimate.json"
        estimate = json.loads(estimate_path.read_text())
        t_ppo = estimate["inputs"]["t_ppo_mean_sd_at_b3_footing"]
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
                    config["campaign_encoder_kwargs"] = {
                        **dict(benchmark.factors), **_resolved_arm_kwargs(benchmark, arm)
                    }
                    runs.append(RunPlan(arm_id, task, seed, config))
            sample = resolved[(benchmark.resolved_arms()[0].arm_id, task)]
            p95_ecore = max((mean + 1.645 * sd) * 2.3172 for key, (mean, sd) in t_ppo.items()
                            if key.endswith("|" + task))
            phases.append(PhasePlan(task, tuple(runs), sample.ppo_timesteps,
                                    sample.ppo_seeds, sample.ppo_seed_aggregation,
                                    timeout_s=4 * p95_ecore))
        return cls(tuple(phases))

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def config_hash(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
