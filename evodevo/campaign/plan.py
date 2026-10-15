"""Declarative, hashable phase plan resolved from the campaign profile."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from evodevo.experiments.benchmark import (CAMPAIGN, CAMPAIGN_VALIDATION,
    BenchmarkConfig, resolve_configs, _resolved_arm_kwargs)


VALIDATION_STAMP = "VALIDATION - NOT A RESULT"
REPO_ROOT = Path(__file__).resolve().parents[2]
# The original campaign's capacity estimate (gitignored, lab host only).
LEGACY_ESTIMATE_PATH = REPO_ROOT / "results/campaign_prep/final_estimate/final_estimate.json"
LEGACY_GUARD_OUTCOME = "all four approved campaign arms"
# Validation-mode sleep scaling. The original two tasks keep their exact
# values; any other task uses the slower UpStepper scale.
LEGACY_VALIDATION_TIME_SCALE = {"Walker-v0": 1 / 100, "UpStepper-v0": 1 / 250}
DEFAULT_VALIDATION_TIME_SCALE = 1 / 250
E_OVER_P_RATIO = 2.3172


class DecisionRecordError(ValueError):
    """The benchmark names a decision record that is missing, not FINAL, or
    disagrees with the YAML's per-task PPO budget or K."""


def resolve_estimate_path(benchmark) -> Path:
    """The estimate a benchmark's plan is built from (repo-relative paths
    resolve against the repository root, never the current directory)."""
    raw = getattr(benchmark, "estimate_path", None)
    if not raw:
        return LEGACY_ESTIMATE_PATH
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_decision_record(benchmark) -> dict | None:
    """Refuse a plan whose YAML disagrees with its FINAL decision record."""
    raw = getattr(benchmark, "decisions_path", None)
    if not raw:
        return None
    path = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
    if not path.is_file():
        raise DecisionRecordError(f"decision record {path} does not exist; run the extension "
                                  "preflight and decision derivation first")
    record = json.loads(path.read_text())
    if record.get("status") != "FINAL":
        raise DecisionRecordError(f"decision record {path} has status {record.get('status')!r}, not 'FINAL'")
    profile = benchmark.resolved_profile_kwargs()
    tasks = record.get("tasks", {})
    if list(benchmark.tasks) != list(record.get("task_order", [])):
        raise DecisionRecordError(f"task order {benchmark.tasks} != decision record {record.get('task_order')}")
    for task in benchmark.tasks:
        decided = tasks.get(task, {})
        for field in ("ppo_timesteps", "ppo_seeds"):
            value = profile[field][task] if isinstance(profile[field], dict) else profile[field]
            if decided.get(field) != value:
                raise DecisionRecordError(f"{task} {field}: YAML has {value!r}, decision record {path} "
                                          f"has {decided.get(field)!r}")
    for field in ("population_size", "offspring_size", "eval_budget", "ppo_seed_aggregation"):
        if field in record.get("shared", {}) and record["shared"][field] != profile[field]:
            raise DecisionRecordError(f"{field}: YAML has {profile[field]!r}, decision record has "
                                      f"{record['shared'][field]!r}")
    if sorted(arm.arm_id for arm in benchmark.resolved_arms()) != sorted(record.get("arms", [])):
        raise DecisionRecordError(f"arms {[a.arm_id for a in benchmark.resolved_arms()]} != decision "
                                  f"record {record.get('arms')}")
    if list(benchmark.seeds) != list(record.get("seeds", [])):
        raise DecisionRecordError(f"seeds {benchmark.seeds} != decision record {record.get('seeds')}")
    return record


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
    guard_outcome: str = LEGACY_GUARD_OUTCOME
    validation_profile: dict | None = None

    @classmethod
    def from_dict(cls, payload):
        return cls(tuple(PhasePlan(p["task"], tuple(RunPlan(**run) for run in p["runs"]),
                                   p["ppo_timesteps"], p["ppo_seeds"], p["aggregation"],
                                   p.get("timeout_s")) for p in payload["phases"]),
                   payload.get("guard_outcome", LEGACY_GUARD_OUTCOME),
                   payload.get("validation_profile"))

    @classmethod
    def from_benchmark(cls, benchmark: BenchmarkConfig, estimate_path=None):
        if benchmark.profile not in {CAMPAIGN, CAMPAIGN_VALIDATION}:
            raise ValueError("CampaignPlan requires the campaign profile")
        verify_decision_record(benchmark)
        resolved = resolve_configs(benchmark)
        estimate_path = Path(estimate_path) if estimate_path is not None else resolve_estimate_path(benchmark)
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
            priors = [(mean + 1.645 * sd) * E_OVER_P_RATIO for key, (mean, sd) in t_ppo.items()
                      if key.endswith("|" + task)]
            if not priors:
                raise ValueError(f"{estimate_path} has no t_PPO prior for task {task!r}; the PPO-task "
                                 "timeout cannot be derived")
            p95_ecore = max(priors)
            phases.append(PhasePlan(task, tuple(runs), sample.ppo_timesteps,
                                    sample.ppo_seeds, sample.ppo_seed_aggregation,
                                    timeout_s=4 * p95_ecore))
        validation = None
        if benchmark.profile == CAMPAIGN_VALIDATION:
            validation = {
                "stamp": VALIDATION_STAMP,
                "time_scale": {task: LEGACY_VALIDATION_TIME_SCALE.get(task, DEFAULT_VALIDATION_TIME_SCALE)
                               for task in benchmark.tasks},
                "e_over_p_ratio": E_OVER_P_RATIO,
                "duration_distribution": t_ppo,
                "distribution_source": str(estimate_path),
            }
        return cls(tuple(phases), guard_outcome=benchmark.guard_outcome or LEGACY_GUARD_OUTCOME,
                   validation_profile=validation)

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def config_hash(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
