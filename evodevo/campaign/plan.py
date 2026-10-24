"""Declarative, hashable phase plan resolved from the campaign profile."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from evodevo.experiments.benchmark import (CAMPAIGN, CAMPAIGN_VALIDATION,
    BenchmarkConfig, resolve_configs, _resolved_arm_kwargs)
from . import units as unit_keys


VALIDATION_STAMP = "VALIDATION - NOT A RESULT"
REPO_ROOT = Path(__file__).resolve().parents[2]
# Alias of the single definition (the gitignored original-campaign estimate);
# resolve_estimate_path reads ``estimate.LEGACY_ESTIMATE_PATH`` at call time.
from . import estimate as _estimate_module
from .estimate import LEGACY_ESTIMATE_PATH
LEGACY_GUARD_OUTCOME = "all four approved campaign arms"
# Validation-mode sleep scaling. The original two tasks keep their exact
# values; any other task uses the slower UpStepper scale.
LEGACY_VALIDATION_TIME_SCALE = {"Walker-v0": 1 / 100, "UpStepper-v0": 1 / 250}
DEFAULT_VALIDATION_TIME_SCALE = 1 / 250
E_OVER_P_RATIO = 2.3172


class DecisionRecordError(ValueError):
    """The benchmark names a decision record that is missing, not FINAL, or
    disagrees with the YAML's per-task PPO budget or K."""


SUPERSEDED_RECORD = REPO_ROOT / "docs/campaign_prep/ext/SUPERSEDED_BY_AMENDMENT_2.json"


def check_not_superseded(path: Path) -> None:
    """Refuse the 3-task extension's preflight record as a launch decision.

    Amendment 2 retired that design. The record is kept unchanged as the
    preflight evidence; SUPERSEDED_BY_AMENDMENT_2.json holds its sha256, and a
    campaign whose ``decisions_path`` is that file, by path or by content,
    can neither be planned nor launched."""
    try:
        retired = json.loads(SUPERSEDED_RECORD.read_text())
    except (OSError, ValueError):
        return
    same_path = path.resolve() == (REPO_ROOT / retired["superseded_path"]).resolve()
    same_content = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == retired["sha256"]
    if same_path or same_content:
        raise DecisionRecordError(
            f"the 3-task extension campaign plan does not exist: decision record {path} is the preflight "
            f"record that Amendment 2 superseded (see {SUPERSEDED_RECORD.relative_to(REPO_ROOT)}); it is "
            "evidence, not a launch decision. Launch evodevo/experiments/configs/campaign_a2.yaml instead")


def resolve_estimate_path(benchmark) -> Path:
    """The estimate a benchmark's plan is built from (repo-relative paths
    resolve against the repository root, never the current directory)."""
    raw = getattr(benchmark, "estimate_path", None)
    if not raw:
        return _estimate_module.LEGACY_ESTIMATE_PATH
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_decision_record(benchmark, allow_test_record: bool = False,
                           allow_superseded: bool = False) -> dict | None:
    """Refuse a plan whose YAML disagrees with its FINAL decision record, and
    any record derived from a non-PRODUCTION (TEST) preflight run unless the
    caller is a rehearsal that passes ``allow_test_record``."""
    raw = getattr(benchmark, "decisions_path", None)
    if not raw:
        return None
    path = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
    if not allow_superseded:
        check_not_superseded(path)
    if not path.is_file():
        raise DecisionRecordError(f"decision record {path} does not exist; run the extension "
                                  "preflight and decision derivation first")
    record = json.loads(path.read_text())
    if record.get("format") == "evodevo-a2-decisions-v1":
        return verify_a2_record(benchmark, record, path, allow_test_record)
    if record.get("status") != "FINAL":
        raise DecisionRecordError(f"decision record {path} has status {record.get('status')!r}, not 'FINAL'")
    if not allow_test_record and "label" in record and record["label"] != "PRODUCTION":
        raise DecisionRecordError(f"decision record {path} is labelled {record['label']!r}, not 'PRODUCTION'; "
                                  "a record derived from a TEST preflight run can never launch a campaign")
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


def verify_a2_record(benchmark, record: dict, path: Path, allow_test_record: bool) -> dict:
    """Amendment 2 decision record: the YAML's ``amendment2`` block and its
    per-task budgets must be exactly what the record states."""
    if record.get("status") != "FINAL":
        raise DecisionRecordError(f"decision record {path} has status {record.get('status')!r}, not 'FINAL'")
    if not allow_test_record and record.get("label") != "PRODUCTION":
        raise DecisionRecordError(f"decision record {path} is labelled {record.get('label')!r}, not "
                                  "'PRODUCTION'; it can never launch a campaign")
    spec = getattr(benchmark, "amendment2", None)
    if not spec:
        raise DecisionRecordError(f"decision record {path} is an Amendment 2 record but the YAML has no "
                                  "amendment2 block")
    if spec != record.get("campaign_spec"):
        raise DecisionRecordError(f"the YAML's amendment2 block differs from campaign_spec in {path}")
    profile = benchmark.resolved_profile_kwargs()
    if list(benchmark.tasks) != list(record.get("task_order", [])):
        raise DecisionRecordError(f"task order {benchmark.tasks} != decision record {record.get('task_order')}")
    for task in benchmark.tasks:
        decided = record.get("tasks", {}).get(task, {})
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
    # Amendment 2: several phases may share a task (one per seed block), so a
    # phase is identified by its id. None for every legacy plan, where the
    # task is the key; the field is then left out of the hashed plan dict.
    phase_id: str | None = None

    @property
    def key(self) -> str:
        return self.phase_id or self.task


@dataclass(frozen=True)
class StagePlan:
    """An Amendment 2 re-training stage: every champion of ``sources``
    trained ``n`` times at ``budget`` PPO steps (``kind`` reeval or robustness)."""
    stage_id: str
    kind: str
    task: str
    n: int
    seed_base: int
    budget: int
    sources: tuple
    timeout_s: float | None = None


@dataclass(frozen=True)
class CampaignPlan:
    phases: tuple[PhasePlan, ...]
    guard_outcome: str = LEGACY_GUARD_OUTCOME
    validation_profile: dict | None = None
    # Amendment 2 only (empty/None for every legacy plan and then omitted from
    # the hashed dict, so the original campaign's plan hash cannot change).
    stages: tuple[StagePlan, ...] = ()
    amendment2: dict | None = None

    @classmethod
    def from_dict(cls, payload):
        return cls(tuple(PhasePlan(p["task"], tuple(RunPlan(**run) for run in p["runs"]),
                                   p["ppo_timesteps"], p["ppo_seeds"], p["aggregation"],
                                   p.get("timeout_s"), p.get("phase_id")) for p in payload["phases"]),
                   payload.get("guard_outcome", LEGACY_GUARD_OUTCOME),
                   payload.get("validation_profile"),
                   tuple(StagePlan(**{**s, "sources": tuple(s["sources"])}) for s in payload.get("stages", ())),
                   payload.get("amendment2"))

    @classmethod
    def from_benchmark(cls, benchmark: BenchmarkConfig, estimate_path=None, allow_test_record: bool = False,
                       allow_superseded: bool = False):
        if benchmark.profile not in {CAMPAIGN, CAMPAIGN_VALIDATION}:
            raise ValueError("CampaignPlan requires the campaign profile")
        verify_decision_record(benchmark, allow_test_record=allow_test_record, allow_superseded=allow_superseded)
        resolved = resolve_configs(benchmark)
        estimate_path = Path(estimate_path) if estimate_path is not None else resolve_estimate_path(benchmark)
        estimate = json.loads(estimate_path.read_text())
        if getattr(benchmark, "amendment2", None):
            return cls._from_amendment2(benchmark, resolved, estimate, estimate_path, allow_test_record)
        if _estimate_module.is_v2(estimate):
            for task in benchmark.tasks:
                if task not in estimate.get("result", {}).get("phase_h", {}):
                    raise ValueError(f"{estimate_path} has no phase band for task {task!r}; the campaign "
                                     "dashboard and calendar cannot be built without one")
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

    @classmethod
    def _from_amendment2(cls, benchmark, resolved, estimate, estimate_path, allow_test_record):
        """Phases are (task, seed block) with their own configs, plus the
        re-training stages, in the strict order ``amendment2.sequence``."""
        from . import amend2
        spec = benchmark.amendment2
        label = spec.get("label")
        if label not in {amend2.PRODUCTION, amend2.TEST}:
            raise DecisionRecordError(f"amendment2.label must be PRODUCTION or TEST, got {label!r}")
        if label == amend2.PRODUCTION and not benchmark.decisions_path:
            raise DecisionRecordError("a PRODUCTION Amendment 2 config must name its decision record "
                                      "(decisions_path)")
        if label == amend2.TEST and benchmark.decisions_path:
            raise DecisionRecordError("a TEST-labelled Amendment 2 config cannot name a decision record")
        a2_estimate = estimate.get("a2") or {}
        timeouts = a2_estimate.get("timeout_s", {})
        bands = estimate.get("result", {}).get("phase_h", {})
        original = amend2.original_manifest(spec["original_campaign"])
        arms = benchmark.resolved_arms()
        arm_by_id = {arm.arm_id: arm for arm in arms}
        phases, spec_phases = [], []
        for item in spec["phases"]:
            task = item["task"]
            if item["id"] not in timeouts or item["id"] not in bands:
                raise ValueError(f"{estimate_path} has no timeout and phase band for {item['id']!r}")
            runs = []
            for arm_id in item.get("arms") or [arm.arm_id for arm in arms]:
                for seed in item["seeds"]:
                    if item["config_source"] == "original":
                        reference = next((r for ph in original["plan"]["phases"] if ph["task"] == task
                                          for r in ph["runs"] if r["arm"] == arm_id), None)
                        if reference is None:
                            raise ValueError(f"the original campaign has no run for ({arm_id}, {task})")
                        config = copy.deepcopy(reference["config"])
                        config["seed"] = seed
                    else:
                        config = dataclasses.replace(resolved[(arm_id, task)], seed=seed).to_dict()
                        config["campaign_encoder_kwargs"] = {
                            **dict(benchmark.factors), **_resolved_arm_kwargs(benchmark, arm_by_id[arm_id])}
                    runs.append(RunPlan(arm_id, task, seed, config))
            sample = resolved[(runs[0].arm, task)]
            phases.append(PhasePlan(task, tuple(runs), sample.ppo_timesteps, sample.ppo_seeds,
                                    sample.ppo_seed_aggregation, timeout_s=timeouts[item["id"]],
                                    phase_id=item["id"]))
            spec_phases.append(item)
        stages = []
        for item in spec["stages"]:
            if item["id"] not in timeouts or item["id"] not in bands:
                raise ValueError(f"{estimate_path} has no timeout and phase band for {item['id']!r}")
            stages.append(StagePlan(item["id"], item["kind"], item["task"], int(item["n"]),
                                    int(item["seed_base"]), int(item["budget"]),
                                    tuple(copy.deepcopy(item["sources"])), timeouts[item["id"]]))
        known = {p.key for p in phases} | {s.stage_id for s in stages}
        if sorted(spec["sequence"]) != sorted(known) or len(set(spec["sequence"])) != len(spec["sequence"]):
            raise ValueError("amendment2.sequence must list every phase and stage exactly once")
        original_campaign = {"path": spec["original_campaign"], "id": original["campaign_id"],
                             "git_commit": original["git_commit"], "config_hash": original["config_hash"]}
        amendment = {"label": label, "original_campaign": original_campaign, "sequence": list(spec["sequence"]),
                     "phases": copy.deepcopy(spec["phases"]), "recorded_versions": dict(spec["recorded_versions"])}
        return cls(tuple(phases), guard_outcome=benchmark.guard_outcome or LEGACY_GUARD_OUTCOME,
                   stages=tuple(stages), amendment2=amendment)

    def to_dict(self):
        payload = dataclasses.asdict(self)
        # Fields added after the original campaign are omitted while unused,
        # so a legacy plan hashes exactly as it did before they existed.
        for phase in payload["phases"]:
            if phase.get("phase_id") is None:
                phase.pop("phase_id", None)
        if not payload.get("stages"):
            payload.pop("stages", None)
        if payload.get("amendment2") is None:
            payload.pop("amendment2", None)
        return payload

    def units(self) -> list:
        """Phases and stages in execution order (legacy: the phases)."""
        if not self.amendment2:
            return list(self.phases)
        by_key = {p.key: p for p in self.phases}
        by_key.update({s.stage_id: s for s in self.stages})
        return [by_key[key] for key in self.amendment2["sequence"]]

    @property
    def config_hash(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
