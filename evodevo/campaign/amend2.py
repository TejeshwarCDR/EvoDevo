"""Amendment 2 machinery: champion re-training stages and the launch checks.

Amendment 2 (docs/campaign_prep/ext/PREREGISTRATION_EXT.md) runs one campaign
of evolutionary phases keyed by (task, seed block) and, between and after
them, *re-training stages*: every run's champion body is trained again from
scratch with fresh PPO seeds (``reeval``, at the task's own budget) and, at the
end, at twice the budget (``robustness``). A re-training is an ordinary
``PPOTask`` on the same pinned workers; nothing here touches how a genome is
evaluated.

Module scope is stdlib only; numpy/evogym/PPO are imported inside the
functions that need them, so the launcher's cheap refusals stay cheap.
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from . import units as unit_keys

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs/campaign_prep/ext"
RECORD_FORMAT = "evodevo-a2-decisions-v1"
PRODUCTION, TEST = "PRODUCTION", "TEST"

CORE_ARMS = ("direct", "nca", "rule_based")
MONOTONE_ARM = "rule_based_monotone"
ORIGINAL_CAMPAIGN_PATH = "results/campaign/20260930T044002Z"

# Seed bases of the two re-training families. Every training's PPO seed is
# ppo_seed(base, k) with k = 0..n-1 (k = 0 is the base itself).
SEED_BASE_REEVAL = 20261101
SEED_BASE_ROBUSTNESS = 20261201
REEVAL_PER_CHAMPION = 10
ROBUSTNESS_PER_CHAMPION = 3
ROBUSTNESS_FACTOR = 2

OUTPUT_FILES = {unit_keys.REEVAL: "reeval.jsonl", unit_keys.ROBUSTNESS: "robustness.jsonl"}
SUPERSEDED_RECORD = DOCS_DIR / "SUPERSEDED_BY_AMENDMENT_2.json"

# Library versions of the original campaign (Gate R, R3: SB3 system_info.txt
# inside the campaign's policy zips, the thermal-study machine profile and
# ENVIRONMENT_REPORT.md). A launch refuses if any installed version differs.
RECORDED_VERSIONS = {
    "python": "3.10.20", "torch": "2.14.0+cpu", "stable-baselines3": "2.9.0", "gymnasium": "1.3.0",
    "numpy": "1.26.4", "cloudpickle": "3.1.2", "evogym": "2.0.0",
    "evogym_commit": "f63c649c688adda1accad5f86731aad96606388a",
}


def retrain_seed(base: int, k: int) -> int:
    from evodevo.evaluation.ppo_runner import ppo_seed
    return ppo_seed(base, k)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- champions ---------------------------------------------------------------

class ChampionError(RuntimeError):
    """A run has no champion that can be re-trained (incomplete or all-invalid)."""


@dataclasses.dataclass(frozen=True)
class Champion:
    task: str
    arm: str
    seed: int
    eval_index: int
    fitness: float
    body: tuple
    body_hash: str
    source_campaign: str
    source_commit: str
    source_kind: str   # "campaign" (this campaign) or "original" (read-only)
    phase_id: str
    valid: bool = True
    invalid_reason: str | None = None

    @property
    def run_key(self) -> str:
        return f"{self.arm}/{self.task}/seed_{self.seed}"


def select_champion(rows: list) -> dict:
    """The milestone definition: the evaluation with the maximum recorded
    fitness, ties to the lowest evaluation index."""
    ordered = sorted(rows, key=lambda row: row["eval_index"])
    best = None
    for row in ordered:
        if best is None or row["fitness"] > best["fitness"]:
            best = row
    return best


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _champion_from_run(run_dir: Path, task: str, arm: str, seed: int, expected_evals: int,
                       campaign: str, commit: str, kind: str, phase_id: str) -> Champion:
    rows = read_jsonl(run_dir / "run_log.jsonl")
    label = f"{arm}/{task}/seed_{seed} in {run_dir}"
    if len(rows) != expected_evals:
        raise ChampionError(f"{label}: {len(rows)} logged evaluations, expected {expected_evals}; "
                            "a champion is only read from a completed run")
    best = select_champion(rows)
    if best.get("body") is None:
        raise ChampionError(f"{label}: best recorded evaluation {best['eval_index']} has no recorded body")
    valid = bool(best.get("valid"))
    return Champion(task, arm, seed, int(best["eval_index"]), float(best["fitness"]),
                    tuple(tuple(int(v) for v in row) for row in best["body"]), best["body_hash"],
                    campaign, commit, kind, phase_id, valid, None if valid else best.get("reason"))


def resolve_repo_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def original_manifest(spec, repo_root: Path | None = None) -> dict:
    """Manifest of the original campaign; ``spec`` is its path or the plan's
    ``original_campaign`` dict."""
    raw = spec["path"] if isinstance(spec, dict) else spec
    path = resolve_repo_path(raw) if repo_root is None else (
        Path(raw) if Path(raw).is_absolute() else repo_root / raw)
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def champions_for_stage(stage: dict, plan: dict, root, repo_root: Path | None = None) -> list:
    """Champions of ``stage`` in a deterministic order: source order, then the
    plan's arm order, then seed. ``plan`` is the plan's dict form."""
    root = Path(root)
    spec = plan["amendment2"]
    phases = {unit_keys.phase_key(p): p for p in plan["phases"]}
    manifest_path = root / "manifest.json"
    own = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    own_commit = own.get("git_commit", "unknown")
    campaign_id = own.get("campaign_id", root.name)
    out = []
    for source in stage["sources"]:
        if "phase" in source:
            phase = phases[source["phase"]]
            arms = list(dict.fromkeys(r["arm"] for r in phase["runs"]))
            budgets = {(r["arm"], r["seed"]): int(r["config"]["eval_budget"]) for r in phase["runs"]}
            for arm in arms:
                for seed in sorted({r["seed"] for r in phase["runs"] if r["arm"] == arm}):
                    run_dir = root / "runs" / arm / phase["task"] / f"seed_{seed}"
                    out.append(_champion_from_run(run_dir, phase["task"], arm, seed, budgets[(arm, seed)],
                                                  campaign_id, own_commit, "campaign", source["phase"]))
        else:
            original = spec["original_campaign"]
            base = resolve_repo_path(original["path"]) if repo_root is None else repo_root / original["path"]
            reference = original_manifest(original, repo_root)
            budgets = {(run["arm"], phase["task"], run["seed"]): int(run["config"]["eval_budget"])
                       for phase in reference["plan"]["phases"] for run in phase["runs"]}
            for arm in source["arms"]:
                for seed in source["seeds"]:
                    run_dir = base / "runs" / arm / stage["task"] / f"seed_{seed}"
                    out.append(_champion_from_run(run_dir, stage["task"], arm, seed,
                                                  budgets[(arm, stage["task"], seed)], original["id"],
                                                  original["git_commit"], "original", "original"))
    return out


# --- paths -------------------------------------------------------------------

def retrain_artifact_dir(root, stage_id: str, task: str, arm: str, seed: int, eval_index: int) -> str:
    """Directory of one champion's trainings in one stage. The worker writes
    ``policies/e<eval>_k<k>.zip`` and ``training_logs/e<eval>_k<k>.json`` below
    it, so a file path is unique per (stage, task, arm, seed, eval index, k)."""
    return os.path.join(str(root), "retrain", unit_keys.slug(stage_id), task, arm, f"seed_{seed}",
                        f"e{eval_index:05d}")


def retrain_file_paths(artifact_dir: str, eval_index: int, k: int) -> tuple:
    stem = f"e{eval_index:05d}_k{k}"
    return (os.path.join(artifact_dir, "policies", stem + ".zip"),
            os.path.join(artifact_dir, "training_logs", stem + ".json"))


def stage_run_key(stage_id: str, champion: Champion) -> str:
    return f"{stage_id}|{champion.run_key}"


def stage_trainings(stage: dict, champions: list, root, eval_episodes: int = 1) -> list:
    """Every (champion, k) training of a stage as (champion, k, ppo_seed,
    PPOTask), k-major so an early stop leaves every champion with some."""
    import numpy as np
    from evodevo.evaluation.ppo_runner import PPOTask
    from evogym import get_full_connectivity
    items = []
    for k in range(stage["n"]):
        seed = retrain_seed(stage["seed_base"], k)
        for champion in champions:
            # An all-invalid evolutionary run is a real sentinel-dominated
            # outcome under the preregistered invalid-genome policy.  It has
            # no structurally valid body that PPO can safely train.  Keep a
            # placeholder in the planned sequence so run_stage can journal a
            # durable sentinel outcome without dispatching it to EvoGym.
            if not champion.valid:
                items.append((champion, k, seed, None))
                continue
            body = np.array(champion.body, dtype=int)
            directory = retrain_artifact_dir(root, stage["stage_id"], champion.task, champion.arm,
                                             champion.seed, champion.eval_index)
            task = PPOTask(stage_run_key(stage["stage_id"], champion), champion.arm, champion.task, seed,
                           0, champion.eval_index, champion.eval_index, k, body,
                           get_full_connectivity(body), int(stage["budget"]), eval_episodes, directory)
            items.append((champion, k, seed, task))
    return items


def planned_trainings(stage: dict, plan: dict, repo_root: Path | None = None) -> int:
    """Training count of a stage from the plan alone (no run logs needed)."""
    return stage["n"] * champion_count(stage, plan)


def champion_count(stage: dict, plan: dict) -> int:
    phases = {unit_keys.phase_key(p): p for p in plan["phases"]}
    count = 0
    for source in stage["sources"]:
        if "phase" in source:
            count += len(phases[source["phase"]]["runs"])
        else:
            count += len(source["arms"]) * len(source["seeds"])
    return count


# --- output rows ---------------------------------------------------------------

def result_row(campaign_id: str, stage: dict, champion: Champion, k: int, seed: int, task, result,
               core_kinds: dict) -> dict:
    cpu_ids = list(result.cpu_ids or [])
    kind = core_kinds.get(cpu_ids[0], "unknown") if cpu_ids else "unknown"
    return {"campaign_id": campaign_id, "stage": stage["stage_id"], "kind": stage["kind"],
            "run_key": champion.run_key, "task": champion.task, "arm": champion.arm, "seed": champion.seed,
            "source_campaign": champion.source_campaign, "source_commit": champion.source_commit,
            "source_kind": champion.source_kind, "phase_id": champion.phase_id,
            "champion_eval_index": champion.eval_index, "champion_fitness": champion.fitness,
            "body_hash": champion.body_hash, "budget": int(stage["budget"]), "k": k, "ppo_seed": seed,
            "champion_valid": champion.valid, "training_attempted": task is not None,
            "fitness": result.fitness, "completed": bool(result.completed), "reason": result.reason,
            "wall_time_s": result.wall_clock_s, "core_kind": kind, "cpu_ids": cpu_ids,
            "started_at": result.started_at, "finished_at": result.finished_at,
            "completed_at": time.time(), "policy_path": result.policy_path,
            "training_log_path": result.training_log_path}


def row_key(row: dict) -> tuple:
    return (row["stage"], row["run_key"], row["champion_eval_index"], row["k"])


def settled(row: dict) -> bool:
    """A row that need not be run again on resume: it completed, or failed in
    the training itself. Infrastructure failures are retried."""
    return bool(row.get("completed")) or not str(row.get("reason") or "").startswith("infra_failure")


def append_row(path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_rows(path) -> dict:
    """{row_key: last row} — a retried infra failure is superseded by its retry."""
    return {row_key(row): row for row in read_jsonl(path)}


def run_stage(root, campaign_id: str, plan: dict, stage: dict, pool, *, workers: int, prefetch: int = 4,
              stop_requested=lambda: False, grace_s: float = 120, core_kinds: dict | None = None,
              eval_episodes: int = 1, on_progress=None, repo_root: Path | None = None) -> dict:
    """Run (or resume) one re-training stage; returns its metrics.

    Idempotent: a training whose row is already settled in the output file is
    never run again. ``pool`` is an EvaluationPool (or anything with its
    submit_ppo_task/next_ppo_result); ``None`` runs the surrogate evaluator
    inline (tests and --mode surrogate)."""
    root = Path(root)
    core_kinds = core_kinds or {}
    output = root / OUTPUT_FILES[stage["kind"]]
    champions = champions_for_stage(stage, plan, root, repo_root)
    done = {key for key, row in latest_rows(output).items() if settled(row)}
    todo = [item for item in stage_trainings(stage, champions, root, eval_episodes)
            if (stage["stage_id"], item[0].run_key, item[0].eval_index, item[1]) not in done]
    started = time.time()
    busy = 0.0
    finished = 0
    stopped = False
    window = workers + prefetch
    pending = collections.deque(todo)
    inflight = {}

    def record(champion, k, seed, task, result):
        nonlocal busy, finished
        append_row(output, result_row(campaign_id, stage, champion, k, seed, task, result, core_kinds))
        if result.started_at is not None and result.finished_at is not None:
            busy += result.finished_at - result.started_at
        finished += 1
        if on_progress:
            on_progress(len(inflight), len(pending))

    def record_invalid_champion(champion, k, seed):
        """Settle an impossible PPO training without hiding the failed seed.

        The sentinel fitness remains in the secondary endpoint, matching the
        campaign's invalid-genome policy, while ``training_attempted`` and the
        reason make clear that no policy or training log can exist.
        """
        from evodevo.evaluation.ppo_runner import PPOTaskResult
        now = time.time()
        reason = "invalid_champion: no structurally valid body; PPO training not attempted"
        if champion.invalid_reason:
            reason += f" ({champion.invalid_reason})"
        record(champion, k, seed, None, PPOTaskResult(
            champion.fitness, False, reason, 0.0, None, None, [], None, None, None, None,
            started_at=now, finished_at=now))

    if pool is None:
        from evodevo.evaluation.ppo_runner import PPOTaskResult, evaluate_body_surrogate
        import numpy as np
        while pending and not stop_requested():
            champion, k, seed, task = pending.popleft()
            if task is None:
                record_invalid_champion(champion, k, seed)
                continue
            surrogate = evaluate_body_surrogate(np.array(champion.body, dtype=int), task=champion.task, seed=seed)
            now = time.time()
            record(champion, k, seed, task, PPOTaskResult(
                float(surrogate.fitness), True, None, 0.0, None, 0, [0], None, None, None, None,
                started_at=now, finished_at=now))
        stopped = bool(pending)
    else:
        while pending or inflight:
            while pending and len(inflight) < window and not stop_requested():
                item = pending.popleft()
                if item[3] is None:
                    record_invalid_champion(item[0], item[1], item[2])
                    continue
                key = (item[3].run_key, item[3].eval_index, item[3].ppo_k)
                inflight[key] = item
                pool.submit_ppo_task(item[3], timeout_s=stage["timeout_s"])
            if not inflight:
                stopped = bool(pending)
                break
            completed = pool.next_ppo_result(should_stop=stop_requested, grace_s=grace_s)
            if completed is None:
                stopped = True
                break
            task, result = completed
            champion, k, seed, _task = inflight.pop((task.run_key, task.eval_index, task.ppo_k))
            record(champion, k, seed, task, result)
    ended = time.time()
    makespan = max(0.0, ended - started)
    return {"status": "stopped" if stopped else "complete", "started_at": started, "ended_at": ended,
            "makespan_s": makespan, "worker_count": workers, "worker_busy_s": busy,
            "worker_utilization": busy / (workers * makespan) if makespan and workers else 0.0,
            "completed_timed_tasks": finished, "trainings_planned": len(todo) + len(done),
            "trainings_skipped_as_done": len(done), "kind": stage["kind"]}


# --- launch checks -------------------------------------------------------------

def installed_versions(repo_root: Path | None = None) -> dict:
    """The versions that matter to PPO training results, read without importing them."""
    repo_root = repo_root or REPO_ROOT

    def dist(name):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.check_output(["git", "-C", str(repo_root / "external/evogym"), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "torch": dist("torch"),
            "stable-baselines3": dist("stable-baselines3"), "gymnasium": dist("gymnasium"),
            "numpy": dist("numpy"), "cloudpickle": dist("cloudpickle"), "evogym": dist("evogym"),
            "evogym_commit": commit}


def version_problems(recorded: dict, installed: dict | None = None) -> list:
    installed = installed if installed is not None else installed_versions()
    return [f"{name}: the original campaign used {want!r}, installed is {installed.get(name)!r}"
            for name, want in sorted(recorded.items()) if installed.get(name) != want]


def config_diff(expected: dict, actual: dict, ignore=("seed",)) -> list:
    """Fields (dotted for nested dicts) at which two run configs differ."""
    problems = []
    for key in sorted(set(expected) | set(actual)):
        if key in ignore:
            continue
        a, b = expected.get(key), actual.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            problems += [f"{key}.{sub}" for sub in config_diff(a, b, ignore=())]
        elif a != b:
            problems.append(key)
    return problems


def config_equality_problems(benchmark, plan, original: dict) -> list:
    """Every Walker-v0/UpStepper-v0 phase must train with the original
    campaign's per-run configuration, differing only in the seed.

    Two independent comparisons: the run configs in the plan against the
    original manifest's, and the configs today's code would build for the
    same arms against the original manifest's (a code or config drift since
    the original campaign shows up here)."""
    import dataclasses as dc
    from evodevo.experiments.benchmark import _resolved_arm_kwargs, resolve_configs

    problems = []
    originals = {}
    for phase in original["plan"]["phases"]:
        for run in phase["runs"]:
            originals.setdefault((run["arm"], phase["task"]), run)
    fresh = resolve_configs(benchmark)
    arms = {arm.arm_id: arm for arm in benchmark.resolved_arms()}
    spec = plan.amendment2 or {}
    source = {p["id"]: p.get("config_source") for p in spec.get("phases", [])}
    for phase in plan.phases:
        if source.get(phase.key) != "original":
            continue
        reference_phase = next((p for p in original["plan"]["phases"] if p["task"] == phase.task), None)
        if reference_phase is None:
            problems.append(f"{phase.key}: the original campaign has no phase for task {phase.task}")
            continue
        for field, mine, theirs in (("ppo_timesteps", phase.ppo_timesteps, reference_phase["ppo_timesteps"]),
                                    ("ppo_seeds", phase.ppo_seeds, reference_phase["ppo_seeds"]),
                                    ("aggregation", phase.aggregation, reference_phase["aggregation"]),
                                    ("timeout_s", phase.timeout_s, reference_phase["timeout_s"])):
            if mine != theirs:
                problems.append(f"{phase.key}: phase {field} {mine!r} != original {theirs!r}")
        for run in phase.runs:
            reference = originals.get((run.arm, run.task))
            if reference is None:
                problems.append(f"{run.run_key}: no original run for ({run.arm}, {run.task})")
                continue
            diff = config_diff(reference["config"], run.config)
            if diff:
                problems.append(f"{run.run_key}: plan config differs from the original in {diff}")
            if (run.arm, run.task) in fresh:
                cfg = dc.replace(fresh[(run.arm, run.task)], seed=run.seed).to_dict()
                cfg["campaign_encoder_kwargs"] = {**dict(benchmark.factors),
                                                  **_resolved_arm_kwargs(benchmark, arms[run.arm])}
                cfg["grid_shape"] = list(cfg["grid_shape"])
                diff = config_diff(reference["config"], cfg)
                if diff:
                    problems.append(f"{run.run_key}: today's code builds a config that differs from the "
                                    f"original in {diff}")
    return problems


def launch_problems(args, benchmark, plan, *, repo_root: Path | None = None) -> tuple:
    """-> (refusals, record) for the Amendment 2 launch checks: library
    versions and per-run config equality. Empty for any plan that is not an
    Amendment 2 plan."""
    spec = plan.amendment2
    if not spec:
        return [], {}
    refusals = []
    label = spec.get("label")
    skip = bool(getattr(args, "skip_config_equality_check", False))
    if skip and label != TEST:
        refusals.append("--skip-config-equality-check is only accepted for a TEST-labelled config; "
                        f"this config is labelled {label!r}")
        skip = False
    versions = installed_versions(repo_root)
    version_issues = version_problems(spec["recorded_versions"], versions)
    refusals += [f"library version differs from the original campaign's: {item}" for item in version_issues]
    equality = []
    if skip:
        equality_record = "skipped (TEST-labelled config, explicit flag)"
    else:
        try:
            equality = config_equality_problems(benchmark, plan, original_manifest(spec["original_campaign"],
                                                                                  repo_root))
        except (OSError, ValueError, KeyError) as exc:
            equality = [f"cannot compare with the original campaign: {type(exc).__name__}: {exc}"]
        refusals += [f"config equality: {item}" for item in equality]
        equality_record = "passed" if not equality else "FAILED"
    return refusals, {"label": label, "installed_versions": versions, "version_check": "passed" if not version_issues
                      else "FAILED", "config_equality": equality_record}
