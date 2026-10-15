"""Extension preflight (B4x learning curves, B5x repeats and cost references).

Run on the lab host only after the original campaign has finished, through
the power-capped production placement (see docs/campaign_prep/ext/RUNBOOK_EXT.md):

    sudo scripts/campaign/with_cpu_power_cap.sh --turbo --p-max-mhz 4200 --p-cpus 0-15 \\
      --pl1-w 125 --pl2-w 160 -- .venv/bin/python -m evodevo.extension preflight all \\
      --reference-campaign results/campaign/<ORIGINAL_ID>

Stages (each resumable; completed trainings are never repeated):
  bodies  6 representative bodies (per core arm: the original campaign's
          Walker-v0 best body, and the first valid random body of the arm's
          campaign encoder)
  curves  B4x: each body x each extension task trained once to 2M timesteps,
          fitness recorded every 100k; then D6-ext fixes each task's budget
  noise   B5x: each body x each extension task trained NOISE_REPS times at
          that budget, plus one Walker-v0 (1M) and one UpStepper-v0 (1.2M)
          training per body as cost references
Every completed training is appended (fsync) to <stage>.jsonl immediately.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import spec


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, row: dict) -> None:
    _repair_tail(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repair_tail(path: Path) -> None:
    """A kill mid-write can leave a last line without its newline. Drop it
    (truncate to the last newline) so the next row starts on a clean line; a
    complete row that only lost its newline is kept."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    try:
        json.loads(data[cut:].decode("utf-8"))
    except ValueError:
        with path.open("r+b") as handle:
            handle.truncate(cut)
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[{_now()}] {path.name}: dropped a truncated last line ({len(data) - cut} bytes)", flush=True)
    else:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def _rows(path: Path) -> list:
    """All complete rows. A malformed LAST line without a trailing newline (a
    crash mid-write) is skipped with a notice; a malformed line anywhere else
    raises."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            if number == len(lines) and not text.endswith("\n"):
                print(f"[{_now()}] {path.name}: skipping a truncated last line (line {number})", flush=True)
                continue
            raise ValueError(f"{path}: malformed line {number}: {exc}") from exc
    return rows


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=spec.REPO_ROOT, text=True).strip()


def run_label(commits: list, params: "Params", previous_label: str | None) -> str:
    """PRODUCTION only for production parameters, a run never labelled TEST,
    no stage run on a dirty tree, and a single git commit across all stages."""
    if previous_label == "TEST" or not params.is_production or not commits:
        return "TEST"
    if any(entry.get("dirty") for entry in commits):
        return "TEST"
    if len({entry.get("git_commit") for entry in commits}) != 1:
        return "TEST"
    return "PRODUCTION"


@dataclasses.dataclass
class Params:
    """Production values come from spec; tests may shrink them. Any run with
    non-production values is labelled TEST and refused by `derive` unless
    explicitly allowed."""
    curve_max_timesteps: int
    curve_eval_interval: int
    noise_reps: int
    reference_timesteps: dict
    budget_override: int | None = None

    @classmethod
    def production(cls):
        from .rules import CURVE_EVAL_INTERVAL, CURVE_MAX_TIMESTEPS
        return cls(CURVE_MAX_TIMESTEPS, CURVE_EVAL_INTERVAL, spec.NOISE_REPS, dict(spec.REFERENCE_TASKS))

    @property
    def is_production(self) -> bool:
        return self == Params.production()


# --- bodies -------------------------------------------------------------------

def campaign_encoder(arm: str, task: str):
    """The exact encoder the extension campaign's runs of ``arm`` will use."""
    from evodevo.campaign.plan import RunPlan
    from evodevo.campaign.scheduler import encoder_and_config_for
    from evodevo.core.config import ExperimentConfig
    from evodevo.experiments.benchmark import BenchmarkConfig, _resolved_arm_kwargs
    bconfig = BenchmarkConfig.from_yaml(str(spec.CAMPAIGN_YAML))
    arm_spec = next(a for a in bconfig.resolved_arms() if a.arm_id == arm)
    cfg = ExperimentConfig.from_method_yaml(bconfig.config_path(arm_spec.encoder, task))
    config = cfg.to_dict()
    config["campaign_encoder_kwargs"] = {**dict(bconfig.factors), **_resolved_arm_kwargs(bconfig, arm_spec)}
    return encoder_and_config_for(RunPlan(arm, task, int(cfg.seed), config))[0]


def build_bodies(run_dir: Path, reference_campaign: Path) -> dict:
    import numpy as np
    from evodevo.morphology.metrics import body_hash
    from evodevo.morphology.validate import validate_body
    path = run_dir / "bodies.json"
    if path.is_file():
        return json.loads(path.read_text())
    bodies = {}
    best_dir = reference_campaign / "milestones" / "Walker-v0" / "best_bodies"
    for index, arm in enumerate(spec.ARMS):
        npz = best_dir / f"{arm}.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"{npz} missing: the original campaign's Walker-v0 milestone package "
                                    "must exist before the extension preflight")
        data = np.load(npz)
        body = np.asarray(data["body"], dtype=int)
        if not validate_body(body).valid:
            raise ValueError(f"{npz}: the Walker-v0 best body is not structurally valid")
        bodies[spec.body_label(arm, "evolved")] = {
            "arm": arm, "kind": "evolved", "body": body.tolist(), "body_hash": body_hash(body),
            "source": str(npz), "walker_fitness": float(data["fitness"])}
        encoder = campaign_encoder(arm, spec.TASK_ORDER[0])
        for attempt in range(spec.MAX_RANDOM_ATTEMPTS):
            rng = np.random.default_rng([spec.RANDOM_BODY_SEED, index, attempt])
            genome = encoder.random_genome(rng)
            candidate = np.asarray(encoder.decode(genome), dtype=int)
            if validate_body(candidate).valid:
                bodies[spec.body_label(arm, "random")] = {
                    "arm": arm, "kind": "random", "body": candidate.tolist(),
                    "body_hash": body_hash(candidate), "attempt": attempt,
                    "rng_seed": [spec.RANDOM_BODY_SEED, index, attempt],
                    "genome": encoder.serialize_genome(genome)}
                break
        else:
            raise RuntimeError(f"no valid random body for {arm} in {spec.MAX_RANDOM_ATTEMPTS} attempts")
    payload = {"created_at": _now(), "git_commit": _git("rev-parse", "HEAD"),
               "reference_campaign": str(reference_campaign), "bodies": bodies}
    _write_json(path, payload)
    return payload


# --- pool ---------------------------------------------------------------------

def make_pool(workers: int, cpu_profile: str):
    from evodevo.campaign.production import PROFILE, placement
    from evodevo.parallel.topology import cpu_slots, detect_topology
    from evodevo.parallel.worker_pool import EvaluationPool
    topology = detect_topology()
    if cpu_profile == PROFILE:
        affinity, _scheduler = placement(topology, workers)
    else:
        affinity = cpu_slots(topology, workers)
    pool = EvaluationPool(affinity.n_workers, affinity, "real", max_retries=1, ignore_sigint=True)
    return pool, topology, affinity


STOP = {"requested": False}
STOP_GRACE_S = 30.0


def _request_stop(signum, _frame):
    if not STOP["requested"]:
        print(f"[{_now()}] signal {signum}: stopping; in-flight trainings get {STOP_GRACE_S:.0f} s, "
              "then workers are terminated. Re-run the same command to resume.", flush=True)
    STOP["requested"] = True


def _terminate_pool(pool) -> None:
    executor = getattr(pool, "_executor", None)
    if executor is not None:
        for process in list(executor._processes.values()):
            if process.is_alive():
                process.terminate()
    pool.shutdown(wait=False)


def run_tasks(pool, items: list, out_path: Path, timeout_s: float, topology, stage: str) -> int:
    """items: [(key, meta, PPOTask)]. Skips keys already in out_path. Stops
    cleanly (completed rows kept, workers terminated) on SIGINT/SIGTERM."""
    done = {row["key"] for row in _rows(out_path)}
    todo = [(key, meta, task) for key, meta, task in items if key not in done]
    if not todo:
        return 0
    kinds = {core.logical_id: core.kind for core in topology.cores}
    by_key = {}
    for key, meta, task in todo:
        task.run_key = key
        by_key[key] = meta
        pool.submit_ppo_task(task, timeout_s=timeout_s)
    finished = 0
    print(f"[{_now()}] {stage}: {len(todo)} trainings submitted ({len(done)} already complete)", flush=True)
    while True:
        completed = pool.next_ppo_result(should_stop=lambda: STOP["requested"], grace_s=STOP_GRACE_S)
        if completed is None:
            break
        task, result = completed
        meta = by_key[task.run_key]
        core = kinds.get(result.cpu_ids[0], "unknown") if result.cpu_ids else "unknown"
        row = {"key": task.run_key, **meta, "seed": task.seed, "timesteps": task.timesteps,
               "completed": bool(result.completed), "fitness": result.fitness, "reason": result.reason,
               "wall_clock_s": result.wall_clock_s, "cpu_ids": result.cpu_ids, "core_kind": core,
               "started_at": result.started_at, "finished_at": result.finished_at,
               "curve": result.curve, "error": result.error}
        _append(out_path, row)
        finished += 1
        print(f"[{_now()}] {stage}: {finished}/{len(todo)} {task.run_key} "
              f"{'ok' if result.completed else 'FAILED'} fitness={result.fitness} "
              f"{result.wall_clock_s / 3600:.2f} h on {core}", flush=True)
    return finished


def _ppo_task(key, task_name, body, seed, timesteps, curve_interval=None):
    import numpy as np
    from evogym import get_full_connectivity
    from evodevo.evaluation.ppo_runner import PPOTask
    body = np.asarray(body, dtype=int)
    return PPOTask(key, "extension_preflight", task_name, int(seed), 0, 0, 0, 0, body,
                   get_full_connectivity(body), int(timesteps), spec.campaign_eval_episodes(), None,
                   curve_eval_interval=curve_interval)


# --- stages -------------------------------------------------------------------

def curve_items(bodies: dict, params: Params) -> list:
    items = []
    for task_name in spec.TASK_ORDER:
        for label, body in bodies["bodies"].items():
            key = f"curve|{task_name}|{label}"
            items.append((key, {"stage": "curves", "task": task_name, "body": label},
                          _ppo_task(key, task_name, body["body"], spec.CURVE_SEED,
                                    params.curve_max_timesteps, params.curve_eval_interval)))
    return items


def noise_items(bodies: dict, budgets: dict, params: Params) -> list:
    from evodevo.evaluation.ppo_runner import ppo_seed
    items = []
    labels = list(bodies["bodies"])
    # Interleave so every task, body and repeat meets both core classes.
    for rep in range(params.noise_reps):
        for task_name in spec.TASK_ORDER:
            for label in labels:
                key = f"noise|{task_name}|{label}|r{rep}"
                items.append((key, {"stage": "noise", "task": task_name, "body": label, "rep": rep},
                              _ppo_task(key, task_name, bodies["bodies"][label]["body"],
                                        ppo_seed(spec.NOISE_SEED_BASE, rep), budgets[task_name])))
    for task_name, timesteps in params.reference_timesteps.items():
        for label in labels:
            key = f"reference|{task_name}|{label}"
            items.append((key, {"stage": "reference", "task": task_name, "body": label, "rep": 0},
                          _ppo_task(key, task_name, bodies["bodies"][label]["body"],
                                    ppo_seed(spec.NOISE_SEED_BASE, 0), timesteps)))
    return items


def budgets_from_curves(run_dir: Path, params: Params) -> dict:
    """D6-ext on the recorded curves. A curve-less (failed) training is excluded,
    reported in a WARNING, and recorded in the decision's details."""
    from .rules import decide_budget
    rows = _rows(run_dir / "curves.jsonl")
    decisions = {}
    for task_name in spec.TASK_ORDER:
        curves = {row["body"]: row["curve"] or [] for row in rows
                  if row["task"] == task_name and row["completed"]}
        failed = [f"{row['key']}: {row.get('reason') or row.get('error') or 'unknown'}"
                  for row in rows if row["task"] == task_name and not row["completed"]]
        if failed:
            print(f"[{_now()}] WARNING: {task_name}: {len(failed)} failed curve training(s) excluded from "
                  f"D6-ext: {'; '.join(failed)}", flush=True)
        if params.budget_override is not None:
            decision = {"ppo_timesteps": params.budget_override, "branch": "TEST_override",
                        "details": {"n_bodies_measured": len(curves), "n_bodies_used": len(curves)}}
        else:
            decision = dataclasses.asdict(decide_budget(task_name, curves))
        used = decision["details"].get("n_bodies_used", len(curves))
        decision["details"]["failed_trainings"] = failed
        decision["details"]["reduced_body_set"] = used < spec.EXPECTED_BODIES
        decisions[task_name] = decision
    _write_json(run_dir / "budgets.json", {"created_at": _now(), "params": dataclasses.asdict(params),
                                           "decisions": decisions})
    return {task: d["ppo_timesteps"] for task, d in decisions.items()}


def status(run_dir: Path) -> str:
    lines = [f"extension preflight {run_dir}"]
    for name in ("bodies.json", "curves.jsonl", "budgets.json", "noise.jsonl"):
        path = run_dir / name
        if not path.exists():
            lines.append(f"  {name}: missing")
        elif name.endswith(".jsonl"):
            rows = _rows(path)
            failed = sum(not r["completed"] for r in rows)
            lines.append(f"  {name}: {len(rows)} trainings ({failed} failed)")
        else:
            lines.append(f"  {name}: present")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evodevo.extension preflight", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("bodies", "curves", "noise", "all", "status"))
    parser.add_argument("--run-id", default="current",
                        help="results/extension_preflight/<run-id> (default: current)")
    parser.add_argument("--reference-campaign", help="results/campaign/<ORIGINAL_ID> (required for bodies)")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--cpu-profile", choices=("thermal-validated", "standard"), default="thermal-validated")
    parser.add_argument("--allow-dirty", action="store_true", help="TEST ONLY")
    # Test-only shrinkage; any use marks the run TEST.
    parser.add_argument("--test-curve-max-timesteps", type=int)
    parser.add_argument("--test-curve-eval-interval", type=int)
    parser.add_argument("--test-noise-reps", type=int)
    parser.add_argument("--test-reference-timesteps", type=int)
    parser.add_argument("--test-budget-override", type=int)
    args = parser.parse_args(argv)

    run_dir = spec.PREFLIGHT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "status":
        print(status(run_dir))
        return 0
    params = Params.production()
    if args.test_curve_max_timesteps:
        params.curve_max_timesteps = args.test_curve_max_timesteps
    if args.test_curve_eval_interval:
        params.curve_eval_interval = args.test_curve_eval_interval
    if args.test_noise_reps:
        params.noise_reps = args.test_noise_reps
    if args.test_reference_timesteps:
        params.reference_timesteps = {t: args.test_reference_timesteps for t in params.reference_timesteps}
    if args.test_budget_override:
        params.budget_override = args.test_budget_override
    manifest_path = run_dir / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    from evodevo.preflight.dirty_check import dirty_paths_outside_results
    dirty = dirty_paths_outside_results(str(spec.REPO_ROOT))
    if dirty and not args.allow_dirty:
        parser.error("refusing a dirty working tree (commit first)")
    recorded = manifest.get("params")
    if recorded and recorded != dataclasses.asdict(params):
        parser.error(f"run {run_dir} was started with different parameters {recorded}")
    manifest.setdefault("git_commits", []).append({"git_commit": _git("rev-parse", "HEAD"), "at": _now(),
                                                   "stage": args.stage, "dirty": bool(dirty)})
    manifest.update({"params": dataclasses.asdict(params), "production_params": params.is_production,
                     "label": run_label(manifest["git_commits"], params, manifest.get("label")),
                     "cpu_profile": args.cpu_profile, "workers": args.workers})
    _write_json(manifest_path, manifest)

    stages = ("bodies", "curves", "noise") if args.stage == "all" else (args.stage,)
    reference = Path(args.reference_campaign) if args.reference_campaign else None
    if "bodies" in stages or not (run_dir / "bodies.json").is_file():
        if reference is None:
            parser.error("--reference-campaign is required to build the representative bodies")
        bodies = build_bodies(run_dir, reference)
    else:
        bodies = json.loads((run_dir / "bodies.json").read_text())
    if stages == ("bodies",):
        print(status(run_dir))
        return 0
    import signal
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    pool, topology, _affinity = make_pool(args.workers, args.cpu_profile)
    try:
        if "curves" in stages:
            run_tasks(pool, curve_items(bodies, params), run_dir / "curves.jsonl",
                      spec.CURVE_TIMEOUT_S, topology, "curves")
            if STOP["requested"]:
                print(status(run_dir))
                return 130
            budgets = budgets_from_curves(run_dir, params)
            print(f"[{_now()}] D6-ext budgets: {budgets}", flush=True)
        if "noise" in stages:
            if not (run_dir / "budgets.json").is_file():
                parser.error("run the curves stage first")
            budgets = {task: d["ppo_timesteps"] for task, d in
                       json.loads((run_dir / "budgets.json").read_text())["decisions"].items()}
            run_tasks(pool, noise_items(bodies, budgets, params), run_dir / "noise.jsonl",
                      spec.NOISE_TIMEOUT_S, topology, "noise")
            if STOP["requested"]:
                print(status(run_dir))
                return 130
    finally:
        if STOP["requested"]:
            _terminate_pool(pool)
        else:
            pool.shutdown(wait=True)
    print(status(run_dir))
    return 0
