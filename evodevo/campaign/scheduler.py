"""Generation-barrier campaign scheduler.

The scheduler intentionally never lets a later generation enter dispatch
until its owning EvolutionRun has received every result from the prior one.
That is the determinism boundary: completion order affects throughput only.
"""
from __future__ import annotations

import dataclasses
import copy
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from bisect import bisect_right
from collections import deque

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import (EvalResult, PPOTask, PPOTaskResult,
                                            aggregate_seed_results, evaluate_body_surrogate,
                                            evaluate_ppo_task, ppo_seed,
                                            validation_task_metadata)
from evodevo.evolution.ea import EvolutionRun
from evodevo.evolution.logging import RunLogger
from evodevo.parallel.affinity import pin_current_process
from evodevo.parallel.topology import allocation_order, cpu_slots, detect_topology
from evodevo.parallel.worker_pool import EvaluationPool
from evodevo.morphology.metrics import body_hash as compute_body_hash
from evodevo.morphology.validate import validate_body

from .checkpoint import latest_checkpoint, write_checkpoint
from .dispatch import DispatchCore, DispatchItem
from .alerts import AlertManager
from .journal import ResultJournal
from .milestone import run_subprocess
from .mirror import sync
from .plan import CampaignPlan


def genome_hash(encoder, genome) -> str:
    payload = json.dumps(encoder.serialize_genome(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def cpu_placement(topology, workers):
    """(worker AffinityPlan, scheduler CPUs): workers take allocation_order
    (P-core primaries, then E-cores, then P-core siblings) one CPU each; the
    scheduler gets every CPU no worker uses. Pure: nothing is pinned here."""
    affinity_plan = cpu_slots(topology, workers)
    used = {cpu for slot in affinity_plan.slots for cpu in slot}
    return affinity_plan, [cpu for cpu in allocation_order(topology) if cpu not in used]


class CampaignScheduler:
    def __init__(self, plan, campaign_id: str, output_root="results/campaign", workers=20,
                 mode="real", grace_s=120, mirror_dir=None, no_mirror=False,
                 allow_same_disk_mirror=False, prefetch=4):
        self.plan, self.campaign_id, self.mode, self.grace_s = plan, campaign_id, mode, grace_s
        self.allow_same_disk_mirror = allow_same_disk_mirror
        self.root = os.path.join(output_root, campaign_id)
        self.journal = ResultJournal(os.path.join(self.root, "journal.jsonl"))
        self.alerts = AlertManager(os.path.join(self.root, "alerts.jsonl"))
        self.mirror_dir, self.no_mirror = mirror_dir, no_mirror
        self.prefetch = prefetch
        topology = detect_topology()
        self.topology = topology
        self.affinity_plan, self.scheduler_cpus = cpu_placement(topology, workers)
        self.scheduler_affinity = pin_current_process(self.scheduler_cpus) if self.scheduler_cpus else None
        self.stop_requested = False

    def _request_stop(self, _signum, _frame):
        self.stop_requested = True

    def _manifest(self, **extra):
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "manifest.json")
        previous = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("config_hash") != self.plan.config_hash:
                raise ValueError("campaign config-hash mismatch")
        current = {**previous, "campaign_id": self.campaign_id, "config_hash": self.plan.config_hash,
                   "plan": self.plan.to_dict(), "guard_outcome": self.plan.guard_outcome,
                   "scheduler_cpus": self.scheduler_cpus,
                   "topology": self.topology.to_dict(),
                   "worker_slots": self.affinity_plan.to_dict(),
                   "scheduler_affinity": dataclasses.asdict(self.scheduler_affinity)
                   if dataclasses.is_dataclass(self.scheduler_affinity) else self.scheduler_affinity,
                   "mirror_dir": self.mirror_dir, "no_mirror": self.no_mirror,
                   "allow_same_disk_mirror": self.allow_same_disk_mirror,
                   "prefetch": self.prefetch, **extra}
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)

    def _dispatch_status(self, running: int, queued: int, task=None, generations=None):
        path = os.path.join(self.root, "dispatch_status.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"tasks_running": running, "tasks_queued": queued,
                       "task": task, "run_generations": generations or {},
                       "updated_at": time.time()}, handle)
        os.replace(temporary, path)

    def _make_run(self, run_plan):
        cfg_dict = dict(run_plan.config)
        factor_kwargs = cfg_dict.pop("campaign_encoder_kwargs", cfg_dict.pop("campaign_factors", {}))
        cfg_dict["grid_shape"] = tuple(cfg_dict["grid_shape"])
        config = ExperimentConfig(**cfg_dict)
        factors, kwargs = extract_factors(factor_kwargs, seed=config.seed)
        encoder_name = "rule_based" if run_plan.arm.startswith("rule_based") else run_plan.arm
        if run_plan.arm == "rule_based_monotone":
            kwargs["action_monotonicity"] = "monotone"
        encoder = get_encoder(encoder_name, factors=factors, grid_shape=config.grid_shape, **kwargs)
        directory = os.path.join(self.root, "runs", run_plan.arm, run_plan.task, f"seed_{run_plan.seed}")
        state = latest_checkpoint(os.path.join(directory, "checkpoint"))
        if os.path.isdir(directory):
            # A SIGKILL can land after tell() flushed its JSONL rows but
            # before its atomic checkpoint. Retain only checkpoint-covered
            # rows; durable PPO outcomes remain in the separate journal.
            limits = {"run_log.jsonl": state["counters"]["evaluations_used"] if state else 0,
                      "generation_log.jsonl": len(state.get("generation_records", [])) if state else 0}
            for name, count in limits.items():
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as handle:
                        lines = handle.readlines()
                    if len(lines) > count:
                        temporary = path + ".reconcile.tmp"
                        with open(temporary, "w", encoding="utf-8") as handle:
                            handle.writelines(lines[:count])
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, path)
        logger = RunLogger(directory, encoder=run_plan.arm, seed=run_plan.seed, append=state is not None)
        run = EvolutionRun.from_state(encoder, config, state) if state is not None else EvolutionRun(encoder, config)
        run.tracker.logger = logger
        return encoder, config, run, logger, directory

    def _run_phase(self, active, pool, phase, outputs, phase_started_at=None):
        """Continuously dispatch one D7 phase with per-run generation barriers."""
        phase_started_at = phase_started_at or time.time()
        scheduler_intervals = []
        task_intervals = []
        component_busy = {name: 0.0 for name in ("prepare", "finish", "journal", "refill")}
        prefetched_submissions = 0
        max_inflight = 0
        journaled = self.journal.indexed()
        core = DispatchCore(self.affinity_plan.n_workers, self.prefetch)
        core.begin_phase(phase.task, (ctx[0].run_key for ctx in active))
        by_run = {ctx[0].run_key: ctx for ctx in active}
        generations = {}
        barriers = deque()
        arm_order = {}
        for rp, *_rest in active:
            arm_order.setdefault(rp.arm, len(arm_order))

        def prepare(ctx):
            wall_started, measured_started = time.time(), time.perf_counter()
            rp, encoder, config, run, _logger, directory = ctx
            candidates = run.propose()
            generations[rp.run_key] = {
                "generation": candidates[0].generation if candidates else run.generation,
                "candidates": candidates,
                "outcomes": {},
            }
            work = []
            for candidate in candidates:
                identity = (rp.run_key, candidate.eval_index)
                h = genome_hash(encoder, candidate.genome)
                entries = [row for key, row in journaled.items() if key[:2] == identity]
                if any(row["genome_hash"] != h for row in entries):
                    raise RuntimeError(f"genome-hash mismatch while resuming {rp.run_key} eval_index={candidate.eval_index}")
                if candidate.cached_result is not None or candidate.cache_source_eval_index is not None:
                    continue
                if self.mode == "surrogate":
                    result = evaluate_body_surrogate(candidate.body, task=config.task, seed=config.seed)
                    result.eval_index = candidate.eval_index
                    generations[rp.run_key]["outcomes"][identity] = result
                    continue
                validation = validate_body(candidate.body)
                if not validation.valid:
                    generations[rp.run_key]["outcomes"][identity] = EvalResult(config.sentinel_fitness, False,
                        "; ".join(validation.reasons), 0.0, compute_body_hash(candidate.body),
                        eval_index=candidate.eval_index, per_seed_fitness=[], per_seed_completed=[],
                        aggregation=config.ppo_seed_aggregation, policy_paths=[], training_log_paths=[])
                    continue
                from evogym import get_full_connectivity
                connections = get_full_connectivity(candidate.body)
                for k in range(config.ppo_seeds):
                    task = PPOTask(rp.run_key, config.encoder, config.task, ppo_seed(config.seed, k),
                        candidate.generation, candidate.genome_id, candidate.eval_index, k,
                        candidate.body, connections, config.ppo_timesteps, config.ppo_eval_episodes, directory)
                    if self.mode == "validation":
                        surrogate = evaluate_body_surrogate(
                            candidate.body, task=config.task, seed=ppo_seed(config.seed, k))
                        validation_profile = {
                            **self.plan.validation_profile,
                            "e_core_ids": list(self.topology.e_core_ids),
                        }
                        task.validation_meta = validation_task_metadata(
                            validation_profile, run_key=rp.run_key, arm=rp.arm,
                            task=config.task, generation=candidate.generation,
                            eval_index=candidate.eval_index, ppo_k=k,
                            fitness=surrogate.fitness,
                        )
                    if (rp.run_key, candidate.eval_index, k) not in journaled:
                        work.append(DispatchItem(
                            task_id=(task.run_key, task.eval_index, task.ppo_k),
                            phase=phase.task, run_key=rp.run_key,
                            generation=candidate.generation,
                            tie_key=(arm_order[rp.arm], rp.seed, candidate.eval_index, k),
                            payload=(task, h),
                        ))
            barrier = core.report_ready(rp.run_key, generations[rp.run_key]["generation"], work)
            if barrier is not None:
                barriers.append(barrier)
            wall_finished = time.time()
            component_busy["prepare"] += time.perf_counter() - measured_started
            scheduler_intervals.append((wall_started, wall_finished, "prepare"))

        def finish_generation(barrier):
            wall_started, measured_started = time.time(), time.perf_counter()
            rp, encoder, config, run, _logger, directory = by_run[barrier.run_key]
            state = generations.pop(rp.run_key)
            if state["generation"] != barrier.generation:
                raise AssertionError("completed generation does not match scheduler state")
            by_index = {}
            results = []
            for candidate in state["candidates"]:
                identity = (rp.run_key, candidate.eval_index)
                if candidate.cached_result is not None:
                    result = candidate.cached_result
                elif candidate.cache_source_eval_index is not None:
                    result = by_index[candidate.cache_source_eval_index]
                elif identity in state["outcomes"]:
                    result = state["outcomes"][identity]
                else:
                    seeds = [PPOTaskResult(**journaled[(rp.run_key, candidate.eval_index, k)]["result"])
                             for k in range(config.ppo_seeds)]
                    fitness, completed, partial = aggregate_seed_results(seeds, config.ppo_seed_aggregation)
                    infra_failures = sum(str(seed.reason).startswith("infra_failure:") for seed in seeds)
                    result = EvalResult(fitness if completed else config.sentinel_fitness, completed,
                        "partial_completion" if completed and partial else
                        (None if completed else ("infra_failure" if infra_failures else "simulation_instability")),
                        sum(seed.wall_clock_s for seed in seeds), compute_body_hash(candidate.body),
                        per_seed_fitness=[seed.fitness for seed in seeds],
                        per_seed_completed=[seed.completed for seed in seeds],
                        aggregation=config.ppo_seed_aggregation,
                        policy_paths=[seed.policy_path for seed in seeds if seed.policy_path],
                        training_log_paths=[seed.training_log_path for seed in seeds if seed.training_log_path],
                        infra_failure_count=infra_failures)
                dispatched = copy.copy(result)
                dispatched.eval_index = candidate.eval_index
                by_index[candidate.eval_index] = result
                results.append(dispatched)
            run.tell(results)
            write_checkpoint(run, os.path.join(directory, "checkpoint"))
            wall_finished = time.time()
            component_busy["finish"] += time.perf_counter() - measured_started
            scheduler_intervals.append((wall_started, wall_finished, "finish"))
            if run.done:
                outputs[rp.run_key] = run.result()
                _logger.close()
                core.mark_run_complete(rp.run_key)
            else:
                prepare(by_run[rp.run_key])

        def refill():
            nonlocal prefetched_submissions, max_inflight
            wall_started, measured_started = time.time(), time.perf_counter()
            if pool is not None and not self.stop_requested:
                while (dispatch_item := core.next_task()) is not None:
                    task, _hash = dispatch_item.payload
                    pool.submit_ppo_task(task, timeout_s=phase.timeout_s)
                    if core.inflight_count > self.affinity_plan.n_workers:
                        prefetched_submissions += 1
                    max_inflight = max(max_inflight, core.inflight_count)
            self._dispatch_status(core.inflight_count, core.ready_count, phase.task,
                                  {key: ctx[3].generation for key, ctx in by_run.items()
                                   if not ctx[3].done})
            wall_finished = time.time()
            component_busy["refill"] += time.perf_counter() - measured_started
            scheduler_intervals.append((wall_started, wall_finished, "refill"))

        for ctx in active:
            if ctx[3].done:
                outputs[ctx[0].run_key] = ctx[3].result()
                ctx[4].close()
                core.mark_run_complete(ctx[0].run_key)
            else:
                prepare(ctx)
            refill()

        while not core.phase_complete:
            while barriers:
                finish_generation(barriers.popleft())
            refill()
            if core.phase_complete:
                break
            if self.stop_requested and core.inflight_count == 0:
                break
            if pool is None:
                if not barriers:
                    raise RuntimeError("surrogate dispatch stalled before its phase barrier")
                continue
            completed = pool.next_ppo_result(should_stop=lambda: self.stop_requested,
                                             grace_s=self.grace_s)
            if completed is None:
                break
            task, result = completed
            if result.started_at is not None and result.finished_at is not None:
                task_intervals.append((result.started_at, result.finished_at,
                                       result.worker_slot, task.run_key,
                                       task.generation, task.eval_index, task.ppo_k))
            wall_started, measured_started = time.time(), time.perf_counter()
            identity = (task.run_key, task.eval_index, task.ppo_k)
            state = generations[task.run_key]
            candidate = next(c for c in state["candidates"] if c.eval_index == task.eval_index)
            h = genome_hash(by_run[task.run_key][1], candidate.genome)
            self.journal.append_task(task, result, h)
            journaled[identity] = {"genome_hash": h, "result": dataclasses.asdict(result)}
            wall_finished = time.time()
            component_busy["journal"] += time.perf_counter() - measured_started
            scheduler_intervals.append((wall_started, wall_finished, "journal"))
            barrier = core.complete(identity)
            if barrier is not None:
                barriers.append(barrier)
            else:
                refill()
        phase_finished_at = time.time()
        makespan = max(0.0, phase_finished_at - phase_started_at)

        def overlap(interval, start, end):
            return max(0.0, min(interval[1], end) - max(interval[0], start))

        steady_start = phase_started_at + 0.05 * makespan
        steady_end = phase_started_at + 0.90 * makespan
        whole_busy = sum(overlap(row, phase_started_at, phase_finished_at)
                         for row in task_intervals)
        steady_busy = sum(overlap(row, steady_start, steady_end) for row in task_intervals)
        merged = []
        for start, end, _component in sorted(scheduler_intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        merged_starts = [row[0] for row in merged]
        starts_during_busy = 0
        for row in task_intervals:
            index = bisect_right(merged_starts, row[0]) - 1
            starts_during_busy += int(index >= 0 and row[0] <= merged[index][1])
        worker_count = self.affinity_plan.n_workers
        steady_span = max(0.0, steady_end - steady_start)
        phase_metrics = {
            "task": phase.task,
            "status": "complete" if core.phase_complete else "stopped",
            "started_at": phase_started_at, "ended_at": phase_finished_at,
            "makespan_s": makespan, "worker_count": worker_count,
            "worker_busy_s": whole_busy,
            "worker_utilization": whole_busy / (worker_count * makespan) if makespan else 0.0,
            "steady_state_window": {"start": steady_start, "end": steady_end,
                                    "excluded_first_fraction": 0.05,
                                    "excluded_last_fraction": 0.10},
            "steady_state_worker_busy_s": steady_busy,
            "steady_state_worker_utilization": (
                steady_busy / (worker_count * steady_span) if steady_span else 0.0),
            "scheduler_busy_s": sum(component_busy.values()),
            "scheduler_busy_fraction": (sum(component_busy.values()) / makespan if makespan else 0.0),
            "scheduler_busy_by_component_s": component_busy,
            "completed_timed_tasks": len(task_intervals),
            "prefetch": self.prefetch,
            "max_inflight": max_inflight,
            "prefetched_submissions": prefetched_submissions,
            "tasks_started_while_scheduler_busy": starts_during_busy,
        }
        metrics_path = os.path.join(self.root, "phase_metrics.json")
        all_metrics = {}
        if os.path.isfile(metrics_path):
            with open(metrics_path, encoding="utf-8") as handle:
                all_metrics = json.load(handle)
        all_metrics[phase.task] = phase_metrics
        temporary = metrics_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metrics_path)
        return core.phase_complete

    def run(self):
        prior = self._read_manifest() if os.path.isfile(os.path.join(self.root, "manifest.json")) else {}
        self._manifest(status="running", started_at=prior.get("started_at", time.time()),
                       phase_status=prior.get("phase_status", {}))
        outputs = {}
        phase_status = dict(prior.get("phase_status", {}))
        old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)
        pool = (EvaluationPool(self.affinity_plan.n_workers, self.affinity_plan, "real", max_retries=2,
                               ignore_sigint=True) if self.mode in {"real", "validation"} else None)
        try:
            for phase in self.plan.phases:  # D7 barrier: no phase interleaving.
                if phase_status.get(phase.task, {}).get("status") == "complete":
                    continue
                phase_start = phase_status.get(phase.task, {}).get("started_at", time.time())
                phase_status[phase.task] = {"status": "running", "started_at": phase_start}
                self._manifest(phase=phase.task, phase_status=phase_status)
                active = [(rp, *self._make_run(rp)) for rp in phase.runs]
                phase_complete = self._run_phase(active, pool, phase, outputs,
                                                 phase_started_at=phase_start)
                if not phase_complete:
                    for _rp, _encoder, _config, run, logger, _directory in active:
                        if not run.done:
                            logger.close()
                    self._manifest(status="stopped", stopped_at=time.time())
                    break
                phase_status[phase.task] = {"status": "complete", "started_at": phase_start,
                                        "ended_at": time.time()}
                self._manifest(status="running", phase=phase.task, phase_status=phase_status)
            # Package generation is isolated in a subprocess.  A report bug
            # cannot roll back completed EvolutionRun checkpoints or logs.
                self.alerts.phase_complete(phase.task)
                milestone = run_subprocess(self.root, phase.task, self.alerts)
                self._manifest(milestone_status={**self._read_manifest().get("milestone_status", {}),
                                             phase.task: milestone})
                if self.mirror_dir:
                    result = sync(self.root, self.mirror_dir, alert_manager=self.alerts,
                                  location_policy={"allow_same_disk": self.allow_same_disk_mirror})
                    if result["ok"]:
                        self._manifest(last_mirror_success=result["timestamp"])
            if not self.stop_requested:
                self._manifest(status="complete", ended_at=time.time())
        finally:
            if pool is not None:
                pool.shutdown(wait=not self.stop_requested)
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
        return outputs

    def _read_manifest(self):
        with open(os.path.join(self.root, "manifest.json"), encoding="utf-8") as handle:
            return json.load(handle)


def resume_command(args, campaign_id):
    command = ["scripts/campaign/run_campaign.sh", "--resume", campaign_id,
               "--config", args.config, "--output-root", args.output_root,
               "--workers", str(args.workers), "--mode", args.mode,
               "--grace-s", str(args.grace_s)]
    if getattr(args, "prefetch", 4) != 4:
        command.extend(["--prefetch", str(args.prefetch)])
    command.extend(["--no-mirror"] if args.no_mirror else ["--mirror-dir", args.mirror_dir])
    if getattr(args, "no_notify", False):
        command.append("--no-notify")
    if getattr(args, "allow_same_disk_mirror", False):
        command.append("--allow-same-disk-mirror")
    if getattr(args, "stamp", None):
        command.extend(["--stamp", args.stamp])
    return shlex.join(command)


def cli_main(argv=None, prog="python -m evodevo.campaign"):
    """Scheduler CLI logic. Invoked ONLY via `python -m evodevo.campaign`
    (evodevo/campaign/__main__.py), which pins thread limits first."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--campaign-id")
    parser.add_argument("--resume")
    parser.add_argument("--config", default="evodevo/experiments/configs/benchmark.yaml")
    parser.add_argument("--output-root", default="results/campaign")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--mode", choices=("real", "surrogate", "validation"), default="real")
    parser.add_argument("--mirror-dir")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--grace-s", type=float, default=120)
    parser.add_argument("--prefetch", type=int, default=4,
                        help="PPO tasks submitted beyond worker slots (default: 4)")
    parser.add_argument("--no-notify", action="store_true",
                        help="explicitly run a REAL campaign with no out-of-band notifier "
                             "(recorded in the manifest; permanent dashboard warning)")
    parser.add_argument("--allow-same-disk-mirror", action="store_true",
                        help="explicitly accept a mirror on the SAME physical disk as the results "
                             "(recorded in the manifest; permanent dashboard warning)")
    parser.add_argument("--stamp", help="label stamped on every artifact, e.g. 'VALIDATION - NOT A RESULT' "
                        "(manifest, STAMP.txt, milestone figures/report/tables); never result-affecting")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan, calendar estimate, disk projection and every launch "
                             "check; write nothing, start nothing")
    parser.add_argument("--launch-at", help="dry-run calendar start, ISO local time "
                        "(e.g. 2026-10-01T00:00); default now")
    parser.add_argument("--check-only", action="store_true",
                        help="run every launch refusal check, print the result, start nothing")
    args = parser.parse_args(argv)
    if args.prefetch < 0:
        parser.error("--prefetch must be >= 0")
    if not args.mirror_dir and not args.no_mirror:
        parser.error("--mirror-dir is required unless --no-mirror is explicitly set")
    if args.mirror_dir and args.no_mirror:
        parser.error("--mirror-dir and --no-mirror are mutually exclusive")
    if args.allow_same_disk_mirror and args.no_mirror:
        parser.error("--allow-same-disk-mirror requires --mirror-dir")
    if args.resume and args.campaign_id:
        parser.error("--resume and --campaign-id are mutually exclusive")
    if args.dry_run:
        from evodevo.experiments.benchmark import BenchmarkConfig
        from .dry_run import render
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(args.config))
        campaign_id = args.resume or args.campaign_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        print(render(args, plan, campaign_id), end="", flush=True)
        return 0
    from evodevo.preflight.dirty_check import project_is_dirty
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    refusals = ["refusing a dirty working tree (commit first)"] if project_is_dirty(repo_root) else []
    campaign_id = args.resume or args.campaign_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if campaign_id == "latest":
        root = args.output_root
        candidates = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest.json"))]
        if not candidates:
            parser.error("no campaign to resume")
        campaign_id = max(candidates, key=lambda d: os.path.getmtime(os.path.join(root, d, "manifest.json")))
    manifest_path = os.path.join(args.output_root, campaign_id, "manifest.json")
    manifest = {}
    if args.resume:
        if not os.path.isfile(manifest_path):
            parser.error(f"no manifest for {campaign_id}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        plan = CampaignPlan.from_dict(manifest["plan"])
        from evodevo.experiments.benchmark import BenchmarkConfig
        current = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(args.config))
        if current.config_hash != plan.config_hash or current.config_hash != manifest.get("config_hash"):
            parser.error("campaign config-hash mismatch")
    else:
        if os.path.isfile(manifest_path):
            parser.error(f"campaign {campaign_id} already exists; use --resume {campaign_id}")
        from evodevo.experiments.benchmark import BenchmarkConfig
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(args.config))
    from .plan import VALIDATION_STAMP
    validation_plan = plan.validation_profile is not None
    if validation_plan != (args.mode == "validation"):
        parser.error("campaign_validation profile and --mode validation must be used together")
    effective_stamp = args.stamp or manifest.get("artifact_stamp")
    if validation_plan and effective_stamp != VALIDATION_STAMP:
        parser.error(f"validation runs require --stamp {VALIDATION_STAMP!r}")
    from .launch_checks import launch_refusals
    more, launch_record = launch_refusals(args)
    refusals += more
    if refusals:
        parser.error("refusing to start:\n  - " + "\n  - ".join(refusals))
    if args.check_only:
        print(f"launch checks passed for {campaign_id}: {json.dumps(launch_record, sort_keys=True)}")
        return 0
    scheduler = CampaignScheduler(plan, campaign_id, args.output_root, args.workers,
        args.mode, args.grace_s, args.mirror_dir, args.no_mirror,
        allow_same_disk_mirror=args.allow_same_disk_mirror, prefetch=args.prefetch)
    scheduler._manifest(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=repo_root, text=True).strip(), launch_checks=launch_record,
        no_notify=bool(args.no_notify))
    stamp = args.stamp or scheduler._read_manifest().get("artifact_stamp")
    if stamp:
        scheduler._manifest(artifact_stamp=stamp)
        with open(os.path.join(scheduler.root, "STAMP.txt"), "w", encoding="utf-8") as handle:
            handle.write(stamp + "\n")
    with open(os.path.join(scheduler.root, "scheduler.pgid"), "w", encoding="ascii") as handle:
        handle.write(str(os.getpgrp()) + "\n")
    scheduler.run()
    if scheduler.stop_requested:
        print(f"Resume: {resume_command(args, campaign_id)}", flush=True)


if __name__ == "__main__":
    # Running this module as __main__ makes every spawned PPO worker re-import
    # it (as __mp_main__) before its initializer pins thread counts -- the
    # BrokenProcessPool bug fixed in the entry-point commit. Refuse outright.
    sys.exit("error: do not run `python -m evodevo.campaign.scheduler`; "
             "use `python -m evodevo.campaign` (evodevo/campaign/__main__.py).")
