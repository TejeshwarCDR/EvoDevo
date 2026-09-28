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

from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import (EvalResult, PPOTask, PPOTaskResult,
                                            aggregate_seed_results, evaluate_body_surrogate,
                                            evaluate_ppo_task, ppo_seed)
from evodevo.evolution.ea import EvolutionRun
from evodevo.evolution.logging import RunLogger
from evodevo.parallel.affinity import pin_current_process
from evodevo.parallel.topology import allocation_order, cpu_slots, detect_topology
from evodevo.parallel.worker_pool import EvaluationPool
from evodevo.morphology.metrics import body_hash as compute_body_hash
from evodevo.morphology.validate import validate_body

from .checkpoint import latest_checkpoint, write_checkpoint
from .alerts import AlertManager
from .journal import ResultJournal
from .milestone import run_subprocess
from .mirror import sync
from .plan import CampaignPlan


def genome_hash(encoder, genome) -> str:
    payload = json.dumps(encoder.serialize_genome(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class CampaignScheduler:
    def __init__(self, plan, campaign_id: str, output_root="results/campaign", workers=20,
                 mode="real", grace_s=120, mirror_dir=None, no_mirror=False):
        self.plan, self.campaign_id, self.mode, self.grace_s = plan, campaign_id, mode, grace_s
        self.root = os.path.join(output_root, campaign_id)
        self.journal = ResultJournal(os.path.join(self.root, "journal.jsonl"))
        self.alerts = AlertManager(os.path.join(self.root, "alerts.jsonl"))
        self.mirror_dir, self.no_mirror = mirror_dir, no_mirror
        topology = detect_topology()
        self.topology = topology
        self.affinity_plan = cpu_slots(topology, workers)
        used = {cpu for slot in self.affinity_plan.slots for cpu in slot}
        self.scheduler_cpus = [cpu for cpu in allocation_order(topology) if cpu not in used]
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
                   "mirror_dir": self.mirror_dir, "no_mirror": self.no_mirror, **extra}
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

    def _phase_round(self, active, pool, timeout_s):
        """Propose all active runs before any PPO dispatch; tell in plan order."""
        journaled = self.journal.indexed()
        batches, outcomes, work = {}, {}, []
        for rp, encoder, config, run, _logger, directory in active:
            candidates = run.propose()
            batches[rp.run_key] = candidates
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
                    outcomes[identity] = result
                    continue
                validation = validate_body(candidate.body)
                if not validation.valid:
                    outcomes[identity] = EvalResult(config.sentinel_fitness, False,
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
                    if (rp.run_key, candidate.eval_index, k) not in journaled:
                        work.append((task, h))
        arm_order = {rp.arm: index for index, (rp, *_rest) in enumerate(active)}
        task_name = active[0][0].task
        run_generations = {rp.run_key: batches[rp.run_key][0].generation
                           for rp, *_rest in active if batches[rp.run_key]}
        work.sort(key=lambda pair: (pair[0].generation, arm_order[pair[0].run_key.split('/')[0]],
                                    int(pair[0].run_key.rsplit('_', 1)[1]), pair[0].eval_index, pair[0].ppo_k))
        if work:
            remaining = len(work)
            self._dispatch_status(min(remaining, self.affinity_plan.n_workers),
                                  max(0, remaining - self.affinity_plan.n_workers),
                                  task_name, run_generations)
            hashes = {(t.run_key, t.eval_index, t.ppo_k): h for t, h in work}
            for task, result in pool.map_ppo_tasks((t for t, _ in work), timeout_s=timeout_s,
                                                    should_stop=lambda: self.stop_requested,
                                                    grace_s=self.grace_s):
                h = hashes[(task.run_key, task.eval_index, task.ppo_k)]
                self.journal.append_task(task, result, h)
                journaled[(task.run_key, task.eval_index, task.ppo_k)] = {
                    "genome_hash": h, "result": dataclasses.asdict(result)}
                remaining -= 1
                running = min(remaining, pool._ppo_inflight)
                self._dispatch_status(running, max(0, remaining - running),
                                      task_name, run_generations)
            if remaining:
                if self.stop_requested:
                    self._dispatch_status(0, remaining, task_name, run_generations)
                    return False
                raise RuntimeError(f"PPO pool returned with {remaining} tasks incomplete")
        else:
            self._dispatch_status(0, 0, task_name, run_generations)
        for rp, encoder, config, run, _logger, _directory in active:
            by_index = {}
            results = []
            for candidate in batches[rp.run_key]:
                identity = (rp.run_key, candidate.eval_index)
                if candidate.cached_result is not None:
                    result = candidate.cached_result
                elif candidate.cache_source_eval_index is not None:
                    result = by_index[candidate.cache_source_eval_index]
                elif identity in outcomes:
                    result = outcomes[identity]
                else:
                    seeds = [PPOTaskResult(**journaled[(rp.run_key, candidate.eval_index, k)]["result"])
                             for k in range(config.ppo_seeds)]
                    fitness, completed, partial = aggregate_seed_results(seeds, config.ppo_seed_aggregation)
                    result = EvalResult(fitness if completed else config.sentinel_fitness, completed,
                        "partial_completion" if completed and partial else (None if completed else "simulation_instability"),
                        sum(seed.wall_clock_s for seed in seeds), compute_body_hash(candidate.body),
                        per_seed_fitness=[seed.fitness for seed in seeds],
                        per_seed_completed=[seed.completed for seed in seeds],
                        aggregation=config.ppo_seed_aggregation,
                        policy_paths=[seed.policy_path for seed in seeds if seed.policy_path],
                        training_log_paths=[seed.training_log_path for seed in seeds if seed.training_log_path],
                        infra_failure_count=sum(not seed.completed for seed in seeds))
                dispatched = copy.copy(result)
                dispatched.eval_index = candidate.eval_index
                by_index[candidate.eval_index] = result
                results.append(dispatched)
            run.tell(results)
            write_checkpoint(run, os.path.join(_directory, "checkpoint"))
        return True

    def run(self):
        prior = self._read_manifest() if os.path.isfile(os.path.join(self.root, "manifest.json")) else {}
        self._manifest(status="running", started_at=prior.get("started_at", time.time()),
                       phase_status=prior.get("phase_status", {}))
        outputs = {}
        phase_status = dict(prior.get("phase_status", {}))
        old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._request_stop)
        signal.signal(signal.SIGTERM, self._request_stop)
        pool = EvaluationPool(self.affinity_plan.n_workers, self.affinity_plan, "real", max_retries=2) if self.mode == "real" else None
        try:
            for phase in self.plan.phases:  # D7 barrier: no phase interleaving.
                if phase_status.get(phase.task, {}).get("status") == "complete":
                    continue
                phase_start = phase_status.get(phase.task, {}).get("started_at", time.time())
                phase_status[phase.task] = {"status": "running", "started_at": phase_start}
                self._manifest(phase=phase.task, phase_status=phase_status)
                active = [(rp, *self._make_run(rp)) for rp in phase.runs]
                while active and not self.stop_requested:
                    lowest = min((item[3].generation for item in active if not item[3].done), default=None)
                    live = [item for item in active if not item[3].done and item[3].generation == lowest]
                    if live:
                        if not self._phase_round(live, pool, phase.timeout_s):
                            break
                    next_active = []
                    for rp, encoder, config, run, logger, directory in active:
                        if run.done:
                            outputs[rp.run_key] = run.result()
                            logger.close()
                        else:
                            next_active.append((rp, encoder, config, run, logger, directory))
                    active = next_active
                if self.stop_requested:
                    for _rp, _encoder, _config, _run, logger, _directory in active:
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
                    result = sync(self.root, self.mirror_dir, alert_manager=self.alerts)
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
    command.extend(["--no-mirror"] if args.no_mirror else ["--mirror-dir", args.mirror_dir])
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
    parser.add_argument("--mode", choices=("real", "surrogate"), default="real")
    parser.add_argument("--mirror-dir")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--grace-s", type=float, default=120)
    args = parser.parse_args(argv)
    if not args.mirror_dir and not args.no_mirror:
        parser.error("--mirror-dir is required unless --no-mirror is explicitly set")
    if args.mirror_dir and args.no_mirror:
        parser.error("--mirror-dir and --no-mirror are mutually exclusive")
    if args.resume and args.campaign_id:
        parser.error("--resume and --campaign-id are mutually exclusive")
    from evodevo.preflight.dirty_check import project_is_dirty
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_is_dirty(repo_root):
        parser.error("refusing a dirty working tree")
    campaign_id = args.resume or args.campaign_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if campaign_id == "latest":
        root = args.output_root
        candidates = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest.json"))]
        if not candidates:
            parser.error("no campaign to resume")
        campaign_id = max(candidates, key=lambda d: os.path.getmtime(os.path.join(root, d, "manifest.json")))
    manifest_path = os.path.join(args.output_root, campaign_id, "manifest.json")
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
        from evodevo.experiments.benchmark import BenchmarkConfig
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(args.config))
    scheduler = CampaignScheduler(plan, campaign_id, args.output_root, args.workers,
        args.mode, args.grace_s, args.mirror_dir, args.no_mirror)
    scheduler._manifest(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=repo_root, text=True).strip())
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
