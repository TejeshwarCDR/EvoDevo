"""Generation-barrier campaign scheduler.

The scheduler intentionally never lets a later generation enter dispatch
until its owning EvolutionRun has received every result from the prior one.
That is the determinism boundary: completion order affects throughput only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

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

from .checkpoint import write_checkpoint
from .journal import ResultJournal


def genome_hash(encoder, genome) -> str:
    payload = json.dumps(encoder.serialize_genome(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class CampaignScheduler:
    def __init__(self, plan, campaign_id: str, output_root="results/campaign", workers=20,
                 mode="real", grace_s=120):
        self.plan, self.campaign_id, self.mode, self.grace_s = plan, campaign_id, mode, grace_s
        self.root = os.path.join(output_root, campaign_id)
        self.journal = ResultJournal(os.path.join(self.root, "journal.jsonl"))
        topology = detect_topology()
        self.affinity_plan = cpu_slots(topology, workers)
        used = {cpu for slot in self.affinity_plan.slots for cpu in slot}
        self.scheduler_cpus = [cpu for cpu in allocation_order(topology) if cpu not in used]
        self.scheduler_affinity = pin_current_process(self.scheduler_cpus) if self.scheduler_cpus else None

    def _manifest(self, **extra):
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "manifest.json")
        current = {"campaign_id": self.campaign_id, "config_hash": self.plan.config_hash,
                   "plan": self.plan.to_dict(), "scheduler_cpus": self.scheduler_cpus,
                   "scheduler_affinity": self.scheduler_affinity, **extra}
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)

    def _make_run(self, run_plan):
        cfg_dict = dict(run_plan.config)
        factor_kwargs = cfg_dict.pop("campaign_factors", {})
        cfg_dict["grid_shape"] = tuple(cfg_dict["grid_shape"])
        config = ExperimentConfig(**cfg_dict)
        factors, kwargs = extract_factors(factor_kwargs, seed=config.seed)
        encoder_name = "rule_based" if run_plan.arm.startswith("rule_based") else run_plan.arm
        if run_plan.arm == "rule_based_monotone":
            kwargs["action_monotonicity"] = "monotone"
        encoder = get_encoder(encoder_name, factors=factors, grid_shape=config.grid_shape, **kwargs)
        directory = os.path.join(self.root, "runs", run_plan.arm, run_plan.task, f"seed_{run_plan.seed}")
        logger = RunLogger(directory, encoder=run_plan.arm, seed=run_plan.seed)
        return encoder, config, EvolutionRun(encoder, config, logger=logger), logger, directory

    def _evaluate_batch(self, run_key, encoder, config, candidates, artifact_dir):
        """Validate in scheduler; results are returned in candidate-index order."""
        results = {}
        futures = []
        with ThreadPoolExecutor(max_workers=self.affinity_plan.n_workers) as executor:
            for candidate in candidates:
                # Candidate already has scheduler-side decoded body.  Invalid
                # morphology is handled by the evaluator's exact sentinel path.
                if self.mode == "surrogate":
                    result = evaluate_body_surrogate(candidate.body, task=config.task, seed=config.seed)
                    result.eval_index = candidate.eval_index
                    results[candidate.eval_index] = result
                    continue
                from evogym import get_full_connectivity
                tasks = [PPOTask(run_key, config.encoder, config.task, ppo_seed(config.seed, k),
                                 candidate.generation, candidate.genome_id, candidate.eval_index, k,
                                 candidate.body, get_full_connectivity(candidate.body), config.ppo_timesteps,
                                 config.ppo_eval_episodes, artifact_dir)
                         for k in range(config.ppo_seeds)]
                futures.append((candidate, tasks, [executor.submit(evaluate_ppo_task, task) for task in tasks]))
            for candidate, tasks, task_futures in futures:
                per_seed = [future.result() for future in task_futures]
                h = genome_hash(encoder, candidate.genome)
                for task, outcome in zip(tasks, per_seed):
                    self.journal.append_task(task, outcome, h)
                fitness, completed, partial = aggregate_seed_results(per_seed, config.ppo_seed_aggregation)
                results[candidate.eval_index] = EvalResult(
                    fitness if completed else config.sentinel_fitness, completed,
                    "partial_completion" if completed and partial else (None if completed else "simulation_instability"),
                    sum(r.wall_clock_s for r in per_seed),
                    __import__("evodevo.morphology.metrics", fromlist=["body_hash"]).body_hash(candidate.body),
                    per_seed_fitness=[r.fitness for r in per_seed],
                    per_seed_completed=[r.completed for r in per_seed], aggregation=config.ppo_seed_aggregation,
                    policy_paths=[r.policy_path for r in per_seed if r.policy_path],
                    training_log_paths=[r.training_log_path for r in per_seed if r.training_log_path],
                    infra_failure_count=sum(not r.completed for r in per_seed), eval_index=candidate.eval_index)
        return [results[c.eval_index] for c in candidates]

    def run(self):
        self._manifest(status="running", started_at=time.time(), phases=[])
        outputs = {}
        for phase in self.plan.phases:  # D7 barrier: no phase interleaving.
            phase_start = time.time()
            active = [(rp, *self._make_run(rp)) for rp in phase.runs]
            while active:
                next_active = []
                for rp, encoder, config, run, logger, directory in active:
                    if not run.done:
                        candidates = run.propose()
                        evaluated = self._evaluate_batch(rp.run_key, encoder, config, candidates, directory)
                        run.tell(evaluated)
                        write_checkpoint(run, os.path.join(directory, "checkpoint"))
                    if run.done:
                        outputs[rp.run_key] = run.result()
                        logger.close()
                    else:
                        next_active.append((rp, encoder, config, run, logger, directory))
                active = next_active
            self._manifest(status="running", phase=phase.task, phase_started_at=phase_start,
                           phase_ended_at=time.time())
        self._manifest(status="complete", ended_at=time.time())
        return outputs
