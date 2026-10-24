"""Replay recorded campaign evaluations through the real worker path.

    python -m evodevo.campaign replay --output-dir DIR [--cpus 0,2,4] \\
        [--campaign results/campaign/20260930T044002Z] [--item TASK:ARM:SEED:EVAL ...]

Amendment 2 reuses the original campaign's Walker-v0/UpStepper-v0 protocol on
today's code. This re-proves it on the code that will run: each requested
recorded evaluation is rebuilt from the original manifest (the exact encoder
and config of the run), its recorded genome is decoded and checked against the
recorded body, and every PPO training of the evaluation (K per genome) runs
through ``EvaluationPool`` -> ``worker_pool._run_ppo_task`` -> ``ppo_runner``
on its own worker, each worker pinned to its own core. The fitness of every
training and the aggregate must equal the recorded values exactly (``==`` on
floats).

Every training writes its policy and log below a directory unique per
(task, arm, seed, evaluation, k): an earlier replay harness lost files to a
shared output directory.

Read-only with respect to the original campaign. Exits 0 only if every
replayed value is bit-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_CAMPAIGN = "results/campaign/20260930T044002Z"
# Gate R's two long replays: a Walker-v0 K=1 evaluation and the last
# evaluation of an UpStepper-v0 run with K=2 (both trainings).
DEFAULT_ITEMS = ("Walker-v0:direct:1:2", "UpStepper-v0:rule_based:0:299")


def parse_item(text: str) -> tuple:
    task, arm, seed, eval_index = text.split(":")
    return task, arm, int(seed), int(eval_index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evodevo.campaign replay", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--item", action="append", help="TASK:ARM:SEED:EVAL_INDEX (repeatable)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpus", default="0,2,4", help="one CPU per training, comma separated")
    parser.add_argument("--timeout-s", type=float, default=6 * 3600)
    args = parser.parse_args(argv)

    import numpy as np
    from evodevo.campaign.plan import RunPlan
    from evodevo.campaign.scheduler import encoder_and_config_for
    from evodevo.evaluation.ppo_runner import PPOTask, PPOTaskResult, aggregate_seed_results, ppo_seed
    from evodevo.morphology.metrics import body_hash
    from evodevo.parallel.topology import AffinityPlan
    from evodevo.parallel.worker_pool import EvaluationPool
    from evogym import get_full_connectivity

    root = Path(args.campaign)
    manifest = json.loads((root / "manifest.json").read_text())
    runs = {(p["task"], r["arm"], r["seed"]): (p, r) for p in manifest["plan"]["phases"] for r in p["runs"]}
    items = [parse_item(text) for text in (args.item or DEFAULT_ITEMS)]
    tasks, expected = [], {}
    for task_name, arm, seed, eval_index in items:
        phase, run = runs[(task_name, arm, seed)]
        encoder, config = encoder_and_config_for(RunPlan(**run))
        rows = [json.loads(line) for line in (root / "runs" / arm / task_name / f"seed_{seed}" /
                                              "run_log.jsonl").read_text().splitlines() if line.strip()]
        row = rows[eval_index]
        assert row["eval_index"] == eval_index and row["valid"] and not row["cache_hit"], row["eval_index"]
        genome = encoder.deserialize_genome(row["genome"])
        body = encoder.decode(genome)
        if body.tolist() != row["body"] or body_hash(body) != row["body_hash"]:
            print(f"REPLAY ERROR: {task_name} {arm} seed_{seed} eval {eval_index}: the decoded body differs "
                  "from the recorded one", file=sys.stderr)
            return 2
        connections = get_full_connectivity(np.array(row["body"], dtype=int))
        directory = os.path.join(args.output_dir, task_name, arm, f"seed_{seed}", f"e{eval_index:05d}")
        for k in range(config.ppo_seeds):
            tasks.append(PPOTask(f"{arm}/{task_name}/seed_{seed}", config.encoder, config.task,
                                 ppo_seed(config.seed, k), row["generation"], row["genome_id"], eval_index, k,
                                 np.array(row["body"], dtype=int), connections, config.ppo_timesteps,
                                 config.ppo_eval_episodes, directory))
        expected[(task_name, arm, seed, eval_index)] = {
            "per_seed": row["per_seed_fitness"], "aggregate": row["fitness"], "aggregation": config.ppo_seed_aggregation,
            "timeout_s": phase["timeout_s"]}
    cpus = [int(c) for c in args.cpus.split(",")]
    if len(cpus) < len(tasks):
        print(f"REPLAY ERROR: {len(tasks)} trainings need {len(tasks)} CPUs, got {cpus}", file=sys.stderr)
        return 2
    cpus = cpus[:len(tasks)]
    plan = AffinityPlan(n_workers=len(tasks), slots=tuple((c,) for c in cpus), oversubscribed=False,
                        allocation_order=tuple(cpus))
    started = time.time()
    pool = EvaluationPool(len(tasks), plan, "real", max_retries=0, ignore_sigint=True)
    results = {}
    try:
        for task in tasks:
            pool.submit_ppo_task(task, timeout_s=args.timeout_s)
        for _ in tasks:
            done = pool.next_ppo_result()
            if done is None:
                break
            task, result = done
            results[(task.task, task.run_key.split("/")[0], task.seed, task.eval_index, task.ppo_k)] = (task, result)
            print(f"finished {task.run_key} eval {task.eval_index} k={task.ppo_k} on cpu {result.cpu_ids}: "
                  f"fitness {result.fitness!r} ({result.wall_clock_s:.0f} s)", flush=True)
    finally:
        pool.shutdown(wait=True)
    report, identical = [], True
    for (task_name, arm, seed, eval_index), want in expected.items():
        mine = sorted((v for key, v in results.items() if key[0] == task_name and key[1] == arm and
                       v[0].run_key == f"{arm}/{task_name}/seed_{seed}" and v[0].eval_index == eval_index),
                      key=lambda pair: pair[0].ppo_k)
        per_seed = [result.fitness for _task, result in mine]
        completed = [result.completed for _task, result in mine]
        aggregate = aggregate_seed_results([result for _task, result in mine], want["aggregation"])[0] if mine else None
        same = (len(per_seed) == len(want["per_seed"]) and all(completed) and per_seed == want["per_seed"]
                and aggregate == want["aggregate"])
        identical &= bool(same)
        report.append({"task": task_name, "arm": arm, "seed": seed, "eval_index": eval_index,
                       "recorded_per_seed": want["per_seed"], "replayed_per_seed": per_seed,
                       "recorded_aggregate": want["aggregate"], "replayed_aggregate": aggregate,
                       "cpu_ids": [result.cpu_ids for _task, result in mine],
                       "wall_clock_s": [result.wall_clock_s for _task, result in mine],
                       "policy_paths": [result.policy_path for _task, result in mine], "bit_identical": bool(same)})
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}: {task_name} {arm} seed_{seed} eval {eval_index}: "
              f"recorded {want['per_seed']!r} replayed {per_seed!r}", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    git = os.popen("git rev-parse HEAD").read().strip()
    (Path(args.output_dir) / "compare.json").write_text(json.dumps(
        {"git_commit": git, "campaign": str(root), "elapsed_s": time.time() - started, "all_identical": identical,
         "items": report}, indent=2, sort_keys=True) + "\n")
    print("ALL BIT-IDENTICAL" if identical else "REPLAY MISMATCH")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
