"""Process-level failure attribution for continuous campaign dispatch."""
from __future__ import annotations

from evodevo.evaluation.ppo_runner import PPOTask
from evodevo.parallel.topology import cpu_slots, detect_topology
from evodevo.parallel.worker_pool import EvaluationPool


def task(index, *, sleep=0.05, crash=False):
    return PPOTask(
        run_key=f"run-{index}", arm="debug", task="debug", seed=index,
        generation=0, genome_id=index, eval_index=index, ppo_k=0,
        body=None, connections=None, timesteps=0, eval_episodes=0,
        artifact_dir=None,
        debug_meta={"sleep_s": sleep, "os_exit": 17 if crash else None,
                    "fitness": float(index)},
    )


def identity(one):
    return one.run_key, one.eval_index, one.ppo_k


def test_one_crash_among_19_healthy_tasks_charges_only_the_crasher():
    tasks = [task(0, sleep=0.2, crash=True)] + [task(i, sleep=0.5) for i in range(1, 20)]
    with EvaluationPool(20, cpu_slots(detect_topology(), 20), "real", max_retries=0) as pool:
        results = list(pool.map_ppo_tasks(tasks, timeout_s=5))
        failures = pool.ppo_failure_counts()

    assert len(results) == 20
    by_id = {identity(one): result for one, result in results}
    assert by_id[identity(tasks[0])].completed is False
    assert by_id[identity(tasks[0])].reason.startswith("infra_failure:")
    assert all(by_id[identity(one)].completed for one in tasks[1:])
    assert failures == {identity(tasks[0]): 1}


def test_hung_task_is_retried_twice_on_fresh_workers():
    hung = task(0, sleep=10)
    with EvaluationPool(2, cpu_slots(detect_topology(), 2), "real", max_retries=2) as pool:
        results = list(pool.map_ppo_tasks([hung], timeout_s=0.1))
        failures = pool.ppo_failure_counts()

    assert len(results) == 1
    assert results[0][1].completed is False
    assert results[0][1].reason.startswith("infra_failure:")
    assert failures == {identity(hung): 3}


def test_always_crashing_task_gets_two_retries_and_other_work_continues():
    crasher = task(0, sleep=0.05, crash=True)
    healthy = [task(i, sleep=0.1) for i in range(1, 5)]
    with EvaluationPool(4, cpu_slots(detect_topology(), 4), "real", max_retries=2) as pool:
        results = list(pool.map_ppo_tasks([crasher, *healthy], timeout_s=5))
        failures = pool.ppo_failure_counts()

    by_id = {identity(one): result for one, result in results}
    assert len(by_id) == 5
    assert by_id[identity(crasher)].completed is False
    assert "after 3 attempt(s)" in by_id[identity(crasher)].reason
    assert all(by_id[identity(one)].completed for one in healthy)
    assert failures == {identity(crasher): 3}
