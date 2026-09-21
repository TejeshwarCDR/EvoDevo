"""Tests for evodevo.parallel.worker_pool / evodevo.parallel.evaluator_entry.

Runs real subprocess pools (spawn context) throughout -- these tests are
slower than the rest of tests/parallel/ but that concurrency is exactly
what's under test. Note: importing numpy/torch at THIS file's own top level
is fine (unlike tests/parallel/test_threading_control.py, which
deliberately controls sys.modules state to test the import-order guard
itself) -- under pytest, sys.modules["__main__"] is pytest's own launcher,
not this file, so nothing this module imports affects what a freshly
spawned worker imports before its initializer runs.
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.parallel.threading_control import THREAD_ENV_VARS
from evodevo.parallel.topology import cpu_slots, detect_topology
from evodevo.parallel.worker_pool import EvaluationPool, EvaluationTask

HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None

GOOD_WALKER = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=int,
)

_FACTORS_META = {"factors": {"dev_steps": 1, "init_condition": "center_seed"}}


def _direct_encoder():
    factors = ExperimentFactors(dev_steps=1, init_condition="center_seed", seed=0)
    return get_encoder("direct", factors=factors)


def _random_tasks(n: int) -> list:
    """A mix of valid and invalid random direct-encoded bodies -- the
    determinism guarantee must hold across both, since an invalid body
    still consumes a real evaluator call (evaluate_body_surrogate) and
    still produces a result."""
    enc = _direct_encoder()
    tasks = []
    for i in range(n):
        rng = np.random.default_rng(i)
        genome = enc.random_genome(rng)
        tasks.append(
            EvaluationTask(
                task_id=i,
                arm="direct",
                strategy="ea",
                task="Walker-v0",
                seed=i,
                genome=enc.serialize_genome(genome),
                ppo_timesteps=0,
                eval_episodes=0,
                meta=dict(_FACTORS_META),
            )
        )
    return tasks


def _good_walker_task(task_id) -> EvaluationTask:
    enc = _direct_encoder()
    genome = GOOD_WALKER.flatten().astype(np.int64)
    return EvaluationTask(
        task_id=task_id,
        arm="direct",
        strategy="ea",
        task="Walker-v0",
        seed=0,
        genome=enc.serialize_genome(genome),
        ppo_timesteps=0,
        eval_episodes=0,
        meta=dict(_FACTORS_META),
    )


def _plan(n_workers: int):
    return cpu_slots(detect_topology(), n_workers)


# --- D7: cross-worker-count determinism --------------------------------


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_determinism_matches_single_worker(n_workers):
    """The explicit D7 assertion: concurrency affects how long a result
    takes, never what the result is. Compares every worker count's
    (task_id -> fitness) mapping against n_workers=1's, computed once via
    the module-scoped fixture below."""
    tasks = _random_tasks(200)
    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=60) as pool:
        results = list(pool.map_tasks(tasks))
    mapping = {r.task_id: r.fitness for r in results}

    with EvaluationPool(n_workers=1, plan=_plan(1), mode="surrogate", timeout_s=60) as pool:
        baseline_results = list(pool.map_tasks(tasks))
    baseline = {r.task_id: r.fitness for r in baseline_results}

    assert mapping == baseline
    assert len(mapping) == 200


# --- thread/affinity pinning proof, surfaced via the first result per worker ---


def _forced_overlap_tasks(n: int, sleep_s: float = 0.2) -> list:
    """Debug-sleep tasks, not real evaluations. Surrogate evaluation is fast
    enough (sub-millisecond) that ProcessPoolExecutor can reuse 2-3 already-
    idle workers to drain a whole task list before a slower-to-spawn Nth
    worker process even finishes its initializer (worker spawn + configure_
    threading + seed_everything's numpy import is tens of milliseconds) --
    confirmed empirically: a 20-task, 4-worker surrogate run used only 3
    distinct worker PIDs. A small per-task delay guarantees a backlog
    persists long enough for every worker to actually get claimed at least
    once, which is what these two tests need to observe."""
    return [
        EvaluationTask(
            task_id=i, arm="", strategy="", task="", seed=0, genome=None,
            ppo_timesteps=0, eval_episodes=0, meta={"_debug_sleep_s": sleep_s},
        )
        for i in range(n)
    ]


def test_workers_report_single_thread_and_pinned_env_vars():
    n_workers = 4
    tasks = _forced_overlap_tasks(20)
    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=60) as pool:
        results = list(pool.map_tasks(tasks))

    reports = [r.threading_report for r in results if r.threading_report is not None]
    assert len(reports) == n_workers, "expected exactly one threading_report per worker (its first task)"
    for report in reports:
        for var in THREAD_ENV_VARS:
            assert report["env_after"][var] == ("FALSE" if var in ("MKL_DYNAMIC", "OMP_DYNAMIC") else "1")
        # configure_threading is step 1 of the initializer, before any
        # heavy import -- at that point neither numpy nor torch has been
        # imported yet in the fresh worker, so the guard always passes.
        assert report["guard_passed"] is True


def test_workers_get_distinct_cpu_affinity_sets():
    n_workers = 4
    topo = detect_topology()
    assert n_workers <= topo.logical_count, "test assumes this machine has >= 4 logical CPUs"

    tasks = _forced_overlap_tasks(20)
    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=60) as pool:
        results = list(pool.map_tasks(tasks))

    affinity_results = [r.affinity_result for r in results if r.affinity_result is not None]
    assert len(affinity_results) == n_workers
    cpu_sets = [tuple(ar["cpu_ids"]) for ar in affinity_results]
    assert len(set(cpu_sets)) == n_workers, f"expected {n_workers} distinct CPU sets, got {cpu_sets}"
    for ar in affinity_results:
        assert ar["applied"] is True


# --- timeout ---------------------------------------------------------------


def test_timeout_yields_timed_out_and_pool_continues():
    n_workers = 4
    tasks = _random_tasks(6)
    tasks.insert(3, EvaluationTask(
        task_id="sleeper", arm="", strategy="", task="", seed=0, genome=None,
        ppo_timesteps=0, eval_episodes=0, meta={"_debug_sleep_s": 3.0},
    ))

    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=1.0) as pool:
        results = list(pool.map_tasks(tasks))

    assert len(results) == len(tasks)
    by_id = {r.task_id: r for r in results}
    assert by_id["sleeper"].timed_out is True
    assert by_id["sleeper"].error is None
    for task_id in range(6):
        assert by_id[task_id].error is None
        assert by_id[task_id].timed_out is False

    stats = pool.stats()
    assert stats["tasks_timed_out"] == 1
    assert stats["tasks_completed"] == 6


# --- crash isolation ---------------------------------------------------------


def test_crash_yields_error_after_retry_and_pool_continues():
    n_workers = 4
    tasks = _random_tasks(6)
    tasks.insert(3, EvaluationTask(
        task_id="crasher", arm="", strategy="", task="", seed=0, genome=None,
        ppo_timesteps=0, eval_episodes=0, meta={"_debug_os_exit": 1},
    ))

    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=30, max_retries=1) as pool:
        results = list(pool.map_tasks(tasks))

    assert len(results) == len(tasks)
    by_id = {r.task_id: r for r in results}
    assert by_id["crasher"].error is not None
    assert by_id["crasher"].timed_out is False
    for task_id in range(6):
        assert by_id[task_id].error is None

    stats = pool.stats()
    assert stats["tasks_failed"] == 1
    assert stats["workers_restarted"] >= 1
    assert stats["tasks_completed"] == 6


# --- peak RSS -----------------------------------------------------------------


def test_peak_rss_populated_and_positive():
    n_workers = 2
    # forced-overlap (not real evaluations): this test only checks that RSS
    # got recorded per worker, not fitness values -- see _forced_overlap_tasks.
    tasks = _forced_overlap_tasks(10)
    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="surrogate", timeout_s=60) as pool:
        results = list(pool.map_tasks(tasks))

    assert len(results) == 10
    for r in results:
        assert r.peak_rss_mb is not None
        assert r.peak_rss_mb > 0

    stats = pool.stats()
    assert stats["peak_rss_mb_aggregate"] > 0
    assert len(stats["peak_rss_mb_by_worker_pid"]) == n_workers


# --- real PPO, slow --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_ppo_pool_smoke():
    n_workers = 2
    tasks = [_good_walker_task(i) for i in range(4)]
    for i, t in enumerate(tasks):
        tasks[i] = EvaluationTask(
            task_id=t.task_id, arm=t.arm, strategy=t.strategy, task=t.task,
            seed=i, genome=t.genome, ppo_timesteps=2000, eval_episodes=1, meta=t.meta,
        )

    with EvaluationPool(n_workers=n_workers, plan=_plan(n_workers), mode="real", timeout_s=120) as pool:
        results = list(pool.map_tasks(tasks))

    assert len(results) == 4
    for r in results:
        assert r.error is None
        assert r.timed_out is False
        assert r.valid is True
        assert math.isfinite(r.fitness)

    stats = pool.stats()
    assert stats["workers_restarted"] == 0
