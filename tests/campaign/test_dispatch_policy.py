"""Executable policy contract for the campaign's continuous dispatcher.

The virtual traces deliberately contain no processes, I/O, sleeps, or wall clock.
"""
from __future__ import annotations

import heapq
import itertools
import random
import statistics

import pytest

from evodevo.campaign.dispatch import DispatchCore, DispatchItem


def item(phase, run, generation, index, duration=1.0, value=None, arm=0, seed=0):
    return DispatchItem(
        task_id=(phase, run, generation, index),
        phase=phase,
        run_key=run,
        generation=generation,
        tie_key=(arm, seed, index, 0),
        payload={"duration": duration, "value": index if value is None else value},
    )


class FakePool:
    """Slot-limited pool with a FIFO submission queue and a virtual clock."""

    def __init__(self, slots):
        self.slots = slots
        self.now = 0.0
        self.running = []
        self.waiting = []
        self.dispatched = []
        self._sequence = itertools.count()

    def submit(self, task):
        self.dispatched.append((self.now, task))
        self.waiting.append(task)
        self._start_waiting()

    def _start_waiting(self):
        while self.waiting and len(self.running) < self.slots:
            task = self.waiting.pop(0)
            finish = self.now + task.payload["duration"]
            heapq.heappush(self.running, (finish, next(self._sequence), task))

    def complete_next(self):
        finish, _sequence, task = heapq.heappop(self.running)
        self.now = finish
        self._start_waiting()
        return task


def refill(core, pool):
    total = core.ready_count + core.inflight_count
    while (task := core.next_task()) is not None:
        pool.submit(task)
    assert core.inflight_count == min(core.capacity, total)
    assert not (core.ready_count and core.inflight_count < core.capacity)


def test_p1_generation_barrier_is_per_run_not_global():
    core = DispatchCore(slots=2, prefetch=0)
    pool = FakePool(slots=2)
    core.begin_phase("walker", ("fast", "slow"))
    core.report_ready("fast", 0, [item("walker", "fast", 0, 0, duration=1)])
    core.report_ready("slow", 0, [item("walker", "slow", 0, 0, duration=10)])
    refill(core, pool)

    completed = pool.complete_next()
    barrier = core.complete(completed.task_id)
    assert (barrier.run_key, barrier.generation, pool.now) == ("fast", 0, 1)
    core.report_ready("fast", 1, [item("walker", "fast", 1, 1, duration=1)])
    refill(core, pool)

    fast_g1_dispatch = next(t for t, task in pool.dispatched if task.run_key == "fast" and task.generation == 1)
    assert fast_g1_dispatch == 1.0
    assert fast_g1_dispatch < 10.0


def test_p2_refills_every_free_slot_without_drained_waves():
    core = DispatchCore(slots=2, prefetch=1)
    pool = FakePool(slots=2)
    core.begin_phase("walker", ("r",))
    tasks = [item("walker", "r", 0, i, duration=10 if i == 0 else 1) for i in range(6)]
    core.report_ready("r", 0, tasks)
    refill(core, pool)
    assert core.inflight_count == 3

    first = pool.complete_next()
    assert pool.now == 1.0
    assert core.complete(first.task_id) is None
    refill(core, pool)
    assert len(pool.dispatched) == 4
    assert pool.dispatched[-1][0] == 1.0
    assert any(task.task_id == tasks[0].task_id for _finish, _seq, task in pool.running)


def test_p3_lowest_generation_then_deterministic_tie_order():
    core = DispatchCore(slots=10)
    core.begin_phase("walker", ("a0", "a1", "b1", "b2"))
    offered = [
        item("walker", "b2", 2, 7, arm=1, seed=0),
        item("walker", "b1", 1, 8, arm=1, seed=0),
        item("walker", "a1", 1, 9, arm=0, seed=1),
        item("walker", "a0", 1, 3, arm=0, seed=0),
    ]
    for task in offered:
        core.report_ready(task.run_key, task.generation, [task])
    assert [core.next_task().task_id for _ in offered] == [
        offered[3].task_id, offered[2].task_id, offered[1].task_id, offered[0].task_id]


def test_p4_phase_barrier_blocks_the_next_d7_task():
    core = DispatchCore(slots=1)
    core.begin_phase("walker", ("r0", "r1"))
    with pytest.raises(RuntimeError, match="phase.*not complete"):
        core.begin_phase("upstepper", ("r0", "r1"))
    with pytest.raises(ValueError, match="active phase"):
        core.report_ready("r0", 0, [item("upstepper", "r0", 0, 0)])
    core.mark_run_complete("r0")
    with pytest.raises(RuntimeError, match="phase.*not complete"):
        core.begin_phase("upstepper", ("r0", "r1"))
    core.mark_run_complete("r1")
    assert core.phase_complete
    core.begin_phase("upstepper", ("r0", "r1"))
    assert core.active_phase == "upstepper"


def _outcome_for_completion_order(order):
    core = DispatchCore(slots=20)
    core.begin_phase("walker", ("a", "b"))
    tasks = [item("walker", run, 0, index, value=(index + 1) * (1 if run == "a" else 10))
             for run in ("a", "b") for index in range(5)]
    for run in ("a", "b"):
        core.report_ready(run, 0, [task for task in tasks if task.run_key == run])
    dispatched = {}
    while (task := core.next_task()) is not None:
        dispatched[task.task_id] = task
    results = {"a": {}, "b": {}}
    for task_id in order:
        task = dispatched[task_id]
        results[task.run_key][task.tie_key[2]] = task.payload["value"]
        core.complete(task_id)
    return {run: tuple(value for _index, value in sorted(values.items())) for run, values in results.items()}


def test_p5_completion_order_cannot_change_run_outcomes():
    ids = [("walker", run, 0, index) for run in ("a", "b") for index in range(5)]
    expected = _outcome_for_completion_order(ids)
    shuffled = list(ids)
    random.Random(20260928).shuffle(shuffled)
    assert _outcome_for_completion_order(shuffled) == expected


def test_p6_no_starvation_in_long_bounded_virtual_trace():
    """Derived trace bound: max lag behind median is <=1 generation.

    Derivation workload: 20 runs x 80 generations x 3 tasks, eight slots plus
    two prefetched tasks, deterministic task durations spanning 1..5 ticks.
    """
    runs = tuple(f"r{i:02d}" for i in range(20))
    generations = 80
    bound = 1
    core = DispatchCore(slots=8, prefetch=2)
    pool = FakePool(slots=8)
    completed_generation = {run: -1 for run in runs}
    observed_lag = 0
    core.begin_phase("walker", runs)

    def offer(run, generation):
        core.report_ready(run, generation, [
            item("walker", run, generation, i,
                 duration=1 + ((int(run[1:]) * 7 + generation * 3 + i) % 5),
                 seed=int(run[1:]))
            for i in range(3)])

    for run in runs:
        offer(run, 0)
    refill(core, pool)
    while pool.running:
        task = pool.complete_next()
        barrier = core.complete(task.task_id)
        if barrier is not None:
            completed_generation[barrier.run_key] = barrier.generation
            if barrier.generation + 1 < generations:
                offer(barrier.run_key, barrier.generation + 1)
            else:
                core.mark_run_complete(barrier.run_key)
            median = statistics.median(completed_generation.values())
            observed_lag = max(observed_lag, int(median - min(completed_generation.values())))
        refill(core, pool)

    assert core.phase_complete
    assert observed_lag == bound, f"derived starvation bound changed: observed {observed_lag}, documented {bound}"
