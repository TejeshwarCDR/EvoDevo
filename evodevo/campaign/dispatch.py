"""Pure scheduling policy shared by the campaign runner and its simulator."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable


@dataclass(frozen=True)
class DispatchItem:
    """One independently dispatchable task and its policy metadata."""

    task_id: Hashable
    phase: str
    run_key: str
    generation: int
    tie_key: tuple
    payload: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class BatchCompletion:
    """A run's generation barrier opened after its final task completed."""

    run_key: str
    generation: int


class DispatchCore:
    """Lowest-generation-first, bounded-prefetch dispatch state machine.

    Callers own execution and time. They report a complete ready batch for a
    run generation, repeatedly ask for the next task, and report completions.
    A new D7 phase cannot begin until every run in the active phase is marked
    complete.
    """

    def __init__(self, slots: int, prefetch: int = 0):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        if prefetch < 0:
            raise ValueError("prefetch must be >= 0")
        self.slots = int(slots)
        self.prefetch = int(prefetch)
        self.capacity = self.slots + self.prefetch
        self.active_phase: str | None = None
        self._phase_runs: set[str] = set()
        self._done_runs: set[str] = set()
        self._ready: list[tuple] = []
        self._ready_ids: set[Hashable] = set()
        self._inflight: dict[Hashable, DispatchItem] = {}
        self._batch_by_run: dict[str, tuple[int, set[Hashable]]] = {}
        self._task_batch: dict[Hashable, tuple[str, int]] = {}
        self._seen_ids: set[Hashable] = set()
        self._sequence = itertools.count()

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def inflight(self) -> frozenset[Hashable]:
        return frozenset(self._inflight)

    @property
    def phase_complete(self) -> bool:
        return self.active_phase is not None and self._done_runs == self._phase_runs

    def begin_phase(self, phase: str, run_keys: Iterable[str]) -> None:
        if self.active_phase is not None and not self.phase_complete:
            raise RuntimeError(f"active phase {self.active_phase!r} is not complete")
        runs = tuple(run_keys)
        if len(set(runs)) != len(runs):
            raise ValueError("phase run keys must be unique")
        self.active_phase = phase
        self._phase_runs = set(runs)
        self._done_runs = set()
        self._ready = []
        self._ready_ids = set()
        self._inflight = {}
        self._batch_by_run = {}
        self._task_batch = {}
        self._seen_ids = set()
        self._sequence = itertools.count()

    def report_ready(
        self, run_key: str, generation: int, items: Iterable[DispatchItem]
    ) -> BatchCompletion | None:
        if self.active_phase is None:
            raise RuntimeError("begin_phase() must be called before reporting work")
        if run_key not in self._phase_runs:
            raise ValueError(f"run {run_key!r} does not belong to active phase {self.active_phase!r}")
        if run_key in self._done_runs:
            raise RuntimeError(f"run {run_key!r} is already complete")
        if run_key in self._batch_by_run:
            prior = self._batch_by_run[run_key][0]
            raise RuntimeError(f"run {run_key!r} still has generation {prior} outstanding")

        batch = tuple(items)
        ids: set[Hashable] = set()
        for item in batch:
            if item.phase != self.active_phase:
                raise ValueError(
                    f"task phase {item.phase!r} does not match active phase {self.active_phase!r}"
                )
            if item.run_key != run_key or item.generation != generation:
                raise ValueError("task metadata does not match its reported run generation")
            if item.task_id in self._seen_ids or item.task_id in ids:
                raise ValueError(f"duplicate dispatch task id {item.task_id!r}")
            ids.add(item.task_id)

        if not batch:
            return BatchCompletion(run_key, generation)
        self._batch_by_run[run_key] = (generation, set(ids))
        for item in batch:
            self._seen_ids.add(item.task_id)
            self._ready_ids.add(item.task_id)
            self._task_batch[item.task_id] = (run_key, generation)
            heapq.heappush(
                self._ready,
                (item.generation, tuple(item.tie_key), next(self._sequence), item),
            )
        return None

    def next_task(self) -> DispatchItem | None:
        """Return the next task if the dispatch window has capacity."""
        if len(self._inflight) >= self.capacity or not self._ready:
            return None
        _generation, _tie, _sequence, item = heapq.heappop(self._ready)
        self._ready_ids.remove(item.task_id)
        self._inflight[item.task_id] = item
        return item

    def complete(self, task_id: Hashable) -> BatchCompletion | None:
        """Retire one task and identify a newly completed run generation."""
        if task_id not in self._inflight:
            raise KeyError(f"task {task_id!r} is not in flight")
        item = self._inflight.pop(task_id)
        run_key, generation = self._task_batch.pop(task_id)
        batch_generation, remaining = self._batch_by_run[run_key]
        if generation != batch_generation:
            raise AssertionError("dispatch batch generation drift")
        remaining.remove(item.task_id)
        if remaining:
            return None
        del self._batch_by_run[run_key]
        return BatchCompletion(run_key, generation)

    def mark_run_complete(self, run_key: str) -> None:
        if run_key not in self._phase_runs:
            raise ValueError(f"run {run_key!r} does not belong to the active phase")
        if run_key in self._batch_by_run:
            raise RuntimeError(f"run {run_key!r} still has an outstanding generation")
        self._done_runs.add(run_key)
