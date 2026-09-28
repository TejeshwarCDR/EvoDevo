"""The concurrent body-evaluation pool.

WHY spawn, never fork: this project's evaluation path loads EvoGym's
compiled C++ simulator (a pybind11 extension with its own internal state
and OpenGL/OSMesa context), links against OpenMP-threaded BLAS libraries,
and -- once GPU support exists anywhere in the stack -- would touch CUDA
contexts. All three are fork-hostile: a forked child inherits the parent's
memory image AS OF THE FORK, including any already-initialized C library
state, background threads that don't survive the fork, and (for CUDA
specifically) a context that is simply invalid in the child and will
segfault or hang on first use. These failures are classic "works on my
laptop, hangs in CI" bugs because they depend on exactly what the parent
process happened to have imported/initialized before the fork. `spawn`
starts each worker as a brand-new Python interpreter that imports only what
it explicitly needs, in an order this module controls -- slower to start
than fork, but the only mode that is actually safe here. See
`multiprocessing.get_context("spawn")` below; this is not configurable per
call, deliberately.

WHY one thread per worker: PyTorch, NumPy's BLAS backend, and OpenMP all
default to using every logical CPU for their own internal thread pool. If
N worker processes each also spawn ~28 OpenMP threads, N*28 threads
compete for 28 logical CPUs -- massive, purely-destructive contention that
produces no error and no crash, just a slow run that looks like nothing is
wrong. Instead, every worker is pinned to exactly one thread
(`configure_threading(1)`, evodevo.parallel.threading_control) AND to a
specific CPU (`pin_current_process`, evodevo.parallel.affinity): the pool's
own concurrency (n_workers processes) is the only source of parallelism,
and each worker gets exactly the CPU resources its pinning was designed to
give it. This is what makes the B2 concurrency sweep this pool exists to
support meaningful: without it, adding workers could make the campaign
SLOWER, not faster, for reasons invisible in any log.

WHY the initializer runs configure_threading() before any other import:
OMP_NUM_THREADS and friends are read once, by OpenMP/MKL's own C
initialization code, the first time numpy or torch is imported in a
process. A worker that imports numpy first and sets the env vars second has
already lost -- see evodevo/parallel/threading_control.py's docstring for
the full mechanism. This module and its initializer are therefore written
so that NOTHING which could transitively import numpy or torch is
imported at worker_pool.py's own module level; the actual evaluation code
(evodevo.parallel.evaluator_entry, which needs numpy and, transitively,
torch) is only imported inside the worker's task loop, which the stdlib
ProcessPoolExecutor protocol guarantees runs strictly after the
initializer has already completed.
"""
from __future__ import annotations

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

from evodevo.parallel.affinity import pin_current_process
from evodevo.parallel.threading_control import configure_threading
from evodevo.parallel.topology import AffinityPlan

# Populated once per worker process by _worker_initializer, read by
# evodevo.parallel.evaluator_entry.run_task. Lives here (not in
# evaluator_entry) specifically so that importing THIS module -- which is
# what the fresh worker must do to resolve the initializer reference --
# never has to import evaluator_entry (and therefore never imports numpy)
# before the initializer has run. See the module docstring.
_WORKER_STATE: dict = {}


def _run_ppo_task(task, timeout_s):
    """Worker entry point for campaign PPO units; heavy imports stay lazy."""
    import dataclasses
    import signal
    import time

    from evodevo.evaluation.ppo_runner import PPOTaskResult, evaluate_ppo_task

    previous = None
    if timeout_s is not None and hasattr(signal, "setitimer"):
        previous = signal.getsignal(signal.SIGALRM)

        def timeout_handler(_signum, _frame):
            raise TimeoutError(f"PPO task timed out after {timeout_s}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    started = time.time()
    try:
        result = evaluate_ppo_task(task)
    except BaseException as exc:  # a task failure must return to the pool
        result = PPOTaskResult(None, False, f"infra_failure: {type(exc).__name__}: {exc}",
                               time.time() - started, None, None, [], None, None, str(exc), None)
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    import os
    import sys

    from evodevo.parallel.threading_control import THREAD_ENV_VARS

    torch_threads = sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None
    return dataclasses.replace(result, worker_slot=_WORKER_STATE.get("slot"),
                               cpu_ids=list(_WORKER_STATE.get("cpu_ids", [])),
                               worker_pid=os.getpid(), torch_num_threads=torch_threads,
                               thread_env={var: os.environ.get(var) for var in THREAD_ENV_VARS})


def _worker_initializer(slot_queue, slots: tuple) -> None:
    """Runs once, in a fresh spawned worker process, before that worker
    processes any task. Order is load-bearing -- see module docstring."""
    # (1) Pin thread counts BEFORE any heavy import. This is the entire
    # reason this function exists as the FIRST thing a worker does.
    threading_report = configure_threading(n_threads=1, strict=True)

    # (2) Claim this worker's slot index (0..n_workers-1) from the shared
    # queue -- whichever worker asks first gets slot 0, etc. Order across
    # workers is nondeterministic (OS scheduling of process startup) but
    # each worker gets exactly one distinct slot.
    slot = slot_queue.get()

    # (3) Pin this process to that slot's CPU set.
    cpu_ids = list(slots[slot])
    affinity_result = pin_current_process(cpu_ids)

    # (4) A segfault in EvoGym's C++ simulator should produce a traceback
    # in this worker's captured stderr, not a silent, unexplained death.
    import faulthandler

    faulthandler.enable()

    # (5) Seed this worker's own baseline RNG state. Lazy import: only now,
    # after (1), because evodevo.core.seeding imports numpy.
    from evodevo.core.seeding import seed_everything

    seed_everything(slot)

    _WORKER_STATE.update(
        {
            "slot": slot,
            "cpu_ids": cpu_ids,
            "threading_report": threading_report,
            "affinity_result": affinity_result,
            "reported": False,
            "tasks_processed": 0,
        }
    )


@dataclass(frozen=True)
class EvaluationTask:
    task_id: Any
    arm: str
    strategy: str
    task: str
    seed: int
    genome: Any  # serialized form, i.e. encoder.serialize_genome(genome)
    ppo_timesteps: int
    eval_episodes: int
    meta: dict = field(default_factory=dict)  # opaque to the pool; see evaluator_entry.py


@dataclass(frozen=True)
class WorkerResult:
    task_id: Any
    fitness: Optional[float]
    valid: bool
    reason: Optional[str]
    wall_clock_s: float
    body_hash: Optional[str]
    worker_slot: Optional[int]
    worker_pid: Optional[int]
    peak_rss_mb: Optional[float]
    cpu_ids: tuple
    timed_out: bool
    error: Optional[str]
    traceback: Optional[str]
    # Attached only for the FIRST result a given worker process produces --
    # proof, in the log, that thread pinning and CPU affinity actually
    # happened in that process, not just that the pool intended them to.
    threading_report: Optional[dict] = None
    affinity_result: Optional[dict] = None
    # Captured stdout/stderr from the task's execution, attached ONLY when
    # the task failed (error set or timed_out) -- see module docstring on
    # evaluator_entry.py. A normal invalid-body (valid=False, no error) is
    # not a "failure" here and carries no captured output.
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    # [(timesteps, fitness), ...] -- populated only when the task requested
    # one via meta["curve_request"] (see
    # evodevo.preflight.stages.b4_learning_curves, the only caller -- this
    # is a real production capability, not a debug-only hook like the
    # "_debug_*" meta keys in evaluator_entry.py); None for every ordinary
    # evaluation.
    curve: Optional[list] = None


class EvaluationPool:
    """Executes EvaluationTasks concurrently across n_workers pinned,
    single-threaded worker processes.

    mode: "real" or "surrogate" -- selects evodevo.evaluation.ppo_runner's
    evaluate_body or evaluate_body_surrogate (via make_evaluator), applied
    uniformly to every task this pool instance runs.

    timeout_s: per-task wall-clock budget, enforced INSIDE the worker via a
    POSIX interval timer (signal.setitimer(ITIMER_REAL), see
    evaluator_entry.py) -- a task that exceeds it is abandoned and the
    worker recovers to process the next task normally; this reliably
    interrupts ordinary Python/PPO/gym execution (including pure sleeps and
    training loops) but cannot interrupt a single blocking call that never
    returns control to the interpreter. None disables the timeout.

    max_tasks_per_child: after any worker has processed this many tasks,
    the ENTIRE pool is recycled (shut down and rebuilt with fresh worker
    processes) before further NEW tasks are submitted. Python 3.10's
    ProcessPoolExecutor has no native per-worker recycling (that arrived in
    3.11); replacing a single worker without tearing down the rest isn't
    possible through the public API, so this approximates it at pool
    granularity. None disables recycling.

    max_retries: how many additional attempts a task gets after its worker
    crashes (segfault, OOM kill, an explicit os._exit) before it is given
    up on and reported as a WorkerResult with `error` set. A stdlib
    ProcessPoolExecutor cannot isolate a crash to just the task that caused
    it -- ALL futures in flight when a worker dies are invalidated together
    (verified empirically against this Python version) -- so every task
    that was in flight at the time of a crash is retried, not only the one
    that actually caused it; retrying an already-successful task merely
    recomputes it.
    """

    def __init__(
        self,
        n_workers: int,
        plan: AffinityPlan,
        mode: str,
        timeout_s: Optional[float] = None,
        max_tasks_per_child: Optional[int] = None,
        max_retries: int = 1,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if mode not in ("real", "surrogate"):
            raise ValueError(f"mode must be 'real' or 'surrogate', got {mode!r}")

        self.n_workers = n_workers
        self.plan = plan
        self.mode = mode
        self.timeout_s = timeout_s
        self.max_tasks_per_child = max_tasks_per_child
        self.max_retries = max_retries

        # Lazy import: safe here, this is parent-process-only code and never
        # runs inside a fresh worker's bootstrap (see module docstring).
        from evodevo.parallel.evaluator_entry import run_task

        self._run_task = run_task

        self._ctx = multiprocessing.get_context("spawn")
        self._executor: Optional[ProcessPoolExecutor] = None
        self._ppo_inflight = 0  # read-only operational telemetry for campaign dashboard
        self._worker_task_counts: dict = {}  # worker_pid -> tasks processed since last rebuild

        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_timed_out": 0,
            "tasks_retried": 0,
            "workers_restarted": 0,
            "peak_rss_mb_aggregate": 0.0,
            "peak_rss_mb_by_worker_pid": {},
        }

        self._rebuild_executor()

    def _rebuild_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._stats["workers_restarted"] += 1

        slot_queue = self._ctx.Queue()
        for slot in range(self.n_workers):
            slot_queue.put(slot)

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=self._ctx,
            initializer=_worker_initializer,
            initargs=(slot_queue, self.plan.slots),
        )
        self._worker_task_counts = {}

    def _submit(self, task: EvaluationTask):
        return self._executor.submit(self._run_task, task, self.mode, self.timeout_s)

    def map_ppo_tasks(self, tasks: Iterable, timeout_s: float | None = None,
                      should_stop=None, grace_s: float = 120):
        """Yield (PPOTask, PPOTaskResult) as they finish on this one pool.

        A crashed worker invalidates all in-flight futures in stdlib's pool;
        those tasks are resubmitted after the pool is rebuilt. Ordinary
        failures/timeouts get two further tries on fresh worker processes.
        """
        from evodevo.evaluation.ppo_runner import PPOTaskResult

        pending = list(tasks)
        self._ppo_inflight = 0
        attempts = {}
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        stop_at = None

        def key(task):
            return (task.run_key, task.eval_index, task.ppo_k)

        while pending:
            if should_stop and should_stop():
                return
            # Drain one wave before recycling. This ensures every failed task
            # gets a genuinely fresh worker, and every completed result can
            # be journaled exactly once even if a sibling worker crashes.
            wave, pending = pending[:self.n_workers], pending[self.n_workers:]
            futures = {}
            for task in wave:
                identity = key(task)
                attempts[identity] = attempts.get(identity, 0) + 1
                futures[self._executor.submit(_run_ppo_task, task, timeout_s)] = task
            self._ppo_inflight = len(futures)
            retry = []
            while futures:
                if should_stop and should_stop() and stop_at is None:
                    import time
                    stop_at = time.monotonic()
                if stop_at is not None:
                    import time
                    remaining = grace_s - (time.monotonic() - stop_at)
                    if remaining <= 0:
                        for process in self._executor._processes.values():
                            if process.is_alive():
                                process.terminate()
                        self._executor.shutdown(wait=False, cancel_futures=True)
                        self._executor = None
                        self._ppo_inflight = 0
                        return
                    done, _ = wait(tuple(futures), timeout=min(1.0, remaining),
                                   return_when=FIRST_COMPLETED)
                else:
                    done, _ = wait(tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    self._ppo_inflight = len(futures)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = PPOTaskResult(None, False, f"infra_failure: {exc}", 0.0,
                                               None, None, [], None, None, str(exc), None)
                    if result.completed or attempts[key(task)] > self.max_retries:
                        yield task, result
                    else:
                        retry.append(task)
            if retry:
                if stop_at is not None:
                    return
                self._rebuild_executor()
                pending = retry + pending
        self._ppo_inflight = 0

    def _record_result_stats(self, result: WorkerResult) -> None:
        if result.timed_out:
            self._stats["tasks_timed_out"] += 1
        elif result.error is not None:
            self._stats["tasks_failed"] += 1
        else:
            self._stats["tasks_completed"] += 1

        if result.peak_rss_mb is not None:
            self._stats["peak_rss_mb_aggregate"] = max(
                self._stats["peak_rss_mb_aggregate"], result.peak_rss_mb
            )
            if result.worker_pid is not None:
                by_pid = self._stats["peak_rss_mb_by_worker_pid"]
                by_pid[result.worker_pid] = max(by_pid.get(result.worker_pid, 0.0), result.peak_rss_mb)

        if result.worker_pid is not None and not result.timed_out:
            self._worker_task_counts[result.worker_pid] = self._worker_task_counts.get(result.worker_pid, 0) + 1

    def _needs_recycle(self) -> bool:
        if self.max_tasks_per_child is None:
            return False
        return any(count >= self.max_tasks_per_child for count in self._worker_task_counts.values())

    def map_tasks(self, tasks: Iterable[EvaluationTask]) -> Iterator[WorkerResult]:
        """Yields WorkerResults as they complete, never holding the full
        result set in memory. Keeps at most n_workers tasks in flight at
        once (bounding how much retry-work a single worker crash can cause,
        since a crash invalidates every future currently in flight)."""
        pending = list(tasks)
        attempts: dict = {}
        in_flight: dict = {}  # future -> EvaluationTask

        def _submit_next() -> None:
            if pending and len(in_flight) < self.n_workers:
                task = pending.pop(0)
                attempts[task.task_id] = attempts.get(task.task_id, 0) + 1
                if attempts[task.task_id] > 1:
                    self._stats["tasks_retried"] += 1
                fut = self._submit(task)
                in_flight[fut] = task

        while pending and len(in_flight) < self.n_workers:
            _submit_next()

        while in_flight:
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)

            crashed = False
            settled: list = []  # (task, result_or_none, crash_exc_or_none)
            for fut in done:
                task = in_flight.pop(fut)
                try:
                    result = fut.result()
                    settled.append((task, result, None))
                except BrokenProcessPool as exc:
                    settled.append((task, None, exc))
                    crashed = True
                except Exception as exc:  # noqa: BLE001 -- defensive: run_task
                    # is designed to never let an ordinary exception escape,
                    # but if it somehow does, this must not kill the pool.
                    settled.append((task, None, exc))

            if crashed:
                # Every future still in flight (not just the ones in `done`)
                # was submitted to the now-broken executor and is poisoned --
                # pull them all back in as retry candidates too.
                for fut, task in list(in_flight.items()):
                    in_flight.pop(fut)
                    settled.append((task, None, BrokenProcessPool()))
                self._rebuild_executor()

            for task, result, exc in settled:
                if exc is not None:
                    # attempts[task_id] counts attempts already made (including
                    # this failed one). max_retries additional attempts are
                    # allowed beyond the first, i.e. total budget = max_retries + 1.
                    if attempts[task.task_id] <= self.max_retries:
                        pending.append(task)
                    else:
                        failure = WorkerResult(
                            task_id=task.task_id,
                            fitness=None,
                            valid=False,
                            reason=None,
                            wall_clock_s=0.0,
                            body_hash=None,
                            worker_slot=None,
                            worker_pid=None,
                            peak_rss_mb=None,
                            cpu_ids=(),
                            timed_out=False,
                            error=f"worker crashed after {attempts[task.task_id]} attempt(s): {exc}",
                            traceback=None,
                        )
                        self._record_result_stats(failure)
                        yield failure
                else:
                    self._record_result_stats(result)
                    yield result
                    if self._needs_recycle():
                        # Recycle before submitting further NEW work; tasks
                        # already in flight in this round are left to finish
                        # naturally (they were submitted before the limit
                        # was hit).
                        self._rebuild_executor()

            while pending and len(in_flight) < self.n_workers:
                _submit_next()

    def stats(self) -> dict:
        return {
            "tasks_completed": self._stats["tasks_completed"],
            "tasks_failed": self._stats["tasks_failed"],
            "tasks_timed_out": self._stats["tasks_timed_out"],
            "tasks_retried": self._stats["tasks_retried"],
            "workers_restarted": self._stats["workers_restarted"],
            "peak_rss_mb_aggregate": self._stats["peak_rss_mb_aggregate"],
            "peak_rss_mb_by_worker_pid": dict(self._stats["peak_rss_mb_by_worker_pid"]),
        }

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)

    def __enter__(self) -> "EvaluationPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)
