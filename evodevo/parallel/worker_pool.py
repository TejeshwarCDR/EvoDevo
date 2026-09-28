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
import itertools
import os
import signal
import time
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


def _run_ppo_task(task, timeout_s, attempt_token=0):
    """Worker entry point for campaign PPO units; heavy imports stay lazy."""
    import dataclasses
    import signal
    from evodevo.evaluation.ppo_runner import PPOTaskResult, _rss_mb, evaluate_ppo_task

    previous = None
    if timeout_s is not None and hasattr(signal, "setitimer"):
        previous = signal.getsignal(signal.SIGALRM)

        def timeout_handler(_signum, _frame):
            raise TimeoutError(f"PPO task timed out after {timeout_s}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    started = time.time()
    slot = _WORKER_STATE.get("slot")
    active_tokens = _WORKER_STATE.get("active_tokens")
    if active_tokens is not None and slot is not None:
        active_tokens[slot] = attempt_token
    try:
        debug = getattr(task, "debug_meta", None)
        validation = getattr(task, "validation_meta", None)
        if validation is not None:
            cpu_ids = list(_WORKER_STATE.get("cpu_ids", []))
            is_e_core = bool(cpu_ids and cpu_ids[0] in set(validation["e_core_ids"]))
            factor = validation["e_over_p_ratio"] if is_e_core else 1.0
            planned = validation["base_duration_s"] * validation["time_scale"] * factor
            time.sleep(planned)
            result = PPOTaskResult(
                validation["fitness"], True, None, time.time() - started,
                _rss_mb(), None, [], None, None, None, None,
                validation_base_duration_s=validation["base_duration_s"],
                validation_planned_duration_s=planned,
            )
        elif debug is not None:
            time.sleep(float(debug.get("sleep_s", 0.0)))
            if debug.get("os_exit"):
                os._exit(int(debug["os_exit"]))
            result = PPOTaskResult(float(debug.get("fitness", 0.0)), True, None,
                                   time.time() - started, None, None, [], None,
                                   None, None, None)
        else:
            result = evaluate_ppo_task(task)
    except BaseException as exc:  # a task failure must return to the pool
        result = PPOTaskResult(None, False, f"infra_failure: {type(exc).__name__}: {exc}",
                               time.time() - started, None, None, [], None, None, str(exc), None)
    finally:
        if active_tokens is not None and slot is not None:
            active_tokens[slot] = 0
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    import sys

    from evodevo.parallel.threading_control import THREAD_ENV_VARS

    torch_threads = sys.modules["torch"].get_num_threads() if "torch" in sys.modules else None
    finished = time.time()
    return dataclasses.replace(result, worker_slot=_WORKER_STATE.get("slot"),
                               cpu_ids=list(_WORKER_STATE.get("cpu_ids", [])),
                               worker_pid=os.getpid(), torch_num_threads=torch_threads,
                               thread_env={var: os.environ.get(var) for var in THREAD_ENV_VARS},
                               started_at=started, finished_at=finished)


def _worker_initializer(slot_queue, slots: tuple, ignore_sigint: bool = False,
                        active_tokens=None, worker_pids=None) -> None:
    """Runs once, in a fresh spawned worker process, before that worker
    processes any task. Order is load-bearing -- see module docstring."""
    # (1) Pin thread counts BEFORE any heavy import. This is the entire
    # reason this function exists as the FIRST thing a worker does.
    threading_report = configure_threading(n_threads=1, strict=True)

    # (1b) Campaign pools: a graceful stop signals the WHOLE process group
    # (scripts/campaign/stop_campaign.sh), workers included. An idle worker
    # interrupted inside call_queue.get() can die holding the queue's
    # process-shared lock; the surviving workers then block on it forever and
    # the parent's interpreter exit hangs joining them. Workers therefore
    # leave SIGINT to the parent, which lets in-flight tasks finish within
    # grace_s and then terminates the workers itself (map_ppo_tasks).
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # The campaign parent blocks SIGINT while eagerly spawning the pool,
        # so a group signal cannot interrupt Python import/bootstrap before
        # this initializer runs. Once SIG_IGN is installed, remove that
        # inherited mask; subsequent group SIGINTs are ignored normally.
        if hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})

    # (2) Claim this worker's slot index (0..n_workers-1) from the shared
    # queue -- whichever worker asks first gets slot 0, etc. Order across
    # workers is nondeterministic (OS scheduling of process startup) but
    # each worker gets exactly one distinct slot.
    slot = slot_queue.get()
    if worker_pids is not None:
        worker_pids[slot] = os.getpid()

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
            "active_tokens": active_tokens,
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
        ignore_sigint: bool = False,
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
        self.ignore_sigint = ignore_sigint

        # Lazy import: safe here, this is parent-process-only code and never
        # runs inside a fresh worker's bootstrap (see module docstring).
        from evodevo.parallel.evaluator_entry import run_task

        self._run_task = run_task

        self._ctx = multiprocessing.get_context("spawn")
        self._executor: Optional[ProcessPoolExecutor] = None
        self._ppo_inflight = 0  # read-only operational telemetry for campaign dashboard
        self._ppo_futures: dict = {}
        self._ppo_results: list = []
        self._ppo_failure_count: dict = {}
        self._ppo_token_sequence = itertools.count(1)
        self._ppo_timeout_s: float | None = None
        self._ppo_stop_at: float | None = None
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

    def _rebuild_executor(self, terminate: bool = False) -> None:
        if self._executor is not None:
            if terminate:
                for process in self._executor._processes.values():
                    if process.is_alive():
                        process.terminate()
            self._executor.shutdown(wait=False, cancel_futures=terminate)
            self._stats["workers_restarted"] += 1

        slot_queue = self._ctx.Queue()
        for slot in range(self.n_workers):
            slot_queue.put(slot)
        self._ppo_active_tokens = self._ctx.Array("q", self.n_workers, lock=False)
        self._ppo_worker_pids = self._ctx.Array("q", self.n_workers, lock=False)

        prior_mask = None
        if self.ignore_sigint and hasattr(signal, "pthread_sigmask"):
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=self._ctx,
                initializer=_worker_initializer,
                initargs=(slot_queue, self.plan.slots, self.ignore_sigint,
                          self._ppo_active_tokens, self._ppo_worker_pids),
            )
            if self.ignore_sigint:
                # ProcessPoolExecutor normally spawns lazily on submit. A
                # campaign group signal could otherwise hit a fresh child
                # during interpreter bootstrap, before the initializer has a
                # chance to ignore SIGINT. Start every worker under the mask
                # and wait until each initializer advertises its PID.
                while len(self._executor._processes) < self.n_workers:
                    self._executor._adjust_process_count()
                deadline = time.monotonic() + 30.0
                while any(pid == 0 for pid in self._ppo_worker_pids):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("campaign workers did not initialize within 30 seconds")
                    time.sleep(0.01)
        finally:
            if prior_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        self._worker_task_counts = {}

    def _submit(self, task: EvaluationTask):
        return self._executor.submit(self._run_task, task, self.mode, self.timeout_s)

    @staticmethod
    def _ppo_key(task):
        return task.run_key, task.eval_index, task.ppo_k

    def submit_ppo_task(self, task, timeout_s: float | None = None) -> None:
        """Submit one task without waiting; completions come from next_ppo_result."""
        # A timeout is uniform only within one active dispatch session.  The
        # campaign intentionally reuses this process pool across D7 phases,
        # whose independently measured p95 durations produce different
        # timeout budgets.  Once every future/result from the prior phase has
        # been consumed, the next submission starts a new session.
        if not self._ppo_futures and not self._ppo_results:
            self._ppo_timeout_s = self.timeout_s if timeout_s is None else timeout_s
        elif timeout_s is not None and timeout_s != self._ppo_timeout_s:
            raise ValueError("a PPO dispatch session cannot mix timeout values")
        identity = self._ppo_key(task)
        if any(self._ppo_key(record[0]) == identity for record in self._ppo_futures.values()):
            raise ValueError(f"PPO task {identity!r} is already submitted")
        self._submit_ppo_attempt(task)

    def _submit_ppo_attempt(self, task) -> None:
        token = next(self._ppo_token_sequence)
        future = self._executor.submit(_run_ppo_task, task, self._ppo_timeout_s, token)
        self._ppo_futures[future] = (task, token)
        self._ppo_inflight = len(self._ppo_futures)

    def _crashed_ppo_tokens(self) -> set[int]:
        """Tokens owned by workers that died independently of pool teardown."""
        deadline = time.monotonic() + 1.0
        while True:
            processes = {process.pid: process for process in self._executor._processes.values()}
            tokens = set()
            for slot in range(self.n_workers):
                pid = int(self._ppo_worker_pids[slot])
                token = int(self._ppo_active_tokens[slot])
                process = processes.get(pid)
                if (token and process is not None and process.exitcode is not None
                        and process.exitcode != -signal.SIGTERM):
                    tokens.add(token)
            if tokens or time.monotonic() >= deadline:
                return tokens
            time.sleep(0.01)

    def _infra_failure(self, task, count: int, detail: str):
        from evodevo.evaluation.ppo_runner import PPOTaskResult

        return PPOTaskResult(None, False,
            f"infra_failure: task failed after {count} attempt(s): {detail}",
            0.0, None, None, [], None, None, detail, None)

    def _charge_ppo_failure(self, task, detail: str):
        identity = self._ppo_key(task)
        count = self._ppo_failure_count.get(identity, 0) + 1
        self._ppo_failure_count[identity] = count
        self._stats["tasks_retried"] += int(count <= self.max_retries)
        if count > self.max_retries:
            return self._infra_failure(task, count, detail), False
        return None, True

    def _rebuild_ppo(self, retry_tasks) -> None:
        self._ppo_futures.clear()
        self._ppo_inflight = 0
        self._rebuild_executor(terminate=True)
        for task in retry_tasks:
            self._submit_ppo_attempt(task)

    def _handle_broken_ppo_pool(self) -> None:
        crashed_tokens = self._crashed_ppo_tokens()
        if not crashed_tokens:
            processes = {process.pid: process.exitcode
                         for process in self._executor._processes.values()}
            slots = [(int(self._ppo_worker_pids[index]),
                      int(self._ppo_active_tokens[index]))
                     for index in range(self.n_workers)]
            raise RuntimeError("broken PPO pool without an attributable crashed-worker token; "
                               f"processes={processes}, slots={slots}")
        retry = []
        for future, (task, token) in list(self._ppo_futures.items()):
            if future.done():
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is not None:
                    self._ppo_results.append((task, result))
                    continue
            if token in crashed_tokens:
                failure, should_retry = self._charge_ppo_failure(task, "worker process crashed")
                if failure is not None:
                    self._ppo_results.append((task, failure))
                if should_retry:
                    retry.append(task)
            else:
                # ProcessPoolExecutor poisons innocent futures after one worker
                # dies. Re-run them, but do not consume their retry budget.
                retry.append(task)
        self._rebuild_ppo(retry)

    def _handle_returned_ppo_failure(self, failed_task, failed_result) -> None:
        retry = []
        failure, should_retry = self._charge_ppo_failure(
            failed_task, failed_result.reason or failed_result.error or "infrastructure failure")
        if failure is not None:
            self._ppo_results.append((failed_task, failure))
        if should_retry:
            retry.append(failed_task)
        # A fresh worker is required for the retry. Every other future is
        # collateral work: resubmit it without charging a failure.
        retry.extend(task for task, _token in self._ppo_futures.values())
        self._rebuild_ppo(retry)

    def next_ppo_result(self, should_stop=None, grace_s: float = 120):
        """Return one final PPO result, transparently handling retries."""
        if self._ppo_results:
            return self._ppo_results.pop(0)
        while self._ppo_futures:
            if should_stop and should_stop() and self._ppo_stop_at is None:
                self._ppo_stop_at = time.monotonic()
            if self._ppo_stop_at is not None:
                remaining = grace_s - (time.monotonic() - self._ppo_stop_at)
                if remaining <= 0:
                    self._rebuild_executor(terminate=True)
                    self._ppo_futures.clear()
                    self._ppo_inflight = 0
                    return None
                done, _ = wait(tuple(self._ppo_futures), timeout=min(1.0, remaining),
                               return_when=FIRST_COMPLETED)
            else:
                done, _ = wait(tuple(self._ppo_futures), timeout=1.0,
                               return_when=FIRST_COMPLETED)
            if not done:
                continue
            broken = False
            for future in done:
                task, _token = self._ppo_futures.pop(future)
                try:
                    result = future.result()
                except BrokenProcessPool:
                    # Put it back: attribution considers the complete poisoned
                    # future set together.
                    self._ppo_futures[future] = (task, _token)
                    broken = True
                    break
                except Exception as exc:
                    from evodevo.evaluation.ppo_runner import PPOTaskResult
                    result = PPOTaskResult(None, False, f"infra_failure: {exc}", 0.0,
                                           None, None, [], None, None, str(exc), None)
                if not result.completed and str(result.reason).startswith("infra_failure:"):
                    self._handle_returned_ppo_failure(task, result)
                    break
                self._ppo_results.append((task, result))
            if broken:
                self._handle_broken_ppo_pool()
            self._ppo_inflight = len(self._ppo_futures)
            if self._ppo_results:
                return self._ppo_results.pop(0)
        self._ppo_stop_at = None
        self._ppo_timeout_s = None
        return None

    def ppo_failure_counts(self) -> dict:
        return {key: value for key, value in self._ppo_failure_count.items() if value}

    def map_ppo_tasks(self, tasks: Iterable, timeout_s: float | None = None,
                      should_stop=None, grace_s: float = 120):
        """Compatibility iterator over the asynchronous PPO interface."""
        if self._ppo_futures:
            raise RuntimeError("a PPO dispatch session is already active")
        for task in tasks:
            self.submit_ppo_task(task, timeout_s=timeout_s)
        while self._ppo_futures or self._ppo_results:
            completed = self.next_ppo_result(should_stop=should_stop, grace_s=grace_s)
            if completed is None:
                return
            yield completed

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
