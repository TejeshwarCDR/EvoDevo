"""The picklable, module-level per-task function a worker process actually
runs (see evodevo/parallel/worker_pool.py for why this must live in a
SEPARATE module from the pool's initializer -- importing this module is
what pulls in numpy and, transitively, torch, and that must only happen
after configure_threading() has already run in the worker).

Reconstructs the encoder from the registry, decodes the task's genome, and
calls the EXISTING evaluator in evodevo.evaluation.ppo_runner
(evaluate_body / evaluate_body_surrogate, selected via make_evaluator) --
the identical function the single-process EA loop calls. This module does
not reimplement validation, sentinel-fitness handling, or fitness
computation; if reusing evaluate_body here required changing its
behaviour, that would be a sign this module's contract is wrong, not a
reason to fork the evaluation logic.
"""
from __future__ import annotations

import os
import resource
import signal
import sys
import tempfile
import time
import traceback as _traceback_module
from contextlib import contextmanager
from typing import Optional

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import DEFAULT_SENTINEL_FITNESS, make_evaluator
from evodevo.parallel import worker_pool as _worker_pool_module
from evodevo.parallel.worker_pool import EvaluationTask, WorkerResult


class _TaskTimeoutError(Exception):
    """Raised inside a worker when a task's SIGALRM timer fires."""


def _alarm_handler(signum, frame) -> None:
    raise _TaskTimeoutError()


def _synthesize_surrogate_curve(final_fitness, valid, curve_request, seed, jitter):
    """Test-only: a plausible rising-then-saturating synthetic curve (no
    real PPO) for exercising a preflight stage's curve-shaped plumbing
    (e.g. select_knee()) end-to-end without paying for training. An
    invalid body has nothing to develop a curve over."""
    if not valid:
        return []
    import random as _random

    max_timesteps = curve_request["max_timesteps"]
    eval_interval = curve_request["eval_interval"]
    rng = _random.Random(seed)
    noise_scale = (jitter or {}).get("fitness_noise_scale", 0.0)

    checkpoints = list(range(eval_interval, max_timesteps + 1, eval_interval))
    if not checkpoints or checkpoints[-1] != max_timesteps:
        checkpoints.append(max_timesteps)

    curve = []
    for t in checkpoints:
        progress = t / max_timesteps
        value = final_fitness * (1 - (1 - progress) ** 2)  # rises, saturates near max_timesteps
        if noise_scale:
            value += rng.gauss(0.0, noise_scale)
        curve.append((t, value))
    return curve


def _build_encoder(task: EvaluationTask):
    """meta is opaque to the pool itself (worker_pool.py never inspects
    it), but this module -- which actually needs to reconstruct an encoder
    -- reads two conventional keys out of it: "factors" (kwargs for
    ExperimentFactors, seed defaults to task.seed if omitted) and
    "encoder_kwargs" (any other constructor kwargs for the named encoder,
    e.g. mutation_rate)."""
    meta = task.meta or {}
    factors_kwargs = dict(meta.get("factors") or {})
    factors_kwargs.setdefault("seed", task.seed)
    factors = ExperimentFactors(**factors_kwargs)
    encoder_kwargs = dict(meta.get("encoder_kwargs") or {})
    return get_encoder(task.arm, factors=factors, **encoder_kwargs)


@contextmanager
def _capture_stdio():
    """Redirects fd 1/2 (not just sys.stdout/sys.stderr) to temp files for
    the duration of the block, so output from EvoGym's compiled simulator
    (which writes directly to the file descriptor, bypassing Python's
    sys.stdout object) is captured too, then restores the original fds.
    Never lets a capture-setup failure take down the task -- falls back to
    "no capture" if fd redirection isn't available."""
    try:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
    except OSError:
        yield None, None
        return

    out_f = tempfile.TemporaryFile(mode="w+b")
    err_f = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(out_f.fileno(), 1)
        os.dup2(err_f.fileno(), 2)
        yield out_f, err_f
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _read_and_close(f) -> Optional[str]:
    if f is None:
        return None
    f.seek(0)
    data = f.read().decode("utf-8", errors="replace")
    f.close()
    return data


def run_task(task: EvaluationTask, mode: str, timeout_s: Optional[float]) -> WorkerResult:
    state = _worker_pool_module._WORKER_STATE
    start = time.time()

    fitness = None
    valid = False
    reason = None
    body_hash_value = None
    timed_out = False
    error = None
    traceback_str = None
    curve = None

    have_alarm = timeout_s is not None and hasattr(signal, "setitimer")
    old_handler = None

    with _capture_stdio() as (out_f, err_f):
        try:
            if have_alarm:
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_s)

            # Debug-only hooks so tests can exercise the pool's timeout and
            # crash-isolation paths deterministically, without those paths
            # ever being reachable from a real evaluation (no real campaign
            # code sets these meta keys). Checked first, before any encoder
            # reconstruction, so a test task doesn't also need a valid arm.
            debug_meta = task.meta or {}
            if "_debug_sleep_s" in debug_meta:
                time.sleep(float(debug_meta["_debug_sleep_s"]))
            elif debug_meta.get("_debug_os_exit"):
                os._exit(int(debug_meta["_debug_os_exit"]))
            else:
                encoder = _build_encoder(task)
                genome = encoder.deserialize_genome(task.genome)
                body = encoder.decode(genome)

                sentinel_fitness = debug_meta.get("sentinel_fitness", DEFAULT_SENTINEL_FITNESS)
                curve_request = debug_meta.get("curve_request")

                if curve_request and mode == "real":
                    # A real production capability (evodevo.preflight.stages
                    # .b4_learning_curves), NOT a "_debug_*" test-only hook:
                    # one PPO training run producing a full fitness-vs-
                    # timesteps curve via a periodic-evaluation callback,
                    # never retraining from scratch per checkpoint.
                    from evodevo.evaluation.ppo_runner import evaluate_body_with_curve

                    curve_result = evaluate_body_with_curve(
                        body,
                        task=task.task,
                        max_timesteps=curve_request["max_timesteps"],
                        eval_interval=curve_request["eval_interval"],
                        eval_episodes=task.eval_episodes,
                        seed=task.seed,
                        sentinel_fitness=sentinel_fitness,
                    )
                    fitness = curve_result.fitness
                    valid = curve_result.valid
                    reason = curve_result.reason
                    body_hash_value = curve_result.body_hash
                    curve = curve_result.curve
                else:
                    evaluator = make_evaluator(mode)
                    eval_result = evaluator(
                        body,
                        task=task.task,
                        ppo_timesteps=task.ppo_timesteps,
                        eval_episodes=task.eval_episodes,
                        seed=task.seed,
                        sentinel_fitness=sentinel_fitness,
                    )
                    fitness = eval_result.fitness
                    valid = eval_result.valid
                    reason = eval_result.reason
                    body_hash_value = eval_result.body_hash

                    if curve_request and mode == "surrogate":
                        # Test-only: synthesizes a plausible rising/
                        # saturating curve (no real PPO) so a preflight
                        # stage's own plumbing/statistics can be exercised
                        # against a "surrogate-with-jitter" curve without
                        # paying for training. Never used in production --
                        # mode="real" is the only path that reaches
                        # evaluate_body_with_curve above.
                        curve = _synthesize_surrogate_curve(
                            fitness, valid, curve_request, task.seed, debug_meta.get("_debug_jitter")
                        )

                # Debug-only hook: lets tests exercise a "surrogate-with-
                # jitter" evaluator -- still the REAL evaluate_body/
                # evaluate_body_surrogate call above, with small, seeded
                # perturbation layered on top ONLY for a valid, non-curve
                # result (curve jitter, if any, was already applied inside
                # _synthesize_surrogate_curve above) -- so preflight stages
                # that need realistic repeat-evaluation variance (B3's
                # timing spread, B5's noise measurement) can be tested
                # without paying for real PPO. No production code path
                # sets this meta key.
                jitter = debug_meta.get("_debug_jitter")
                if jitter and valid and curve is None:
                    import random as _random

                    jitter_rng = _random.Random(task.seed)
                    noise_scale = jitter.get("fitness_noise_scale", 0.0)
                    if noise_scale:
                        fitness = fitness + jitter_rng.gauss(0.0, noise_scale)
                    sleep_s = jitter.get("sleep_s", 0.0)
                    if sleep_s:
                        time.sleep(sleep_s * jitter_rng.uniform(0.5, 1.5))
        except _TaskTimeoutError:
            timed_out = True
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: this is
            # the boundary between "this task failed" (caught here, worker
            # process itself is fine, continues to the next task) and "this
            # worker crashed" (a segfault/os._exit/OOM-kill, which by
            # definition cannot be caught by a try/except running inside the
            # process that just died -- see EvaluationPool's crash handling).
            error = f"{type(exc).__name__}: {exc}"
            traceback_str = _traceback_module.format_exc()
        finally:
            if have_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        stdout_text = _read_and_close(out_f)
        stderr_text = _read_and_close(err_f)

    wall_clock_s = time.time() - start
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    is_failure = error is not None or timed_out

    threading_report_dict = None
    affinity_result_dict = None
    if not state.get("reported"):
        tr = state.get("threading_report")
        ar = state.get("affinity_result")
        threading_report_dict = tr.to_dict() if tr is not None else None
        affinity_result_dict = ar.to_dict() if ar is not None else None
        state["reported"] = True

    state["tasks_processed"] = state.get("tasks_processed", 0) + 1

    return WorkerResult(
        task_id=task.task_id,
        fitness=fitness,
        valid=valid,
        reason=reason,
        wall_clock_s=wall_clock_s,
        body_hash=body_hash_value,
        worker_slot=state.get("slot"),
        worker_pid=os.getpid(),
        peak_rss_mb=peak_rss_mb,
        cpu_ids=tuple(state.get("cpu_ids") or ()),
        timed_out=timed_out,
        error=error,
        traceback=traceback_str,
        threading_report=threading_report_dict,
        affinity_result=affinity_result_dict,
        stdout=stdout_text if is_failure else None,
        stderr=stderr_text if is_failure else None,
        curve=curve,
    )
