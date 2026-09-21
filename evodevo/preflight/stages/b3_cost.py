"""D5 B3: evaluation cost.

Stage `evaluation_cost` -- cost "expensive", depends_on ("concurrency_sweep",).
Runs at the concurrency B2's knee identified (overridable) and measures, at
the FULL campaign ppo_timesteps (never reduced -- see MUST NOT below), what
a real PPO evaluation actually costs, per arm and per task separately.

WHY per task, never extrapolated: harder terrain can simulate at a
genuinely different rate (more contacts, different episode lengths before
termination); assuming Walker-v0's t_PPO transfers to UpStepper-v0 is
exactly the confound this stage exists to rule out, so every (arm, task)
pair gets its own independent measurement -- there is no code path in this
module that infers one task's cost from another's.

THREE TIMINGS, not one, because "how much does an evaluation cost" is not
one number:
  - t_ppo: a real, full-budget PPO train+eval call on a VALID body -- the
    expensive path, and the only one whose cost scales with ppo_timesteps.
  - t_sentinel: decode() + validate_body() on an INVALID body -- the path
    every campaign genome that fails validation actually takes (see
    evodevo.evaluation.ppo_runner.evaluate_body's own early return). This
    measures whether that path really is close to free, rather than
    asserting it.
  - t_develop: decode() alone (no validation call at all) -- isolates the
    developmental process's own cost (NCA's rollout / rule_based's
    interpreter, both running T_primary=100 steps) from everything else,
    to confirm it is negligible against t_ppo. Meaningful for every arm,
    including direct, whose decode() is a pure reshape and should measure
    as close to zero (a useful negative control, not a wasted measurement).

t_ppo / t_sentinel is recorded explicitly per (arm, task) -- the single
number that explains why invalid genomes are (or are not) effectively free
compared to a real evaluation.

MUST NOT reduce ppo_timesteps here -- that reduction is B2's job (measuring
the SHAPE of the concurrency curve cheaply); conflating the two is exactly
how a wrong cost projection gets built from a cheap-mode number mistaken
for t_PPO. See evodevo/preflight/stages/b2_sweep.py's own
ppo_timesteps_is_reduced_not_campaign_budget field for the B2 side of this
same discipline.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

from evodevo.preflight.run_dir import read_json
from evodevo.preflight.stages.b2_sweep import CAMPAIGN_EVAL_EPISODES, _resolved_dev_steps_and_dynamics

DEFAULT_ARMS = ("direct", "nca", "rule_based")
DEFAULT_TASKS = ("Walker-v0", "UpStepper-v0")
DEFAULT_COST_REPS = 3
DEFAULT_PPO_TIMESTEPS = 1_000_000
DEFAULT_COST_MAX_DRAWS = 5000
DEFAULT_T_SENTINEL_N = 100
DEFAULT_T_DEVELOP_N = 100
DEFAULT_TASK_TIMEOUT_S = 7200.0

# Wide enough that per-arm internal seed ranges (develop/sentinel/valid-draw,
# each bounded by max(cost_max_draws, t_sentinel_n, t_develop_n), all far
# below this stride) can never collide between two different arms' ranges.
ARM_SEED_STRIDE = 100_000_000
_DEVELOP_OFFSET = 1_000_000
_SENTINEL_OFFSET = 2_000_000
_VALID_DRAW_OFFSET = 3_000_000


def describe_values(values: list) -> dict:
    """mean/median/sd/min/max/n -- the fixed reporting shape every timing
    in this stage (and evodevo.preflight.stages.b5_noise's noise numbers)
    uses. "Never a bare mean" (per this stage's own instruction): n alone
    tells a reader whether the rest of the row can be trusted, and sd is
    always present (None, not 0, for a single-sample n=1 -- a lone sample
    has no meaningful spread, and 0 would misrepresent that as "measured
    exactly zero variance")."""
    n = len(values)
    if n == 0:
        return {"mean": None, "median": None, "sd": None, "min": None, "max": None, "n": 0}
    mean = sum(values) / n
    sd = None
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = variance**0.5
    sorted_values = sorted(values)
    mid = n // 2
    median = sorted_values[mid] if n % 2 == 1 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    return {"mean": mean, "median": median, "sd": sd, "min": min(values), "max": max(values), "n": n}


def _percentile(values: list, p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    import math

    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _read_knee_level(ctx) -> int:
    sweep = read_json(ctx.run_dir.stage_output_path("concurrency_sweep"))
    knee = sweep.get("knee_level")
    if knee is None:
        raise RuntimeError(
            "concurrency_sweep's stage output has no knee_level -- cannot resolve a default "
            "--concurrency for evaluation_cost; pass --concurrency explicitly."
        )
    return knee


def _draw_valid_genomes(encoder, n_needed: int, max_draws: int, seed_base: int):
    import numpy as np

    from evodevo.morphology.validate import validate_body

    valid = []
    draws = 0
    while len(valid) < n_needed and draws < max_draws:
        genome_seed = seed_base + draws
        draws += 1
        genome = encoder.random_genome(np.random.default_rng(genome_seed))
        body = encoder.decode(genome)
        if validate_body(body).valid:
            valid.append((genome_seed, genome))
    return valid, draws


def _measure_develop(encoder, n: int, seed_base: int) -> list:
    import numpy as np

    times = []
    for i in range(n):
        genome = encoder.random_genome(np.random.default_rng(seed_base + i))
        t0 = time.perf_counter()
        encoder.decode(genome)
        times.append(time.perf_counter() - t0)
    return times


def _measure_sentinel(encoder, n_needed: int, max_draws: int, seed_base: int):
    import numpy as np

    from evodevo.morphology.validate import validate_body

    times = []
    draws = 0
    while len(times) < n_needed and draws < max_draws:
        genome_seed = seed_base + draws
        draws += 1
        genome = encoder.random_genome(np.random.default_rng(genome_seed))
        t0 = time.perf_counter()
        body = encoder.decode(genome)
        validation = validate_body(body)
        elapsed = time.perf_counter() - t0
        if not validation.valid:
            times.append(elapsed)
    return times, draws


class EvaluationCostStage:
    name = "evaluation_cost"
    description = (
        "Evaluation cost (D5 B3): t_PPO per arm x task at full ppo_timesteps under B2's chosen "
        "concurrency, plus t_sentinel (invalid-genome fast path) and t_develop (decode-only cost), "
        "and the t_PPO/t_sentinel ratio."
    )
    estimated_cost = "expensive"
    depends_on = ("concurrency_sweep",)

    def __init__(
        self,
        concurrency: Optional[int] = None,
        arms=DEFAULT_ARMS,
        tasks=DEFAULT_TASKS,
        cost_reps: int = DEFAULT_COST_REPS,
        ppo_timesteps: int = DEFAULT_PPO_TIMESTEPS,
        cost_max_draws: int = DEFAULT_COST_MAX_DRAWS,
        t_sentinel_n: int = DEFAULT_T_SENTINEL_N,
        t_develop_n: int = DEFAULT_T_DEVELOP_N,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        evaluator_mode: str = "real",
        workload_seed: int = 0,
        jitter: Optional[dict] = None,
        self_check_stamp: Optional[str] = None,
    ):
        if evaluator_mode not in ("real", "surrogate"):
            raise ValueError(f"evaluator_mode must be 'real' or 'surrogate', got {evaluator_mode!r}")
        self.concurrency = concurrency
        self.arms = tuple(arms)
        self.tasks = tuple(tasks)
        self.cost_reps = cost_reps
        self.ppo_timesteps = ppo_timesteps
        self.cost_max_draws = cost_max_draws
        self.t_sentinel_n = t_sentinel_n
        self.t_develop_n = t_develop_n
        self.task_timeout_s = task_timeout_s
        # "real" in every production path -- see module docstring. A
        # test-only override so this stage's own plumbing (statistics,
        # partial-result handling, ratio computation) can be exercised
        # without paying for PPO. Never set to "surrogate" from
        # evodevo/preflight/__main__.py's CLI wiring.
        self.evaluator_mode = evaluator_mode
        self.workload_seed = workload_seed
        self.jitter = jitter  # test-only, forwarded into EvaluationTask.meta["_debug_jitter"]
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        from evodevo.core.factors import extract_factors
        from evodevo.encoders import get_encoder
        from evodevo.parallel.worker_pool import EvaluationTask

        concurrency = self.concurrency
        concurrency_source = "override"
        if concurrency is None:
            concurrency = _read_knee_level(ctx)
            concurrency_source = "b2_knee"

        dev_steps, init_condition, dynamics_by_arm = _resolved_dev_steps_and_dynamics()

        ctx.events.record(self.name, "measurement", {"concurrency_used": concurrency, "concurrency_source": concurrency_source})

        arm_results = {}
        stage_partial = False

        for arm_index, arm_id in enumerate(self.arms):
            if arm_id not in dynamics_by_arm:
                raise ValueError(f"unknown arm '{arm_id}'; available: {sorted(dynamics_by_arm)}")
            encoder_name, dynamics_kwargs = dynamics_by_arm[arm_id]
            combined = {"dev_steps": dev_steps, "init_condition": init_condition, **dynamics_kwargs}
            arm_seed_base = self.workload_seed * 1_000_000_000 + arm_index * ARM_SEED_STRIDE

            factors, encoder_kwargs = extract_factors(combined, seed=arm_seed_base)
            encoder = get_encoder(encoder_name, factors=factors, **encoder_kwargs)
            factors_dict = {k: v for k, v in dataclasses.asdict(factors).items() if k != "seed"}
            meta_common = {"factors": factors_dict, "encoder_kwargs": encoder_kwargs}

            develop_start = time.time()
            develop_times = _measure_develop(encoder, self.t_develop_n, arm_seed_base + _DEVELOP_OFFSET)
            develop_end = time.time()
            t_develop_stats = describe_values(develop_times)
            t_develop_telemetry = ctx.telemetry.summarise(window=(develop_start, develop_end)) if ctx.telemetry is not None else None

            sentinel_start = time.time()
            sentinel_times, sentinel_draws = _measure_sentinel(
                encoder, self.t_sentinel_n, self.cost_max_draws, arm_seed_base + _SENTINEL_OFFSET
            )
            sentinel_end = time.time()
            t_sentinel_stats = describe_values(sentinel_times)
            t_sentinel_p99 = _percentile(sentinel_times, 99)
            t_sentinel_telemetry = ctx.telemetry.summarise(window=(sentinel_start, sentinel_end)) if ctx.telemetry is not None else None
            sentinel_shortfall = max(0, self.t_sentinel_n - len(sentinel_times))

            valid_genomes, n_draws = _draw_valid_genomes(
                encoder, self.cost_reps, self.cost_max_draws, arm_seed_base + _VALID_DRAW_OFFSET
            )
            draw_shortfall = max(0, self.cost_reps - len(valid_genomes))
            arm_partial = draw_shortfall > 0
            if arm_partial:
                ctx.events.record(
                    self.name, "anomaly",
                    {"arm": arm_id, "reason": "valid_genome_shortfall", "needed": self.cost_reps,
                     "drawn": len(valid_genomes), "draw_attempts": n_draws},
                )

            task_results = {}
            for task_name in self.tasks:
                if not valid_genomes:
                    task_results[task_name] = {
                        "t_ppo": describe_values([]),
                        "telemetry": None,
                        "observations": [],
                        "n_requested": self.cost_reps,
                        "n_successful": 0,
                        "n_failed": 0,
                        "failed_task_ids": [],
                        "t_ppo_over_t_sentinel_ratio": None,
                        "partial": True,
                    }
                    continue

                genome_by_seed = {}
                eval_tasks = []
                for genome_seed, genome in valid_genomes:
                    serialized = encoder.serialize_genome(genome)
                    genome_by_seed[genome_seed] = serialized
                    meta = dict(meta_common)
                    if self.jitter:
                        meta["_debug_jitter"] = self.jitter
                    eval_tasks.append(
                        EvaluationTask(
                            task_id=genome_seed, arm=encoder_name, strategy="preflight_b3",
                            task=task_name, seed=genome_seed, genome=serialized,
                            ppo_timesteps=self.ppo_timesteps, eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=meta,
                        )
                    )

                window_start = time.time()
                pool = ctx.make_pool(n_workers=concurrency, mode=self.evaluator_mode, timeout_s=self.task_timeout_s, max_retries=1)
                with pool:
                    results = list(pool.map_tasks(eval_tasks))
                window_end = time.time()

                successful = [r for r in results if r.error is None and not r.timed_out]
                failed = [r for r in results if r.error is not None or r.timed_out]
                t_ppo_times = [r.wall_clock_s for r in successful]
                t_ppo_stats = describe_values(t_ppo_times)
                telemetry_summary = ctx.telemetry.summarise(window=(window_start, window_end)) if ctx.telemetry is not None else None

                ratio = None
                if t_sentinel_stats["mean"] and t_ppo_stats["mean"] is not None:
                    ratio = t_ppo_stats["mean"] / t_sentinel_stats["mean"]

                observations = [
                    {
                        "arm_label": arm_id,
                        "encoder": encoder_name,
                        "genome_seed": r.task_id,
                        "genome": genome_by_seed[r.task_id],
                        "meta": meta_common,
                        "fitness": r.fitness,
                        "wall_clock_s": r.wall_clock_s,
                        "body_hash": r.body_hash,
                    }
                    for r in successful
                ]

                task_partial = len(successful) < self.cost_reps
                if task_partial:
                    arm_partial = True
                    ctx.events.record(
                        self.name, "anomaly",
                        {"arm": arm_id, "task": task_name, "reason": "ppo_execution_shortfall",
                         "requested": self.cost_reps, "successful": len(successful), "failed": len(failed)},
                    )
                if failed:
                    ctx.events.record(
                        self.name, "failure",
                        {"arm": arm_id, "task": task_name,
                         "failed_task_ids": [f.task_id for f in failed],
                         "errors": [f.error for f in failed]},
                    )

                task_results[task_name] = {
                    "t_ppo": t_ppo_stats,
                    "telemetry": telemetry_summary,
                    "observations": observations,
                    "n_requested": self.cost_reps,
                    "n_successful": len(successful),
                    "n_failed": len(failed),
                    "failed_task_ids": [f.task_id for f in failed],
                    "t_ppo_over_t_sentinel_ratio": ratio,
                    "partial": task_partial,
                }

            if arm_partial:
                stage_partial = True

            arm_results[arm_id] = {
                "encoder": encoder_name,
                "meta": meta_common,
                "t_develop": t_develop_stats,
                "t_develop_telemetry": t_develop_telemetry,
                "t_sentinel": t_sentinel_stats,
                "t_sentinel_p99": t_sentinel_p99,
                "t_sentinel_telemetry": t_sentinel_telemetry,
                "t_sentinel_draw_attempts": sentinel_draws,
                "t_sentinel_shortfall": sentinel_shortfall,
                "valid_genome_draw": {
                    "n_needed": self.cost_reps,
                    "n_drawn": len(valid_genomes),
                    "n_draw_attempts": n_draws,
                    "shortfall": draw_shortfall,
                },
                "tasks": task_results,
                "partial": arm_partial,
            }
            ctx.events.record(
                self.name, "measurement",
                {"arm": arm_id, "t_develop_mean_s": t_develop_stats["mean"], "t_sentinel_mean_s": t_sentinel_stats["mean"],
                 "valid_genomes_drawn": len(valid_genomes), "partial": arm_partial},
            )

        return {
            "concurrency_used": concurrency,
            "concurrency_source": concurrency_source,
            "ppo_timesteps": self.ppo_timesteps,
            "ppo_timesteps_is_campaign_budget": True,
            "eval_episodes": CAMPAIGN_EVAL_EPISODES,
            "arms": arm_results,
            "partial": stage_partial,
            "evaluator_mode": self.evaluator_mode,
            "self_check_stamp": self.self_check_stamp,
        }
