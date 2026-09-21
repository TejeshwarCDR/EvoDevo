"""D5 B5: evaluation noise.

Stage `evaluation_noise` -- cost "expensive", depends_on ("evaluation_cost",).
B5 is the stage most often skipped, and the one that determines whether
the campaign can answer its own question at all: if repeated PPO
evaluations of the SAME body disagree by more than the campaign hopes to
detect BETWEEN bodies or arms, no amount of population size fixes that --
the experiment is measuring noise, not signal. This stage exists to put a
real number on that risk before the campaign is scoped, not after.

WHAT IT MEASURES: for each task, three bodies spanning the low, middle and
high fitness values B3 actually observed (pooled across every arm B3
tested, NOT three arbitrary or hand-picked bodies -- see _select_bodies())
are each re-evaluated --noise-reps times with DIFFERENT PPO seeds and an
otherwise IDENTICAL configuration (same genome, same arm's resolved
factors/encoder_kwargs, same task, same ppo_timesteps and concurrency B3
used). Every individual fitness value is kept, not just the summary --see
each body's own "fitness_values" list in the stage output.

MUST NOT reuse a cached fitness: a fitness cache keyed by body hash alone
(evodevo.evaluation.cache.FitnessCache, the same cache the EA loop can
optionally enable) would return the SAME value for every repeated
evaluation of the SAME body, making the measured "noise" exactly zero by
construction and silently defeating this entire stage. This stage never
constructs or threads a FitnessCache anywhere in its own code (the worker-
pool path it uses, evodevo.parallel.worker_pool.EvaluationPool, has no
caching layer at all); cache_enabled=True is refused loudly at the top of
run() regardless, as a self-documenting guard against a future change
introducing one.

DOES NOT RESOLVE D16. This stage's decision_input output block states, in
plain sentences, what the measured noise implies (defensibility of a
single evaluation per genome, the cost multiplier of 2-3 PPO seeds) -- it
never picks one. That choice is made by Tejeshwar at the scoping gate.
"""
from __future__ import annotations

import os
from typing import Optional

from evodevo.preflight.run_dir import read_json
from evodevo.preflight.stages.b2_sweep import CAMPAIGN_EVAL_EPISODES
from evodevo.preflight.stages.b3_cost import describe_values

DEFAULT_NOISE_BODIES = 3
DEFAULT_NOISE_REPS = 5
DEFAULT_TASK_TIMEOUT_S = 7200.0


class CacheEnabledError(RuntimeError):
    """Raised when evaluation_noise is constructed/run with cache_enabled=True
    -- see module docstring for why a cache hit would silently zero out the
    entire measurement."""


def _tasks_from_b3(b3_output: dict) -> list:
    tasks = set()
    for arm_data in b3_output.get("arms", {}).values():
        tasks.update(arm_data.get("tasks", {}).keys())
    return sorted(tasks)


def _pool_observations_for_task(b3_output: dict, task_name: str) -> list:
    pooled = []
    for arm_data in b3_output.get("arms", {}).values():
        task_data = arm_data.get("tasks", {}).get(task_name)
        if not task_data:
            continue
        pooled.extend(task_data.get("observations", []))
    return pooled


def _select_bodies(pooled: list, n_bodies: int) -> list:
    """Picks n_bodies observations spanning the fitness range B3 actually
    observed for this task (pooled across every arm), evenly spaced by
    RANK, not by arbitrary selection. For the default n_bodies=3 this is
    exactly low/middle/high; for other values it generalises to evenly
    spaced quantiles of the observed distribution."""
    if not pooled:
        return []
    sorted_obs = sorted(pooled, key=lambda o: o["fitness"])
    n_available = len(sorted_obs)
    k = min(n_bodies, n_available)
    if k == 1:
        indices = [n_available // 2]
    else:
        indices = sorted({round(i * (n_available - 1) / (k - 1)) for i in range(k)})
    chosen = [sorted_obs[i] for i in indices]
    if len(chosen) == 3:
        labels = ["low", "middle", "high"]
    else:
        labels = [f"rank_{i + 1}_of_{len(chosen)}" for i in range(len(chosen))]
    return [dict(obs, label=label) for obs, label in zip(chosen, labels)]


def _pooled_sd(sd_n_pairs: list) -> Optional[float]:
    """Standard pooled-variance combination of several groups' own sample
    sds (each with its own n) into one within-group sd estimate."""
    numerator = 0.0
    denominator = 0
    for sd, n in sd_n_pairs:
        if n > 1 and sd is not None:
            numerator += (n - 1) * sd**2
            denominator += n - 1
    if denominator == 0:
        return None
    return (numerator / denominator) ** 0.5


def _evaluate_body_reps(ctx, task_name, body, noise_reps, ppo_timesteps, concurrency, evaluator_mode, jitter, seed_base, task_timeout_s):
    from evodevo.parallel.worker_pool import EvaluationTask

    eval_tasks = []
    for rep in range(noise_reps):
        meta = dict(body["meta"])
        if jitter:
            meta["_debug_jitter"] = jitter
        eval_tasks.append(
            EvaluationTask(
                task_id=rep, arm=body["encoder"], strategy="preflight_b5", task=task_name,
                seed=seed_base + rep, genome=body["genome"], ppo_timesteps=ppo_timesteps,
                eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=meta,
            )
        )
    n_workers = max(1, min(concurrency, noise_reps))
    pool = ctx.make_pool(n_workers=n_workers, mode=evaluator_mode, timeout_s=task_timeout_s, max_retries=1)
    with pool:
        return list(pool.map_tasks(eval_tasks))


def _build_decision_input(task_results: dict, b3_output: dict) -> dict:
    """Plain-sentence framing of what the measured noise implies for D16 --
    never a bare-numbers dump, and never a resolution of D16 itself."""
    sentences = []
    fractions = {
        t: r["within_over_between_fraction"]
        for t, r in task_results.items()
        if r.get("within_over_between_fraction") is not None
    }

    single_eval_defensible = None
    if fractions:
        for task_name, fraction in fractions.items():
            sentences.append(
                f"On {task_name}, the pooled within-body repeated-evaluation standard deviation is "
                f"{fraction:.1%} of the between-body fitness spread B3 observed on that task."
            )
        worst_task = max(fractions, key=lambda t: fractions[t])
        worst_fraction = fractions[worst_task]
        if worst_fraction < 0.10:
            single_eval_defensible = True
            sentences.append(
                f"The largest of these fractions ({worst_fraction:.1%}, on {worst_task}) is small "
                f"relative to the between-body spread, so a single PPO evaluation per genome appears "
                f"defensible on the evidence measured here."
            )
        elif worst_fraction < 0.30:
            single_eval_defensible = "borderline"
            sentences.append(
                f"The largest of these fractions ({worst_fraction:.1%}, on {worst_task}) is a "
                f"non-trivial share of the between-body spread -- not clearly negligible, and worth "
                f"weighing against the cost of repeated evaluation before treating a single "
                f"evaluation per genome as safe."
            )
        else:
            single_eval_defensible = False
            sentences.append(
                f"The largest of these fractions ({worst_fraction:.1%}, on {worst_task}) is NOT small "
                f"relative to the between-body spread the experiment hopes to distinguish -- a single "
                f"PPO evaluation per genome risks mistaking PPO training noise for a genuine "
                f"between-body or between-arm difference on that task."
            )
    else:
        sentences.append(
            "No between-body spread could be computed from B3's observations, so this stage cannot "
            "assess whether single-evaluation-per-genome is defensible."
        )

    t_ppo_means = [
        task_data["t_ppo"]["mean"]
        for arm_data in b3_output.get("arms", {}).values()
        for task_data in arm_data.get("tasks", {}).values()
        if task_data.get("t_ppo", {}).get("mean") is not None
    ]
    mean_t_ppo_s = (sum(t_ppo_means) / len(t_ppo_means)) if t_ppo_means else None
    if mean_t_ppo_s is not None:
        sentences.append(
            f"B3 measured a mean t_PPO of {mean_t_ppo_s:.1f}s per genome, averaged across the arms "
            f"and tasks it tested; evaluating with 2 PPO seeds per genome instead of 1 multiplies the "
            f"PPO evaluation budget by 2x, and 3 seeds multiplies it by 3x, with no change to any "
            f"other part of the cost structure."
        )

    sentences.append(
        "This stage produces the noise number and this framing only -- it does not choose between "
        "single-evaluation, 2-seed, or 3-seed evaluation for D16; that decision is made by "
        "Tejeshwar at the scoping gate."
    )

    return {
        "text": sentences,
        "within_over_between_fraction_by_task": fractions,
        "single_eval_defensible": single_eval_defensible,
        "cost_multiplier_2_seeds": 2,
        "cost_multiplier_3_seeds": 3,
        "mean_t_ppo_s_from_b3": mean_t_ppo_s,
    }


def _make_figure(path: str, task_results: dict, self_check_stamp: Optional[str] = None) -> None:
    import random as _random

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(task_results.keys())
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(5.5 * n_tasks, 4.5), squeeze=False)
    axes = axes[0]

    for ax, task_name in zip(axes, tasks):
        bodies = task_results[task_name]["bodies"]
        rng = _random.Random(0)
        for i, body in enumerate(bodies):
            values = body["fitness_values"]
            xs = [i + rng.uniform(-0.08, 0.08) for _ in values]
            ax.scatter(xs, values, alpha=0.7, color="tab:blue", zorder=2, label="individual eval" if i == 0 else None)
            if body["mean"] is not None:
                ax.hlines(
                    body["mean"], i - 0.22, i + 0.22, color="tab:red", linewidth=2.5, zorder=3,
                    label="mean" if i == 0 else None,
                )
        ax.set_xticks(range(len(bodies)))
        ax.set_xticklabels(
            [f"{b['label']}\n(B3 fitness={b['b3_fitness']:.2f})" for b in bodies], fontsize=8
        )
        ax.set_xlabel("body (selected by B3-observed fitness rank)")
        ax.set_ylabel("fitness (repeated evaluation)")
        ax.set_title(task_name)
        if bodies:
            ax.legend(fontsize=8, loc="best")

    suptitle = "B5 evaluation noise: repeated-evaluation fitness spread per body"
    if self_check_stamp:
        suptitle = f"{self_check_stamp} — {suptitle}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class EvaluationNoiseStage:
    name = "evaluation_noise"
    description = (
        "Evaluation noise (D5 B5): repeated-PPO-seed fitness spread for low/middle/high-fitness "
        "bodies per task, pooled-within-body sd as a fraction of B3's between-body spread, and a "
        "plain-language D16 decision-input block. Determines whether the campaign can answer its "
        "own question -- treated as first-class, not an appendix."
    )
    estimated_cost = "expensive"
    depends_on = ("evaluation_cost",)

    def __init__(
        self,
        noise_bodies: int = DEFAULT_NOISE_BODIES,
        noise_reps: int = DEFAULT_NOISE_REPS,
        concurrency: Optional[int] = None,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        evaluator_mode: str = "real",
        cache_enabled: bool = False,
        jitter: Optional[dict] = None,
        self_check_stamp: Optional[str] = None,
    ):
        if evaluator_mode not in ("real", "surrogate"):
            raise ValueError(f"evaluator_mode must be 'real' or 'surrogate', got {evaluator_mode!r}")
        self.noise_bodies = noise_bodies
        self.noise_reps = noise_reps
        self.concurrency = concurrency
        self.task_timeout_s = task_timeout_s
        # "real" in every production path -- see module docstring's own
        # MUST-NOT reasoning (shared with b3_cost.py/b2_sweep.py). Test-only.
        self.evaluator_mode = evaluator_mode
        # MUST default False -- see CacheEnabledError / module docstring.
        # Never set True from evodevo/preflight/__main__.py's CLI wiring.
        self.cache_enabled = cache_enabled
        self.jitter = jitter  # test-only, forwarded into EvaluationTask.meta["_debug_jitter"]
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        if self.cache_enabled:
            raise CacheEnabledError(
                "evaluation_noise requires cache_enabled=False -- a fitness-cache hit on a repeated "
                "evaluation of the same body would silently return the SAME cached fitness for "
                "every rep, making the measured noise exactly zero and defeating this stage's "
                "entire purpose. Refusing to run rather than silently producing a meaningless number."
            )

        b3_output = read_json(ctx.run_dir.stage_output_path("evaluation_cost"))
        ppo_timesteps = b3_output["ppo_timesteps"]
        concurrency = self.concurrency if self.concurrency is not None else b3_output["concurrency_used"]

        ctx.events.record(self.name, "measurement", {"ppo_timesteps": ppo_timesteps, "concurrency_used": concurrency})

        task_results = {}
        stage_partial = False

        for task_index, task_name in enumerate(_tasks_from_b3(b3_output)):
            pooled = _pool_observations_for_task(b3_output, task_name)
            bodies = _select_bodies(pooled, self.noise_bodies)

            if not bodies:
                task_results[task_name] = {
                    "bodies": [], "pooled_within_body_sd": None,
                    "between_body_spread_from_b3": describe_values([]),
                    "within_over_between_fraction": None,
                    "partial": True, "reason": "no B3 observations available for this task",
                }
                stage_partial = True
                ctx.events.record(self.name, "anomaly", {"task": task_name, "reason": "no_b3_observations"})
                continue

            body_records = []
            within_sd_n_pairs = []
            for body_index, body in enumerate(bodies):
                seed_base = 700_000_000 + task_index * 10_000_000 + body_index * 100_000
                results = _evaluate_body_reps(
                    ctx, task_name, body, self.noise_reps, ppo_timesteps, concurrency,
                    self.evaluator_mode, self.jitter, seed_base, self.task_timeout_s,
                )
                successful = [r for r in results if r.error is None and not r.timed_out]
                failed = [r for r in results if r.error is not None or r.timed_out]
                fitness_values = [r.fitness for r in successful]
                stats = describe_values(fitness_values)
                cv = None
                if stats["sd"] is not None and stats["mean"]:
                    cv = stats["sd"] / abs(stats["mean"])
                value_range = (stats["max"] - stats["min"]) if stats["min"] is not None else None

                body_partial = len(successful) < self.noise_reps
                if body_partial:
                    stage_partial = True
                    ctx.events.record(
                        self.name, "anomaly",
                        {"task": task_name, "body": body["label"], "reason": "rep_shortfall",
                         "requested": self.noise_reps, "successful": len(successful), "failed": len(failed)},
                    )
                if failed:
                    ctx.events.record(
                        self.name, "failure",
                        {"task": task_name, "body": body["label"], "errors": [f.error for f in failed]},
                    )

                body_records.append(
                    {
                        "label": body["label"],
                        "arm": body["arm_label"],
                        "encoder": body["encoder"],
                        "genome_seed": body["genome_seed"],
                        "body_hash": body.get("body_hash"),
                        "b3_fitness": body["fitness"],
                        "fitness_values": fitness_values,
                        "mean": stats["mean"],
                        "sd": stats["sd"],
                        "cv": cv,
                        "min": stats["min"],
                        "max": stats["max"],
                        "range": value_range,
                        "n_requested": self.noise_reps,
                        "n_successful": len(successful),
                        "n_failed": len(failed),
                        "failed_task_ids": [f.task_id for f in failed],
                        "partial": body_partial,
                    }
                )
                if stats["sd"] is not None:
                    within_sd_n_pairs.append((stats["sd"], stats["n"]))

            pooled_within_sd = _pooled_sd(within_sd_n_pairs)
            between_spread_stats = describe_values([o["fitness"] for o in pooled])
            between_body_sd = between_spread_stats["sd"]
            within_over_between = None
            if pooled_within_sd is not None and between_body_sd:
                within_over_between = pooled_within_sd / between_body_sd

            task_partial = any(b["partial"] for b in body_records)
            if task_partial:
                stage_partial = True

            task_results[task_name] = {
                "bodies": body_records,
                "pooled_within_body_sd": pooled_within_sd,
                "between_body_spread_from_b3": between_spread_stats,
                "within_over_between_fraction": within_over_between,
                "partial": task_partial,
            }

        figure_path = None
        try:
            figure_path = os.path.join(ctx.run_dir.report_dir, "b5_noise.png")
            _make_figure(figure_path, task_results, self_check_stamp=self.self_check_stamp)
        except Exception as e:  # noqa: BLE001 -- a figure failure must not lose the measured data
            ctx.events.record(self.name, "failure", {"error": f"figure generation failed: {type(e).__name__}: {e}"})
            figure_path = None

        decision_input = _build_decision_input(task_results, b3_output)

        return {
            "ppo_timesteps": ppo_timesteps,
            "concurrency_used": concurrency,
            "noise_bodies": self.noise_bodies,
            "noise_reps": self.noise_reps,
            "cache_enabled": self.cache_enabled,
            "tasks": task_results,
            "decision_input": decision_input,
            "figure_path": figure_path,
            "partial": stage_partial,
            "evaluator_mode": self.evaluator_mode,
            "self_check_stamp": self.self_check_stamp,
        }
