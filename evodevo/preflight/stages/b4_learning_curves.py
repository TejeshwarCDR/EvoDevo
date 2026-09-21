"""D5 B4: learning curves.

Stage `learning_curves` -- cost "expensive", depends_on ("evaluation_cost",).
For each task, trains --curve-bodies representative bodies (spanning the
low/middle/high fitness B3 actually observed for that task, recorded by
body hash so the selection is auditable -- see _select_bodies, reused from
evodevo.preflight.stages.b5_noise) to --curve-max-timesteps, recording
fitness at every --curve-eval-interval steps via ONE periodic-evaluation
PPO training run per body (evodevo.evaluation.ppo_runner.
evaluate_body_with_curve) -- never retraining from scratch per checkpoint,
which would multiply cost by the number of checkpoints for no additional
information.

D6 KNEE-SELECTION RULE, implemented exactly as pre-registered in
select_knee() below: the smallest budget at which the median marginal
fitness improvement per additional 100k steps falls below 1% of the
fitness achieved so far, sustained over three consecutive windows.
select_knee() is a pure function of a single fitness-vs-timesteps series
(the across-bodies median at each checkpoint, computed by _median_series()
below) and is unit-tested independently of any PPO run.

D6 permits the selected budget to differ BETWEEN TASKS (harder terrain may
genuinely need more training before it plateaus) but every arm's bodies on
a given task are trained to the identical curve_max_timesteps here, and
the one budget selected per task applies uniformly across arms on that
task -- see each task's own "budget_note" field in the stage output.
"""
from __future__ import annotations

import os
from statistics import median
from typing import Optional

from evodevo.preflight.run_dir import read_json
from evodevo.preflight.stages.b2_sweep import CAMPAIGN_EVAL_EPISODES
from evodevo.preflight.stages.b5_noise import _pool_observations_for_task, _select_bodies, _tasks_from_b3

DEFAULT_CURVE_BODIES = 3
DEFAULT_CURVE_MAX_TIMESTEPS = 2_000_000
DEFAULT_CURVE_EVAL_INTERVAL = 100_000
DEFAULT_TASK_TIMEOUT_S = 14400.0

# D6's own wording, quoted verbatim -- evodevo/preflight/__main__.py's
# --dry-run and this stage's own JSON output both surface this exact text
# so a reviewer can check the rule that was actually applied against the
# rule that was pre-registered, not a paraphrase of it.
SELECT_KNEE_RULE_TEXT = (
    "The smallest budget at which the median marginal fitness improvement per additional "
    "100k steps falls below 1% of the fitness achieved so far, sustained over three "
    "consecutive windows."
)

BUDGET_NOTE = (
    "D6 permits differing selected budgets BETWEEN TASKS (harder terrain may need more "
    "training before it plateaus) but not between arms on the same task -- every arm's bodies "
    "on a given task are trained to the identical curve_max_timesteps here, and the budget "
    "selected for a task applies uniformly to every arm measured on that task."
)


def select_knee(series: list, threshold: float = 0.01, sustained_windows: int = 3) -> dict:
    """D6's pre-registered rule -- see module docstring for the exact
    wording this implements. `series` is a single fitness-vs-timesteps
    curve: a list of (timesteps, fitness) pairs (already the across-bodies
    median at each checkpoint when called from this stage's run(); a
    hand-built single curve is exactly as valid here, which is why this
    function is unit-tested directly against synthetic series rather than
    only through a real PPO run).

    Never raises -- a still-rising series (no 3 consecutive windows below
    threshold) is a real, reportable outcome ("knee_found": False), not an
    error; the module's own WARNING ("no plateau observed, budget is a
    floor") is built from this at the call site (this function does not
    know about --curve-max-timesteps, only about the series it was given).
    """
    sorted_series = sorted(series, key=lambda p: p[0])

    windows = []
    for (t0, f0), (t1, f1) in zip(sorted_series, sorted_series[1:]):
        step = t1 - t0
        if step <= 0:
            continue
        raw_delta = f1 - f0
        normalized_per_100k = raw_delta / step * 100_000.0
        denom = abs(f0)
        if denom > 1e-12:
            relative = normalized_per_100k / denom
        else:
            relative = 0.0 if abs(normalized_per_100k) < 1e-12 else float("inf")
        windows.append({"window_start": t0, "window_end": t1, "relative_improvement": relative})

    knee_timesteps = None
    knee_window_index = None
    n = len(windows)
    for i in range(n - sustained_windows + 1):
        candidate = windows[i : i + sustained_windows]
        if all(w["relative_improvement"] < threshold for w in candidate):
            knee_timesteps = candidate[0]["window_start"]
            knee_window_index = i
            break

    return {
        "knee_timesteps": knee_timesteps,
        "knee_found": knee_timesteps is not None,
        "knee_window_index": knee_window_index,
        "threshold": threshold,
        "sustained_windows": sustained_windows,
        "windows": windows,
        "max_timesteps_in_series": sorted_series[-1][0] if sorted_series else None,
        "rule_text": SELECT_KNEE_RULE_TEXT,
    }


def _median_series(curves: list) -> list:
    """Aligns several bodies' own curves BY CHECKPOINT INDEX -- every body
    on a task is trained under the identical max_timesteps/eval_interval
    schedule, so each curve's i-th point is "the same" checkpoint even
    though PPO's own rollout-buffer batching (n_steps, unchanged from
    evaluate_body()'s own default -- see evaluate_body_with_curve's
    docstring) can make the exact timestep counts differ by a handful of
    steps between bodies. Returns the per-checkpoint MEDIAN fitness (and
    the median of the actual per-body timesteps at that index, robust to
    that small cross-body drift), across whichever bodies still have a
    point at that index (a body whose training failed partway through
    simply contributes fewer points)."""
    if not curves:
        return []
    max_len = max(len(c) for c in curves)
    series = []
    for i in range(max_len):
        points_at_i = [curve[i] for curve in curves if i < len(curve)]
        if not points_at_i:
            continue
        t = median(p[0] for p in points_at_i)
        f = median(p[1] for p in points_at_i)
        series.append((t, f))
    return series


def _make_figure(path: str, task_results: dict, rule_text: str, self_check_stamp: Optional[str] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(task_results.keys())
    n_tasks = max(1, len(tasks))
    fig, axes = plt.subplots(1, n_tasks, figsize=(6.5 * n_tasks, 5), squeeze=False)
    axes = axes[0]

    for ax, task_name in zip(axes, tasks):
        data = task_results[task_name]
        for body in data["bodies"]:
            if body["curve"]:
                ts = [p[0] for p in body["curve"]]
                fs = [p[1] for p in body["curve"]]
                ax.plot(ts, fs, marker="o", markersize=3, alpha=0.6, linewidth=1,
                         label=f"{body['label']} ({body['arm']})")
        if data["median_series"]:
            ts = [p[0] for p in data["median_series"]]
            fs = [p[1] for p in data["median_series"]]
            ax.plot(ts, fs, color="black", linewidth=2, linestyle="--", label="median across bodies")

        budget = data["selected_budget"]
        if budget is not None:
            ax.axvline(x=budget, color="red", linestyle=":", linewidth=2)
            tag = "floor (no plateau)" if data["no_plateau_observed"] else "knee"
            ax.annotate(
                f"{tag}\nt={budget:,}", xy=(budget, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=7, color="red", va="bottom",
            )

        ax.set_xlabel("PPO timesteps")
        ax.set_ylabel("fitness")
        ax.set_title(task_name)
        if data["bodies"]:
            ax.legend(fontsize=7, loc="best")

    suptitle = f"B4 learning curves -- D6 knee rule: {rule_text}"
    if self_check_stamp:
        suptitle = f"{self_check_stamp} — {suptitle}"
    fig.suptitle(suptitle, fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class LearningCurvesStage:
    name = "learning_curves"
    description = (
        "Learning curves (D5 B4): fitness-vs-timesteps per representative body per task (one PPO "
        "training run per body, periodic-eval callback), the D6 knee-selection rule applied per "
        "task, and a WARNING wherever no plateau was observed within curve_max_timesteps."
    )
    estimated_cost = "expensive"
    depends_on = ("evaluation_cost",)

    def __init__(
        self,
        curve_bodies: int = DEFAULT_CURVE_BODIES,
        curve_max_timesteps: int = DEFAULT_CURVE_MAX_TIMESTEPS,
        curve_eval_interval: int = DEFAULT_CURVE_EVAL_INTERVAL,
        concurrency: Optional[int] = None,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        knee_threshold: float = 0.01,
        knee_sustained_windows: int = 3,
        evaluator_mode: str = "real",
        jitter: Optional[dict] = None,
        self_check_stamp: Optional[str] = None,
    ):
        if evaluator_mode not in ("real", "surrogate"):
            raise ValueError(f"evaluator_mode must be 'real' or 'surrogate', got {evaluator_mode!r}")
        self.curve_bodies = curve_bodies
        self.curve_max_timesteps = curve_max_timesteps
        self.curve_eval_interval = curve_eval_interval
        self.concurrency = concurrency
        self.task_timeout_s = task_timeout_s
        self.knee_threshold = knee_threshold
        self.knee_sustained_windows = knee_sustained_windows
        # "real" in every production path -- test-only override, same
        # discipline as b2_sweep/b3_cost/b5_noise. Never set to "surrogate"
        # from evodevo/preflight/__main__.py's CLI wiring.
        self.evaluator_mode = evaluator_mode
        self.jitter = jitter  # test-only, forwarded into EvaluationTask.meta["_debug_jitter"]
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        from evodevo.parallel.worker_pool import EvaluationTask

        b3_output = read_json(ctx.run_dir.stage_output_path("evaluation_cost"))
        concurrency = self.concurrency if self.concurrency is not None else b3_output["concurrency_used"]

        ctx.events.record(
            self.name, "measurement",
            {"concurrency_used": concurrency, "curve_max_timesteps": self.curve_max_timesteps,
             "curve_eval_interval": self.curve_eval_interval},
        )

        task_results = {}
        stage_partial = False

        for task_name in _tasks_from_b3(b3_output):
            pooled = _pool_observations_for_task(b3_output, task_name)
            bodies = _select_bodies(pooled, self.curve_bodies)

            if not bodies:
                task_results[task_name] = {
                    "bodies": [], "median_series": [], "knee": None, "selected_budget": None,
                    "no_plateau_observed": None, "warning": None,
                    "partial": True, "reason": "no B3 observations available for this task",
                }
                stage_partial = True
                ctx.events.record(self.name, "anomaly", {"task": task_name, "reason": "no_b3_observations"})
                continue

            eval_tasks = []
            for body in bodies:
                meta = dict(body["meta"])
                if self.jitter:
                    meta["_debug_jitter"] = self.jitter
                meta["curve_request"] = {
                    "max_timesteps": self.curve_max_timesteps, "eval_interval": self.curve_eval_interval,
                }
                eval_tasks.append(
                    EvaluationTask(
                        task_id=body["genome_seed"], arm=body["encoder"], strategy="preflight_b4",
                        task=task_name, seed=body["genome_seed"], genome=body["genome"],
                        ppo_timesteps=self.curve_max_timesteps, eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=meta,
                    )
                )

            n_workers = max(1, min(concurrency, len(eval_tasks)))
            pool = ctx.make_pool(n_workers=n_workers, mode=self.evaluator_mode, timeout_s=self.task_timeout_s, max_retries=1)
            with pool:
                results = list(pool.map_tasks(eval_tasks))
            result_by_id = {r.task_id: r for r in results}

            body_records = []
            curves_for_median = []
            n_failed = 0
            for body in bodies:
                r = result_by_id.get(body["genome_seed"])
                failed = r is None or r.error is not None or r.timed_out
                curve = list(r.curve) if (r is not None and r.curve) else []
                if failed:
                    n_failed += 1
                elif curve:
                    curves_for_median.append(curve)
                body_records.append(
                    {
                        "label": body["label"],
                        "arm": body["arm_label"],
                        "encoder": body["encoder"],
                        "genome_seed": body["genome_seed"],
                        "body_hash": (r.body_hash if r is not None else body.get("body_hash")),
                        "b3_fitness": body["fitness"],
                        "curve": curve,
                        "final_fitness": (curve[-1][1] if curve else None),
                        "failed": failed,
                        "error": (r.error if r is not None else "no result returned"),
                    }
                )

            if n_failed:
                stage_partial = True
                ctx.events.record(
                    self.name, "anomaly",
                    {"task": task_name, "reason": "body_training_failed", "n_failed": n_failed, "n_requested": len(bodies)},
                )

            median_series = _median_series(curves_for_median)
            knee = (
                select_knee(median_series, threshold=self.knee_threshold, sustained_windows=self.knee_sustained_windows)
                if median_series
                else None
            )
            knee_found = bool(knee and knee["knee_found"])
            no_plateau_observed = not knee_found
            selected_budget = knee["knee_timesteps"] if knee_found else self.curve_max_timesteps

            warning = None
            if no_plateau_observed:
                warning = (
                    f"On {task_name}, no sustained plateau was observed within "
                    f"curve_max_timesteps={self.curve_max_timesteps:,} -- the reported budget "
                    f"({selected_budget:,}) is a FLOOR, not a knee: fitness may still be improving "
                    f"past this point. Stated plainly here, not averaged away."
                )
                ctx.events.record(self.name, "anomaly", {"task": task_name, "warning": warning})

            task_results[task_name] = {
                "bodies": body_records,
                "median_series": median_series,
                "knee": knee,
                "selected_budget": selected_budget,
                "no_plateau_observed": no_plateau_observed,
                "warning": warning,
                "partial": n_failed > 0,
            }

        figure_path = None
        try:
            figure_path = os.path.join(ctx.run_dir.report_dir, "b4_learning_curves.png")
            _make_figure(figure_path, task_results, SELECT_KNEE_RULE_TEXT, self_check_stamp=self.self_check_stamp)
        except Exception as e:  # noqa: BLE001 -- a figure failure must not lose the measured data
            ctx.events.record(self.name, "failure", {"error": f"figure generation failed: {type(e).__name__}: {e}"})
            figure_path = None

        return {
            "concurrency_used": concurrency,
            "curve_max_timesteps": self.curve_max_timesteps,
            "curve_eval_interval": self.curve_eval_interval,
            "knee_rule_text": SELECT_KNEE_RULE_TEXT,
            "knee_threshold": self.knee_threshold,
            "knee_sustained_windows": self.knee_sustained_windows,
            "budget_note": BUDGET_NOTE,
            "tasks": task_results,
            "figure_path": figure_path,
            "partial": stage_partial,
            "evaluator_mode": self.evaluator_mode,
            "self_check_stamp": self.self_check_stamp,
        }
