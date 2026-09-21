"""D5 B6: the projection table.

Stage `projection` -- cost "free" (pure arithmetic over already-measured
numbers; no PPO, no new evaluation of any kind), depends_on
("learning_curves", "evaluation_noise"). This is the artifact actually
shown to the supervisor: report/projection.md, report/projection.csv, and
report/b6_projection.png.

WHAT THIS STAGE DOES: combines B2's measured throughput/overhead, B3's
measured t_PPO/t_sentinel (with their measured SPREAD, not just their
mean), and this project's own already-established per-arm validity rates
(loaded from results/, never re-measured -- see _load_validity_rates())
into:

    expected_cost_per_eval(arm) = validity_rate * t_PPO + (1 - validity_rate) * t_sentinel

then projects wall-clock for every (population_size, generations, seeds,
arm, task) cell in the CLI-supplied candidate grids, using B2's REALISED
throughput at the chosen concurrency (via a measured overhead factor, not
an assumption of perfect linear scaling -- see _measured_overhead_factor).

EVERY cell carries a RANGE (low/mean/high), derived from B3's own measured
t_PPO spread, never a bare point estimate -- exactly the same discipline
D11 already requires for the campaign's own ETA, applied here to the
pre-flight projection itself.

MUST NOT pick a configuration: this stage enumerates the full cross
product in a fixed, deterministic order (population, generations, seeds,
arm, task -- never sorted by cost, never flagged with a "winner") and
states the D14 option costs as explicit deltas against ONE reference grid
cell (the smallest of each supplied list, labelled plainly as a reference
point for comparison, not a recommendation). D15 (which cell to actually
run) is resolved by Tejeshwar at the scoping gate -- if any code path here
ever computed "the recommended row", that would be a bug in this module.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

from evodevo.preflight.run_dir import read_json

DEFAULT_POPULATIONS = (15,)
DEFAULT_GENERATIONS = (20,)
DEFAULT_SEEDS = (5,)
DEFAULT_ARMS = ("direct", "nca", "rule_based")
DEFAULT_TASKS = ("Walker-v0", "UpStepper-v0")

# Already-measured, already-established per-arm validity rates -- see
# _load_validity_rates()'s own docstring for exactly where each number
# comes from. Never re-measured by this stage (cost="free" is only
# possible because these numbers already exist).
_INSPECTION_JSON_PATH = "results/inspection/{arm}/{arm}_sanity.json"
_MONOTONE_VALID_RATE = 0.6530  # results/rule_based/MONOTONE_ABLATION.md, "monotone" column, N=2000,
# "Structural cost: monotone vs unrestricted" table -- cited directly (not
# derived from that same table's "unrestricted" column, which is a
# different-vintage sample than results/inspection/rule_based/
# rule_based_sanity.json's own canonical rate; monotone's own column is a
# real, disclosed, direct measurement in its own right).
_MONOTONE_VALID_RATE_N = 2000
_MONOTONE_VALID_RATE_SOURCE = (
    "results/rule_based/MONOTONE_ABLATION.md, 'monotone' column of the "
    "'Structural cost: monotone vs unrestricted' table"
)

D14_OPTIONS = (
    {"name": "more_seeds_5_to_10", "description": "Double seeds per (arm, task) from 5 to 10.", "axis": "seeds", "to_value": 10},
    {"name": "second_grid_size", "description": "Add a second grid size (e.g. 7x7) alongside 5x5 -- doubles the whole cross product.", "multiplier": 2.0},
    {"name": "direct_repaired_arm", "description": "Add a hypothetical 'direct_repaired' arm to the arm list -- one more arm in the cross product.", "extra_arms": 1},
    {"name": "nca_compact_arm", "description": "Add a hypothetical 'nca_compact' arm to the arm list -- one more arm in the cross product.", "extra_arms": 1},
    {"name": "larger_timestep_budget", "description": "Double ppo_timesteps from the B3-measured baseline -- roughly doubles t_PPO, and therefore roughly doubles wall-clock.", "multiplier": 2.0},
)


def _load_validity_rates() -> tuple:
    """Loads the ALREADY-MEASURED per-arm validity rate for every arm this
    project has registered/ablated, from the committed results this
    project already produced -- never re-measured here. direct/nca/
    rule_based come from results/inspection/{arm}/{arm}_sanity.json's own
    valid_rate field (N=2000 each, this project's canonical per-arm
    inspection report); rule_based_monotone has no separate inspection
    report (it is not a separately registered encoder class), so its rate
    is cited directly from results/rule_based/MONOTONE_ABLATION.md's own
    "monotone" column instead."""
    rates: dict = {}
    sources: dict = {}
    for arm in ("direct", "nca", "rule_based"):
        path = _INSPECTION_JSON_PATH.format(arm=arm)
        try:
            data = read_json(path)
            rates[arm] = data[arm]["valid_rate"]
            sources[arm] = {"path": path, "n": data[arm]["n"]}
        except (FileNotFoundError, KeyError) as e:
            rates[arm] = None
            sources[arm] = {"path": path, "error": f"{type(e).__name__}: {e}"}

    rates["rule_based_monotone"] = _MONOTONE_VALID_RATE
    sources["rule_based_monotone"] = {"path": _MONOTONE_VALID_RATE_SOURCE, "n": _MONOTONE_VALID_RATE_N}

    return rates, sources


def _expected_cost_per_eval_range(validity_rate: Optional[float], t_ppo_stats: dict, t_sentinel_stats: dict) -> Optional[dict]:
    """expected_cost_per_eval(arm) = validity_rate * t_PPO + (1 -
    validity_rate) * t_sentinel, computed three times (against t_PPO's
    measured min/mean/max) to carry B3's measured SPREAD through as a
    range, never a bare point estimate. t_sentinel is held at its own mean
    throughout -- it is the comparatively negligible term (see B3's own
    t_PPO/t_sentinel ratio), so its own small spread is not the dominant
    source of uncertainty here."""
    if validity_rate is None:
        return None
    t_sentinel_mean = t_sentinel_stats.get("mean")
    if t_sentinel_mean is None:
        return None
    result = {}
    for key, ppo_key in (("low", "min"), ("mean", "mean"), ("high", "max")):
        t_ppo_value = t_ppo_stats.get(ppo_key)
        if t_ppo_value is None:
            return None
        result[key] = validity_rate * t_ppo_value + (1 - validity_rate) * t_sentinel_mean
    return result


def _measured_overhead_factor(b2_output: dict, concurrency: int) -> dict:
    """How far the pool's REALISED throughput at `concurrency` (B2's own
    measurement) falls short of naive perfect linear scaling from the
    smallest level B2 actually tested. Returns {"factor": ..., "source":
    ...} -- factor=1.0 with an explicit "not measured" source if
    `concurrency` wasn't one of B2's tested levels (e.g. --concurrency was
    overridden to something B2 never swept), so the projection still runs,
    with the assumption disclosed rather than silently wrong."""
    levels = {lvl["level"]: lvl for lvl in b2_output.get("levels", [])}
    if not levels or concurrency not in levels:
        return {"factor": 1.0, "source": "not measured (concurrency not among B2's tested levels) -- assuming perfect scaling, disclosed"}

    baseline_level = min(levels.keys())
    baseline = levels[baseline_level]
    if baseline_level == concurrency:
        return {"factor": 1.0, "source": f"concurrency ({concurrency}) equals B2's smallest tested level; no scaling correction to measure"}
    if not baseline.get("genomes_per_minute"):
        return {"factor": 1.0, "source": "not measured (B2 baseline level had zero throughput) -- assuming perfect scaling, disclosed"}

    per_worker_at_baseline = baseline["genomes_per_minute"] / baseline_level
    naive_prediction = per_worker_at_baseline * concurrency
    realised = levels[concurrency]["genomes_per_minute"]
    if naive_prediction <= 0:
        return {"factor": 1.0, "source": "not measured (degenerate B2 baseline) -- assuming perfect scaling, disclosed"}

    factor = realised / naive_prediction
    return {
        "factor": factor,
        "source": (
            f"measured: B2's realised throughput at concurrency={concurrency} "
            f"({realised:.2f} genomes/min) versus naive linear scaling from its level="
            f"{baseline_level} baseline ({naive_prediction:.2f} genomes/min)"
        ),
    }


def _build_projection_table(populations, generations, seeds_list, arms, tasks, cost_ranges, overhead_factor) -> list:
    """The full cross product, in a FIXED deterministic order (population,
    generations, seeds, arm, task) -- never sorted by cost, never
    annotated with a preferred row. evaluations_per_seed = population_size
    * generations is a disclosed simplifying model of a (mu+lambda) EA's
    evaluation count (one population evaluated per generation); real
    offspring_size/eval_budget accounting is D15's own open question, not
    reintroduced here."""
    rows = []
    for population in populations:
        for gens in generations:
            for seeds in seeds_list:
                for arm in arms:
                    for task in tasks:
                        cost_range = cost_ranges.get((arm, task))
                        evaluations_per_seed = population * gens
                        total_evaluations = evaluations_per_seed * seeds
                        row = {
                            "population_size": population,
                            "generations": gens,
                            "seeds": seeds,
                            "arm": arm,
                            "task": task,
                            "evaluations_per_seed": evaluations_per_seed,
                            "total_evaluations": total_evaluations,
                            "cost_per_eval_low_s": cost_range["low"] if cost_range else None,
                            "cost_per_eval_mean_s": cost_range["mean"] if cost_range else None,
                            "cost_per_eval_high_s": cost_range["high"] if cost_range else None,
                            "overhead_factor": overhead_factor,
                        }
                        for label, key in (("low", "cost_per_eval_low_s"), ("mean", "cost_per_eval_mean_s"), ("high", "cost_per_eval_high_s")):
                            cost = row[key]
                            if cost is None:
                                row[f"wall_clock_hours_{label}"] = None
                            else:
                                serial_s = total_evaluations * cost
                                # overhead_factor here already folds in concurrency
                                # (see caller): parallel_s = serial_s / (concurrency * factor).
                                # No flooring at 1.0 -- a genuinely poor
                                # realised-throughput factor must make the
                                # projection WORSE than naive serial, not
                                # silently clip back to "no worse than serial".
                                parallel_s = serial_s / overhead_factor if overhead_factor > 0 else serial_s
                                row[f"wall_clock_hours_{label}"] = parallel_s / 3600.0
                        rows.append(row)
    return rows


def _build_d14_deltas(baseline_row: dict, concurrency: int, overhead_factor: float) -> list:
    """Explicit wall-clock deltas for each named D14 option, against ONE
    reference cell (the smallest supplied grid point -- a reference point
    for comparison, never a recommendation; see module docstring)."""
    deltas = []
    baseline_hours = baseline_row.get("wall_clock_hours_mean")
    for option in D14_OPTIONS:
        if baseline_hours is None:
            deltas.append({**option, "baseline_hours": None, "new_hours": None, "delta_hours": None})
            continue
        if "to_value" in option:
            factor = option["to_value"] / baseline_row["seeds"]
        elif "extra_arms" in option:
            factor = (1 + option["extra_arms"]) / 1  # this option adds arms alongside a 1-arm reference cell
        else:
            factor = option.get("multiplier", 1.0)
        new_hours = baseline_hours * factor
        deltas.append(
            {
                **option,
                "baseline_hours": baseline_hours,
                "new_hours": new_hours,
                "delta_hours": new_hours - baseline_hours,
            }
        )
    return deltas


def _write_markdown(
    path: str, rows: list, validity_sources: dict, overhead: dict, concurrency: int, deltas: list, baseline_row: dict,
    self_check_stamp: Optional[str] = None,
) -> None:
    lines = []
    title = "# D5 B6 -- Evaluation Cost Projection"
    if self_check_stamp:
        title = f"# {self_check_stamp} — {title[2:]}"
    lines.append(title)
    lines.append("")
    lines.append(
        f"Every wall-clock figure below is a RANGE (low/mean/high hours), derived from B3's own "
        f"measured t_PPO spread (min/mean/max), never a single point estimate -- see D11's own "
        f"identical discipline for the campaign ETA. Concurrency: {concurrency} (B2's measured knee "
        f"unless overridden). Overhead factor applied: {overhead['factor']:.3f} ({overhead['source']})."
    )
    lines.append("")
    lines.append("**This table does not pick a configuration.** It is the full candidate grid, in a fixed "
                  "order; D15 (which cell to run) is resolved by Tejeshwar at the scoping gate.")
    lines.append("")
    lines.append("| population | generations | seeds | arm | task | evaluations | wall-clock (low h) | wall-clock (mean h) | wall-clock (high h) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        def fmt(v):
            return f"{v:.2f}" if v is not None else "n/a"

        lines.append(
            f"| {row['population_size']} | {row['generations']} | {row['seeds']} | {row['arm']} | {row['task']} | "
            f"{row['total_evaluations']} | {fmt(row['wall_clock_hours_low'])} | {fmt(row['wall_clock_hours_mean'])} | "
            f"{fmt(row['wall_clock_hours_high'])} |"
        )
    lines.append("")
    lines.append("## Per-arm validity rate provenance (already measured, not re-measured here)")
    lines.append("")
    lines.append("| arm | valid_rate | n | source |")
    lines.append("|---|---|---|---|")
    for arm, source in validity_sources.items():
        n = source.get("n", "n/a")
        # NOTE: intentionally NOT named `path` -- this function's own
        # `path` parameter is the markdown file's OUTPUT path; shadowing
        # it here previously caused this function to write its content to
        # whatever this source citation string happened to be instead of
        # the real output path (caught in testing -- see git history).
        source_path = source.get("path", source.get("error", "n/a"))
        rate = source.get("valid_rate", "")
        lines.append(f"| {arm} | {rate} | {n} | {source_path} |")
    lines.append("")
    baseline_hours = baseline_row.get("wall_clock_hours_mean")
    baseline_hours_str = f"{baseline_hours:.4f}" if baseline_hours is not None else "n/a"
    lines.append(f"## D14 option costs (deltas against the reference cell: population={baseline_row['population_size']}, "
                  f"generations={baseline_row['generations']}, seeds={baseline_row['seeds']}, arm={baseline_row['arm']}, "
                  f"task={baseline_row['task']}, {baseline_hours_str} h)")
    lines.append("")
    lines.append("| option | description | new wall-clock (mean h) | delta (h) |")
    lines.append("|---|---|---|---|")
    for d in deltas:
        new_h = f"{d['new_hours']:.2f}" if d["new_hours"] is not None else "n/a"
        delta_h = f"{d['delta_hours']:+.2f}" if d["delta_hours"] is not None else "n/a"
        lines.append(f"| {d['name']} | {d['description']} | {new_h} | {delta_h} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_csv(path: str, rows: list) -> None:
    fieldnames = [
        "population_size", "generations", "seeds", "arm", "task", "evaluations_per_seed",
        "total_evaluations", "cost_per_eval_low_s", "cost_per_eval_mean_s", "cost_per_eval_high_s",
        "overhead_factor", "wall_clock_hours_low", "wall_clock_hours_mean", "wall_clock_hours_high",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _make_figure(path: str, cost_ranges: dict, self_check_stamp: Optional[str] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{arm}\n{task}" for (arm, task) in cost_ranges]
    means = [r["mean"] for r in cost_ranges.values()]
    lows = [r["mean"] - r["low"] for r in cost_ranges.values()]
    highs = [r["high"] - r["mean"] for r in cost_ranges.values()]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    ax.bar(labels, means, yerr=[lows, highs], capsize=4, color="tab:blue", alpha=0.8)
    ax.set_ylabel("expected cost per evaluation (s)")
    title = "B6: expected_cost_per_eval by arm x task (range from B3's measured t_PPO spread)"
    if self_check_stamp:
        title = f"{self_check_stamp} — {title}"
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class ProjectionStage:
    name = "projection"
    description = (
        "Projection table (D5 B6): expected_cost_per_eval per arm (validity_rate x t_PPO + "
        "(1-validity_rate) x t_sentinel), projected wall-clock with a measured range for the full "
        "CLI-supplied candidate grid, and D14 option-cost deltas. Never picks a configuration."
    )
    estimated_cost = "free"
    depends_on = ("learning_curves", "evaluation_noise")

    def __init__(
        self,
        populations=DEFAULT_POPULATIONS,
        generations=DEFAULT_GENERATIONS,
        seeds=DEFAULT_SEEDS,
        arms=DEFAULT_ARMS,
        tasks=DEFAULT_TASKS,
        overhead: Optional[float] = None,
        concurrency: Optional[int] = None,
        self_check_stamp: Optional[str] = None,
    ):
        self.populations = tuple(populations)
        self.generations = tuple(generations)
        self.seeds = tuple(seeds)
        self.arms = tuple(arms)
        self.tasks = tuple(tasks)
        self.overhead = overhead  # None -> measured from B2; explicit override otherwise
        self.concurrency = concurrency  # None -> B3's own concurrency_used
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        b2_output = read_json(ctx.run_dir.stage_output_path("concurrency_sweep"))
        b3_output = read_json(ctx.run_dir.stage_output_path("evaluation_cost"))

        concurrency = self.concurrency if self.concurrency is not None else b3_output["concurrency_used"]

        if self.overhead is not None:
            overhead = {"factor": self.overhead, "source": "explicit --proj-overhead override"}
        else:
            overhead = _measured_overhead_factor(b2_output, concurrency)
        # Fold concurrency into the single divisor _build_projection_table applies.
        effective_overhead = concurrency * overhead["factor"]

        validity_rates, validity_sources = _load_validity_rates()

        cost_ranges = {}
        for arm in self.arms:
            arm_data = b3_output.get("arms", {}).get(arm)
            if not arm_data:
                continue
            t_sentinel_stats = arm_data.get("t_sentinel", {})
            for task in self.tasks:
                task_data = arm_data.get("tasks", {}).get(task)
                if not task_data:
                    continue
                cost_range = _expected_cost_per_eval_range(validity_rates.get(arm), task_data.get("t_ppo", {}), t_sentinel_stats)
                if cost_range is not None:
                    cost_ranges[(arm, task)] = cost_range
                    validity_sources.setdefault(arm, {})["valid_rate"] = validity_rates.get(arm)

        for arm in self.arms:
            validity_sources.setdefault(arm, {})["valid_rate"] = validity_rates.get(arm)

        rows = _build_projection_table(
            self.populations, self.generations, self.seeds, self.arms, self.tasks, cost_ranges, effective_overhead
        )

        baseline_row = rows[0] if rows else None
        deltas = _build_d14_deltas(baseline_row, concurrency, overhead["factor"]) if baseline_row else []

        report_dir = ctx.run_dir.report_dir
        md_path = os.path.join(report_dir, "projection.md")
        csv_path = os.path.join(report_dir, "projection.csv")
        figure_path = os.path.join(report_dir, "b6_projection.png")

        _write_markdown(
            md_path, rows, validity_sources, overhead, concurrency, deltas, baseline_row or {},
            self_check_stamp=self.self_check_stamp,
        )
        _write_csv(csv_path, rows)
        try:
            if cost_ranges:
                _make_figure(figure_path, cost_ranges, self_check_stamp=self.self_check_stamp)
            else:
                figure_path = None
        except Exception as e:  # noqa: BLE001 -- a figure failure must not lose the table
            ctx.events.record(self.name, "failure", {"error": f"figure generation failed: {type(e).__name__}: {e}"})
            figure_path = None

        ctx.events.record(
            self.name, "measurement",
            {"n_rows": len(rows), "concurrency_used": concurrency, "overhead_factor": overhead["factor"]},
        )

        return {
            "concurrency_used": concurrency,
            "overhead": overhead,
            "validity_rates": {arm: validity_sources.get(arm, {}).get("valid_rate") for arm in self.arms},
            "validity_sources": validity_sources,
            "cost_per_eval_ranges": {f"{arm}::{task}": r for (arm, task), r in cost_ranges.items()},
            "grid": {
                "populations": list(self.populations), "generations": list(self.generations),
                "seeds": list(self.seeds), "arms": list(self.arms), "tasks": list(self.tasks),
            },
            "rows": rows,
            "d14_deltas": deltas,
            "report_paths": {"markdown": md_path, "csv": csv_path, "figure": figure_path},
            "partial": not bool(rows) or not bool(cost_ranges),
            "self_check_stamp": self.self_check_stamp,
        }
