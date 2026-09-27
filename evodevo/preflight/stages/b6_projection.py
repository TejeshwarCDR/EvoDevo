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

from evodevo.parallel.topology import cpu_slots
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

# For the two "add a hypothetical new arm" D14 options: neither
# 'direct_repaired' nor 'nca_compact' has ever actually been measured (they
# are hypothetical variants under consideration, not registered encoders --
# see _load_validity_rates()'s docstring on which arms DO have real
# measurements), so this stage costs them using their nearest already-
# measured relative's own cost_per_eval as an explicitly-disclosed proxy,
# rather than inventing a number. If either hypothetical arm is ever
# actually implemented and measured, this mapping should be replaced by a
# real cost_ranges[(arm, task)] entry for that arm instead.
_D14_ARM_PROXY = {
    "direct_repaired_arm": "direct",
    "nca_compact_arm": "nca",
}

# D13 follow-up fix (the P/E-core bias in this stage's wall-clock model):
# B3's t_PPO was measured entirely on P-cores -- at concurrency=8,
# cpu_slots()/allocation_order() (evodevo.parallel.topology) assign every
# worker to a P-core physical primary on this machine's hybrid 8P/12E
# topology, so B3 never exercised an E-core at all. A projection run at
# any concurrency ABOVE the P-core boundary (e.g. 20, this run's own
# corrected B2 knee) therefore cannot assume every worker is as fast as a
# P-core -- see _pe_worker_split()/_mixed_core_hours() below for the
# corrected model, and the module docstring's new "P/E-core bias" section
# for why this matters.
#
# MEASURED_PE_RATIO's source (do not replace with a fresh literal without
# updating this citation): a direct, targeted single-task probe, NOT
# derived from B2's sweep throughput (that number is pool-aggregate,
# confounded by workload mix and scheduling). Raw measurement persisted at
# results/preflight/20260922T122730Z_4be8402b/pe_core_ratio_measurement.json.
# Methodology: a 1-worker EvaluationPool (mode="real") pinned first to
# cpu_id=0 (a P-core physical primary) then to cpu_id=16 (an E-core),
# given the SAME identical valid 'nca' genome, ppo_timesteps=20000
# (b2_sweep.DEFAULT_TIMESTEPS -- a PROBE-scale budget, NOT the campaign's
# real ~1,000,000-timestep t_PPO; see the provenance file's own "caveat"
# field for why extrapolating this ratio to full-length PPO training is an
# approximation, not a certainty):
#   P-core wall_clock_s = 28.925267219543457
#   E-core wall_clock_s = 62.014686822891235
#   ratio (E/P)          = 2.1439624516585556
#
# CORROBORATED (not superseded) at PRODUCTION scale: a direct follow-up
# measurement at the campaign's real ppo_timesteps=1,000,000 (n=3 per core
# class, one genome each from direct/nca/rule_based, reused from B3's own
# evaluation_cost.json observations) found ratio_e_over_p=2.31715026119251
# -- an 8.08% difference from this probe-scale value, within the ~15%
# agreement threshold used to decide this ratio was fit for use, so
# MEASURED_PE_RATIO below was deliberately left unchanged. See
# results/preflight/20260922T122730Z_4be8402b/pe_core_ratio_measurement_
# production_scale.json for the full raw measurement.
MEASURED_PE_RATIO = 62.014686822891235 / 28.925267219543457
MEASURED_PE_RATIO_SOURCE = (
    "results/preflight/20260922T122730Z_4be8402b/pe_core_ratio_measurement.json -- single-task probe, "
    "P-core (cpu 0) vs E-core (cpu 16), identical valid nca genome, ppo_timesteps=20000 "
    "(b2_sweep.DEFAULT_TIMESTEPS, not the campaign's real t_PPO budget): "
    "28.925267219543457s (P) vs 62.014686822891235s (E)"
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


def _pe_worker_split(topology, concurrency: int) -> dict:
    """How many of `concurrency` workers land on P-cores vs E-cores --
    via cpu_slots()/allocation_order() (evodevo.parallel.topology), the
    EXACT SAME functions the real EvaluationPool (and the B2 sweep) use to
    place workers, never hand-derived here: this projection and the real
    launcher must never be able to disagree about which cores get used.

    n_p_workers counts every worker NOT confirmed to be an E-core (i.e.
    "P-core or unclassified"), not just cores explicitly kind=="P" --
    on a non-hybrid machine (topology.e_core_ids empty, every core
    kind=="unknown"), B3's own t_PPO measurement was ALSO taken on
    "unclassified" cores via this exact same allocation order, so treating
    them as the same speed class B3 measured is the self-consistent
    default, not an optimistic guess. Only cores this machine's own
    detection affirmatively classifies as "E" get the measured slowdown."""
    plan = cpu_slots(topology, concurrency)
    e_ids = set(topology.e_core_ids)
    n_e = sum(1 for slot in plan.slots if slot[0] in e_ids)
    n_p = len(plan.slots) - n_e
    return {
        "concurrency": concurrency,
        "n_p_workers": n_p,
        "n_e_workers": n_e,
        "oversubscribed": plan.oversubscribed,
    }


def _mixed_core_hours(
    total_evaluations: int,
    p_cost_s: Optional[float],
    e_cost_s: Optional[float],
    n_p_workers: int,
    n_e_workers: int,
) -> Optional[float]:
    """Wall-clock HOURS for `total_evaluations` independent evaluations run
    across a MIXED pool of n_p_workers P-core workers (each completing one
    evaluation every p_cost_s seconds) and n_e_workers E-core workers (one
    every e_cost_s seconds) -- a workers-in-parallel throughput model, NOT
    a simple average of the two per-core times:

        aggregate_rate (evaluations/second) = n_p_workers / p_cost_s
                                             + n_e_workers / e_cost_s
        hours = (total_evaluations / aggregate_rate) / 3600

    Every P-core worker and every E-core worker is assumed to run
    continuously and independently (no cross-worker contention beyond
    what's already implicit in B3's own per-core-class measurement) --
    this is the SAME idealised full-parallelism assumption the old
    concurrency-only divisor already made, just no longer pretending every
    worker is P-core-fast.

    INVARIANT (see test_b6_pe_model_reduces_to_p_core_only_formula_*):
    when n_e_workers == 0 (concurrency at or below this machine's P-core
    boundary, so cpu_slots() never assigns an E-core), this reduces
    EXACTLY to total_evaluations * p_cost_s / n_p_workers -- the old
    P-core-only formula -- since the second term of aggregate_rate drops
    out entirely."""
    if p_cost_s is None or p_cost_s <= 0:
        return None
    rate = 0.0
    if n_p_workers > 0:
        rate += n_p_workers / p_cost_s
    if n_e_workers > 0 and e_cost_s is not None and e_cost_s > 0:
        rate += n_e_workers / e_cost_s
    if rate <= 0:
        return None
    return (total_evaluations / rate) / 3600.0


def _build_pe_core_bias_summary(baseline_row: Optional[dict], rows: list) -> dict:
    """Step 2 of the D13 follow-up fix: makes the SIZE of the P/E-core
    correction visible in the report itself, not just applied silently --
    naive (uniform P-core-speed) vs. corrected (mixed-rate) wall-clock,
    side by side, for the reference cell and for the full campaign total
    (sum of every row), with the percentage difference. This is the number
    that shows how wrong the PREVIOUS projection.md for this run actually
    was (it used the naive figures as if they were final)."""

    def _pct(naive, corrected):
        if naive is None or corrected is None or naive == 0:
            return None
        return (corrected - naive) / naive * 100.0

    reference_cell = None
    if baseline_row is not None:
        naive_h = baseline_row.get("wall_clock_hours_naive_mean")
        corrected_h = baseline_row.get("wall_clock_hours_mean")
        reference_cell = {
            "arm": baseline_row.get("arm"),
            "task": baseline_row.get("task"),
            "naive_hours": naive_h,
            "corrected_hours": corrected_h,
            "delta_hours": (corrected_h - naive_h) if (naive_h is not None and corrected_h is not None) else None,
            "pct_difference": _pct(naive_h, corrected_h),
        }

    campaign_total = None
    if rows:
        naive_vals = [r["wall_clock_hours_naive_mean"] for r in rows if r.get("wall_clock_hours_naive_mean") is not None]
        corrected_vals = [r["wall_clock_hours_mean"] for r in rows if r.get("wall_clock_hours_mean") is not None]
        if naive_vals and corrected_vals:
            naive_total = sum(naive_vals)
            corrected_total = sum(corrected_vals)
            campaign_total = {
                "naive_hours": naive_total,
                "corrected_hours": corrected_total,
                "delta_hours": corrected_total - naive_total,
                "pct_difference": _pct(naive_total, corrected_total),
            }

    return {"reference_cell": reference_cell, "campaign_total": campaign_total}


def _build_projection_table(populations, generations, seeds_list, arms, tasks, cost_ranges, overhead_factor, pe_split, pe_ratio) -> list:
    """The full cross product, in a FIXED deterministic order (population,
    generations, seeds, arm, task) -- never sorted by cost, never
    annotated with a preferred row. evaluations_per_seed = population_size
    * generations is a disclosed simplifying model of a (mu+lambda) EA's
    evaluation count (one population evaluated per generation); real
    offspring_size/eval_budget accounting is D15's own open question, not
    reintroduced here."""
    rows = []
    n_p_workers = pe_split["n_p_workers"]
    n_e_workers = pe_split["n_e_workers"]
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
                            "n_p_workers": n_p_workers,
                            "n_e_workers": n_e_workers,
                            "pe_ratio": pe_ratio,
                        }
                        for label, key in (("low", "cost_per_eval_low_s"), ("mean", "cost_per_eval_mean_s"), ("high", "cost_per_eval_high_s")):
                            p_cost = row[key]
                            if p_cost is None:
                                row[f"wall_clock_hours_naive_{label}"] = None
                                row[f"wall_clock_hours_{label}"] = None
                                continue

                            # NAIVE (legacy, kept only for the Step 2
                            # side-by-side comparison -- see
                            # _build_pe_core_bias_summary()): assumes every
                            # worker runs at P-core speed uniformly,
                            # regardless of how many are actually E-cores.
                            # No flooring at 1.0 -- a genuinely poor
                            # realised-throughput factor must make the
                            # projection WORSE than naive serial, not
                            # silently clip back to "no worse than serial".
                            serial_s = total_evaluations * p_cost
                            naive_parallel_s = serial_s / overhead_factor if overhead_factor > 0 else serial_s
                            row[f"wall_clock_hours_naive_{label}"] = naive_parallel_s / 3600.0

                            # CORRECTED (primary, D13 follow-up fix): an
                            # explicit P/E-aware mixed-rate model -- see
                            # MEASURED_PE_RATIO's own citation and
                            # _mixed_core_hours()'s docstring.
                            e_cost = p_cost * pe_ratio
                            row[f"wall_clock_hours_{label}"] = _mixed_core_hours(
                                total_evaluations, p_cost, e_cost, n_p_workers, n_e_workers
                            )
                        rows.append(row)
    return rows


def _build_d14_deltas(
    baseline_row: dict,
    rows: list,
    cost_ranges: dict,
    t_ppo_stats_by_arm_task: dict,
    t_sentinel_mean_by_arm: dict,
    validity_rates: dict,
    pe_split: dict,
    pe_ratio: float,
) -> list:
    """Explicit wall-clock deltas for each named D14 option -- each option
    is modeled according to WHAT it structurally changes, not by applying
    the same multiplier to the same single cell (that was the bug: every
    option previously collapsed to baseline_hours * 2.0, see git history /
    D14 fix commit):

      - more_seeds_5_to_10, larger_timestep_budget: SINGLE-CELL changes.
        Only the reference (arm, task) cell is affected, so the delta is
        against that cell's own wall_clock_hours_mean.
          * seeds simply multiplies total_evaluations for that cell.
          * timesteps scales t_PPO only (not t_sentinel -- an invalid
            body's sentinel-fitness early exit doesn't run PPO longer just
            because the budget grew), re-deriving cost_per_eval from
            validity_rate * (t_ppo * multiplier) + (1 - validity_rate) *
            t_sentinel, so the delta's SIZE correctly depends on that arm's
            own validity_rate (a low-validity arm barely moves; a
            high-validity arm moves close to the full 2x).

      - second_grid_size, direct_repaired_arm, nca_compact_arm:
        CAMPAIGN-WIDE changes. None of these affect only the reference
        cell -- a second grid size reruns the WHOLE existing cross product
        again, and a new arm adds ITS OWN total cost across the relevant
        tasks. Both are deltas against the CAMPAIGN TOTAL (sum of every
        row's wall_clock_hours_mean), not the single reference cell --
        "add that arm/grid's own total cost to the campaign total, not
        double an unrelated single cell."

    Each returned dict carries "scope" ("single_cell" or "campaign_total")
    so a reader knows which baseline a given delta is measured against.
    """
    # baseline_row/rows' own wall_clock_hours_mean is already the
    # CORRECTED (P/E-aware) figure -- see _build_projection_table() -- so
    # every delta below inherits the correction automatically wherever it
    # is computed from those fields; only the two branches that derive a
    # brand-new hours figure from scratch (larger_timestep_budget,
    # extra_arms) need to call _mixed_core_hours() explicitly themselves.
    baseline_cell_hours = baseline_row.get("wall_clock_hours_mean") if baseline_row else None
    campaign_total_hours = None
    if rows:
        hours = [r["wall_clock_hours_mean"] for r in rows if r.get("wall_clock_hours_mean") is not None]
        if hours:
            campaign_total_hours = sum(hours)

    n_p_workers = pe_split["n_p_workers"]
    n_e_workers = pe_split["n_e_workers"]

    deltas = []
    for option in D14_OPTIONS:
        if baseline_row is None or baseline_cell_hours is None:
            deltas.append({**option, "scope": None, "baseline_hours": None, "new_hours": None, "delta_hours": None})
            continue

        if "to_value" in option:
            scope = "single_cell"
            factor = option["to_value"] / baseline_row["seeds"]
            baseline_hours = baseline_cell_hours
            new_hours = baseline_cell_hours * factor

        elif option["name"] == "larger_timestep_budget":
            scope = "single_cell"
            baseline_hours = baseline_cell_hours
            arm, task = baseline_row["arm"], baseline_row["task"]
            t_ppo_stats = t_ppo_stats_by_arm_task.get((arm, task), {})
            t_sentinel_mean = t_sentinel_mean_by_arm.get(arm)
            validity = validity_rates.get(arm)
            t_ppo_mean = t_ppo_stats.get("mean")
            if validity is None or t_ppo_mean is None or t_sentinel_mean is None:
                new_hours = None
            else:
                new_p_cost_per_eval_mean = validity * (t_ppo_mean * option["multiplier"]) + (1 - validity) * t_sentinel_mean
                new_e_cost_per_eval_mean = new_p_cost_per_eval_mean * pe_ratio
                total_evaluations = baseline_row["total_evaluations"]
                new_hours = _mixed_core_hours(
                    total_evaluations, new_p_cost_per_eval_mean, new_e_cost_per_eval_mean, n_p_workers, n_e_workers
                )

        elif option["name"] == "second_grid_size":
            # Adds one more full copy of the campaign total (a second grid
            # size means re-running the ENTIRE existing cross product
            # again at that size), not a multiple of one cell.
            scope = "campaign_total"
            baseline_hours = campaign_total_hours
            new_hours = campaign_total_hours * option["multiplier"] if campaign_total_hours is not None else None

        elif "extra_arms" in option:
            # Adds the new arm's OWN total cost (priced via its proxy arm,
            # see _D14_ARM_PROXY) across every task this projection
            # actually costed, at the reference cell's own population /
            # generations / seeds -- added ON TOP of the campaign total,
            # never multiplying an existing unrelated cell.
            scope = "campaign_total"
            baseline_hours = campaign_total_hours
            proxy_arm = _D14_ARM_PROXY[option["name"]]
            tasks_costed = sorted({task for (_arm, task) in cost_ranges})
            new_arm_total_hours = 0.0
            proxy_missing = False
            for task in tasks_costed:
                cost_range = cost_ranges.get((proxy_arm, task))
                if cost_range is None:
                    proxy_missing = True
                    continue
                total_evaluations = baseline_row["evaluations_per_seed"] * baseline_row["seeds"]
                p_cost_mean = cost_range["mean"]
                e_cost_mean = p_cost_mean * pe_ratio
                task_hours = _mixed_core_hours(total_evaluations, p_cost_mean, e_cost_mean, n_p_workers, n_e_workers)
                new_arm_total_hours += task_hours if task_hours is not None else 0.0
            if campaign_total_hours is None or proxy_missing or not tasks_costed:
                new_hours = None
            else:
                new_hours = campaign_total_hours + new_arm_total_hours

        else:
            scope = "single_cell"
            baseline_hours = baseline_cell_hours
            new_hours = baseline_cell_hours * option.get("multiplier", 1.0)

        delta_hours = (new_hours - baseline_hours) if (new_hours is not None and baseline_hours is not None) else None
        deltas.append(
            {
                **option,
                "scope": scope,
                "baseline_hours": baseline_hours,
                "new_hours": new_hours,
                "delta_hours": delta_hours,
            }
        )
    return deltas


def _write_markdown(
    path: str, rows: list, validity_sources: dict, overhead: dict, concurrency: int, deltas: list, baseline_row: dict,
    pe_split: dict, pe_ratio: float, pe_ratio_source: str, pe_core_bias: dict,
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
        f"unless overridden), split across {pe_split['n_p_workers']} P-core and {pe_split['n_e_workers']} "
        f"E-core workers (via cpu_slots()/allocation_order(), the same placement the real pool uses). "
        f"Overhead factor applied to the naive column only: {overhead['factor']:.3f} ({overhead['source']})."
    )
    lines.append("")
    lines.append(
        "**Wall-clock below is the P/E-CORE-CORRECTED figure** (D13 follow-up fix) -- it accounts for "
        f"this machine's hybrid P/E-core topology using a measured P-core-vs-E-core speed ratio of "
        f"{pe_ratio:.4f}x ({pe_ratio_source}), NOT a uniform-worker-speed assumption. See the "
        "'P/E-core correction: naive vs. corrected' section below for exactly how large that correction "
        "is."
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
    lines.append("## P/E-core correction: naive vs. corrected")
    lines.append("")
    lines.append(
        "B3's t_PPO was measured entirely on P-cores (at concurrency=8, cpu_slots() puts every worker "
        "on a P-core physical primary on this machine). The PREVIOUS projection.md for this run "
        f"projected wall-clock at concurrency={concurrency} assuming every worker ran at that P-core "
        f"speed -- but {pe_split['n_p_workers']} of those {pe_split['n_p_workers'] + pe_split['n_e_workers']} "
        f"workers are actually E-cores, measured {pe_ratio:.4f}x slower ({pe_ratio_source}). The table "
        "below is the size of that error: 'naive' is the old uniform-speed figure, 'corrected' is the "
        "P/E-aware mixed-rate model now used everywhere else in this document."
    )
    lines.append("")
    lines.append("| scope | naive (h) | corrected (h) | delta (h) | % difference |")
    lines.append("|---|---|---|---|---|")

    def _fmt_bias_row(label: str, entry: Optional[dict]) -> str:
        if entry is None:
            return f"| {label} | n/a | n/a | n/a | n/a |"
        naive = f"{entry['naive_hours']:.2f}" if entry.get("naive_hours") is not None else "n/a"
        corrected = f"{entry['corrected_hours']:.2f}" if entry.get("corrected_hours") is not None else "n/a"
        delta = f"{entry['delta_hours']:+.2f}" if entry.get("delta_hours") is not None else "n/a"
        pct = f"{entry['pct_difference']:+.1f}%" if entry.get("pct_difference") is not None else "n/a"
        return f"| {label} | {naive} | {corrected} | {delta} | {pct} |"

    reference_cell = pe_core_bias.get("reference_cell")
    ref_label = "reference cell"
    if reference_cell:
        ref_label = f"reference cell ({reference_cell.get('arm')}, {reference_cell.get('task')})"
    lines.append(_fmt_bias_row(ref_label, reference_cell))
    lines.append(_fmt_bias_row("full campaign total (sum of every row above)", pe_core_bias.get("campaign_total")))
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
    lines.append(f"## D14 option costs (population={baseline_row['population_size']}, "
                  f"generations={baseline_row['generations']}, seeds={baseline_row['seeds']}, arm={baseline_row['arm']}, "
                  f"task={baseline_row['task']}, reference cell = {baseline_hours_str} h)")
    lines.append("")
    lines.append(
        "Each option's baseline/delta is measured against the scope that option actually changes -- "
        "`single_cell` options (seeds, timestep budget) change only the reference cell above; "
        "`campaign_total` options (a new grid size, a new arm) add their own cost on top of the FULL "
        "campaign total (the sum of every row in the table above), since they don't affect the "
        "reference cell in isolation."
    )
    lines.append("")
    lines.append("| option | description | scope | baseline (mean h) | new wall-clock (mean h) | delta (h) |")
    lines.append("|---|---|---|---|---|---|")
    for d in deltas:
        baseline_h = f"{d['baseline_hours']:.2f}" if d.get("baseline_hours") is not None else "n/a"
        new_h = f"{d['new_hours']:.2f}" if d["new_hours"] is not None else "n/a"
        delta_h = f"{d['delta_hours']:+.2f}" if d["delta_hours"] is not None else "n/a"
        lines.append(f"| {d['name']} | {d['description']} | {d.get('scope') or 'n/a'} | {baseline_h} | {new_h} | {delta_h} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_csv(path: str, rows: list) -> None:
    fieldnames = [
        "population_size", "generations", "seeds", "arm", "task", "evaluations_per_seed",
        "total_evaluations", "cost_per_eval_low_s", "cost_per_eval_mean_s", "cost_per_eval_high_s",
        "overhead_factor", "n_p_workers", "n_e_workers", "pe_ratio",
        # wall_clock_hours_* is the P/E-CORRECTED figure (primary, D13
        # follow-up fix); wall_clock_hours_naive_* is the legacy
        # uniform-worker-speed figure, kept for the same naive-vs-corrected
        # comparison projection.md's own "P/E-core correction" section makes.
        "wall_clock_hours_low", "wall_clock_hours_mean", "wall_clock_hours_high",
        "wall_clock_hours_naive_low", "wall_clock_hours_naive_mean", "wall_clock_hours_naive_high",
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
        # Fold concurrency into the single divisor _build_projection_table applies
        # for the legacy NAIVE column only -- see MEASURED_PE_RATIO's own citation
        # for why the PRIMARY (corrected) wall-clock no longer uses this divisor.
        effective_overhead = concurrency * overhead["factor"]
        # D13 follow-up fix: how many of `concurrency` workers actually land on
        # P-cores vs E-cores, via the SAME allocation logic the real pool uses.
        pe_split = _pe_worker_split(ctx.topology, concurrency)

        validity_rates, validity_sources = _load_validity_rates()

        cost_ranges = {}
        t_ppo_stats_by_arm_task = {}
        t_sentinel_mean_by_arm = {}
        for arm in self.arms:
            arm_data = b3_output.get("arms", {}).get(arm)
            if not arm_data:
                continue
            t_sentinel_stats = arm_data.get("t_sentinel", {})
            t_sentinel_mean_by_arm[arm] = t_sentinel_stats.get("mean")
            for task in self.tasks:
                task_data = arm_data.get("tasks", {}).get(task)
                if not task_data:
                    continue
                t_ppo_stats_by_arm_task[(arm, task)] = task_data.get("t_ppo", {})
                cost_range = _expected_cost_per_eval_range(validity_rates.get(arm), task_data.get("t_ppo", {}), t_sentinel_stats)
                if cost_range is not None:
                    cost_ranges[(arm, task)] = cost_range
                    validity_sources.setdefault(arm, {})["valid_rate"] = validity_rates.get(arm)

        for arm in self.arms:
            validity_sources.setdefault(arm, {})["valid_rate"] = validity_rates.get(arm)

        rows = _build_projection_table(
            self.populations, self.generations, self.seeds, self.arms, self.tasks, cost_ranges, effective_overhead,
            pe_split, MEASURED_PE_RATIO,
        )

        baseline_row = rows[0] if rows else None
        deltas = (
            _build_d14_deltas(
                baseline_row, rows, cost_ranges, t_ppo_stats_by_arm_task, t_sentinel_mean_by_arm,
                validity_rates, pe_split, MEASURED_PE_RATIO,
            )
            if baseline_row
            else []
        )
        pe_core_bias = _build_pe_core_bias_summary(baseline_row, rows)

        report_dir = ctx.run_dir.report_dir
        md_path = os.path.join(report_dir, "projection.md")
        csv_path = os.path.join(report_dir, "projection.csv")
        figure_path = os.path.join(report_dir, "b6_projection.png")

        _write_markdown(
            md_path, rows, validity_sources, overhead, concurrency, deltas, baseline_row or {},
            pe_split, MEASURED_PE_RATIO, MEASURED_PE_RATIO_SOURCE, pe_core_bias,
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
            # D13 follow-up fix: the P/E-aware mixed-core throughput model's
            # own inputs/outputs, so a reader of the JSON (not just
            # projection.md) can see the correction and its provenance.
            "pe_split": pe_split,
            "pe_ratio": MEASURED_PE_RATIO,
            "pe_ratio_source": MEASURED_PE_RATIO_SOURCE,
            "pe_core_bias": pe_core_bias,
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
