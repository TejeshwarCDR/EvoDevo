"""D5 B2: the concurrency sweep.

Stage `concurrency_sweep` -- cost "cheap" relative to the primary PPO
matrix (reduced timesteps, see CAMPAIGN_EVAL_EPISODES/timesteps below),
but the ONLY B1-B7 stage so far that trains real PPO. depends_on
("machine_profile",): the topology/thread-pinning facts B1 measured are
what this stage's whole affinity-plan machinery is built on.

WHAT THIS MEASURES: for each worker count in --sweep-levels, this pool-
evaluates the SAME fixed set of genomes (see sample_workload()) and
records throughput (genomes/min), latency, CPU/thermal conditions, and
worker failures. The purpose is to find where adding workers stops paying
for itself -- contention from N processes sharing 28 logical CPUs is a
property of REAL PPO training (real numeric load on real cores), so this
stage never uses the surrogate evaluator in production; the `evaluator_
mode` constructor parameter that allows overriding to "surrogate" exists
ONLY for tests exercising the sweep's own plumbing (level ordering,
cooldown, figure generation) without paying for PPO, and is never wired to
any production CLI flag.

CONTROLS AGAINST CONFOUNDS, EXPLICITLY:
  - Level VISIT ORDER is randomised by default (--sweep-order=shuffled),
    not ascending, and the actual order used is recorded in the stage
    output. A monotonic drift confound (e.g. the CPU is simply hotter by
    hour 3 than hour 1, or a slow thermal ramp) would otherwise be
    indistinguishable from "throughput really falls as concurrency rises"
    if levels were always visited in the same increasing order -- a
    reviewer who sees ascending-only order in the method has no way to
    rule that out. Shuffling (and disclosing the order used) is what
    makes "this is a concurrency effect, not a time-of-day effect" a
    claim this stage can actually support.
  - A cooldown (--sweep-cooldown-s, default 60s) runs between every level,
    and package temperature is recorded at the START and END of each
    level, so sustained heat carried over from one level into the next is
    visible rather than silently blended into the next level's numbers.

This stage does NOT assume the answer. Nowhere in this module is 20 or 28
(this machine's all-physical/all-logical core counts) treated as the
expected or preferred optimum -- see _boundaries(), which derives those
numbers from ctx.topology every time, and _optimum_warning(), which fires
precisely when the measured optimum does NOT match them, because that is
the case worth a reader's attention, not the case to be assumed away.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Optional

from evodevo.preflight.telemetry import read_cpu_package_temp_c, read_thermal_throttle_count, stats

DEFAULT_LEVELS = (8, 12, 16, 20, 24, 28)
DEFAULT_N_GENOMES = 24
DEFAULT_TIMESTEPS = 20000
DEFAULT_COOLDOWN_S = 60.0
DEFAULT_TOLERANCE = 0.05
DEFAULT_TASK = "Walker-v0"
DEFAULT_TASK_TIMEOUT_S = 300.0

# evodevo/experiments/configs/shared/base.yaml's ppo_eval_episodes=1 -- a
# SHARED_BASE_KEYS field that does NOT vary between the fast_dev/
# primary_matrix profiles (unlike population_size/offspring_size/
# eval_budget/ppo_timesteps, which are exactly D15's open question). Safe
# to cite as a fixed campaign constant without reading a profile-scoped
# config: this stage reduces ppo_timesteps only (see DEFAULT_TIMESTEPS,
# always far below the campaign's 1,000,000) and records that reduction in
# the stage output, precisely so this number is never mistaken for t_PPO.
CAMPAIGN_EVAL_EPISODES = 1

VALID_ORDERS = ("ascending", "shuffled")


def _resolved_dev_steps_and_dynamics():
    """Sources dev_steps/init_condition and each registered arm's resolved
    architecture kwargs from the same places the real campaign does
    (evodevo/experiments/configs/benchmark.yaml's `factors:` block, and
    each encoder's public resolved_config.py PRIMARY/MONOTONE_ABLATION) --
    so sampled genomes are representative of what the real campaign will
    actually evaluate, not stale encoder-class defaults or an invented
    configuration. Reads ONLY `.factors` and each PRIMARY object's
    architecture fields -- never population_size/offspring_size/
    eval_budget/ppo_timesteps/seeds from that file (D15 stays open; see
    evodevo/preflight/context.py's own docstring)."""
    from evodevo.encoders.nca.resolved_config import PRIMARY as NCA_PRIMARY
    from evodevo.encoders.rule_based.resolved_config import (
        MONOTONE_ABLATION as RB_MONOTONE,
        PRIMARY as RB_PRIMARY,
    )
    from evodevo.experiments.benchmark import BenchmarkConfig
    # Same path evodevo.experiments.lock_hash.DEFAULT_BENCHMARK_CONFIG
    # names -- benchmark.py itself has no equivalent constant to import.
    from evodevo.experiments.lock_hash import DEFAULT_BENCHMARK_CONFIG

    bconfig = BenchmarkConfig.from_yaml(DEFAULT_BENCHMARK_CONFIG)
    dev_steps = bconfig.factors["dev_steps"]
    init_condition = bconfig.factors["init_condition"]

    dynamics_by_arm = {
        "direct": ("direct", {}),
        "nca": ("nca", {"hidden_channels": NCA_PRIMARY.hidden_channels}),
        "rule_based": (
            "rule_based",
            {"readout_mode": RB_PRIMARY.readout_mode, "readout_window": RB_PRIMARY.readout_window},
        ),
        "rule_based_monotone": (
            "rule_based",
            {
                "readout_mode": RB_MONOTONE.readout_mode,
                "readout_window": RB_MONOTONE.readout_window,
                "action_monotonicity": RB_MONOTONE.action_monotonicity,
            },
        ),
    }
    return dev_steps, init_condition, dynamics_by_arm


def sample_workload(n_genomes: int, seed: int = 0) -> list:
    """N genomes drawn round-robin across every registered arm (direct,
    nca, rule_based, rule_based_monotone), each via that arm's OWN
    random_genome() under its real resolved dynamics -- not hand-picked
    valid bodies. direct's real validity rate under this configuration is
    only a few percent (most raw random genomes are disconnected) while
    nca/rule_based's is roughly two-thirds -- round-robin sampling across
    all four arms therefore produces a genuinely mixed, representative
    population of valid and invalid bodies without this function ever
    curating for validity itself.

    Deterministic for a fixed (n_genomes, seed): calling this twice
    returns byte-identical results (workload_hash() below proves it), which
    is what lets every concurrency level in the sweep evaluate the
    IDENTICAL workload -- a difference in measured throughput between
    levels can only be a concurrency effect, never a workload difference.
    """
    import dataclasses

    import numpy as np

    from evodevo.core.factors import extract_factors
    from evodevo.encoders import get_encoder

    dev_steps, init_condition, dynamics_by_arm = _resolved_dev_steps_and_dynamics()
    arm_ids = sorted(dynamics_by_arm)  # fixed order: direct, nca, rule_based, rule_based_monotone

    workload = []
    for i in range(n_genomes):
        arm_id = arm_ids[i % len(arm_ids)]
        encoder_name, dynamics_kwargs = dynamics_by_arm[arm_id]
        genome_seed = seed * 1_000_000 + i
        combined = {"dev_steps": dev_steps, "init_condition": init_condition, **dynamics_kwargs}
        factors, encoder_kwargs = extract_factors(combined, seed=genome_seed)
        encoder = get_encoder(encoder_name, factors=factors, **encoder_kwargs)
        genome = encoder.random_genome(np.random.default_rng(genome_seed))
        # Store the FULL resolved factors extract_factors() actually
        # produced (e.g. hidden_channels for nca), not just the two keys
        # this function started from -- evaluator_entry.py's _build_encoder
        # reconstructs the encoder from exactly this dict inside the worker
        # pool, so an incomplete factors dict here would silently evaluate
        # a DIFFERENT architecture than the one random_genome() was just
        # sampled from (e.g. nca's real hidden_channels=0 vs. ExperimentFactors'
        # own unrelated default of 3). seed is intentionally excluded --
        # EvaluationTask.seed (below) is the single source of truth for it.
        factors_dict = {k: v for k, v in dataclasses.asdict(factors).items() if k != "seed"}
        workload.append(
            {
                "task_id": i,
                "arm_label": arm_id,
                "encoder": encoder_name,
                "seed": genome_seed,
                "genome": encoder.serialize_genome(genome),
                "meta": {"factors": factors_dict, "encoder_kwargs": encoder_kwargs},
            }
        )
    return workload


def workload_hash(workload: list) -> str:
    """A stable content hash of the exact task set (arm/seed/genome for
    every entry) -- proves every concurrency level evaluated the IDENTICAL
    workload, not merely "a same-sized workload"."""
    canonical = json.dumps(
        [{"arm_label": w["arm_label"], "seed": w["seed"], "genome": w["genome"]} for w in workload],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _boundaries(topology) -> dict:
    """The three physically-meaningful worker counts this machine's
    topology implies (see evodevo.parallel.topology.allocation_order's own
    docstring for the mechanism) -- derived from ctx.topology every call,
    never a hard-coded literal, so a reader on a differently-shaped machine
    gets that machine's own real boundaries, not this one's."""
    p_primary_count = sum(1 for c in topology.cores if c.kind == "P" and c.is_physical_primary)
    return {
        "p_core_only": p_primary_count,
        "all_physical": topology.physical_count,
        "all_logical": topology.logical_count,
    }


def _visit_order(levels: list, order: str, rng: random.Random) -> list:
    if order == "ascending":
        return sorted(levels)
    if order == "shuffled":
        shuffled = list(levels)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"sweep_order must be one of {VALID_ORDERS}, got {order!r}")


def _run_level(ctx, level: int, eval_tasks: list, timeout_s: float, evaluator_mode: str) -> dict:
    temp_start = read_cpu_package_temp_c()
    throttle_start = read_thermal_throttle_count()
    start_epoch = time.time()

    pool = ctx.make_pool(n_workers=level, mode=evaluator_mode, timeout_s=timeout_s, max_retries=1)
    with pool:
        results = list(pool.map_tasks(eval_tasks))
    pool_stats = pool.stats()
    affinity_plan = pool.plan.to_dict()

    end_epoch = time.time()
    temp_end = read_cpu_package_temp_c()
    throttle_end = read_thermal_throttle_count()

    wall_clock_s = end_epoch - start_epoch
    n_genomes = len(results)
    n_valid = sum(1 for r in results if r.valid)
    n_errored = sum(1 for r in results if r.error is not None)
    n_timed_out = sum(1 for r in results if r.timed_out)
    latencies = [r.wall_clock_s for r in results if r.wall_clock_s is not None]
    latency_stats = stats(latencies)

    genomes_per_minute = (n_genomes / wall_clock_s * 60.0) if wall_clock_s > 0 else 0.0
    valid_genomes_per_minute = (n_valid / wall_clock_s * 60.0) if wall_clock_s > 0 else 0.0

    cpu_percent_mean = cpu_percent_max = None
    if ctx.telemetry is not None:
        telemetry_summary = ctx.telemetry.summarise(window=(start_epoch, end_epoch))
        cpu_percent_mean = telemetry_summary["cpu_percent_overall"]["mean"]
        cpu_percent_max = telemetry_summary["cpu_percent_overall"]["max"]

    throttled = None
    if throttle_start is not None and throttle_end is not None:
        throttled = throttle_end > throttle_start

    return {
        "level": level,
        "n_genomes": n_genomes,
        "n_valid": n_valid,
        "n_invalid": n_genomes - n_valid,
        "worker_failures": n_errored + n_timed_out,
        "genomes_per_minute": genomes_per_minute,
        "valid_genomes_per_minute": valid_genomes_per_minute,
        "efficiency_genomes_per_minute_per_worker": (genomes_per_minute / level) if level > 0 else 0.0,
        "latency_mean_s": latency_stats["mean"],
        "latency_p95_s": latency_stats["p95"],
        "wall_clock_s": wall_clock_s,
        "cpu_percent_mean": cpu_percent_mean,
        "cpu_percent_max": cpu_percent_max,
        "package_temp_start_c": temp_start,
        "package_temp_end_c": temp_end,
        "throttled": throttled,
        "peak_rss_mb": pool_stats["peak_rss_mb_aggregate"],
        "workers_restarted": pool_stats["workers_restarted"],
        "affinity_plan": affinity_plan,
    }


def compute_optimum_and_knee(level_records: list, tolerance: float) -> dict:
    """optimum: the level with the highest measured genomes_per_minute.

    knee: the SMALLEST level whose throughput is within `tolerance`
    (relative) of that maximum -- the point past which adding workers
    stops meaningfully paying for itself. Reported separately from the raw
    optimum because the raw maximum can sit at an expensive level that
    only marginally beats a much cheaper one; the knee is the number a
    reader actually wants when choosing a concurrency level for the real
    campaign, and it is exactly what efficiency = throughput / level also
    exposes from a different angle (see each level record's own
    efficiency_genomes_per_minute_per_worker field)."""
    if not level_records:
        return {"optimum_level": None, "optimum_throughput": None, "knee_level": None, "knee_throughput": None, "tolerance": tolerance}

    by_level = sorted(level_records, key=lambda r: r["level"])
    optimum = max(by_level, key=lambda r: r["genomes_per_minute"])
    threshold = optimum["genomes_per_minute"] * (1.0 - tolerance)
    within_tolerance = [r for r in by_level if r["genomes_per_minute"] >= threshold]
    knee = min(within_tolerance, key=lambda r: r["level"])

    return {
        "optimum_level": optimum["level"],
        "optimum_throughput": optimum["genomes_per_minute"],
        "knee_level": knee["level"],
        "knee_throughput": knee["genomes_per_minute"],
        "tolerance": tolerance,
    }


def _optimum_warning(optimum_level: Optional[int], boundaries: dict) -> Optional[str]:
    if optimum_level is None:
        return None
    if optimum_level in (boundaries["all_physical"], boundaries["all_logical"]):
        return None
    return (
        f"measured throughput-maximising concurrency level is {optimum_level}, which does not "
        f"coincide with this machine's all-physical-cores boundary ({boundaries['all_physical']}) or "
        f"its all-logical/hyperthreaded boundary ({boundaries['all_logical']}). This is not "
        f"necessarily wrong -- it is the interesting case, and must reach the report rather than "
        f"being averaged away or assumed to be measurement noise."
    )


def _make_figure(
    path: str, level_records: list, boundaries: dict, timesteps: int, topology_detection_method: str,
    self_check_stamp: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_level = sorted(level_records, key=lambda r: r["level"])
    levels = [r["level"] for r in by_level]
    throughput = [r["genomes_per_minute"] for r in by_level]
    efficiency = [r["efficiency_genomes_per_minute_per_worker"] for r in by_level]

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(levels, throughput, marker="o", color="tab:blue", label="throughput (genomes/min)")
    ax1.set_xlabel("concurrency level (worker processes)")
    ax1.set_ylabel("throughput (genomes/min)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(levels, efficiency, marker="s", linestyle="--", color="tab:orange", label="efficiency (genomes/min per worker)")
    ax2.set_ylabel("efficiency (genomes/min per worker)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    boundary_labels = {
        boundaries["p_core_only"]: "P-core only",
        boundaries["all_physical"]: "all physical cores",
        boundaries["all_logical"]: "with hyperthreads",
    }
    for x, label in boundary_labels.items():
        ax1.axvline(x=x, color="gray", linestyle=":", linewidth=1)
        ax1.annotate(
            label, xy=(x, ax1.get_ylim()[1]), xytext=(2, -10), textcoords="offset points",
            rotation=90, va="top", ha="left", fontsize=8, color="gray",
        )

    title = "B2 concurrency sweep: throughput and efficiency vs. worker count"
    if self_check_stamp:
        title = f"{self_check_stamp} — {title}"
    ax1.set_title(title)
    fig.text(
        0.5, -0.02,
        f"Reduced-cost measurement: ppo_timesteps={timesteps} per genome (NOT t_PPO / the campaign "
        f"budget). Topology boundaries via {topology_detection_method}.",
        ha="center", va="top", fontsize=8, wrap=True,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class ConcurrencySweepStage:
    name = "concurrency_sweep"
    description = (
        "Concurrency sweep (D5 B2): real-PPO throughput/efficiency vs. worker count across "
        "--sweep-levels, on a fixed genome workload, with randomised visit order and inter-level "
        "cooldown to control for thermal drift."
    )
    estimated_cost = "cheap"
    depends_on = ("machine_profile",)

    def __init__(
        self,
        levels=DEFAULT_LEVELS,
        n_genomes: int = DEFAULT_N_GENOMES,
        timesteps: int = DEFAULT_TIMESTEPS,
        order: str = "shuffled",
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        tolerance: float = DEFAULT_TOLERANCE,
        task: str = DEFAULT_TASK,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        order_seed: Optional[int] = None,
        workload_seed: int = 0,
        evaluator_mode: str = "real",
        self_check_stamp: Optional[str] = None,
    ):
        if order not in VALID_ORDERS:
            raise ValueError(f"order must be one of {VALID_ORDERS}, got {order!r}")
        if evaluator_mode not in ("real", "surrogate"):
            raise ValueError(f"evaluator_mode must be 'real' or 'surrogate', got {evaluator_mode!r}")
        self.levels = tuple(levels)
        self.n_genomes = n_genomes
        self.timesteps = timesteps
        self.order = order
        self.cooldown_s = cooldown_s
        self.tolerance = tolerance
        self.task = task
        self.task_timeout_s = task_timeout_s
        self.order_seed = order_seed
        self.workload_seed = workload_seed
        # "real" in every production path (see module docstring); a
        # test-only override so the sweep's own plumbing can be exercised
        # without paying for PPO. Never set to "surrogate" from
        # evodevo/preflight/__main__.py's CLI wiring.
        self.evaluator_mode = evaluator_mode
        # Set only by --self-check (see evodevo/preflight/__main__.py); stamps
        # this stage's JSON output and figure title so nobody mistakes the
        # trivial-settings self-check run for a real B2 measurement.
        self.self_check_stamp = self_check_stamp

    def run(self, ctx) -> dict:
        from evodevo.parallel.worker_pool import EvaluationTask

        boundaries = _boundaries(ctx.topology)

        workload = sample_workload(self.n_genomes, seed=self.workload_seed)
        workload_hash_value = workload_hash(workload)
        eval_tasks = [
            EvaluationTask(
                task_id=w["task_id"], arm=w["encoder"], strategy="preflight_b2",
                task=self.task, seed=w["seed"], genome=w["genome"],
                ppo_timesteps=self.timesteps, eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=w["meta"],
            )
            for w in workload
        ]

        order_rng = random.Random(self.order_seed)
        visit_order = _visit_order(list(self.levels), self.order, order_rng)

        ctx.events.record(
            self.name, "measurement",
            {"visit_order": visit_order, "workload_hash": workload_hash_value, "n_genomes": self.n_genomes},
        )

        level_records = []
        for i, level in enumerate(visit_order):
            ctx.events.record(self.name, "measurement", {"level_start": level})
            record = _run_level(ctx, level, eval_tasks, self.task_timeout_s, self.evaluator_mode)
            level_records.append(record)
            ctx.events.record(self.name, "measurement", record)
            if record["worker_failures"] > 0:
                ctx.events.record(self.name, "anomaly", {"level": level, "worker_failures": record["worker_failures"]})
            if record["throttled"]:
                ctx.events.record(self.name, "anomaly", {"level": level, "thermal_throttled": True})

            is_last = i == len(visit_order) - 1
            if self.cooldown_s > 0 and not is_last:
                time.sleep(self.cooldown_s)

        optimum_and_knee = compute_optimum_and_knee(level_records, self.tolerance)
        warning = _optimum_warning(optimum_and_knee["optimum_level"], boundaries)
        warnings_list = [warning] if warning else []
        if warning:
            ctx.events.record(self.name, "anomaly", {"warning": warning})

        figure_path = None
        try:
            figure_path = os.path.join(ctx.run_dir.report_dir, "b2_concurrency.png")
            _make_figure(
                figure_path, level_records, boundaries, self.timesteps, ctx.topology.detection_method,
                self_check_stamp=self.self_check_stamp,
            )
        except Exception as e:  # noqa: BLE001 -- a figure failure must not lose the measured data
            ctx.events.record(self.name, "failure", {"error": f"figure generation failed: {type(e).__name__}: {e}"})
            figure_path = None

        result = {
            "levels_requested": list(self.levels),
            "visit_order": visit_order,
            "sweep_order_mode": self.order,
            "n_genomes": self.n_genomes,
            "workload_hash": workload_hash_value,
            "ppo_timesteps": self.timesteps,
            "ppo_timesteps_is_reduced_not_campaign_budget": True,
            "eval_episodes": CAMPAIGN_EVAL_EPISODES,
            "cooldown_s": self.cooldown_s,
            "tolerance": self.tolerance,
            "boundaries": boundaries,
            "levels": level_records,
            "optimum_level": optimum_and_knee["optimum_level"],
            "optimum_throughput_genomes_per_minute": optimum_and_knee["optimum_throughput"],
            "knee_level": optimum_and_knee["knee_level"],
            "knee_throughput_genomes_per_minute": optimum_and_knee["knee_throughput"],
            "warnings": warnings_list,
            "figure_path": figure_path,
            "evaluator_mode": self.evaluator_mode,
            "self_check_stamp": self.self_check_stamp,
        }
        return result
