"""Tests for evodevo.preflight.stages.b4_learning_curves / b6_projection / b7_reliability."""
from __future__ import annotations

import json
import math
import os

import pytest

from evodevo.parallel.topology import detect_topology
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import atomic_write_json, create_or_resume_run_dir, read_json
from evodevo.preflight.stages import DependencyNotDoneError, StageRegistry, run_stages
from evodevo.preflight.stages.b1_machine import MachineProfileStage
from evodevo.preflight.stages.b2_sweep import ConcurrencySweepStage
from evodevo.preflight.stages.b3_cost import EvaluationCostStage
from evodevo.preflight.stages.b4_learning_curves import LearningCurvesStage, select_knee
from evodevo.preflight.stages.b5_noise import EvaluationNoiseStage
from evodevo.preflight.stages.b6_projection import (
    D14_OPTIONS,
    MEASURED_PE_RATIO,
    ProjectionStage,
    _build_projection_table,
    _expected_cost_per_eval_range,
    _mixed_core_hours,
    _pe_worker_split,
)
from evodevo.preflight.stages.b7_reliability import ReliabilityStage, _classify_error

_JITTER = {"fitness_noise_scale": 1.0, "sleep_s": 0.005}


def _make_ctx(tmp_path) -> PreflightContext:
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path / "runs"))
    events = EventLog(run_dir.events_path)
    return PreflightContext(run_dir=run_dir, events=events, telemetry=None, topology=detect_topology(), config={})


def _mark_done(ctx, stage_name: str, output: dict) -> None:
    atomic_write_json(ctx.run_dir.stage_output_path(stage_name), output)
    manifest = read_json(ctx.run_dir.manifest_path)
    manifest["stages"][stage_name] = {"status": "done"}
    atomic_write_json(ctx.run_dir.manifest_path, manifest)


_ARM_DYNAMICS = {"direct": {}, "nca": {"hidden_channels": 0}}


def _find_valid_genome(arm: str, max_seeds: int = 500):
    """A REAL valid genome for `arm`, found by actually sampling and
    validating -- each encoder has its own genome FORMAT (direct's is 25
    material-type ints; nca's is hundreds of network weights), so a
    genome from one arm cannot stand in for another (using direct's shape
    for an 'nca'-labelled observation would make the worker's own decode()
    fail or silently produce garbage -- caught in testing, see git
    history)."""
    import numpy as np

    from evodevo.core.factors import extract_factors
    from evodevo.encoders import get_encoder
    from evodevo.morphology.validate import validate_body

    factors, enc_kwargs = extract_factors({"dev_steps": 100, "init_condition": "center_seed", **_ARM_DYNAMICS[arm]}, seed=0)
    encoder = get_encoder(arm, factors=factors, **enc_kwargs)
    for seed in range(max_seeds):
        genome = encoder.random_genome(np.random.default_rng(seed))
        if validate_body(encoder.decode(genome)).valid:
            return encoder.serialize_genome(genome)
    raise RuntimeError(f"no valid genome found for arm {arm!r} within {max_seeds} draws")


_VALID_GENOME_BY_ARM = {arm: _find_valid_genome(arm) for arm in ("direct", "nca")}


def _synthetic_b3_output(tasks=("Walker-v0",), ppo_timesteps: int = 0, concurrency: int = 2) -> dict:
    def obs(arm, seed, fitness):
        return {
            "arm_label": arm, "encoder": arm, "genome_seed": seed, "genome": _VALID_GENOME_BY_ARM[arm],
            "meta": {"factors": {"dev_steps": 100, "init_condition": "center_seed", **_ARM_DYNAMICS[arm]}, "encoder_kwargs": {}},
            "fitness": fitness, "wall_clock_s": 0.01, "body_hash": f"hash-{seed}",
        }

    arms = {}
    for arm in ("direct", "nca"):
        tasks_dict = {}
        for task in tasks:
            observations = [obs(arm, i, f) for i, f in enumerate([1.0, 5.0, 9.0])]
            tasks_dict[task] = {
                "t_ppo": {"mean": 0.02, "median": 0.02, "sd": 0.001, "min": 0.019, "max": 0.021, "n": 3},
                "observations": observations, "n_requested": 3, "n_successful": 3, "n_failed": 0,
            }
        arms[arm] = {
            "encoder": arm, "tasks": tasks_dict,
            "t_sentinel": {"mean": 0.001, "median": 0.001, "sd": 0.0001, "min": 0.0009, "max": 0.0012, "n": 100},
        }
    return {"concurrency_used": concurrency, "ppo_timesteps": ppo_timesteps, "arms": arms}


# --- B4: learning_curves --------------------------------------------------


def test_b4_depends_on_evaluation_cost():
    stage = LearningCurvesStage()
    assert stage.depends_on == ("evaluation_cost",)


def test_b4_registers_expensive_with_correct_dependency():
    registry = StageRegistry()
    registry.register(LearningCurvesStage())
    stage = registry.get("learning_curves")
    assert stage.estimated_cost == "expensive"
    assert stage.depends_on == ("evaluation_cost",)


def test_b4_production_default_evaluator_mode_is_real():
    stage = LearningCurvesStage()
    assert stage.evaluator_mode == "real"
    assert stage.curve_max_timesteps == 2_000_000
    assert stage.curve_eval_interval == 100_000


def test_b4_refuses_to_run_if_evaluation_cost_is_not_done(tmp_path):
    ctx = _make_ctx(tmp_path)
    registry = StageRegistry()
    registry.register(MachineProfileStage())
    registry.register(ConcurrencySweepStage(evaluator_mode="surrogate"))
    registry.register(EvaluationCostStage(evaluator_mode="surrogate"))
    registry.register(LearningCurvesStage(evaluator_mode="surrogate"))

    with pytest.raises(DependencyNotDoneError) as excinfo:
        run_stages(ctx, registry, only=["learning_curves"])
    assert "evaluation_cost" in str(excinfo.value)


def test_b4_runs_end_to_end_with_surrogate_jitter_and_produces_a_complete_result_and_figure(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "concurrency_sweep", {"knee_level": 2, "optimum_level": 2})
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0",)))

    stage = LearningCurvesStage(
        curve_bodies=3, curve_max_timesteps=1000, curve_eval_interval=200,
        evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)

    assert result["concurrency_used"] == 2
    assert result["curve_max_timesteps"] == 1000
    assert result["curve_eval_interval"] == 200
    assert "median marginal fitness improvement" in result["knee_rule_text"]
    assert "BETWEEN TASKS" in result["budget_note"]

    assert "Walker-v0" in result["tasks"]
    task_data = result["tasks"]["Walker-v0"]
    assert len(task_data["bodies"]) == 3
    for body in task_data["bodies"]:
        assert body["curve"], "expected a non-empty curve for a valid body"
        assert body["final_fitness"] is not None
        assert not body["failed"]
    assert task_data["median_series"]
    assert task_data["selected_budget"] is not None
    assert isinstance(task_data["no_plateau_observed"], bool)

    assert result["figure_path"] is not None
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0


def test_b4_warns_when_no_plateau_is_observed(tmp_path):
    """A curve that only ever rises (huge, unbounded jitter noise relative
    to a tiny max_timesteps) should never claim a knee it didn't find."""
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "concurrency_sweep", {"knee_level": 1, "optimum_level": 1})
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0",), concurrency=1))

    stage = LearningCurvesStage(
        curve_bodies=3, curve_max_timesteps=200, curve_eval_interval=100,  # only 2 checkpoints -- too few for select_knee to ever find a sustained-3-window plateau
        evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)
    task_data = result["tasks"]["Walker-v0"]
    assert task_data["no_plateau_observed"] is True
    assert task_data["warning"] is not None
    assert "FLOOR" in task_data["warning"]
    assert task_data["selected_budget"] == 200


# --- select_knee: unit-tested against synthetic curves, D6 rule -----------


def _series(levels: dict) -> list:
    return [(t, v) for t, v in levels.items()]


def test_select_knee_clean_plateau_finds_the_knee():
    series = [(i * 100_000, v) for i, v in enumerate([0, 10, 18, 24, 28, 29, 29.3, 29.5, 29.6, 29.65])]
    result = select_knee(series)
    assert result["knee_found"] is True
    assert result["knee_timesteps"] == 600_000


def test_select_knee_still_rising_curve_finds_no_knee():
    series = [(i * 100_000, v) for i, v in enumerate([0, 10, 20, 32, 46, 62, 80, 100, 122, 146])]
    result = select_knee(series)
    assert result["knee_found"] is False
    assert result["knee_timesteps"] is None


def test_select_knee_noisy_plateau_not_derailed_by_one_bad_window():
    series = [(i * 100_000, v) for i, v in enumerate([0, 10, 18, 24, 28, 28.1, 28.2, 45, 28.3, 28.35, 28.4, 28.42])]
    result = select_knee(series)
    assert result["knee_found"] is True
    # The spike at index 7 must not have been mistaken for sustained
    # improvement -- the found knee must be AFTER the spike, from a
    # genuinely sustained 3-window run.
    assert result["knee_timesteps"] == 700_000


def test_select_knee_plateau_then_rises_again_finds_no_knee():
    """Only 2 consecutive low-improvement windows before the curve rises
    again -- D6 requires 3 SUSTAINED windows, so this must NOT report a
    false-positive knee at the brief plateau."""
    series = [(i * 100_000, v) for i, v in enumerate([0, 10, 18, 24, 24.05, 24.1, 40, 60, 85])]
    result = select_knee(series)
    assert result["knee_found"] is False


def test_select_knee_rule_text_matches_d6_wording():
    result = select_knee([(0, 1), (100_000, 1.0001)])
    assert result["rule_text"] == (
        "The smallest budget at which the median marginal fitness improvement per additional "
        "100k steps falls below 1% of the fitness achieved so far, sustained over three "
        "consecutive windows."
    )


def test_select_knee_handles_short_series_without_raising():
    assert select_knee([])["knee_found"] is False
    assert select_knee([(0, 1.0)])["knee_found"] is False


# --- B6: projection --------------------------------------------------------


def test_b6_depends_on_learning_curves_and_evaluation_noise():
    stage = ProjectionStage()
    assert set(stage.depends_on) == {"learning_curves", "evaluation_noise"}


def test_b6_registers_free_with_correct_dependencies():
    registry = StageRegistry()
    registry.register(ProjectionStage())
    stage = registry.get("projection")
    assert stage.estimated_cost == "free"
    assert set(stage.depends_on) == {"learning_curves", "evaluation_noise"}


def test_b6_refuses_to_run_if_dependencies_are_not_done(tmp_path):
    ctx = _make_ctx(tmp_path)
    registry = StageRegistry()
    registry.register(MachineProfileStage())
    registry.register(ConcurrencySweepStage(evaluator_mode="surrogate"))
    registry.register(EvaluationCostStage(evaluator_mode="surrogate"))
    registry.register(LearningCurvesStage(evaluator_mode="surrogate"))
    registry.register(EvaluationNoiseStage(evaluator_mode="surrogate"))
    registry.register(ProjectionStage())

    with pytest.raises(DependencyNotDoneError) as excinfo:
        run_stages(ctx, registry, only=["projection"])
    message = str(excinfo.value)
    assert "learning_curves" in message or "evaluation_noise" in message


def test_expected_cost_per_eval_range_matches_hand_computed_example():
    """validity_rate=0.5, t_PPO={min:10,mean:20,max:30}, t_sentinel mean=2:
    expected = validity_rate*t_PPO + (1-validity_rate)*t_sentinel."""
    t_ppo_stats = {"min": 10.0, "mean": 20.0, "max": 30.0}
    t_sentinel_stats = {"mean": 2.0}

    result = _expected_cost_per_eval_range(0.5, t_ppo_stats, t_sentinel_stats)

    assert math.isclose(result["low"], 0.5 * 10.0 + 0.5 * 2.0)  # 6.0
    assert math.isclose(result["mean"], 0.5 * 20.0 + 0.5 * 2.0)  # 11.0
    assert math.isclose(result["high"], 0.5 * 30.0 + 0.5 * 2.0)  # 16.0


def test_projection_wall_clock_matches_hand_computed_example():
    """population=10, generations=5, seeds=2 -> evaluations_per_seed=50,
    total_evaluations=100. cost_per_eval_mean=11 (from the fixture above).

    NAIVE column (legacy, unchanged): concurrency folded into
    overhead_factor=4.0 (perfect 4x scaling assumed) -> parallel_s =
    100*11/4 = 275s = 275/3600 h.

    CORRECTED column (D13 follow-up fix, P/E-aware): 3 P-core workers at
    t_p=11s/eval, 1 E-core worker at t_e=11*2=22s/eval (pe_ratio=2.0) ->
    aggregate_rate = 3/11 + 1/22 evaluations/s -> hours =
    (100/aggregate_rate)/3600."""
    cost_ranges = {("direct", "Walker-v0"): {"low": 6.0, "mean": 11.0, "high": 16.0}}
    pe_split = {"n_p_workers": 3, "n_e_workers": 1}
    pe_ratio = 2.0
    rows = _build_projection_table(
        populations=[10], generations=[5], seeds_list=[2], arms=["direct"], tasks=["Walker-v0"],
        cost_ranges=cost_ranges, overhead_factor=4.0, pe_split=pe_split, pe_ratio=pe_ratio,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["evaluations_per_seed"] == 50
    assert row["total_evaluations"] == 100

    expected_naive_mean_hours = (100 * 11.0 / 4.0) / 3600.0
    assert math.isclose(row["wall_clock_hours_naive_mean"], expected_naive_mean_hours, rel_tol=1e-9)
    expected_naive_low_hours = (100 * 6.0 / 4.0) / 3600.0
    expected_naive_high_hours = (100 * 16.0 / 4.0) / 3600.0
    assert math.isclose(row["wall_clock_hours_naive_low"], expected_naive_low_hours, rel_tol=1e-9)
    assert math.isclose(row["wall_clock_hours_naive_high"], expected_naive_high_hours, rel_tol=1e-9)

    for label, cost in (("low", 6.0), ("mean", 11.0), ("high", 16.0)):
        aggregate_rate = 3 / cost + 1 / (cost * pe_ratio)
        expected_corrected_hours = (100 / aggregate_rate) / 3600.0
        assert math.isclose(row[f"wall_clock_hours_{label}"], expected_corrected_hours, rel_tol=1e-9)
    # The corrected figure must be WORSE (larger) than a naive estimate
    # that pretends the E-core worker is P-core-fast.
    assert row["wall_clock_hours_mean"] > (100 * 11.0 / 4.0) / 3600.0


def test_b6_runs_end_to_end_and_produces_markdown_csv_and_figure(tmp_path):
    ctx = _make_ctx(tmp_path)
    b2_output = {
        "levels": [
            {"level": 1, "genomes_per_minute": 100.0, "efficiency_genomes_per_minute_per_worker": 100.0},
            {"level": 2, "genomes_per_minute": 150.0, "efficiency_genomes_per_minute_per_worker": 75.0},
        ]
    }
    _mark_done(ctx, "concurrency_sweep", b2_output)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0",), concurrency=2))
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})

    stage = ProjectionStage(populations=[15], generations=[10], seeds=[5], arms=["direct", "nca"], tasks=["Walker-v0"])
    result = stage.run(ctx)

    assert result["concurrency_used"] == 2
    assert 0 < result["overhead"]["factor"] <= 1.0
    assert result["rows"]

    md_path = result["report_paths"]["markdown"]
    csv_path = result["report_paths"]["csv"]
    figure_path = result["report_paths"]["figure"]
    assert os.path.exists(md_path) and os.path.getsize(md_path) > 0
    assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    assert os.path.exists(figure_path) and os.path.getsize(figure_path) > 0

    md_text = open(md_path).read()
    assert "does not pick a configuration" in md_text
    assert "Tejeshwar" in md_text


def test_b6_every_cell_carries_a_range_never_a_bare_point_estimate(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "concurrency_sweep", {"levels": [{"level": 2, "genomes_per_minute": 100.0}]})
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0", "UpStepper-v0"), concurrency=2))
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})

    stage = ProjectionStage(populations=[10, 20], generations=[5], seeds=[3], arms=["direct", "nca"], tasks=["Walker-v0", "UpStepper-v0"])
    result = stage.run(ctx)

    assert len(result["rows"]) == 2 * 1 * 1 * 2 * 2  # populations x generations x seeds x arms x tasks
    for row in result["rows"]:
        assert row["wall_clock_hours_low"] is not None
        assert row["wall_clock_hours_mean"] is not None
        assert row["wall_clock_hours_high"] is not None
        assert row["wall_clock_hours_low"] <= row["wall_clock_hours_mean"] <= row["wall_clock_hours_high"]


def test_b6_never_selects_or_ranks_a_configuration(tmp_path):
    """MUST NOT: no row is flagged as chosen/recommended/best, and rows
    appear in the fixed (population, generations, seeds, arm, task) order
    -- never sorted by cost."""
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "concurrency_sweep", {"levels": [{"level": 2, "genomes_per_minute": 100.0}]})
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0",), concurrency=2))
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})

    stage = ProjectionStage(populations=[30, 10], generations=[5], seeds=[3], arms=["nca", "direct"], tasks=["Walker-v0"])
    result = stage.run(ctx)

    for row in result["rows"]:
        assert not any("recommend" in str(k).lower() or "chosen" in str(k).lower() or "best" in str(k).lower() for k in row)
    # Fixed grid order preserved exactly (30 before 10, nca before direct --
    # the CLI's OWN order, never re-sorted by wall-clock cost).
    assert [r["population_size"] for r in result["rows"]] == [30, 30, 10, 10]
    assert [r["arm"] for r in result["rows"]] == ["nca", "direct", "nca", "direct"]


# --- P/E-core-aware throughput model (D13 follow-up fix) ------------------


def test_pe_worker_split_matches_cpu_slots_allocation():
    """_pe_worker_split must never hand-derive the P/E split -- it has to
    come from the SAME cpu_slots()/allocation_order() the real pool uses,
    so the projection and the real launcher can never disagree."""
    from evodevo.parallel.topology import cpu_slots

    topology = detect_topology()
    concurrency = min(topology.logical_count, 6)
    plan = cpu_slots(topology, concurrency)
    e_ids = set(topology.e_core_ids)
    expected_n_e = sum(1 for slot in plan.slots if slot[0] in e_ids)

    pe_split = _pe_worker_split(topology, concurrency)
    assert pe_split["n_e_workers"] == expected_n_e
    assert pe_split["n_p_workers"] + pe_split["n_e_workers"] == concurrency


def test_mixed_core_hours_hand_computed_weighted_harmonic():
    """4 P-core workers at 10s/eval, 2 E-core workers at 25s/eval, 100
    evaluations total. aggregate_rate = 4/10 + 2/25 = 0.4 + 0.08 = 0.48
    evaluations/s -> hours = (100/0.48)/3600."""
    hours = _mixed_core_hours(total_evaluations=100, p_cost_s=10.0, e_cost_s=25.0, n_p_workers=4, n_e_workers=2)
    expected = (100 / (4 / 10.0 + 2 / 25.0)) / 3600.0
    assert math.isclose(hours, expected, rel_tol=1e-9)
    # Sanity: strictly worse than if all 6 workers were P-core-fast.
    all_p_core_hours = (100 * 10.0 / 6) / 3600.0
    assert hours > all_p_core_hours


def test_mixed_core_hours_with_zero_e_workers_is_pure_p_core_rate():
    hours = _mixed_core_hours(total_evaluations=100, p_cost_s=10.0, e_cost_s=25.0, n_p_workers=4, n_e_workers=0)
    expected = (100 * 10.0 / 4) / 3600.0
    assert math.isclose(hours, expected, rel_tol=1e-9)


def test_b6_pe_model_reduces_to_p_core_only_formula_at_or_below_p_core_boundary():
    """Step 1e's correctness invariant: for any concurrency where
    cpu_slots() places every worker on a P-core (or an unclassified core
    on a non-hybrid machine -- see _pe_worker_split's own docstring), the
    corrected mixed-rate model must reduce EXACTLY to the old P-core-only
    formula (total_evaluations * p_cost_s / concurrency), not an
    approximation of it."""
    topology = detect_topology()
    p_core_boundary = sum(1 for c in topology.cores if c.kind == "P" and c.is_physical_primary) or topology.logical_count
    total_evaluations = 750
    p_cost_s = 27.9
    e_cost_s = p_cost_s * MEASURED_PE_RATIO

    for concurrency in range(1, min(p_core_boundary, 8) + 1):
        pe_split = _pe_worker_split(topology, concurrency)
        assert pe_split["n_e_workers"] == 0, f"concurrency={concurrency} unexpectedly included an E-core worker"
        corrected_hours = _mixed_core_hours(
            total_evaluations, p_cost_s, e_cost_s, pe_split["n_p_workers"], pe_split["n_e_workers"]
        )
        expected_hours = (total_evaluations * p_cost_s / concurrency) / 3600.0
        assert math.isclose(corrected_hours, expected_hours, rel_tol=1e-9)


def test_b6_d14_options_cover_the_five_named_items():
    names = {opt["name"] for opt in D14_OPTIONS}
    assert names == {
        "more_seeds_5_to_10", "second_grid_size", "direct_repaired_arm", "nca_compact_arm", "larger_timestep_budget",
    }


def test_b6_d14_options_produce_genuinely_different_deltas(tmp_path):
    """Regression test for the D14 projection bug: every option used to
    collapse to baseline_hours * 2.0 (all five identical, 2.91h/+1.45h
    against ANY grid). Hand-computes two structurally different options
    (more_seeds_5_to_10, a SINGLE-CELL seeds-axis change, vs
    larger_timestep_budget, a SINGLE-CELL change that only scales t_PPO,
    weighted by the reference arm's own validity_rate) against the
    reference cell's own committed numbers, and checks they now differ."""
    ctx = _make_ctx(tmp_path)
    b2_output = {
        "levels": [
            {"level": 1, "genomes_per_minute": 100.0, "efficiency_genomes_per_minute_per_worker": 100.0},
            {"level": 2, "genomes_per_minute": 150.0, "efficiency_genomes_per_minute_per_worker": 75.0},
        ]
    }
    _mark_done(ctx, "concurrency_sweep", b2_output)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output(tasks=("Walker-v0",), concurrency=2))
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})

    stage = ProjectionStage(populations=[15], generations=[10], seeds=[5], arms=["direct", "nca"], tasks=["Walker-v0"])
    result = stage.run(ctx)

    deltas_by_name = {d["name"]: d for d in result["d14_deltas"]}

    # None of the five may still collapse to one shared value -- the bug's
    # own symptom (identical 2.91h / +1.45h everywhere).
    new_hours_values = {d["new_hours"] for d in result["d14_deltas"]}
    assert len(new_hours_values) > 1, "D14 options must not all collapse to the same new wall-clock"

    # --- hand-computed reference cell: population=15, generations=10,
    # seeds=5, arm="direct" (validity_rate=0.0215, the real committed
    # results/inspection/direct/direct_sanity.json value), task="Walker-v0".
    # cost_per_eval_mean = validity*t_ppo_mean + (1-validity)*t_sentinel_mean
    #                    = 0.0215*0.02 + 0.9785*0.001 = 0.0014085
    # total_evaluations = 15*10*5 = 750
    #
    # baseline_hours is the P/E-AWARE CORRECTED figure (D13 follow-up fix):
    # at concurrency=2, cpu_slots() on this machine's own real topology
    # places 2 workers via the exact same allocation logic the real pool
    # uses -- reuse it here rather than assuming a P/E split by hand.
    total_evaluations = 15 * 10 * 5
    validity_direct = 0.0215
    cost_per_eval_mean = validity_direct * 0.02 + (1 - validity_direct) * 0.001
    pe_split = _pe_worker_split(detect_topology(), concurrency=2)
    baseline_cell_hours = _mixed_core_hours(
        total_evaluations, cost_per_eval_mean, cost_per_eval_mean * MEASURED_PE_RATIO,
        pe_split["n_p_workers"], pe_split["n_e_workers"],
    )
    assert math.isclose(deltas_by_name["more_seeds_5_to_10"]["baseline_hours"], baseline_cell_hours, rel_tol=1e-9)

    # more_seeds_5_to_10: doubles ONLY this cell's seeds axis (5 -> 10),
    # so total_evaluations doubles and wall-clock exactly doubles.
    expected_more_seeds_hours = baseline_cell_hours * 2.0
    assert math.isclose(deltas_by_name["more_seeds_5_to_10"]["new_hours"], expected_more_seeds_hours, rel_tol=1e-9)
    assert deltas_by_name["more_seeds_5_to_10"]["scope"] == "single_cell"

    # larger_timestep_budget: doubles t_PPO ONLY (not t_sentinel), so the
    # new cost_per_eval is validity*(t_ppo*2) + (1-validity)*t_sentinel --
    # NOT a bare 2x of the cell, because most of direct's evaluations are
    # invalid (validity_rate=0.0215) and pay t_sentinel, which does not
    # grow with the timestep budget. Also P/E-aware, same pe_split.
    expected_new_cost_per_eval = validity_direct * (0.02 * 2.0) + (1 - validity_direct) * 0.001
    expected_timestep_hours = _mixed_core_hours(
        total_evaluations, expected_new_cost_per_eval, expected_new_cost_per_eval * MEASURED_PE_RATIO,
        pe_split["n_p_workers"], pe_split["n_e_workers"],
    )
    assert math.isclose(deltas_by_name["larger_timestep_budget"]["new_hours"], expected_timestep_hours, rel_tol=1e-9)
    assert deltas_by_name["larger_timestep_budget"]["scope"] == "single_cell"

    # The two options must now produce genuinely different deltas -- under
    # the bug, both were baseline_hours * 2.0 and therefore identical.
    assert not math.isclose(
        deltas_by_name["more_seeds_5_to_10"]["delta_hours"],
        deltas_by_name["larger_timestep_budget"]["delta_hours"],
        rel_tol=1e-6,
    )
    # larger_timestep_budget's delta must be SMALLER than a bare doubling --
    # direct's low validity_rate means most evaluations don't pay t_PPO at
    # all, so growing the timestep budget barely moves the average cost.
    assert deltas_by_name["larger_timestep_budget"]["delta_hours"] < deltas_by_name["more_seeds_5_to_10"]["delta_hours"]

    # second_grid_size and the two new-arm options are CAMPAIGN-WIDE, not
    # single-cell -- their baseline must be the full campaign total (sum of
    # every row), not the one reference cell.
    campaign_total_hours = sum(r["wall_clock_hours_mean"] for r in result["rows"])
    for name in ("second_grid_size", "direct_repaired_arm", "nca_compact_arm"):
        assert deltas_by_name[name]["scope"] == "campaign_total"
        assert math.isclose(deltas_by_name[name]["baseline_hours"], campaign_total_hours, rel_tol=1e-9)


# --- B7: reliability --------------------------------------------------------


def test_b7_depends_on_projection():
    stage = ReliabilityStage()
    assert stage.depends_on == ("projection",)


def test_b7_registers_free_with_correct_dependency():
    registry = StageRegistry()
    registry.register(ReliabilityStage())
    stage = registry.get("reliability")
    assert stage.estimated_cost == "free"
    assert stage.depends_on == ("projection",)


def test_b7_refuses_to_run_if_projection_is_not_done(tmp_path):
    ctx = _make_ctx(tmp_path)
    registry = StageRegistry()
    registry.register(MachineProfileStage())
    registry.register(ConcurrencySweepStage(evaluator_mode="surrogate"))
    registry.register(EvaluationCostStage(evaluator_mode="surrogate"))
    registry.register(LearningCurvesStage(evaluator_mode="surrogate"))
    registry.register(EvaluationNoiseStage(evaluator_mode="surrogate"))
    registry.register(ProjectionStage())
    registry.register(ReliabilityStage())

    with pytest.raises(DependencyNotDoneError) as excinfo:
        run_stages(ctx, registry, only=["reliability"])
    assert "projection" in str(excinfo.value)


def test_classify_error_keyword_matching():
    assert _classify_error(None) == "none"
    assert _classify_error("A process in the process pool was terminated abruptly") == "worker_crash"
    assert _classify_error("worker crashed after 2 attempt(s): BrokenProcessPool") == "worker_crash"
    assert _classify_error("MemoryError: cannot allocate memory") == "oom"
    assert _classify_error("task timed out after 30s") == "timeout"
    assert _classify_error("ValueError: something unrelated") == "other"


def test_b7_aggregates_correctly_from_synthetic_prior_stage_outputs_and_telemetry(tmp_path):
    ctx = _make_ctx(tmp_path)

    # machine.json (as write_machine_snapshot + B1 would have produced)
    atomic_write_json(
        ctx.run_dir.machine_path,
        {
            "ram_total_mb": 10000.0,
            "b1_machine_profile": {
                "risk_state": {
                    "swap_in_use": False, "ulimit_n_soft": 65536, "ulimit_c_soft": 0,
                    "sdb1_mounted": False, "nvidia_driver_loaded": False,
                    "unattended_upgrades_enabled": True, "coretemp_readable": True,
                }
            },
        },
    )

    # concurrency_sweep: 2 levels, 4 genomes each, known peak_rss/workers_restarted
    _mark_done(ctx, "concurrency_sweep", {
        "levels": [
            {"level": 1, "n_genomes": 4, "n_valid": 2, "n_invalid": 2, "peak_rss_mb": 500.0, "workers_restarted": 0},
            {"level": 2, "n_genomes": 4, "n_valid": 3, "n_invalid": 1, "peak_rss_mb": 600.0, "workers_restarted": 1},
        ]
    })
    # evaluation_cost: 1 arm, 1 task, 3 requested, 2 successful (1 failure)
    _mark_done(ctx, "evaluation_cost", {
        "arms": {"direct": {"tasks": {"Walker-v0": {"n_requested": 3, "n_successful": 2}}}}
    })
    _mark_done(ctx, "learning_curves", {"tasks": {}})
    _mark_done(ctx, "evaluation_noise", {"tasks": {}})
    _mark_done(ctx, "projection", {"rows": [{"total_evaluations": 1000}, {"total_evaluations": 2000}]})

    # events.jsonl: one worker_crash-shaped failure, recorded exactly as a
    # real stage would (see e.g. b3_cost.py's own ctx.events.record call).
    ctx.events.record("evaluation_cost", "failure", {"errors": ["worker crashed after 2 attempt(s): BrokenProcessPool"]})

    # telemetry.jsonl: 3 hand-written samples spanning a temperature rise
    # and a measurable disk_free_gb decrease.
    telemetry_lines = [
        {"ts_epoch": 1000.0, "cpu_package_temp_c": 60.0, "cpu_freq_per_core_mhz": [4000, 4000], "disk_free_gb": 100.0},
        {"ts_epoch": 1005.0, "cpu_package_temp_c": 90.0, "cpu_freq_per_core_mhz": [3800, 3800], "disk_free_gb": 99.0},
        {"ts_epoch": 1010.0, "cpu_package_temp_c": 92.0, "cpu_freq_per_core_mhz": [3600, 3600], "disk_free_gb": 98.0},
    ]
    with open(ctx.run_dir.telemetry_path, "w") as f:
        for line in telemetry_lines:
            f.write(json.dumps(line) + "\n")

    stage = ReliabilityStage(thermal_threshold_c=85.0)
    result = stage.run(ctx)

    failures = result["failures"]
    # total_attempted = concurrency_sweep(4+4=8) + evaluation_cost(3) = 11
    assert failures["total_attempted"] == 11
    # total_successful = concurrency_sweep((2+2)+(3+1)=8) + evaluation_cost(2) = 10
    assert failures["total_successful"] == 10
    assert failures["failures_by_class"]["worker_crash"] == 1
    assert failures["workers_restarted_total"] == 1

    assert result["concurrency_info"]["peak_rss_mb"] == 600.0
    assert result["concurrency_info"]["max_safe_concurrency"] == int(10000.0 * 0.8 // 600.0)

    assert result["thermal"]["max_temp_c"] == 92.0
    assert result["thermal"]["time_above_threshold_s"] > 0  # two samples >= 85C

    assert result["disk"]["bytes_written_gb"] == pytest.approx(2.0)  # 100 -> 98
    assert result["disk"]["campaign_total_evaluations"] == 3000
    assert result["disk"]["safety_margin"] == 3.0

    # risk table: every row measured or explicitly not, per AC4
    for row in result["risk_table"]:
        assert row["measured"] in (True, False)
        if not row["measured"]:
            assert row["note"] and row["note"] != ""


def test_b7_risk_table_includes_frequency_policy_rows_from_b1(tmp_path):
    """D13: a measurement whose frequency policy is unrecorded cannot be
    reproduced -- B7's risk table must surface every field B1's
    _cpu_freq_policy() records, and flag turbo being enabled
    (no_turbo=False) as contradicting the plan."""
    ctx = _make_ctx(tmp_path)
    atomic_write_json(
        ctx.run_dir.machine_path,
        {
            "ram_total_mb": 10000.0,
            "b1_machine_profile": {
                "risk_state": {"coretemp_readable": True},
                "cpu": {
                    "governor": "performance",
                    "freq_policy": {
                        "scaling_driver": "intel_pstate",
                        "energy_performance_preference_p_core": "performance",
                        "energy_performance_preference_e_core": "performance",
                        "epp_p_core_cpu_id": 0,
                        "epp_e_core_cpu_id": 16,
                        "intel_pstate_no_turbo": False,
                        "intel_pstate_status": "active",
                        "idle_freq_mhz_by_core": {"P_core_cpu0": 1775.4, "E_core_cpu16": 2316.6},
                    },
                },
            },
        },
    )
    _mark_done(ctx, "concurrency_sweep", {})
    _mark_done(ctx, "evaluation_cost", {})
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})
    _mark_done(ctx, "projection", {"rows": []})

    stage = ReliabilityStage()
    result = stage.run(ctx)

    rows_by_name = {row["row"]: row for row in result["risk_table"]}
    for expected in (
        "cpufreq_governor", "cpufreq_scaling_driver",
        "energy_performance_preference_p_core", "energy_performance_preference_e_core",
        "intel_pstate_no_turbo", "intel_pstate_status",
    ):
        assert expected in rows_by_name, f"missing D13 row: {expected}"
        assert rows_by_name[expected]["measured"] is True
        assert "IDENTICAL" in rows_by_name[expected]["note"]

    assert rows_by_name["cpufreq_governor"]["value"] == "performance"
    assert rows_by_name["cpufreq_scaling_driver"]["value"] == "intel_pstate"
    assert rows_by_name["intel_pstate_no_turbo"]["value"] is False
    assert rows_by_name["intel_pstate_no_turbo"]["contradicts_plan"] is True


def test_b7_risk_table_every_row_measured_or_has_a_reason(tmp_path):
    """AC4, directly: an entirely-empty run directory (nothing measured
    anywhere) must still produce a risk table where every unmeasured row
    carries an explicit reason, never a blank cell."""
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "concurrency_sweep", {})
    _mark_done(ctx, "evaluation_cost", {})
    _mark_done(ctx, "learning_curves", {})
    _mark_done(ctx, "evaluation_noise", {})
    _mark_done(ctx, "projection", {"rows": []})

    stage = ReliabilityStage()
    result = stage.run(ctx)

    assert len(result["risk_table"]) > 0
    for row in result["risk_table"]:
        assert "measured" in row and "value" in row and "note" in row
        if not row["measured"]:
            assert row["note"], f"row {row['row']!r} is unmeasured but has no reason"
    assert result["partial"] is True  # since most/all rows are unmeasured here
