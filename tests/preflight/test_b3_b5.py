"""Tests for evodevo.preflight.stages.b3_cost / b5_noise.

Runs real subprocess pools under a test-only surrogate-with-jitter
evaluator (see evodevo.parallel.evaluator_entry's "_debug_jitter" meta
hook) -- these tests are slower than most of tests/preflight/, but real
concurrent execution is what's under test. Note: importing numpy/torch at
THIS file's own top level is fine, same reasoning as
tests/parallel/test_worker_pool.py.
"""
from __future__ import annotations

import math
import os

import pytest

from evodevo.parallel.topology import detect_topology
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import atomic_write_json, create_or_resume_run_dir, read_json
from evodevo.preflight.stages import DependencyNotDoneError, StageRegistry, run_stages
from evodevo.preflight.stages.b3_cost import EvaluationCostStage, describe_values
from evodevo.preflight.stages.b5_noise import CacheEnabledError, EvaluationNoiseStage

_JITTER = {"fitness_noise_scale": 1.0, "sleep_s": 0.01}


def _make_ctx(tmp_path) -> PreflightContext:
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path / "runs"))
    events = EventLog(run_dir.events_path)
    return PreflightContext(run_dir=run_dir, events=events, telemetry=None, topology=detect_topology(), config={})


def _mark_done(ctx, stage_name: str, output: dict) -> None:
    atomic_write_json(ctx.run_dir.stage_output_path(stage_name), output)
    manifest = read_json(ctx.run_dir.manifest_path)
    manifest["stages"][stage_name] = {"status": "done"}
    atomic_write_json(ctx.run_dir.manifest_path, manifest)


def _fake_concurrency_sweep_done(ctx, knee_level: int = 2) -> None:
    _mark_done(ctx, "concurrency_sweep", {"knee_level": knee_level, "optimum_level": knee_level})


# A real, VALID direct-encoder genome (GOOD_WALKER, flattened) -- using an
# arbitrary/invalid genome here would make every re-evaluation hit the
# sentinel fast path deterministically (no jitter applied to invalid
# results), collapsing every "noise" measurement to a constant -10.0 and
# defeating the point of these fixtures.
_GOOD_WALKER_GENOME = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0]


def _synthetic_b3_output(task: str = "Walker-v0", ppo_timesteps: int = 0, concurrency: int = 2) -> dict:
    """A hand-built B3-shaped output with known fitness values, for B5
    tests that don't need to actually run B3 first."""

    def obs(arm, seed, fitness):
        return {
            "arm_label": arm, "encoder": arm, "genome_seed": seed,
            "genome": _GOOD_WALKER_GENOME,
            "meta": {"factors": {"dev_steps": 1, "init_condition": "center_seed"}, "encoder_kwargs": {}},
            "fitness": fitness, "wall_clock_s": 0.01, "body_hash": f"hash-{seed}",
        }

    observations = [obs("direct", i, f) for i, f in enumerate([1.0, 5.0, 9.0])]
    return {
        "concurrency_used": concurrency,
        "ppo_timesteps": ppo_timesteps,
        "arms": {
            "direct": {
                "tasks": {
                    task: {
                        "t_ppo": {"mean": 0.02, "median": 0.02, "sd": 0.001, "min": 0.019, "max": 0.021, "n": 3},
                        "observations": observations,
                    }
                }
            }
        },
    }


# --- B3: evaluation_cost --------------------------------------------------


def test_b3_depends_on_concurrency_sweep():
    stage = EvaluationCostStage()
    assert stage.depends_on == ("concurrency_sweep",)


def test_b3_registers_expensive_with_correct_dependency():
    registry = StageRegistry()
    registry.register(EvaluationCostStage())
    stage = registry.get("evaluation_cost")
    assert stage.estimated_cost == "expensive"
    assert stage.depends_on == ("concurrency_sweep",)


def test_b3_production_default_evaluator_mode_is_real():
    """MUST NOT: B3 never reduces timesteps, and never uses the surrogate
    evaluator in production -- see the stage's own module docstring."""
    stage = EvaluationCostStage()
    assert stage.evaluator_mode == "real"
    assert stage.ppo_timesteps == 1_000_000


def test_b3_runs_end_to_end_with_surrogate_jitter_and_produces_a_complete_result(tmp_path):
    ctx = _make_ctx(tmp_path)
    _fake_concurrency_sweep_done(ctx, knee_level=2)

    stage = EvaluationCostStage(
        arms=["direct", "nca"], tasks=["Walker-v0", "UpStepper-v0"], cost_reps=2, ppo_timesteps=0,
        t_sentinel_n=8, t_develop_n=8, evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)

    assert result["concurrency_used"] == 2
    assert result["concurrency_source"] == "b2_knee"
    assert result["ppo_timesteps_is_campaign_budget"] is True
    assert result["partial"] is False
    assert set(result["arms"].keys()) == {"direct", "nca"}

    for arm_data in result["arms"].values():
        for key in ("t_develop", "t_sentinel", "t_sentinel_p99", "valid_genome_draw", "tasks", "partial"):
            assert key in arm_data
        for stat_key in ("mean", "median", "sd", "min", "max", "n"):
            assert stat_key in arm_data["t_develop"]
            assert stat_key in arm_data["t_sentinel"]
        assert arm_data["t_develop"]["n"] == 8
        assert arm_data["t_sentinel"]["n"] == 8
        assert arm_data["partial"] is False

        assert set(arm_data["tasks"].keys()) == {"Walker-v0", "UpStepper-v0"}
        for task_data in arm_data["tasks"].values():
            for key in ("t_ppo", "telemetry", "observations", "n_requested", "n_successful",
                        "n_failed", "t_ppo_over_t_sentinel_ratio", "partial"):
                assert key in task_data
            for stat_key in ("mean", "median", "sd", "min", "max", "n"):
                assert stat_key in task_data["t_ppo"]
            assert task_data["n_successful"] == 2
            assert task_data["t_ppo"]["n"] == 2
            assert task_data["t_ppo_over_t_sentinel_ratio"] is not None
            assert task_data["partial"] is False


def test_b3_measures_tasks_independently_never_extrapolated(tmp_path):
    """Different tasks must get their OWN timing measurement -- the SAME
    genomes are used (a deliberate design choice, isolating the task's own
    effect), but each task's t_ppo must be independently measured, never
    copied from another task."""
    ctx = _make_ctx(tmp_path)
    _fake_concurrency_sweep_done(ctx)
    stage = EvaluationCostStage(
        arms=["direct"], tasks=["Walker-v0", "UpStepper-v0"], cost_reps=2, ppo_timesteps=0,
        t_sentinel_n=5, t_develop_n=5, evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)

    walker_obs = result["arms"]["direct"]["tasks"]["Walker-v0"]["observations"]
    upstepper_obs = result["arms"]["direct"]["tasks"]["UpStepper-v0"]["observations"]

    walker_seeds = {o["genome_seed"] for o in walker_obs}
    upstepper_seeds = {o["genome_seed"] for o in upstepper_obs}
    assert walker_seeds == upstepper_seeds, "the same genomes should be evaluated on every task"

    walker_wall_clocks = [o["wall_clock_s"] for o in walker_obs]
    upstepper_wall_clocks = [o["wall_clock_s"] for o in upstepper_obs]
    assert walker_wall_clocks != upstepper_wall_clocks, (
        "each task's timing must be its own independent measurement, not one copied from another"
    )


def test_b3_partial_result_when_valid_genomes_are_scarce(tmp_path):
    ctx = _make_ctx(tmp_path)
    _fake_concurrency_sweep_done(ctx)

    stage = EvaluationCostStage(
        arms=["direct"], tasks=["Walker-v0"], cost_reps=3, ppo_timesteps=0,
        cost_max_draws=0,  # deterministic: zero draw attempts allowed -> zero valid genomes, always
        t_sentinel_n=5, t_develop_n=5, evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)

    assert result["partial"] is True, "a cost estimate from fewer than cost_reps valid genomes must be marked partial"
    arm = result["arms"]["direct"]
    assert arm["partial"] is True
    assert arm["valid_genome_draw"] == {"n_needed": 3, "n_drawn": 0, "n_draw_attempts": 0, "shortfall": 3}
    task = arm["tasks"]["Walker-v0"]
    assert task["partial"] is True
    assert task["n_successful"] == 0
    assert task["n_requested"] == 3


def test_b3_does_not_silently_proceed_on_fewer_valid_genomes(tmp_path):
    """Even when SOME valid genomes are found (just fewer than cost_reps),
    the result must still be marked partial -- not silently reported as a
    complete cost_reps-sample estimate."""
    ctx = _make_ctx(tmp_path)
    _fake_concurrency_sweep_done(ctx)

    stage = EvaluationCostStage(
        arms=["direct"], tasks=["Walker-v0"], cost_reps=5, ppo_timesteps=0,
        cost_max_draws=3,  # far too few draws to reliably find 5 valid direct genomes (~2% valid rate)
        t_sentinel_n=3, t_develop_n=3, evaluator_mode="surrogate", jitter=_JITTER,
    )
    result = stage.run(ctx)

    arm = result["arms"]["direct"]
    assert arm["valid_genome_draw"]["n_draw_attempts"] <= 3
    assert arm["valid_genome_draw"]["n_drawn"] < 5
    assert arm["valid_genome_draw"]["shortfall"] > 0
    assert result["partial"] is True


# --- statistics, hand-computed fixtures -------------------------------


def test_describe_values_matches_hand_computed_fixture():
    values = [2.0, 4.0, 4.0, 4.0, 5.0]
    expected_mean = 3.8
    expected_variance = sum((v - expected_mean) ** 2 for v in values) / 4  # sample sd, n-1=4
    expected_sd = expected_variance**0.5

    result = describe_values(values)

    assert result["n"] == 5
    assert math.isclose(result["mean"], expected_mean)
    assert math.isclose(result["sd"], expected_sd)
    assert result["min"] == 2.0
    assert result["max"] == 5.0
    assert result["median"] == 4.0


def test_describe_values_single_sample_has_null_sd_not_zero():
    result = describe_values([7.0])
    assert result["n"] == 1
    assert result["mean"] == 7.0
    assert result["sd"] is None


def test_describe_values_empty_input():
    result = describe_values([])
    assert result == {"mean": None, "median": None, "sd": None, "min": None, "max": None, "n": 0}


def test_cv_matches_hand_computed_fixture():
    """mean=14, variance=((4)^2+(2)^2+0+(2)^2+(4)^2)/4=10, sd=sqrt(10)."""
    values = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = describe_values(values)

    assert math.isclose(stats["mean"], 14.0)
    assert math.isclose(stats["sd"], 10**0.5, rel_tol=1e-9)

    cv = stats["sd"] / abs(stats["mean"])
    assert math.isclose(cv, (10**0.5) / 14.0, rel_tol=1e-9)


# --- B5: evaluation_noise ------------------------------------------------


def test_b5_depends_on_evaluation_cost():
    stage = EvaluationNoiseStage()
    assert stage.depends_on == ("evaluation_cost",)


def test_b5_production_defaults_are_safe():
    """MUST NOT: real PPO by default, cache never enabled by default."""
    stage = EvaluationNoiseStage()
    assert stage.evaluator_mode == "real"
    assert stage.cache_enabled is False


def test_b5_refuses_to_run_if_cache_enabled(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output())

    stage = EvaluationNoiseStage(cache_enabled=True, evaluator_mode="surrogate")
    with pytest.raises(CacheEnabledError):
        stage.run(ctx)


def test_b5_refuses_to_run_if_evaluation_cost_is_not_done(tmp_path):
    from evodevo.preflight.stages.b1_machine import MachineProfileStage
    from evodevo.preflight.stages.b2_sweep import ConcurrencySweepStage

    ctx = _make_ctx(tmp_path)  # evaluation_cost never marked done

    registry = StageRegistry()
    # registry.ordered() validates the full depends_on chain up front, so
    # evaluation_cost's own dependency chain must be registered too (never
    # run, since only=["evaluation_noise"] restricts execution to just B5).
    registry.register(MachineProfileStage())
    registry.register(ConcurrencySweepStage(evaluator_mode="surrogate"))
    registry.register(EvaluationCostStage(evaluator_mode="surrogate"))
    registry.register(EvaluationNoiseStage(evaluator_mode="surrogate"))

    with pytest.raises(DependencyNotDoneError) as excinfo:
        run_stages(ctx, registry, only=["evaluation_noise"])
    assert "evaluation_cost" in str(excinfo.value)


def test_b5_registers_expensive_with_correct_dependency():
    registry = StageRegistry()
    registry.register(EvaluationNoiseStage())
    stage = registry.get("evaluation_noise")
    assert stage.estimated_cost == "expensive"
    assert stage.depends_on == ("evaluation_cost",)


def test_b5_runs_end_to_end_and_produces_a_complete_result_and_figure(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output())

    stage = EvaluationNoiseStage(noise_bodies=3, noise_reps=4, evaluator_mode="surrogate", jitter=_JITTER)
    result = stage.run(ctx)

    assert result["ppo_timesteps"] == 0
    assert result["concurrency_used"] == 2
    assert result["cache_enabled"] is False
    assert result["partial"] is False

    assert "Walker-v0" in result["tasks"]
    task_data = result["tasks"]["Walker-v0"]
    assert len(task_data["bodies"]) == 3
    assert {b["label"] for b in task_data["bodies"]} == {"low", "middle", "high"}
    for body in task_data["bodies"]:
        assert body["n_successful"] == 4
        assert len(body["fitness_values"]) == 4
        for key in ("mean", "sd", "cv", "min", "max", "range"):
            assert key in body
    assert task_data["pooled_within_body_sd"] is not None
    assert task_data["within_over_between_fraction"] is not None

    decision = result["decision_input"]
    assert isinstance(decision["text"], list) and len(decision["text"]) >= 2
    for sentence in decision["text"]:
        assert isinstance(sentence, str)
        assert sentence.strip().endswith((".", "!"))
        assert len(sentence.split()) > 4, f"looks like a bare number/fragment, not a sentence: {sentence!r}"
    joined = " ".join(decision["text"])
    assert "Tejeshwar" in joined  # explicitly defers D16
    assert "D16" not in "".join(str(v) for v in [decision["single_eval_defensible"]])  # doesn't itself pick an answer

    assert result["figure_path"] is not None
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0


def test_b5_selects_low_middle_high_from_b3_pooled_fitness_not_arbitrary(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output())  # fitness values 1.0, 5.0, 9.0

    stage = EvaluationNoiseStage(noise_bodies=3, noise_reps=1, evaluator_mode="surrogate")
    result = stage.run(ctx)

    bodies = {b["label"]: b["b3_fitness"] for b in result["tasks"]["Walker-v0"]["bodies"]}
    assert bodies == {"low": 1.0, "middle": 5.0, "high": 9.0}


def test_b5_reports_every_individual_fitness_value_not_just_summary(tmp_path):
    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output())

    stage = EvaluationNoiseStage(noise_bodies=3, noise_reps=5, evaluator_mode="surrogate", jitter=_JITTER)
    result = stage.run(ctx)

    for body in result["tasks"]["Walker-v0"]["bodies"]:
        assert len(body["fitness_values"]) == 5
        # each rep used a DIFFERENT seed -> jittered values should not all be identical
        assert len(set(body["fitness_values"])) > 1


def test_b5_on_a_run_whose_evaluation_cost_is_done_executes_without_rerunning_b3(tmp_path):
    """AC2: --only evaluation_noise on a run whose evaluation_cost is done
    executes without re-running B3."""
    from evodevo.preflight.stages.b1_machine import MachineProfileStage
    from evodevo.preflight.stages.b2_sweep import ConcurrencySweepStage

    ctx = _make_ctx(tmp_path)
    _mark_done(ctx, "evaluation_cost", _synthetic_b3_output())

    registry = StageRegistry()
    # A B3 that would raise if actually invoked (concurrency_sweep was
    # never marked done) -- proves it is never run.
    registry.register(MachineProfileStage())
    registry.register(ConcurrencySweepStage(evaluator_mode="surrogate"))
    registry.register(EvaluationCostStage(evaluator_mode="surrogate"))
    registry.register(EvaluationNoiseStage(evaluator_mode="surrogate", jitter=_JITTER))

    b3_status_before = read_json(ctx.run_dir.manifest_path)["stages"]["evaluation_cost"]

    run_stages(ctx, registry, only=["evaluation_noise"])

    manifest_after = read_json(ctx.run_dir.manifest_path)
    assert manifest_after["stages"]["evaluation_cost"] == b3_status_before
    assert manifest_after["stages"]["evaluation_noise"]["status"] == "done"
