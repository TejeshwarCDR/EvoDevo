"""Tests for evodevo.preflight.stages.b1_machine / b2_sweep.

Runs real subprocess pools (b2) and a real subprocess probe (b1) -- these
tests are slower than the rest of tests/preflight/, but that's what's
under test. Note: importing numpy/torch at THIS file's own top level is
fine, same reasoning as tests/parallel/test_worker_pool.py -- under
pytest, sys.modules["__main__"] is pytest's own launcher, not this file.
"""
from __future__ import annotations

import os

import pytest

from evodevo.parallel.topology import detect_topology
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import create_or_resume_run_dir, read_json
from evodevo.preflight.stages import StageRegistry
from evodevo.preflight.stages.b1_machine import (
    MachineProfileStage,
    _measure_disk_write_rate,
    _probe_unconfigured_threading,
)
from evodevo.preflight.stages.b2_sweep import (
    ConcurrencySweepStage,
    compute_optimum_and_knee,
    sample_workload,
    workload_hash,
)


def _make_ctx(tmp_path, telemetry=None) -> PreflightContext:
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path / "runs"))
    events = EventLog(run_dir.events_path)
    return PreflightContext(run_dir=run_dir, events=events, telemetry=telemetry, topology=detect_topology(), config={})


# --- B1: machine_profile ----------------------------------------------

_B1_TOP_LEVEL_KEYS = {
    "topology", "memory_disk", "unconfigured_threading", "environment", "accelerator", "cpu", "risk_state",
    "self_check_stamp",
}


def test_b1_runs_to_completion_and_produces_every_documented_key(tmp_path):
    ctx = _make_ctx(tmp_path)
    stage = MachineProfileStage(disk_probe_mb=8)

    result = stage.run(ctx)

    assert set(result.keys()) == _B1_TOP_LEVEL_KEYS

    topo = result["topology"]
    for key in ("logical_count", "physical_count", "p_core_physical_count", "p_core_physical_ids",
                "e_core_count", "e_core_ids", "hybrid", "detection_method"):
        assert key in topo
    assert topo["logical_count"] == detect_topology().logical_count

    mem = result["memory_disk"]
    for key in ("ram_total_mb", "ram_available_mb", "swap_total_mb", "swap_used_mb",
                "disk_free_root_gb", "disk_free_run_dir_gb", "sdb1_mounted", "sdb1_mount_point",
                "sdb1_free_gb", "disk_probe_mb", "disk_write_rate_mb_s"):
        assert key in mem
    assert mem["disk_write_rate_mb_s"] is not None and mem["disk_write_rate_mb_s"] > 0

    thr = result["unconfigured_threading"]
    assert "torch_num_threads" in thr and "env" in thr

    env = result["environment"]
    for key in ("python_version", "numpy_version", "torch_version", "gymnasium_version",
                "stable_baselines3_version", "evogym_version", "evogym_commit",
                "project_git_commit", "project_dirty", "project_dirty_files", "lock_hash"):
        assert key in env
    assert env["python_version"]
    assert env["project_git_commit"]

    accel = result["accelerator"]
    assert "cuda_available" in accel and "ppo_runner_device" in accel

    cpu = result["cpu"]
    assert "model" in cpu and "max_freq_mhz" in cpu and "governor" in cpu

    risk = result["risk_state"]
    for key in ("swap_total_mb", "swap_used_mb", "swap_in_use", "ulimit_n_soft", "ulimit_n_hard",
                "ulimit_c_soft", "ulimit_c_hard", "sdb1_mounted", "nvidia_driver_loaded",
                "sleep_targets", "unattended_upgrades_enabled", "coretemp_readable"):
        assert key in risk

    # Merged into machine.json alongside the static snapshot -- not clobbering it.
    machine = read_json(ctx.run_dir.machine_path)
    assert machine["b1_machine_profile"] == result
    assert "topology" in machine  # the original write_machine_snapshot() content is still there


def test_b1_registers_free_with_no_dependencies():
    registry = StageRegistry()
    registry.register(MachineProfileStage())
    stage = registry.get("machine_profile")
    assert stage.estimated_cost == "free"
    assert stage.depends_on == ()


def test_b1_disk_probe_cleans_up_temp_file_even_when_interrupted(tmp_path, monkeypatch):
    from evodevo.preflight.stages import b1_machine

    class _FailingOnFsync:
        """Proxies every os.* attribute to the real os module except
        fsync, which fails -- narrower than monkeypatching the global os
        module (which would affect unrelated code, e.g. pytest's own
        internals, for the test's duration)."""

        def fsync(self, fd):
            raise OSError("disk full (injected)")

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(b1_machine, "os", _FailingOnFsync())

    run_dir_path = str(tmp_path)
    with pytest.raises(OSError):
        _measure_disk_write_rate(run_dir_path, 1)

    leftover = [f for f in os.listdir(run_dir_path) if f.startswith(".b1_disk_probe_")]
    assert leftover == [], f"temp file(s) left behind after an interrupted write: {leftover}"


def test_unconfigured_threading_probe_shows_the_real_multi_threaded_default():
    result = _probe_unconfigured_threading()

    assert result.get("error") is None, result
    assert result["torch_num_threads"] is not None
    assert result["torch_num_threads"] > 1, (
        "expected the genuinely UNCONFIGURED torch default to exceed 1 thread on this "
        "multi-core machine -- a value of 1 here would mean the probe isn't really "
        "measuring the unconfigured state (e.g. env vars leaked in from this process)"
    )
    for var, value in result["env"].items():
        assert value is None, f"{var} was not stripped from the probe subprocess's environment"


# --- B2: concurrency_sweep ----------------------------------------------


def test_b2_depends_on_machine_profile():
    stage = ConcurrencySweepStage()
    assert stage.depends_on == ("machine_profile",)


def test_b2_production_default_evaluator_mode_is_real():
    """The production default MUST be real PPO -- see MUST NOT in the
    module docstring: contention is a property of real work. evaluator_
    mode="surrogate" exists only as an explicit, test-only override."""
    stage = ConcurrencySweepStage()
    assert stage.evaluator_mode == "real"


def test_workload_is_byte_identical_across_repeated_calls(tmp_path):
    w1 = sample_workload(8, seed=0)
    w2 = sample_workload(8, seed=0)
    assert workload_hash(w1) == workload_hash(w2)
    # Also a genuine mix of arms, not a single hand-picked encoder.
    assert len({w["arm_label"] for w in w1}) > 1


def test_workload_mix_includes_both_valid_and_invalid_bodies():
    from evodevo.core.factors import ExperimentFactors
    from evodevo.encoders import get_encoder
    from evodevo.morphology.validate import validate_body

    workload = sample_workload(24, seed=0)
    validity = []
    for w in workload:
        factors = ExperimentFactors(seed=w["seed"], **w["meta"]["factors"])
        encoder = get_encoder(w["encoder"], factors=factors, **w["meta"]["encoder_kwargs"])
        genome = encoder.deserialize_genome(w["genome"])
        body = encoder.decode(genome)
        validity.append(validate_body(body).valid)

    assert any(validity), "expected at least one valid body in a 24-genome sample -- not hand-picked, but shouldn't be zero"
    assert not all(validity), "expected at least one invalid body -- a realistic mix, not cherry-picked valid walkers"


def test_b2_runs_with_surrogate_evaluator_and_produces_a_complete_result_and_figure(tmp_path):
    ctx = _make_ctx(tmp_path)
    stage = ConcurrencySweepStage(
        levels=[1, 2],
        n_genomes=4,
        timesteps=0,
        cooldown_s=0,
        evaluator_mode="surrogate",  # test-only override -- see stage docstring
    )

    result = stage.run(ctx)

    assert result["levels_requested"] == [1, 2]
    assert set(result["visit_order"]) == {1, 2}
    assert result["n_genomes"] == 4
    assert result["workload_hash"] == workload_hash(sample_workload(4, seed=stage.workload_seed))
    assert result["ppo_timesteps_is_reduced_not_campaign_budget"] is True
    assert result["evaluator_mode"] == "surrogate"

    assert len(result["levels"]) == 2
    for level_record in result["levels"]:
        assert level_record["level"] in (1, 2)
        assert level_record["n_genomes"] == 4
        assert level_record["n_valid"] + level_record["n_invalid"] == 4
        assert level_record["genomes_per_minute"] >= 0
        assert level_record["worker_failures"] == 0
        assert "affinity_plan" in level_record
        assert level_record["affinity_plan"]["n_workers"] == level_record["level"]

    assert result["optimum_level"] in (1, 2)
    assert result["knee_level"] in (1, 2)
    assert isinstance(result["warnings"], list)

    assert result["figure_path"] is not None
    assert os.path.exists(result["figure_path"])
    assert os.path.getsize(result["figure_path"]) > 0

    # Also written under stages/concurrency_sweep.json via the normal
    # orchestration path -- exercised here directly since this test calls
    # stage.run() itself rather than going through run_stages().
    from evodevo.preflight.run_dir import atomic_write_json

    atomic_write_json(ctx.run_dir.stage_output_path(stage.name), result)
    reloaded = read_json(ctx.run_dir.stage_output_path(stage.name))
    assert reloaded["optimum_level"] == result["optimum_level"]


def test_b2_ascending_order_visits_levels_in_order(tmp_path):
    ctx = _make_ctx(tmp_path)
    stage = ConcurrencySweepStage(
        levels=[2, 1], n_genomes=2, timesteps=0, cooldown_s=0,
        order="ascending", evaluator_mode="surrogate",
    )
    result = stage.run(ctx)
    assert result["visit_order"] == [1, 2]


def test_b2_rejects_invalid_order():
    with pytest.raises(ValueError):
        ConcurrencySweepStage(order="descending")


def test_b2_rejects_invalid_evaluator_mode():
    with pytest.raises(ValueError):
        ConcurrencySweepStage(evaluator_mode="fake")


# --- knee computation, unit-tested against synthetic curves --------------


def _records(levels, throughputs):
    return [{"level": lv, "genomes_per_minute": tp} for lv, tp in zip(levels, throughputs)]


def test_knee_on_a_saturating_curve():
    """Throughput rises then plateaus -- knee should land well below the
    literal maximum, at the point where further workers stop paying off."""
    levels = [1, 2, 4, 8, 16, 20]
    throughputs = [10, 20, 28, 38, 40, 40]
    result = compute_optimum_and_knee(_records(levels, throughputs), tolerance=0.05)

    assert result["optimum_level"] == 16
    assert result["optimum_throughput"] == 40
    assert result["knee_level"] == 8
    assert result["knee_throughput"] == 38


def test_knee_on_a_peaked_then_declining_curve():
    """A genuine single peak with real degradation after it -- optimum and
    knee should coincide, since nothing else comes within tolerance."""
    levels = [1, 2, 4, 8, 12, 16, 20]
    throughputs = [10, 18, 24, 28, 25, 20, 15]
    result = compute_optimum_and_knee(_records(levels, throughputs), tolerance=0.05)

    assert result["optimum_level"] == 8
    assert result["knee_level"] == 8


def test_knee_on_a_flat_curve():
    """No real concurrency benefit anywhere -- knee (and optimum) should
    land at the SMALLEST level tested, not an arbitrary tie-break."""
    levels = [1, 2, 4, 8, 12, 16, 20]
    throughputs = [20.0] * len(levels)
    result = compute_optimum_and_knee(_records(levels, throughputs), tolerance=0.05)

    assert result["optimum_level"] == 1
    assert result["knee_level"] == 1


def test_knee_on_empty_input():
    result = compute_optimum_and_knee([], tolerance=0.05)
    assert result["optimum_level"] is None
    assert result["knee_level"] is None


# --- the "must not assume 20/28" guarantee, directly ---------------------


def test_optimum_warning_fires_for_a_non_boundary_optimum():
    from evodevo.preflight.stages.b2_sweep import _optimum_warning

    boundaries = {"p_core_only": 8, "all_physical": 20, "all_logical": 28}
    assert _optimum_warning(12, boundaries) is not None
    assert _optimum_warning(20, boundaries) is None
    assert _optimum_warning(28, boundaries) is None
