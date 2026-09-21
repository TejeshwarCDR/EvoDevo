"""Tests for evodevo.preflight.report -- must be runnable at any time,
including against a still-in-progress or interrupted run, labeling every
artifact with its completion fraction (D9).
"""
from __future__ import annotations

from evodevo.parallel.topology import detect_topology
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.report import generate_report, generate_report_markdown, regenerate_figures
from evodevo.preflight.run_dir import atomic_write_json, create_or_resume_run_dir, read_json
from evodevo.preflight.stages.b1_machine import MachineProfileStage
from evodevo.preflight.stages.b2_sweep import ConcurrencySweepStage
from evodevo.preflight.telemetry import TelemetrySampler


def _mark_done(ctx, name, output):
    atomic_write_json(ctx.run_dir.stage_output_path(name), output)
    manifest = read_json(ctx.run_dir.manifest_path)
    manifest["stages"][name] = {"status": "done", "duration_s": 1.5}
    atomic_write_json(ctx.run_dir.manifest_path, manifest)


def _make_partial_run(tmp_path):
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))
    events = EventLog(run_dir.events_path)
    telemetry = TelemetrySampler(run_dir.telemetry_path, interval_s=0.5, disk_path=run_dir.path)
    ctx = PreflightContext(run_dir=run_dir, events=events, telemetry=telemetry, topology=detect_topology(), config={})

    b1 = MachineProfileStage(disk_probe_mb=4)
    _mark_done(ctx, "machine_profile", b1.run(ctx))

    b2 = ConcurrencySweepStage(levels=[1], n_genomes=1, timesteps=0, cooldown_s=0, evaluator_mode="surrogate")
    _mark_done(ctx, "concurrency_sweep", b2.run(ctx))

    # evaluation_cost/learning_curves/evaluation_noise/projection/reliability
    # left pending -- a genuinely partial run.
    events.close()
    return run_dir


def test_report_runs_on_a_partial_run_and_labels_completion_fraction(tmp_path):
    run_dir = _make_partial_run(tmp_path)

    markdown = generate_report_markdown(run_dir)
    assert "Completion: 2/7 stages done." in markdown
    assert "## machine_profile -- status: `done`" in markdown
    assert "## concurrency_sweep -- status: `done`" in markdown
    assert "## learning_curves -- status: `pending`" in markdown
    assert "Not yet complete" in markdown


def test_report_regenerates_figures_only_for_completed_figure_producing_stages(tmp_path):
    run_dir = _make_partial_run(tmp_path)
    written = regenerate_figures(run_dir)

    assert "concurrency_sweep" in written
    assert written["concurrency_sweep"][0].endswith("b2_concurrency.png")
    import os

    assert os.path.exists(written["concurrency_sweep"][0])

    # No B4/B5/B6 output exists yet in this partial run -- report.py must
    # not attempt to regenerate their figures (and must not crash trying).
    assert "learning_curves" not in written
    assert "evaluation_noise" not in written
    assert "projection" not in written


def test_generate_report_writes_preflight_report_md(tmp_path):
    run_dir = _make_partial_run(tmp_path)
    out_path = generate_report(run_dir)
    import os

    assert os.path.exists(out_path)
    assert out_path.endswith("PREFLIGHT_REPORT.md")
    with open(out_path) as f:
        content = f.read()
    assert "Completion: 2/7" in content


def test_report_is_safe_to_run_on_a_run_with_zero_completed_stages(tmp_path):
    """The extreme case of 'partial': nothing has finished yet at all."""
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))
    markdown = generate_report_markdown(run_dir)
    assert "Completion: 0/7 stages done." in markdown
    for name in ("machine_profile", "concurrency_sweep", "evaluation_cost", "reliability"):
        assert f"## {name} -- status: `pending`" in markdown
