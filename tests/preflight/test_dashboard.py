"""Tests for evodevo.preflight.dashboard.

Dashboard/PlainLogger are ordinary, numpy/torch-free-at-import modules --
safe to import and exercise directly in this process, unlike
evodevo.preflight.__main__ (see test_cli.py's own docstring for why that
one is different).
"""
from __future__ import annotations

import inspect
import time

from evodevo.parallel.topology import detect_topology
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.dashboard import Dashboard, PlainLogger, _spark
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import create_or_resume_run_dir
from evodevo.preflight.stages import StageRegistry
from evodevo.preflight.telemetry import TelemetrySampler


class _DummyStage:
    def __init__(self, name, depends_on=()):
        self.name = name
        self.description = "dummy stage for dashboard tests"
        self.estimated_cost = "free"
        self.depends_on = depends_on

    def run(self, ctx):
        return {}


def _make_ctx(tmp_path):
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))
    events = EventLog(run_dir.events_path)
    telemetry = TelemetrySampler(run_dir.telemetry_path, interval_s=0.2, disk_path=run_dir.path)
    ctx = PreflightContext(run_dir=run_dir, events=events, telemetry=telemetry, topology=detect_topology(), config={})
    return ctx


def _make_registry():
    registry = StageRegistry()
    registry.register(_DummyStage("a"))
    registry.register(_DummyStage("b", depends_on=("a",)))
    return registry


def test_dashboard_renders_a_snapshot_without_a_tty_or_any_events(tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    registry = _make_registry()
    dash = Dashboard(ctx, registry, interval_s=0.1)
    snapshot = dash._snapshot()
    dash._render(snapshot)  # must not raise even with zero events/telemetry samples so far
    ctx.events.close()


def test_plain_logger_renders_one_line_to_stderr(tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    registry = _make_registry()
    logger = PlainLogger(ctx, registry, interval_s=0.1)
    snapshot = logger._snapshot()
    logger._render(snapshot)
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert "preflight" in captured.err
    ctx.events.close()


def test_dashboard_reflects_running_stage_and_failure_from_events(tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    registry = _make_registry()

    manifest_path = ctx.run_dir.manifest_path
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["stages"]["a"] = {"status": "running", "duration_s": None}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    ctx.events.record("a", "measurement", {"level": 4, "genomes_per_minute": 12.5})
    ctx.events.record("a", "failure", {"error": "boom: something broke"})

    dash = Dashboard(ctx, registry, interval_s=0.1)
    snapshot = dash._snapshot()
    assert snapshot["current_stage"] == "a"
    assert snapshot["failure_count"] == 1
    assert snapshot["last_failure"]["payload"]["error"] == "boom: something broke"
    assert snapshot["last_measurement"]["payload"]["level"] == 4

    dash._render(snapshot)
    logger = PlainLogger(ctx, registry, interval_s=0.1)
    logger._render(snapshot)
    captured = capsys.readouterr()
    assert "failures=1" in captured.err
    ctx.events.close()


def test_dashboard_start_stop_thread_lifecycle_is_clean(tmp_path):
    ctx = _make_ctx(tmp_path)
    registry = _make_registry()
    logger = PlainLogger(ctx, registry, interval_s=0.05)
    logger.start()
    time.sleep(0.2)
    logger.stop()
    assert logger._thread is None
    ctx.events.close()


def test_eta_is_always_reported_as_a_range_never_a_single_number(tmp_path):
    ctx = _make_ctx(tmp_path)
    registry = _make_registry()

    manifest_path = ctx.run_dir.manifest_path
    import json

    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["stages"]["a"] = {"status": "done", "duration_s": 5.0}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    dash = Dashboard(ctx, registry, interval_s=0.1)
    snapshot = dash._snapshot()
    eta_str = dash._eta_str(snapshot)
    assert "-" in eta_str  # a range, not a point estimate
    assert snapshot["eta_low_s"] is not None
    assert snapshot["eta_high_s"] is not None
    assert snapshot["eta_low_s"] < snapshot["eta_high_s"]
    ctx.events.close()


def test_spark_handles_empty_and_none_values():
    assert _spark([]) == ""
    assert _spark([0, 50, 100]) != ""
    assert _spark([None, 10]) != ""


def test_dashboard_module_never_touches_worker_stdout_or_stderr():
    """Structural guard for 'never write worker stdout into the dashboard
    region': neither renderer has any code path that could read a
    WorkerResult's own captured stdout/stderr, because it never imports or
    references WorkerResult at all."""
    import evodevo.preflight.dashboard as dashboard_module

    source = inspect.getsource(dashboard_module)
    code_only = source.split('"""', 2)[-1]  # strip the module docstring, which discusses this by name
    assert "WorkerResult" not in code_only
    assert "worker_pool" not in code_only
