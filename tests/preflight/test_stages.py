from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import create_or_resume_run_dir
from evodevo.preflight.stages import (
    DependencyCycleError,
    DependencyNotDoneError,
    DuplicateStageError,
    StageRegistry,
    UnknownStageError,
    load_manifest,
    run_stages,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ctx(tmp_path) -> PreflightContext:
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path / "runs"))
    events = EventLog(run_dir.events_path)
    return PreflightContext(run_dir=run_dir, events=events, telemetry=None, topology=None, config={})


class RecordingStage:
    """A dummy Stage: satisfies the Stage protocol via plain attributes,
    does not inherit from anything -- proving the protocol is structural."""

    def __init__(self, name, depends_on=(), cost="cheap", calls=None):
        self.name = name
        self.description = f"dummy stage {name}"
        self.estimated_cost = cost
        self.depends_on = tuple(depends_on)
        self.calls = calls if calls is not None else []

    def run(self, ctx):
        self.calls.append(self.name)
        return {"stage": self.name}


class FlakyStage:
    name = "flaky"
    description = "fails on its first attempt, then succeeds"
    estimated_cost = "cheap"
    depends_on = ()

    def __init__(self):
        self.attempts = 0

    def run(self, ctx):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("boom")
        return {"attempts": self.attempts}


# --- registry mechanics ------------------------------------------------


def test_registry_rejects_duplicate_name():
    registry = StageRegistry()
    registry.register(RecordingStage("a"))
    with pytest.raises(DuplicateStageError):
        registry.register(RecordingStage("a"))


def test_registry_rejects_bad_estimated_cost():
    registry = StageRegistry()
    stage = RecordingStage("a")
    stage.estimated_cost = "moderate"
    with pytest.raises(ValueError):
        registry.register(stage)


def test_registry_rejects_stage_missing_required_attribute():
    class Incomplete:
        name = "x"
        description = "x"
        estimated_cost = "free"
        # no depends_on, no run

    registry = StageRegistry()
    with pytest.raises(TypeError):
        registry.register(Incomplete())


def test_ordered_respects_dependencies_regardless_of_registration_order():
    registry = StageRegistry()
    registry.register(RecordingStage("b", depends_on=("a",)))
    registry.register(RecordingStage("a"))
    order = registry.ordered()
    assert order.index("a") < order.index("b")


def test_ordered_detects_unregistered_dependency():
    registry = StageRegistry()
    registry.register(RecordingStage("a", depends_on=("ghost",)))
    with pytest.raises(UnknownStageError):
        registry.ordered()


def test_ordered_detects_cycle():
    registry = StageRegistry()
    registry.register(RecordingStage("a", depends_on=("b",)))
    registry.register(RecordingStage("b", depends_on=("a",)))
    with pytest.raises(DependencyCycleError):
        registry.ordered()


# --- orchestration -------------------------------------------------------


def test_two_dummy_stages_execute_in_dependency_order(tmp_path):
    calls = []
    registry = StageRegistry()
    # Registered in reverse order deliberately -- execution order must
    # come from depends_on, not registration order.
    registry.register(RecordingStage("b", depends_on=("a",), calls=calls))
    registry.register(RecordingStage("a", calls=calls))
    ctx = _make_ctx(tmp_path)

    manifest = run_stages(ctx, registry)

    assert calls == ["a", "b"]
    assert manifest["stages"]["a"]["status"] == "done"
    assert manifest["stages"]["b"]["status"] == "done"
    for name in ("a", "b"):
        entry = manifest["stages"][name]
        assert entry["duration_s"] is not None
        assert entry["output_path"] == ctx.run_dir.stage_output_path(name)
        assert entry["git_commit"] is not None
        assert os.path.exists(entry["output_path"])


def test_rerunning_does_not_repeat_a_done_stage(tmp_path):
    calls = []
    registry = StageRegistry()
    registry.register(RecordingStage("a", calls=calls))
    ctx = _make_ctx(tmp_path)

    run_stages(ctx, registry)
    assert calls == ["a"]

    run_stages(ctx, registry)  # no --force: must be silently skipped, not re-run
    assert calls == ["a"]


def test_force_reruns_a_done_stage(tmp_path):
    calls = []
    registry = StageRegistry()
    registry.register(RecordingStage("a", calls=calls))
    ctx = _make_ctx(tmp_path)

    run_stages(ctx, registry)
    run_stages(ctx, registry, force=["a"])

    assert calls == ["a", "a"]


def test_only_runs_exactly_one_stage(tmp_path):
    calls = []
    registry = StageRegistry()
    registry.register(RecordingStage("a", calls=calls))
    registry.register(RecordingStage("b", calls=calls))
    ctx = _make_ctx(tmp_path)

    run_stages(ctx, registry, only=["a"])

    assert calls == ["a"]
    manifest = load_manifest(ctx.run_dir)
    assert manifest["stages"]["a"]["status"] == "done"
    assert "b" not in manifest["stages"]


def test_skip_excludes_named_stage(tmp_path):
    calls = []
    registry = StageRegistry()
    registry.register(RecordingStage("a", calls=calls))
    registry.register(RecordingStage("b", calls=calls))
    ctx = _make_ctx(tmp_path)

    run_stages(ctx, registry, skip=["b"])

    assert calls == ["a"]
    manifest = load_manifest(ctx.run_dir)
    assert "b" not in manifest["stages"]


def test_unknown_stage_name_in_only_raises():
    registry = StageRegistry()
    registry.register(RecordingStage("a"))
    from evodevo.preflight.stages import resolve_plan

    with pytest.raises(UnknownStageError):
        resolve_plan(registry, {"stages": {}}, only=["typo"])


def test_unmet_dependency_refuses_with_message_naming_it(tmp_path):
    registry = StageRegistry()
    registry.register(RecordingStage("a"))
    registry.register(RecordingStage("b", depends_on=("a",)))
    ctx = _make_ctx(tmp_path)

    with pytest.raises(DependencyNotDoneError) as excinfo:
        run_stages(ctx, registry, only=["b"])

    message = str(excinfo.value)
    assert "'a'" in message
    assert "'b'" in message

    manifest = load_manifest(ctx.run_dir)
    assert manifest["stages"]["b"]["status"] == "failed"


def test_failed_stage_is_rerun_on_next_invocation(tmp_path):
    stage = FlakyStage()
    registry = StageRegistry()
    registry.register(stage)
    ctx = _make_ctx(tmp_path)

    with pytest.raises(RuntimeError):
        run_stages(ctx, registry)
    manifest = load_manifest(ctx.run_dir)
    assert manifest["stages"]["flaky"]["status"] == "failed"
    assert "boom" in manifest["stages"]["flaky"]["error"]

    run_stages(ctx, registry)  # resume: a failed stage is re-run automatically
    manifest = load_manifest(ctx.run_dir)
    assert manifest["stages"]["flaky"]["status"] == "done"
    assert stage.attempts == 2


# --- kill-and-resume (real subprocess, real SIGKILL) ------------------------

_DRIVER_SCRIPT = f"""
import sys
sys.path.insert(0, {REPO_ROOT!r})
import time
from evodevo.preflight.context import PreflightContext
from evodevo.preflight.events import EventLog
from evodevo.preflight.run_dir import create_or_resume_run_dir
from evodevo.preflight.stages import StageRegistry, run_stages


class FastStage:
    name = "fast"
    description = "fast dummy"
    estimated_cost = "free"
    depends_on = ()

    def run(self, ctx):
        return {{"ok": True}}


class SlowStage:
    name = "slow"
    description = "slow dummy"
    estimated_cost = "cheap"
    depends_on = ("fast",)

    def __init__(self, sleep_s):
        self.sleep_s = sleep_s

    def run(self, ctx):
        time.sleep(self.sleep_s)
        return {{"ok": True}}


def main():
    base_dir = sys.argv[1]
    run_id = sys.argv[2] or None
    sleep_s = float(sys.argv[3])

    registry = StageRegistry()
    registry.register(FastStage())
    registry.register(SlowStage(sleep_s))

    run_dir = create_or_resume_run_dir(base_dir=base_dir, run_id=run_id)
    print(run_dir.run_id, flush=True)

    events = EventLog(run_dir.events_path)
    ctx = PreflightContext(run_dir=run_dir, events=events, telemetry=None, topology=None, config={{}})
    try:
        run_stages(ctx, registry)
    finally:
        events.close()


if __name__ == "__main__":
    main()
"""


def test_kill_during_stage_is_rerun_on_resume_while_done_stage_is_skipped(tmp_path):
    script_path = tmp_path / "driver.py"
    script_path.write_text(_DRIVER_SCRIPT)
    base_dir = str(tmp_path / "runs")

    proc = subprocess.Popen(
        [sys.executable, str(script_path), base_dir, "", "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        run_id = proc.stdout.readline().strip()
        assert run_id, f"driver did not print a run_id (stderr: {proc.stderr.read()})"

        manifest_path = os.path.join(base_dir, run_id, "manifest.json")
        deadline = time.time() + 15
        fast_done_and_slow_running = False
        while time.time() < deadline:
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                stages = manifest.get("stages", {})
                if stages.get("fast", {}).get("status") == "done" and stages.get("slow", {}).get("status") == "running":
                    fast_done_and_slow_running = True
                    break
            time.sleep(0.05)

        assert fast_done_and_slow_running, "slow stage never reached 'running' status in time"

        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["stages"]["fast"]["status"] == "done"
    assert manifest["stages"]["slow"]["status"] == "running"
    fast_start_time_before_resume = manifest["stages"]["fast"]["start_time"]

    resumed = subprocess.run(
        [sys.executable, str(script_path), base_dir, run_id, "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resumed.returncode == 0, resumed.stderr

    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["stages"]["fast"]["status"] == "done"
    assert manifest["stages"]["fast"]["start_time"] == fast_start_time_before_resume, (
        "the already-done 'fast' stage must not have been re-run on resume"
    )
    assert manifest["stages"]["slow"]["status"] == "done"
