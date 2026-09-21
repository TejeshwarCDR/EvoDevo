"""Tests for evodevo.preflight.__main__ -- ALWAYS via subprocess.

evodevo.preflight.__main__ must never be imported directly in this (or any
other) test process: its own top-level configure_threading(1, strict=True)
call raises ThreadingConfiguredTooLateError the instant numpy/torch is
already loaded, which is true of nearly every other test in this suite by
the time it runs. See that module's own docstring. Every test here
launches a real `python -m evodevo.preflight ...` subprocess instead,
exactly as an operator would.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from evodevo.preflight.dirty_check import project_is_dirty

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_REQUIRED_FLAGS = [
    "--run-id", "--only", "--skip", "--force", "--dry-run", "--self-check",
    "--concurrency", "--sweep-levels", "--sweep-genomes", "--sweep-timesteps",
    "--sweep-order", "--sweep-cooldown-s", "--arms", "--tasks", "--ppo-timesteps",
    "--cost-reps", "--noise-bodies", "--noise-reps", "--curve-bodies",
    "--curve-max-timesteps", "--curve-eval-interval", "--proj-populations",
    "--proj-generations", "--proj-seeds", "--proj-overhead", "--telemetry-interval",
    "--timeout-s", "--no-dashboard", "--dashboard-interval",
]


def _run(args, timeout=60, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "evodevo.preflight", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, **kwargs,
    )


def test_help_exits_zero_and_starts_nothing(tmp_path):
    result = _run(["--help"])
    assert result.returncode == 0
    assert not os.path.exists(tmp_path / "results")


def test_help_lists_every_documented_flag():
    result = _run(["--help"])
    for flag in _REQUIRED_FLAGS:
        assert flag in result.stdout, f"{flag!r} missing from --help output"


def test_dry_run_prints_stage_plan_and_writes_no_measurement_data(tmp_path):
    base_dir = tmp_path / "preflight"
    result = _run(["--dry-run", "--base-dir", str(base_dir)])
    assert result.returncode == 0
    assert "machine_profile" in result.stdout
    assert "reliability" in result.stdout

    run_dirs = list(base_dir.iterdir())
    assert len(run_dirs) == 1
    stages_dir = run_dirs[0] / "stages"
    assert list(stages_dir.glob("*.json")) == []
    assert not (run_dirs[0] / "resolved_config.json").exists()


def test_non_tty_stdout_falls_back_to_plain_line_logging(tmp_path):
    """subprocess.PIPE-captured stdout is never a TTY, so this exercises
    the exact same fallback __main__.py takes under --no-dashboard or a
    genuinely non-interactive terminal -- without needing a real pty."""
    base_dir = tmp_path / "preflight"
    result = _run(["--only", "machine_profile", "--base-dir", str(base_dir), "--allow-dirty"], timeout=60)
    assert result.returncode == 0
    assert "[preflight t=" in result.stderr
    assert "stage=" in result.stderr


def test_self_check_dry_run_forces_trivial_settings_and_fresh_run_id(tmp_path):
    base_dir = tmp_path / "preflight"
    result = _run(["--self-check", "--dry-run", "--base-dir", str(base_dir)])
    assert result.returncode == 0
    assert "selfcheck_" in result.stdout


def test_unknown_run_id_raises_run_dir_not_found(tmp_path):
    base_dir = tmp_path / "preflight"
    result = _run(["--dry-run", "--base-dir", str(base_dir), "--run-id", "does-not-exist"])
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_sigint_during_a_stage_leaves_a_resumable_manifest_and_resume_completes_it(tmp_path):
    base_dir = tmp_path / "preflight"
    common = [
        "--base-dir", str(base_dir),
        "--only", "machine_profile,concurrency_sweep",
        "--sweep-levels", "1,2",
        "--sweep-genomes", "1",
        "--sweep-timesteps", "500",
        "--no-dashboard",
        "--allow-dirty",
    ]
    proc = subprocess.Popen(
        [sys.executable, "-m", "evodevo.preflight", *common, "--sweep-cooldown-s", "20"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 90
        running_stage = None
        run_dir_path = None
        while time.time() < deadline:
            if base_dir.exists():
                candidates = list(base_dir.iterdir())
                if candidates:
                    run_dir_path = candidates[0]
                    manifest_path = run_dir_path / "manifest.json"
                    if manifest_path.exists():
                        try:
                            manifest = json.loads(manifest_path.read_text())
                        except ValueError:
                            manifest = {"stages": {}}
                        for name, entry in manifest.get("stages", {}).items():
                            if entry.get("status") == "running":
                                running_stage = name
                                break
            if running_stage:
                break
            time.sleep(0.1)

        assert running_stage is not None, "no stage reached status='running' before the poll deadline"
        assert run_dir_path is not None

        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert proc.returncode == 130

    manifest_path = run_dir_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["stages"][running_stage]["status"] == "running"

    resolved_config_path = run_dir_path / "resolved_config.json"
    assert resolved_config_path.exists()

    resume = _run(
        [
            "--base-dir", str(base_dir), "--run-id", run_dir_path.name,
            "--only", "machine_profile,concurrency_sweep",
            "--sweep-levels", "1,2", "--sweep-genomes", "1", "--sweep-timesteps", "500",
            "--sweep-cooldown-s", "0", "--no-dashboard", "--allow-dirty",
        ],
        timeout=120,
    )
    assert resume.returncode == 0, resume.stderr

    manifest = json.loads(manifest_path.read_text())
    assert manifest["stages"]["machine_profile"]["status"] == "done"
    assert manifest["stages"]["concurrency_sweep"]["status"] == "done"


_IMPORT_ORDER_SCRIPT = """
import builtins, sys

import_order = []
_real_import = builtins.__import__


def _tracing_import(name, *args, **kwargs):
    if name not in import_order:
        import_order.append(name)
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _tracing_import
import evodevo.preflight.__main__ as m
builtins.__import__ = _real_import

assert "numpy" not in sys.modules, (
    "numpy was imported merely by importing evodevo.preflight.__main__: %r" % (import_order[:40],)
)
assert "torch" not in sys.modules
assert "evodevo.parallel.threading_control" in import_order
assert m._THREADING_REPORT.guard_passed is True
print("IMPORT_ORDER_OK")
"""

_IMPORT_TOO_LATE_SCRIPT = """
import numpy  # noqa: F401 -- simulate the exact ordering mistake the module docstring warns about
import evodevo.preflight.__main__
"""


def test_import_order_regression_numpy_never_imported_before_configure_threading(tmp_path):
    """The guard against a future refactor silently breaking the whole
    thing: importing evodevo.preflight.__main__ (with nothing else
    imported first) must never itself pull in numpy, and
    configure_threading's own guard must report it ran cleanly."""
    script_path = tmp_path / "check_import_order.py"
    script_path.write_text(_IMPORT_ORDER_SCRIPT)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_ORDER_OK" in result.stdout


def test_importing_main_after_numpy_already_loaded_raises_too_late_error(tmp_path):
    """The other half of the guard: if numpy IS already loaded (simulating
    exactly the mistake this module's docstring warns about -- something
    importing it in-process instead of via `python -m evodevo.preflight`),
    the import must fail loudly (sys.exit(1) with a clear message), never
    silently proceed with a misconfigured orchestrator process."""
    script_path = tmp_path / "check_too_late.py"
    script_path.write_text(_IMPORT_TOO_LATE_SCRIPT)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode != 0
    assert "load-bearing" in result.stderr


# --- dirty-tree launch refusal --------------------------------------------
#
# evodevo.preflight.__main__'s dirty-tree check always inspects THIS real
# repository (its REPO_ROOT is derived from __file__, not from cwd or any
# --base-dir), so these tests query the repo's OWN current state via
# evodevo.preflight.dirty_check (safe to import directly -- see that
# module's own docstring) and assert the CLI's behaviour is consistent
# with it, rather than assuming a fixed clean/dirty state. This makes them
# correct both before and after this project's own work gets committed.


def test_a_real_launch_is_refused_exactly_when_the_tree_is_currently_dirty(tmp_path):
    base_dir = tmp_path / "preflight"
    currently_dirty = project_is_dirty(REPO_ROOT)

    result = _run(["--only", "machine_profile", "--base-dir", str(base_dir)], timeout=60)

    if currently_dirty:
        assert result.returncode == 1
        assert "refusing to launch" in result.stderr
        assert "offending paths" in result.stderr
        assert not base_dir.exists(), "a refused launch must not create any run directory"
    else:
        assert result.returncode == 0
        assert base_dir.exists()


def test_dry_run_is_never_refused_even_when_the_tree_is_dirty(tmp_path):
    """--dry-run is exempt: it runs no stage and only ever writes a run
    directory skeleton, so it must keep working while iterating on code,
    same as it did throughout this whole project's own uncommitted
    development."""
    base_dir = tmp_path / "preflight"
    result = _run(["--dry-run", "--base-dir", str(base_dir)])
    assert result.returncode == 0


def test_self_check_is_exempt_from_the_dirty_tree_refusal_but_records_it(tmp_path):
    base_dir = tmp_path / "preflight"
    currently_dirty = project_is_dirty(REPO_ROOT)

    result = _run(["--self-check", "--base-dir", str(base_dir), "--only", "machine_profile"], timeout=60)
    assert result.returncode == 0, result.stderr

    run_dirs = list(base_dir.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["project_dirty"] is currently_dirty


def test_allow_dirty_permits_a_real_launch_and_records_project_dirty(tmp_path):
    base_dir = tmp_path / "preflight"
    currently_dirty = project_is_dirty(REPO_ROOT)

    result = _run(["--only", "machine_profile", "--base-dir", str(base_dir), "--allow-dirty"], timeout=60)
    assert result.returncode == 0, result.stderr

    run_dirs = list(base_dir.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["project_dirty"] is currently_dirty

    resolved_config = json.loads((run_dirs[0] / "resolved_config.json").read_text())
    assert resolved_config["project_dirty"] is currently_dirty
    assert resolved_config["allow_dirty"] is True
