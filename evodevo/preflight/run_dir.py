"""The pre-flight run directory: layout, creation, resume, and atomic
writes.

D8 (crash safety): pre-flight runs unattended, in tmux, for roughly half a
day. It WILL be interrupted -- a killed tmux pane, a machine reboot, an
OOM-killed process. Every JSON artifact this package writes must therefore
never be observed half-written: every write goes to `<path>.tmp` first,
then `os.replace(tmp, path)`, which POSIX guarantees is atomic (the
destination path always shows either the old complete content or the new
complete content, never a partial write). A crash between the two events
either lands before the write (old content, or no file) or after it (new
content) -- there is no in-between state a reader can observe.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

RUN_DIR_CONTENTS = (
    "manifest.json",
    "machine.json",
    "stages",
    "events.jsonl",
    "telemetry.jsonl",
    "logs",
    "report",
)


class RunDirNotFoundError(FileNotFoundError):
    """Raised when --run-id names a directory that does not exist -- resume
    must never silently fall back to creating a new run under an old name."""


def atomic_write_json(path: str, data: Any) -> None:
    """Writes `data` as JSON to `path`, never leaving a partially-written
    file at `path` itself (see module docstring). Leaves no `.tmp` file
    behind on success; a crash mid-write leaves the `.tmp` file as debris
    but `path` itself is untouched."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class RunDir:
    """A resolved pre-flight run directory and its fixed sub-paths. Every
    path here is a plain string join -- this class does not itself decide
    whether the directory exists; see create_or_resume_run_dir()."""

    path: str

    @property
    def run_id(self) -> str:
        return os.path.basename(os.path.normpath(self.path))

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.path, "manifest.json")

    @property
    def machine_path(self) -> str:
        return os.path.join(self.path, "machine.json")

    @property
    def stages_dir(self) -> str:
        return os.path.join(self.path, "stages")

    @property
    def events_path(self) -> str:
        return os.path.join(self.path, "events.jsonl")

    @property
    def telemetry_path(self) -> str:
        return os.path.join(self.path, "telemetry.jsonl")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.path, "logs")

    @property
    def report_dir(self) -> str:
        return os.path.join(self.path, "report")

    def stage_output_path(self, stage_name: str) -> str:
        return os.path.join(self.stages_dir, f"{stage_name}.json")

    def stage_log_path(self, stage_name: str) -> str:
        return os.path.join(self.logs_dir, f"{stage_name}.log")

    def ensure_skeleton(self) -> None:
        """Creates every directory this run needs. Idempotent -- safe to
        call again on resume."""
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.stages_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        # events.jsonl/telemetry.jsonl are append-only logs, not JSON
        # documents -- touch them into existence so every expected path in
        # RUN_DIR_CONTENTS is present immediately, even before the first
        # event/sample is written.
        for path in (self.events_path, self.telemetry_path):
            if not os.path.exists(path):
                open(path, "a").close()


def write_machine_snapshot(run_dir: RunDir) -> None:
    """Writes machine.json ONCE: a static snapshot of what this machine
    looked like when this pre-flight run started (hostname, CPU topology,
    Python/library versions, git commit). This is provenance, not a
    measurement -- no B1-B7 timing/throughput number belongs here."""
    import platform
    import sys

    from evodevo.parallel.topology import detect_topology

    snapshot: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "os_cpu_count": os.cpu_count(),
        "git_commit": git_short_hash(),
        "topology": detect_topology().to_dict(),
    }
    try:
        import psutil

        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        snapshot["ram_total_mb"] = round(vmem.total / (1024 * 1024), 1)
        snapshot["swap_total_mb"] = round(swap.total / (1024 * 1024), 1)
    except Exception:
        snapshot["ram_total_mb"] = None
        snapshot["swap_total_mb"] = None

    atomic_write_json(run_dir.machine_path, snapshot)


def create_or_resume_run_dir(base_dir: str = "results/preflight", run_id: Optional[str] = None) -> RunDir:
    """The single entry point for getting a RunDir to work with.

    run_id=None (the default): creates a NEW run directory named
    `<UTC timestamp>_<git short hash>`, writes machine.json, and
    initializes an empty manifest.json.

    run_id=<name>: resumes an existing run directory -- it MUST already
    exist (RunDirNotFoundError otherwise; resume must never silently fall
    back to creating a new run under a name that doesn't exist, which would
    look like a resume but actually discard the expectation that prior
    stage results are still there). machine.json and manifest.json are
    left exactly as they are; ensure_skeleton() only fills in any
    directories/log files that might be missing.
    """
    if run_id is not None:
        run_dir = RunDir(os.path.join(base_dir, run_id))
        if not os.path.isdir(run_dir.path):
            raise RunDirNotFoundError(
                f"--run-id {run_id!r} does not exist under {base_dir!r} -- resume requires an "
                f"existing run directory; omit --run-id to start a new run."
            )
        run_dir.ensure_skeleton()
        return run_dir

    os.makedirs(base_dir, exist_ok=True)
    new_id = f"{_utc_timestamp()}_{git_short_hash()}"
    run_dir = RunDir(os.path.join(base_dir, new_id))
    if os.path.exists(run_dir.path):
        # Two invocations in the same second, same commit -- vanishingly
        # unlikely but must not silently overwrite a real run.
        raise FileExistsError(f"run directory {run_dir.path!r} already exists")
    run_dir.ensure_skeleton()
    write_machine_snapshot(run_dir)
    atomic_write_json(run_dir.manifest_path, {"run_id": new_id, "stages": {}})
    return run_dir
