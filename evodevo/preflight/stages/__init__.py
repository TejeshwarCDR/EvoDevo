"""The stage protocol, registry, manifest, and orchestration engine.

A pre-flight run is a DAG of named stages, each independently resumable.
The manifest (run_dir/manifest.json) is the single source of truth for
"what has already happened" -- every status transition (pending -> running
-> done/failed) is written to it immediately, atomically (see run_dir.py),
before this process does anything else. That is what makes resuming after
a kill -9 safe: on restart, a stage the manifest still shows as "running"
can only mean the process that was running it is gone, so it is treated
exactly like "failed" and re-run -- never silently skipped, and never
assumed to have secretly finished.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from evodevo.preflight.run_dir import RunDir, atomic_write_json, git_short_hash, read_json
from evodevo.preflight.run_dir import RunDirNotFoundError  # noqa: F401 -- re-exported for convenience

VALID_STATUSES = ("pending", "running", "done", "failed", "skipped")
VALID_COSTS = ("free", "cheap", "expensive")

# The exact stamp text required, verbatim, on every artifact (JSON key and
# figure title) produced by --self-check. Shared so every stage and
# report/readiness consumer stamps and recognises the identical string.
SELF_CHECK_STAMP = "SELF-CHECK — NOT A MEASUREMENT"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@runtime_checkable
class Stage(Protocol):
    name: str
    description: str
    estimated_cost: str  # "free" | "cheap" | "expensive"
    depends_on: tuple

    def run(self, ctx: Any) -> dict:
        ...


class DuplicateStageError(ValueError):
    """Raised when two stages register under the same name."""


class UnknownStageError(KeyError):
    """Raised for a stage name that was never registered -- covers a typo
    in --only/--skip/--force, and a depends_on referencing a stage that
    doesn't exist."""


class DependencyCycleError(ValueError):
    """Raised when depends_on edges form a cycle -- no valid run order exists."""


class DependencyNotDoneError(RuntimeError):
    """Raised when a stage is about to run but one of its declared
    dependencies is not (yet) status="done" in the manifest."""


_REQUIRED_STAGE_ATTRS = ("name", "description", "estimated_cost", "depends_on", "run")


class StageRegistry:
    """Explicit registration only -- never filesystem/module scanning, so a
    broken stage import fails loudly at registration time, not silently at
    dry-run time by simply not appearing."""

    def __init__(self):
        self._stages: dict = {}

    def register(self, stage: Any) -> None:
        for attr in _REQUIRED_STAGE_ATTRS:
            if not hasattr(stage, attr):
                raise TypeError(f"stage {stage!r} is missing required attribute/method {attr!r}")
        if stage.estimated_cost not in VALID_COSTS:
            raise ValueError(
                f"stage '{stage.name}': estimated_cost must be one of {VALID_COSTS}, "
                f"got {stage.estimated_cost!r}"
            )
        if stage.name in self._stages:
            raise DuplicateStageError(f"stage '{stage.name}' is already registered")
        self._stages[stage.name] = stage

    def get(self, name: str) -> Any:
        try:
            return self._stages[name]
        except KeyError:
            raise UnknownStageError(f"unknown stage '{name}'; available: {sorted(self._stages)}")

    def all_names(self) -> list:
        return list(self._stages)

    def ordered(self) -> list:
        """A dependency-respecting order: every stage appears after all of
        its depends_on. Deterministic for a fixed registration order (DFS
        post-order), so two runs of the same registry always print/execute
        stages in the same order."""
        for name, stage in self._stages.items():
            for dep in stage.depends_on:
                if dep not in self._stages:
                    raise UnknownStageError(f"stage '{name}' depends on unregistered stage '{dep}'")

        order: list = []
        visited: set = set()
        in_progress: set = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in in_progress:
                raise DependencyCycleError(f"dependency cycle detected involving stage '{name}'")
            in_progress.add(name)
            for dep in self._stages[name].depends_on:
                visit(dep)
            in_progress.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._stages:
            visit(name)
        return order


def _default_stage_entry() -> dict:
    return {
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "duration_s": None,
        "output_path": None,
        "error": None,
        "git_commit": None,
    }


def load_manifest(run_dir: RunDir) -> dict:
    try:
        return read_json(run_dir.manifest_path)
    except FileNotFoundError:
        return {"run_id": run_dir.run_id, "stages": {}}


def save_manifest(run_dir: RunDir, manifest: dict) -> None:
    atomic_write_json(run_dir.manifest_path, manifest)


def _set_stage_status(run_dir: RunDir, manifest: dict, stage_name: str, **updates) -> dict:
    entry = manifest["stages"].setdefault(stage_name, _default_stage_entry())
    entry.update(updates)
    save_manifest(run_dir, manifest)
    return entry


def _stage_status(manifest: dict, name: str) -> str:
    entry = manifest["stages"].get(name)
    return entry["status"] if entry else "pending"


def resolve_plan(
    registry: StageRegistry,
    manifest: dict,
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
    force: Optional[Iterable[str]] = None,
) -> list:
    """The ordered list of stage names this invocation will actually run.

    Resume semantics: a stage already "done" is skipped UNLESS named in
    `force`. "failed" and "running" stages are always re-run when selected
    (a "running" status with no live process behind it means the process
    that was running it died -- see module docstring). --only restricts to
    exactly the named stages (still in dependency order); --skip excludes
    the named stages from an otherwise-full run. Every name in only/skip/
    force is validated against the registry up front -- a typo must raise,
    never be silently ignored.
    """
    only = list(only or [])
    skip = list(skip or [])
    force = set(force or [])

    for name in list(only) + list(skip) + list(force):
        registry.get(name)  # raises UnknownStageError if not registered

    order = registry.ordered()
    if only:
        only_set = set(only)
        candidates = [n for n in order if n in only_set]
    else:
        skip_set = set(skip)
        candidates = [n for n in order if n not in skip_set]

    plan = []
    for name in candidates:
        status = _stage_status(manifest, name)
        if status == "done" and name not in force:
            continue
        plan.append(name)
    return plan


def run_stages(
    ctx: Any,
    registry: StageRegistry,
    only: Optional[Iterable[str]] = None,
    skip: Optional[Iterable[str]] = None,
    force: Optional[Iterable[str]] = None,
) -> dict:
    """Runs every stage in `resolve_plan(...)`'s order, updating
    manifest.json and events.jsonl as it goes. Stops (re-raising) at the
    first failure -- a stopped run is recovered by re-invoking with the
    same --run-id, which resume semantics will pick back up correctly, not
    by continuing past a failure in the same process."""
    manifest = load_manifest(ctx.run_dir)
    plan = resolve_plan(registry, manifest, only=only, skip=skip, force=force)

    for name in plan:
        stage = registry.get(name)

        for dep in stage.depends_on:
            dep_status = _stage_status(manifest, dep)
            if dep_status != "done":
                message = (
                    f"stage '{name}' requires dependency '{dep}' to be done, but it is "
                    f"'{dep_status}' -- run '{dep}' first (or include it in this invocation)."
                )
                _set_stage_status(
                    ctx.run_dir, manifest, name,
                    status="failed", end_time=_utc_now_iso(), duration_s=0.0, error=message,
                )
                ctx.events.record(name, "failure", {"error": message})
                raise DependencyNotDoneError(message)

        start = time.time()
        _set_stage_status(
            ctx.run_dir, manifest, name,
            status="running", start_time=_utc_now_iso(), end_time=None,
            duration_s=None, error=None, git_commit=git_short_hash(),
        )
        ctx.events.record(name, "stage_start", {})

        try:
            result = stage.run(ctx)
        except Exception as e:
            duration = time.time() - start
            error_message = f"{type(e).__name__}: {e}"
            _set_stage_status(
                ctx.run_dir, manifest, name,
                status="failed", end_time=_utc_now_iso(), duration_s=duration, error=error_message,
            )
            ctx.events.record(name, "failure", {"error": error_message})
            raise

        duration = time.time() - start
        output_path = ctx.run_dir.stage_output_path(name)
        atomic_write_json(output_path, result)
        _set_stage_status(
            ctx.run_dir, manifest, name,
            status="done", end_time=_utc_now_iso(), duration_s=duration,
            output_path=output_path, error=None,
        )
        ctx.events.record(name, "stage_end", {"duration_s": duration})

    return manifest
