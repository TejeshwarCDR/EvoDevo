"""Ordered execution units of a campaign plan, read from the plan's dict form.

A legacy plan (the original campaign and the 3-task extension) is one
evolutionary phase per task, keyed by the task name. An Amendment 2 plan has
phases keyed by ``phase_id`` ("ObstacleTraverser-v1:s0-4"), plus re-training
stages ("reeval:..." and "robustness:...") interleaved in ``amendment2.sequence``.
Every reader of phase state (scheduler, dashboard, monitor, milestone, alerts)
goes through this module, so the key of a unit is defined in exactly one place.

Stdlib only: imported by modules that must stay numpy/torch free.
"""
from __future__ import annotations

PHASE, REEVAL, ROBUSTNESS = "phase", "reeval", "robustness"
STAGE_KINDS = (REEVAL, ROBUSTNESS)


def phase_key(phase: dict) -> str:
    """Status/metrics key of an evolutionary phase (its task for a legacy plan)."""
    return phase.get("phase_id") or phase["task"]


def slug(key: str) -> str:
    """A key as a file name. Colons are legal on ext4 but not on the NTFS backup disk."""
    return key.replace(":", "__")


def plan_units(plan: dict) -> list:
    """[{"key", "kind", "task", "phase"|"stage"}] in execution order."""
    phases = {phase_key(p): p for p in plan.get("phases", [])}
    stages = {s["stage_id"]: s for s in plan.get("stages", [])}
    sequence = (plan.get("amendment2") or {}).get("sequence")
    if not sequence:
        return [{"key": key, "kind": PHASE, "task": p["task"], "phase": p} for key, p in phases.items()]
    units = []
    for key in sequence:
        if key in phases:
            units.append({"key": key, "kind": PHASE, "task": phases[key]["task"], "phase": phases[key]})
        elif key in stages:
            units.append({"key": key, "kind": stages[key]["kind"], "task": stages[key]["task"],
                          "stage": stages[key]})
        else:
            raise ValueError(f"plan sequence names unknown unit {key!r}")
    return units


def unit_by_key(plan: dict, key: str):
    return next((u for u in plan_units(plan) if u["key"] == key), None)


def task_of_key(plan: dict, key: str) -> str:
    """The task a status/alert key belongs to (a plain task name is its own key)."""
    unit = unit_by_key(plan, key)
    return unit["task"] if unit else key


def is_stage(unit: dict) -> bool:
    return unit["kind"] in STAGE_KINDS
