"""Pins the current process to specific logical CPUs.

Affinity is a performance optimization, never a run-correctness
requirement: a platform without os.sched_setaffinity (anything non-Linux)
must degrade to "the OS scheduler decides", not crash a campaign run. Every
function here is designed to never raise for that reason.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AffinityResult:
    applied: bool
    cpu_ids: tuple[int, ...]
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "cpu_ids": list(self.cpu_ids),
            "reason": self.reason,
        }


def pin_current_process(cpu_ids: list[int]) -> AffinityResult:
    """Restricts the current process to the given logical CPU ids via
    os.sched_setaffinity. Never raises: on a platform without that call
    (macOS, Windows), or if the OS itself rejects the requested set (e.g.
    an id outside the machine's actual CPU set), this returns
    applied=False with a reason and logs a warning, rather than crashing
    the worker it was meant to speed up."""
    cpu_ids = tuple(cpu_ids)

    if not hasattr(os, "sched_setaffinity"):
        warnings.warn(
            f"pin_current_process({list(cpu_ids)}): os.sched_setaffinity is not available on this "
            f"platform -- continuing without CPU pinning (the OS scheduler will place this process)."
        )
        return AffinityResult(applied=False, cpu_ids=cpu_ids, reason="unsupported_platform")

    try:
        os.sched_setaffinity(0, cpu_ids)
    except OSError as e:
        warnings.warn(
            f"pin_current_process({list(cpu_ids)}) failed ({e}) -- continuing without CPU pinning."
        )
        return AffinityResult(applied=False, cpu_ids=cpu_ids, reason=f"os_error:{e}")

    return AffinityResult(applied=True, cpu_ids=cpu_ids, reason=None)


def current_affinity() -> list[int]:
    """The current process's CPU affinity set, sorted. Returns [] on a
    platform without os.sched_getaffinity rather than raising."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    return sorted(os.sched_getaffinity(0))
