"""Environment-aware idle gate for the timed Step-5 conformance run.

The lab machine is operated remotely, so the display/remote-control/editor
stack is part of the idle baseline rather than a competing workload.  That
baseline is bounded here; process names are not an unlimited exemption.
"""
from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import Iterable, Sequence


OPERATOR_PROCESS_NAMES = frozenset({
    "anydesk", "claude", "code", "codex", "gnome-shell", "xorg",
})
SAMPLE_INTERVAL_S = 5.0
SAMPLE_COUNT = 3
SUSTAINED_SAMPLE_COUNT = 2

# psutil's system percentage is normalized over all logical CPUs; process
# percentages use 100% for one logical CPU.  The observed remote baseline is
# about 3.2--3.9% system / 85--95% of one CPU in the allowlisted stack.
MAX_SYSTEM_PERCENT = 12.0
MAX_OPERATOR_CPU_PERCENT = 150.0
MAX_COMPETING_CPU_PERCENT = 25.0


@dataclass(frozen=True)
class ActivitySample:
    system_percent: float
    operator_cpu_percent: float
    competing_cpu_percent: float
    top_operator: tuple[tuple[str, float], ...] = ()
    top_competing: tuple[tuple[str, float], ...] = ()


def classify_process_cpu(rows: Iterable[tuple[str, float]]):
    """Split `(process name, % of one CPU)` rows by the exact allowlist."""
    operator_rows = []
    competing_rows = []
    for name, cpu_percent in rows:
        normalized = (name or "<unknown>").casefold()
        row = (normalized, float(cpu_percent))
        (operator_rows if normalized in OPERATOR_PROCESS_NAMES else competing_rows).append(row)
    operator_rows.sort(key=lambda row: row[1], reverse=True)
    competing_rows.sort(key=lambda row: row[1], reverse=True)
    return (sum(row[1] for row in operator_rows),
            sum(row[1] for row in competing_rows),
            tuple(operator_rows[:8]), tuple(competing_rows[:8]))


def collect_samples(interval_s: float = SAMPLE_INTERVAL_S,
                    count: int = SAMPLE_COUNT) -> list[ActivitySample]:
    """Measure system and per-process CPU over consecutive windows."""
    import psutil  # lazy: campaign entry-point thread limits are already fixed

    tracked = {}

    def prime_new_processes():
        current = {}
        for process in psutil.process_iter(["pid", "name", "create_time"]):
            try:
                identity = (process.pid, process.info["create_time"])
                current[identity] = process
                if identity not in tracked:
                    process.cpu_percent(None)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        tracked.clear()
        tracked.update(current)

    psutil.cpu_percent(interval=None)
    prime_new_processes()
    samples = []
    for _index in range(count):
        time.sleep(interval_s)
        system_percent = float(psutil.cpu_percent(interval=None))
        rows = []
        for process in tuple(tracked.values()):
            try:
                rows.append((process.name(), process.cpu_percent(None)))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        operator, competing, top_operator, top_competing = classify_process_cpu(rows)
        samples.append(ActivitySample(system_percent, operator, competing,
                                      top_operator, top_competing))
        prime_new_processes()
    return samples


def assess_samples(samples: Sequence[ActivitySample]):
    """Apply the sustained-load policy and return `(refusals, evidence)`."""
    if len(samples) < SUSTAINED_SAMPLE_COUNT:
        raise ValueError(f"need at least {SUSTAINED_SAMPLE_COUNT} activity samples")
    checks = (
        ("total system CPU", "system_percent", MAX_SYSTEM_PERCENT,
         "% of total logical-CPU capacity"),
        ("operator baseline CPU", "operator_cpu_percent", MAX_OPERATOR_CPU_PERCENT,
         "% of one logical CPU"),
        ("competing process CPU", "competing_cpu_percent", MAX_COMPETING_CPU_PERCENT,
         "% of one logical CPU"),
    )
    refusals = []
    violations = {}
    for label, field, limit, unit in checks:
        values = [float(getattr(sample, field)) for sample in samples]
        count = sum(value > limit for value in values)
        violations[field] = count
        if count >= SUSTAINED_SAMPLE_COUNT:
            refusals.append(
                f"machine-idle gate: sustained {label} exceeded {limit:.1f} {unit} "
                f"in {count}/{len(samples)} samples (observed "
                + ", ".join(f"{value:.1f}" for value in values) + ")")
    record = {
        "required": True,
        "passed": not refusals,
        "policy": {
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "sample_count": len(samples),
            "sustained_sample_count": SUSTAINED_SAMPLE_COUNT,
            "max_system_percent": MAX_SYSTEM_PERCENT,
            "max_operator_cpu_percent": MAX_OPERATOR_CPU_PERCENT,
            "max_competing_cpu_percent": MAX_COMPETING_CPU_PERCENT,
        },
        "operator_process_names": sorted(OPERATOR_PROCESS_NAMES),
        "samples": [dataclasses.asdict(sample) for sample in samples],
        "violating_sample_counts": violations,
        "logical_cpu_count": os.cpu_count(),
    }
    return refusals, record


def check_machine_idle():
    return assess_samples(collect_samples())
