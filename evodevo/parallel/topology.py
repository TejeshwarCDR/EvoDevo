"""CPU topology detection for deliberate worker placement.

Reads Linux sysfs to build a picture of this machine's physical cores,
hyperthread siblings, and (on Intel hybrid parts) the P-core/E-core split --
the information `evodevo.parallel.affinity` needs to pin worker processes to
specific CPUs instead of letting the scheduler bounce them around.

Never imports numpy or torch (see evodevo/parallel/threading_control.py for
why that matters); this module only reads small text files under /sys.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

# Real sysfs roots. Exposed as module attributes (not bound into function
# default-argument values) specifically so tests can monkeypatch them --
# `monkeypatch.setattr(topology, "DEFAULT_CPU_SYSFS_ROOT", fixture_path)`
# then `detect_topology()` with no arguments -- and have detect_topology()
# pick up the patched value, which a bound default argument would not.
DEFAULT_CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
DEFAULT_HYBRID_SYSFS_ROOT = "/sys/devices"

_CPU_DIR_RE = re.compile(r"^cpu(\d+)$")


@dataclass(frozen=True)
class CoreInfo:
    logical_id: int
    core_id: int
    package_id: int
    is_physical_primary: bool
    sibling_ids: tuple[int, ...]
    kind: str  # "P" | "E" | "unknown"
    max_freq_khz: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "core_id": self.core_id,
            "package_id": self.package_id,
            "is_physical_primary": self.is_physical_primary,
            "sibling_ids": list(self.sibling_ids),
            "kind": self.kind,
            "max_freq_khz": self.max_freq_khz,
        }


@dataclass(frozen=True)
class Topology:
    cores: tuple[CoreInfo, ...]
    logical_count: int
    physical_count: int
    p_core_ids: tuple[int, ...]
    e_core_ids: tuple[int, ...]
    sibling_map: dict[int, tuple[int, ...]]
    hybrid: bool
    detection_method: str  # "cpu_core_cpu_atom" | "freq_cluster" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cores": [c.to_dict() for c in self.cores],
            "logical_count": self.logical_count,
            "physical_count": self.physical_count,
            "p_core_ids": list(self.p_core_ids),
            "e_core_ids": list(self.e_core_ids),
            "sibling_map": {str(k): list(v) for k, v in sorted(self.sibling_map.items())},
            "hybrid": self.hybrid,
            "detection_method": self.detection_method,
        }


@dataclass(frozen=True)
class AffinityPlan:
    n_workers: int
    slots: tuple[tuple[int, ...], ...]
    oversubscribed: bool
    allocation_order: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_workers": self.n_workers,
            "slots": [list(s) for s in self.slots],
            "oversubscribed": self.oversubscribed,
            "allocation_order": list(self.allocation_order),
        }


def _parse_cpu_list(text: str) -> list[int]:
    """Parses a Linux sysfs cpulist string, e.g. "0-1", "16", "0-1,4,8-9",
    into a sorted list of distinct ints. Returns [] for blank input."""
    text = text.strip()
    if not text:
        return []
    ids: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(chunk))
    return sorted(ids)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> Optional[int]:
    text = _read_text(path)
    return int(text) if text is not None and text != "" else None


def _discover_logical_cpus(cpu_sysfs_root: str) -> list[str]:
    try:
        names = os.listdir(cpu_sysfs_root)
    except OSError:
        return []
    found = []
    for name in names:
        match = _CPU_DIR_RE.match(name)
        if match and os.path.isdir(os.path.join(cpu_sysfs_root, name, "topology")):
            found.append(name)
    return sorted(found, key=lambda n: int(_CPU_DIR_RE.match(n).group(1)))


def _collect_raw_cores(cpu_sysfs_root: str) -> dict[int, dict[str, Any]]:
    raw: dict[int, dict[str, Any]] = {}
    for cpu_name in _discover_logical_cpus(cpu_sysfs_root):
        logical_id = int(_CPU_DIR_RE.match(cpu_name).group(1))
        cpu_dir = os.path.join(cpu_sysfs_root, cpu_name)
        topo_dir = os.path.join(cpu_dir, "topology")
        core_id = _read_int(os.path.join(topo_dir, "core_id"))
        package_id = _read_int(os.path.join(topo_dir, "physical_package_id"))
        siblings_text = _read_text(os.path.join(topo_dir, "thread_siblings_list"))
        sibling_ids = tuple(_parse_cpu_list(siblings_text)) if siblings_text else (logical_id,)
        if logical_id not in sibling_ids:
            sibling_ids = tuple(sorted(set(sibling_ids) | {logical_id}))
        max_freq_khz = _read_int(os.path.join(cpu_dir, "cpufreq", "cpuinfo_max_freq"))
        raw[logical_id] = {
            "core_id": core_id if core_id is not None else logical_id,
            "package_id": package_id if package_id is not None else 0,
            "sibling_ids": sibling_ids,
            "max_freq_khz": max_freq_khz,
        }
    return raw


def _detect_hybrid_kinds(
    hybrid_sysfs_root: str, logical_ids: list[int]
) -> tuple[dict[int, str], str, bool]:
    """Returns (kind_by_cpu, detection_method, hybrid). Tries the explicit
    Intel cpu_core/cpu_atom sysfs classification first, then falls back to
    clustering on max_freq_khz, then gives up with everything "unknown"."""
    core_cpus_text = _read_text(os.path.join(hybrid_sysfs_root, "cpu_core", "cpus"))
    atom_cpus_text = _read_text(os.path.join(hybrid_sysfs_root, "cpu_atom", "cpus"))
    if core_cpus_text is not None and atom_cpus_text is not None:
        p_ids = set(_parse_cpu_list(core_cpus_text))
        e_ids = set(_parse_cpu_list(atom_cpus_text))
        if p_ids or e_ids:
            kind_by_cpu = {cid: ("P" if cid in p_ids else "E" if cid in e_ids else "unknown") for cid in logical_ids}
            return kind_by_cpu, "cpu_core_cpu_atom", bool(p_ids and e_ids)
    return {}, "", False


def _detect_kinds_by_freq(raw: dict[int, dict[str, Any]]) -> tuple[dict[int, str], str, bool]:
    freqs = {cid: entry["max_freq_khz"] for cid, entry in raw.items() if entry["max_freq_khz"] is not None}
    distinct = sorted(set(freqs.values()))
    if len(distinct) < 2:
        return {cid: "unknown" for cid in raw}, "unknown", False

    # Largest gap between consecutive distinct frequencies splits the two
    # clusters -- robust to per-core turbo-bin variance within a cluster
    # (e.g. this machine's P-cores range 5500-5600MHz, not a single value).
    gaps = [(distinct[i + 1] - distinct[i], distinct[i], distinct[i + 1]) for i in range(len(distinct) - 1)]
    _, low, high = max(gaps, key=lambda g: g[0])
    threshold = (low + high) / 2

    kind_by_cpu: dict[int, str] = {}
    for cid, entry in raw.items():
        freq = entry["max_freq_khz"]
        if freq is None:
            kind_by_cpu[cid] = "unknown"
        else:
            kind_by_cpu[cid] = "P" if freq >= threshold else "E"
    return kind_by_cpu, "freq_cluster", True


def _fallback_topology(detection_method: str = "unknown") -> Topology:
    """Used when sysfs is unavailable at all (e.g. non-Linux). Best-effort:
    every logical CPU reported by os.cpu_count() is its own physical
    primary of unknown kind -- never raises."""
    logical_count = os.cpu_count() or 1
    cores = tuple(
        CoreInfo(
            logical_id=i,
            core_id=i,
            package_id=0,
            is_physical_primary=True,
            sibling_ids=(i,),
            kind="unknown",
            max_freq_khz=None,
        )
        for i in range(logical_count)
    )
    return Topology(
        cores=cores,
        logical_count=logical_count,
        physical_count=logical_count,
        p_core_ids=(),
        e_core_ids=(),
        sibling_map={c.logical_id: c.sibling_ids for c in cores},
        hybrid=False,
        detection_method=detection_method,
    )


def detect_topology(
    cpu_sysfs_root: Optional[str] = None,
    hybrid_sysfs_root: Optional[str] = None,
) -> Topology:
    """Detects this machine's CPU topology from Linux sysfs.

    Reads /sys/devices/system/cpu/cpu*/topology/{core_id,
    physical_package_id, thread_siblings_list} for the physical-core /
    hyperthread-sibling structure. For the P-core/E-core split on Intel
    hybrid parts, tries /sys/devices/cpu_core/cpus and
    /sys/devices/cpu_atom/cpus first (the kernel's own classification);
    if those aren't present, falls back to clustering logical CPUs by
    cpufreq/cpuinfo_max_freq (higher cluster = P); if that also can't
    produce two clusters (e.g. a non-hybrid machine), every core is
    reported with kind="unknown" rather than guessing. Never raises --
    a machine with no sysfs at all (e.g. non-Linux) gets a best-effort
    Topology built from os.cpu_count() alone.
    """
    root = cpu_sysfs_root if cpu_sysfs_root is not None else DEFAULT_CPU_SYSFS_ROOT
    hybrid_root = hybrid_sysfs_root if hybrid_sysfs_root is not None else DEFAULT_HYBRID_SYSFS_ROOT

    raw = _collect_raw_cores(root)
    if not raw:
        return _fallback_topology()

    logical_ids = sorted(raw.keys())
    kind_by_cpu, detection_method, hybrid = _detect_hybrid_kinds(hybrid_root, logical_ids)
    if not kind_by_cpu:
        kind_by_cpu, detection_method, hybrid = _detect_kinds_by_freq(raw)

    cores = []
    sibling_map: dict[int, tuple[int, ...]] = {}
    for cid in logical_ids:
        entry = raw[cid]
        sibling_ids = entry["sibling_ids"]
        sibling_map[cid] = sibling_ids
        cores.append(
            CoreInfo(
                logical_id=cid,
                core_id=entry["core_id"],
                package_id=entry["package_id"],
                is_physical_primary=(cid == min(sibling_ids)),
                sibling_ids=sibling_ids,
                kind=kind_by_cpu.get(cid, "unknown"),
                max_freq_khz=entry["max_freq_khz"],
            )
        )

    physical_count = sum(1 for c in cores if c.is_physical_primary)
    p_core_ids = tuple(sorted(c.logical_id for c in cores if c.kind == "P"))
    e_core_ids = tuple(sorted(c.logical_id for c in cores if c.kind == "E"))

    return Topology(
        cores=tuple(cores),
        logical_count=len(cores),
        physical_count=physical_count,
        p_core_ids=p_core_ids,
        e_core_ids=e_core_ids,
        sibling_map=sibling_map,
        hybrid=hybrid,
        detection_method=detection_method,
    )


def allocation_order(topology: Topology) -> list[int]:
    """The ordered list of logical CPU ids to hand to workers, one at a
    time, as worker count grows.

    Order: P-core primaries, then E-cores, then P-core hyperthread
    siblings, then anything unclassified.

    WHY this order matters: on this project's target machine (a hybrid
    8 P-core / 12 E-core / 28-logical-thread part), it makes the worker
    counts used in the B2 concurrency sweep land on physically meaningful
    boundaries instead of arbitrary points on a curve --

      8 workers  -> P-core primaries only (every worker owns a full,
                    non-hyperthreaded core, the fastest per-worker config)
      20 workers -> every physical core, P and E alike, still one worker
                    per physical core (no hyperthread sharing yet)
      28 workers -> every physical core PLUS every P-core's hyperthread
                    sibling (the point where two workers start sharing one
                    physical core's execution resources)

    So a sweep across worker counts 1..28 has a mechanical explanation for
    any knee in its throughput curve at 8 or 20 -- "that's where the P-cores
    ran out" / "that's where hyperthreading started" -- rather than just an
    unexplained shape a reader has to take on faith.
    """
    p_primary = sorted(c.logical_id for c in topology.cores if c.kind == "P" and c.is_physical_primary)
    e_all = sorted(c.logical_id for c in topology.cores if c.kind == "E")
    p_sibling = sorted(c.logical_id for c in topology.cores if c.kind == "P" and not c.is_physical_primary)
    unclassified = sorted(c.logical_id for c in topology.cores if c.kind == "unknown")
    return p_primary + e_all + p_sibling + unclassified


def cpu_slots(topology: Topology, n_workers: int) -> AffinityPlan:
    """Builds the per-worker CPU affinity assignment for n_workers workers,
    taken in order from allocation_order(topology).

    If n_workers exceeds the number of logical CPUs, this does NOT raise --
    it wraps round-robin through allocation_order and sets
    AffinityPlan.oversubscribed=True, because the B2 sweep must be able to
    deliberately probe worker counts past the core count (that's the whole
    point of a saturation sweep) rather than being blocked from doing so.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    order = allocation_order(topology)
    logical_count = len(order)
    oversubscribed = n_workers > logical_count
    slots = tuple((order[i % logical_count],) for i in range(n_workers))
    return AffinityPlan(
        n_workers=n_workers,
        slots=slots,
        oversubscribed=oversubscribed,
        allocation_order=tuple(order),
    )
