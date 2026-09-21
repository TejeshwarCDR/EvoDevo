"""Tests for evodevo.parallel.topology.

The fixture-based tests monkeypatch DEFAULT_CPU_SYSFS_ROOT/
DEFAULT_HYBRID_SYSFS_ROOT to a captured i7-14700K sysfs snapshot
(tests/parallel/fixtures/i7_14700k_sysfs/), so they assert the SAME
20-physical/28-logical/8P/12E result regardless of what machine actually
runs the test suite (acceptance criterion: must pass on a non-i7-14700K
machine too).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from evodevo.parallel import topology

FIXTURE_DEVICES_ROOT = Path(__file__).parent / "fixtures" / "i7_14700k_sysfs" / "devices"
FIXTURE_CPU_ROOT = FIXTURE_DEVICES_ROOT / "system" / "cpu"


def _patch_to_fixture(monkeypatch):
    monkeypatch.setattr(topology, "DEFAULT_CPU_SYSFS_ROOT", str(FIXTURE_CPU_ROOT))
    monkeypatch.setattr(topology, "DEFAULT_HYBRID_SYSFS_ROOT", str(FIXTURE_DEVICES_ROOT))


# --- real-machine detection -------------------------------------------------


def test_detect_topology_logical_count_matches_os_cpu_count():
    t = topology.detect_topology()
    assert t.logical_count == os.cpu_count()


def test_allocation_order_is_a_permutation_of_logical_ids():
    t = topology.detect_topology()
    order = topology.allocation_order(t)
    expected = sorted(c.logical_id for c in t.cores)
    assert sorted(order) == expected
    assert len(order) == len(set(order)), "allocation_order contains duplicates"


def test_cpu_slots_no_oversubscription_assigns_distinct_cpus():
    t = topology.detect_topology()
    n = t.logical_count
    plan = topology.cpu_slots(t, n)
    assert plan.oversubscribed is False
    assert len(plan.slots) == n
    assigned = [slot[0] for slot in plan.slots]
    assert len(set(assigned)) == n, "not every worker got a distinct CPU"


def test_cpu_slots_oversubscription_wraps_round_robin():
    t = topology.detect_topology()
    n = t.logical_count + 4
    plan = topology.cpu_slots(t, n)
    assert plan.oversubscribed is True
    assert len(plan.slots) == n
    order = topology.allocation_order(t)
    logical_count = len(order)
    for i, slot in enumerate(plan.slots):
        assert slot[0] == order[i % logical_count]


def test_cpu_slots_rejects_nonpositive_n_workers():
    t = topology.detect_topology()
    with pytest.raises(ValueError):
        topology.cpu_slots(t, 0)


def test_detect_topology_falls_back_gracefully_when_sysfs_missing(monkeypatch, tmp_path):
    missing = str(tmp_path / "does_not_exist")
    monkeypatch.setattr(topology, "DEFAULT_CPU_SYSFS_ROOT", missing)
    monkeypatch.setattr(topology, "DEFAULT_HYBRID_SYSFS_ROOT", missing)

    t = topology.detect_topology()

    assert t.logical_count == os.cpu_count()
    assert t.detection_method == "unknown"
    assert t.hybrid is False


# --- fixture-based (machine-independent) detection --------------------------


def test_i7_14700k_fixture_yields_20_physical_28_logical(monkeypatch):
    _patch_to_fixture(monkeypatch)

    t = topology.detect_topology()

    assert t.logical_count == 28
    assert t.physical_count == 20
    assert t.hybrid is True
    assert t.detection_method == "cpu_core_cpu_atom"


def test_i7_14700k_fixture_yields_8_p_cores_and_12_e_cores(monkeypatch):
    _patch_to_fixture(monkeypatch)

    t = topology.detect_topology()

    p_primaries = [c for c in t.cores if c.kind == "P" and c.is_physical_primary]
    e_primaries = [c for c in t.cores if c.kind == "E" and c.is_physical_primary]
    assert len(p_primaries) == 8, "expected 8 physical P-cores"
    assert len(e_primaries) == 12, "expected 12 physical E-cores"
    # P-cores are hyperthreaded (2 logical ids per physical core), E-cores are not.
    assert len(t.p_core_ids) == 16
    assert len(t.e_core_ids) == 12


def test_i7_14700k_fixture_allocation_order_lands_on_8_20_28(monkeypatch):
    _patch_to_fixture(monkeypatch)

    t = topology.detect_topology()
    order = topology.allocation_order(t)
    p_primary_ids = {c.logical_id for c in t.cores if c.kind == "P" and c.is_physical_primary}

    assert len(order) == 28
    first_8 = order[:8]
    first_20 = order[:20]
    assert set(first_8) == p_primary_ids, "first 8 slots should be exactly the 8 P-core primaries"
    assert set(first_20) == p_primary_ids | set(t.e_core_ids), (
        "first 20 slots should be every physical core (P primaries + all E-cores)"
    )


def test_i7_14700k_fixture_topology_to_dict_round_trips_key_fields(monkeypatch):
    _patch_to_fixture(monkeypatch)

    t = topology.detect_topology()
    d = t.to_dict()

    assert d["logical_count"] == 28
    assert d["physical_count"] == 20
    assert len(d["cores"]) == 28
    assert d["hybrid"] is True


def test_cpu_slots_to_dict(monkeypatch):
    _patch_to_fixture(monkeypatch)

    t = topology.detect_topology()
    plan = topology.cpu_slots(t, 32)
    d = plan.to_dict()

    assert d["n_workers"] == 32
    assert d["oversubscribed"] is True
    assert len(d["slots"]) == 32
