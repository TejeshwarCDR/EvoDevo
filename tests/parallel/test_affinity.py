"""Tests for evodevo.parallel.affinity.

The pin/restore test is skipped entirely on platforms without
os.sched_setaffinity (anything non-Linux) -- affinity is a performance
optimization there, never a correctness requirement, and this module must
never crash a run for lacking it (see affinity.py's module docstring).
"""
from __future__ import annotations

import os

import pytest

from evodevo.parallel.affinity import AffinityResult, current_affinity, pin_current_process

_HAS_SCHED_AFFINITY = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")


@pytest.mark.skipif(not _HAS_SCHED_AFFINITY, reason="platform lacks os.sched_setaffinity/sched_getaffinity")
def test_pin_to_single_cpu_and_restore():
    original = current_affinity()
    assert original, "expected at least one CPU in the original affinity set"
    target_cpu = original[0]

    try:
        result = pin_current_process([target_cpu])
        assert isinstance(result, AffinityResult)
        assert result.applied is True
        assert result.reason is None
        assert current_affinity() == [target_cpu]
    finally:
        pin_current_process(original)

    assert current_affinity() == original


@pytest.mark.skipif(not _HAS_SCHED_AFFINITY, reason="platform lacks os.sched_setaffinity/sched_getaffinity")
def test_affinity_result_to_dict():
    original = current_affinity()
    result = pin_current_process(original)
    d = result.to_dict()
    try:
        assert d["applied"] is True
        assert d["cpu_ids"] == list(original)
        assert d["reason"] is None
    finally:
        pin_current_process(original)


def test_pin_current_process_never_raises_on_unsupported_platform(monkeypatch):
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)

    with pytest.warns(UserWarning):
        result = pin_current_process([0])

    assert result.applied is False
    assert result.reason == "unsupported_platform"


def test_current_affinity_returns_empty_list_on_unsupported_platform(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)

    assert current_affinity() == []
