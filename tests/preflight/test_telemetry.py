from __future__ import annotations

import json
import math
import time

import pytest

from evodevo.preflight.telemetry import TelemetrySampler
from evodevo.preflight.telemetry import percentile as _percentile
from evodevo.preflight.telemetry import stats as _stats


def test_percentile_and_stats_helpers():
    assert _percentile([], 95) is None
    assert _percentile([10], 95) == 10
    assert _percentile([1, 2, 3, 4, 5], 50) == 3

    assert _stats([]) == {"mean": None, "max": None, "p95": None}
    result = _stats([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["max"] == 3.0


def test_sampler_starts_writes_samples_and_stops_cleanly(tmp_path):
    path = str(tmp_path / "telemetry.jsonl")
    sampler = TelemetrySampler(path, interval_s=0.05, disk_path=str(tmp_path))

    sampler.start()
    deadline = time.time() + 5
    while sampler.sample_count() < 3 and time.time() < deadline:
        time.sleep(0.02)
    sampler.stop(timeout=5)

    assert sampler.sample_count() >= 3

    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 3
    for sample in lines:
        assert "ts" in sample
        assert "ts_epoch" in sample


def test_sampler_can_be_used_as_a_context_manager(tmp_path):
    path = str(tmp_path / "telemetry.jsonl")
    with TelemetrySampler(path, interval_s=0.05, disk_path=str(tmp_path)) as sampler:
        deadline = time.time() + 5
        while sampler.sample_count() < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert sampler.sample_count() >= 3


def test_summarise_returns_finite_values_over_the_full_run(tmp_path):
    path = str(tmp_path / "telemetry.jsonl")
    sampler = TelemetrySampler(path, interval_s=0.05, disk_path=str(tmp_path))
    sampler.start()
    deadline = time.time() + 5
    while sampler.sample_count() < 5 and time.time() < deadline:
        time.sleep(0.02)
    sampler.stop(timeout=5)

    summary = sampler.summarise()
    assert summary["n_samples"] >= 5
    # ram_used_mb is always obtainable via psutil on every platform this
    # project runs on -- a good canary that stats propagate correctly.
    ram_stats = summary["ram_used_mb"]
    assert ram_stats["mean"] is not None
    assert math.isfinite(ram_stats["mean"])
    assert math.isfinite(ram_stats["max"])
    assert math.isfinite(ram_stats["p95"])


def test_summarise_window_filters_by_time_range(tmp_path):
    path = str(tmp_path / "telemetry.jsonl")
    sampler = TelemetrySampler(path, interval_s=0.05, disk_path=str(tmp_path))
    sampler.start()
    deadline = time.time() + 5
    while sampler.sample_count() < 6 and time.time() < deadline:
        time.sleep(0.02)
    sampler.stop(timeout=5)

    full = sampler.summarise()
    # A window entirely before any sample was taken -- must return zero
    # samples, not raise or silently include everything.
    empty_window = sampler.summarise(window=(0.0, 1.0))

    assert full["n_samples"] >= 6
    assert empty_window["n_samples"] == 0
    assert empty_window["ram_used_mb"] == {"mean": None, "max": None, "p95": None}


def test_sampler_degrades_gracefully_when_sensors_raise(tmp_path, monkeypatch):
    import psutil

    def _raise(*args, **kwargs):
        raise OSError("no sensors on this box")

    monkeypatch.setattr(psutil, "sensors_temperatures", _raise)

    path = str(tmp_path / "telemetry.jsonl")
    sampler = TelemetrySampler(path, interval_s=0.05, disk_path=str(tmp_path))

    with pytest.warns(UserWarning, match="cpu_package_temp_c"):
        sampler.start()
        deadline = time.time() + 5
        while sampler.sample_count() < 3 and time.time() < deadline:
            time.sleep(0.02)
        sampler.stop(timeout=5)

    assert sampler.sample_count() >= 3
    with sampler._lock:
        samples = list(sampler._samples)
    assert all(s["cpu_package_temp_c"] is None for s in samples)
    # Every OTHER metric must still be recorded -- one dead sensor must not
    # take the rest of the sampler down with it.
    assert any(s["ram_used_mb"] is not None for s in samples)


def test_sampler_warns_only_once_per_metric(tmp_path, monkeypatch):
    import psutil

    def _raise(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(psutil, "sensors_temperatures", _raise)

    path = str(tmp_path / "telemetry.jsonl")
    sampler = TelemetrySampler(path, interval_s=0.02, disk_path=str(tmp_path))

    with pytest.warns(UserWarning) as record:
        sampler.start()
        deadline = time.time() + 3
        while sampler.sample_count() < 10 and time.time() < deadline:
            time.sleep(0.02)
        sampler.stop(timeout=5)

    temp_warnings = [w for w in record.list if "cpu_package_temp_c" in str(w.message)]
    assert len(temp_warnings) == 1
