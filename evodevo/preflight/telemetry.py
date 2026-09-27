"""Background machine-condition sampler.

A throughput number without the conditions that produced it is not a
measurement: "B2 got N evaluations/sec" is meaningless on its own if the
machine was thermal-throttling, swapping, or sharing the box with an
unrelated process at the time. This module runs a background thread that
periodically samples CPU/memory/thermal state to telemetry.jsonl for the
whole pre-flight run's duration; any stage can later call summarise() over
its own [start, end] time window to attach "what the machine was doing"
to its own result.

Degrades gracefully by design: every individual metric is read through
_safe(), which turns any exception (or a legitimate "no data" result, e.g.
no temperature sensors on this platform) into a recorded null plus a
ONE-TIME warning -- a missing sensor must never take down a half-day
unattended run.
"""
from __future__ import annotations

import glob
import json
import math
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

_SCALAR_METRICS = (
    "cpu_percent_overall",
    "load_avg_1m",
    "load_avg_5m",
    "load_avg_15m",
    "ram_used_mb",
    "ram_available_mb",
    "swap_used_mb",
    "disk_free_gb",
    "process_count",
    "cpu_package_temp_c",
    "thermal_throttle_count",
)
_LIST_METRICS = ("cpu_percent_per_core", "cpu_freq_per_core_mhz")


def percentile(values: list, p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def stats(values: list) -> dict:
    if not values:
        return {"mean": None, "max": None, "p95": None}
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "p95": percentile(values, 95),
    }


def read_cpu_package_temp_c() -> Optional[float]:
    """A single point-in-time CPU package temperature reading, in Celsius,
    or None if no sensor is readable. Module-level (not just a
    TelemetrySampler method) so a stage that wants "temperature right now"
    (e.g. B2's per-level before/after readings) doesn't need a running
    sampler to get one -- TelemetrySampler._take_sample uses this exact
    function for its own periodic samples, so there is one implementation,
    not two that could drift."""
    import psutil

    temps = psutil.sensors_temperatures()
    if not temps:
        return None
    if "coretemp" in temps:
        for entry in temps["coretemp"]:
            if "package" in (entry.label or "").lower():
                return entry.current
        if temps["coretemp"]:
            return temps["coretemp"][0].current
    for entries in temps.values():
        if entries:
            return entries[0].current
    return None


# Fallback critical/Tjmax threshold, ONLY used when the platform's own
# sensor exposes neither .critical nor .high (see
# read_cpu_package_temp_critical_c()) -- a generic, conservative value for
# modern Intel parts. This project's own target machine reports a real
# coretemp .critical of 100.0 via psutil, so this constant is a last
# resort, not the primary source of truth.
FALLBACK_CRITICAL_TEMP_C = 100.0

# How far below the critical/Tjmax threshold a measured temperature must
# still be to rule OUT genuine thermal throttling -- see
# thermal_throttle_is_genuine()'s own docstring for why this margin exists
# at all (kernel's package_throttle_count conflates true over-temperature
# throttling with RAPL power/current-limit throttling, which asserts the
# SAME hardware "Thermal Status" bit under ordinary sustained multi-core
# load, nowhere near Tjmax).
DEFAULT_THROTTLE_TEMP_MARGIN_C = 15.0


def read_cpu_package_temp_critical_c() -> Optional[float]:
    """The critical/Tjmax threshold for the SAME sensor entry
    read_cpu_package_temp_c() reads its current value from (mirrors that
    function's own package/coretemp preference order exactly, so the two
    are always talking about the same physical sensor) -- psutil exposes
    this as shwtemp.critical for platforms that report it (this project's
    own target machine reports 100.0 C via its 'coretemp' / 'Package id 0'
    entry). Falls back to shwtemp.high (a lower, non-critical warning
    threshold some platforms report instead), then to None if the
    platform's sensor exposes neither -- callers combine None with
    FALLBACK_CRITICAL_TEMP_C rather than treating it as an error, since a
    missing critical-temp reading must never take down a run any more than
    a missing current-temp reading does (see this module's own docstring)."""
    import psutil

    temps = psutil.sensors_temperatures()
    if not temps:
        return None
    entry = None
    if "coretemp" in temps:
        for candidate in temps["coretemp"]:
            if "package" in (candidate.label or "").lower():
                entry = candidate
                break
        if entry is None and temps["coretemp"]:
            entry = temps["coretemp"][0]
    if entry is None:
        for entries in temps.values():
            if entries:
                entry = entries[0]
                break
    if entry is None:
        return None
    return entry.critical if entry.critical is not None else entry.high


def thermal_throttle_is_genuine(
    counter_incremented: bool,
    max_temp_c: Optional[float],
    critical_temp_c: Optional[float],
    margin_c: float = DEFAULT_THROTTLE_TEMP_MARGIN_C,
) -> Optional[bool]:
    """Whether a kernel-reported thermal_throttle_count increment (see
    read_thermal_throttle_count()) reflects GENUINE over-temperature
    throttling, rather than the identical hardware "Thermal Status" bit
    being asserted by RAPL power/current-limit enforcement under ordinary
    sustained multi-core load -- both conditions increment the SAME sysfs
    counter (/sys/.../thermal_throttle/package_throttle_count), and Linux
    does not expose them separately without an MSR read (IA32_PACKAGE_
    THERM_STATUS bit 10, "Power Limitation Status", vs. bit 0, "Thermal
    Status") which requires root and is not available in every deployment.

    Empirically diagnosed on this project's own target machine: B2's D13
    "throttled": true fired at EVERY concurrency level, including package
    temperatures of 40-55C -- nowhere near this machine's own measured
    critical/Tjmax of 100C (read_cpu_package_temp_critical_c()) -- while
    the counter itself incremented by thousands of brief (~ms-scale, see
    package_throttle_total_time_ms / package_throttle_count) events under
    any sustained heavy multi-core load, even well below Tjmax. That
    pattern -- frequent, brief events, far from Tjmax -- is the signature
    of power/current-limit throttling, not overheating.

    Only returns True when the counter incremented AND the measured
    temperature was genuinely close (within `margin_c`) to the real
    critical/Tjmax threshold -- i.e. a throttle event that temperature
    alone can plausibly explain. Returns False (not throttled) the moment
    counter_incremented is False, regardless of temperature. Returns None
    only when the counter DID increment but no temperature reading exists
    to correlate against -- "unknown", never silently coerced to True or
    False, consistent with every other sensor in this module degrading to
    None rather than guessing (see the module docstring)."""
    if not counter_incremented:
        return False
    if max_temp_c is None:
        return None
    threshold = (critical_temp_c if critical_temp_c is not None else FALLBACK_CRITICAL_TEMP_C) - margin_c
    return max_temp_c >= threshold


def read_thermal_throttle_count() -> Optional[int]:
    """A single point-in-time cumulative thermal-throttle event count
    (Linux's own /sys/.../thermal_throttle/package_throttle_count
    counters, max across packages) -- monotonically non-decreasing, so
    comparing two readings taken before/after a measurement window answers
    "did this machine throttle during that window", which is the D13
    question that matters, not the absolute count."""
    paths = glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count")
    if not paths:
        return None
    values = []
    for p in paths:
        try:
            with open(p) as f:
                values.append(int(f.read().strip()))
        except (OSError, ValueError):
            continue
    return max(values) if values else None


class TelemetrySampler:
    def __init__(self, output_path: str, interval_s: float = 5.0, disk_path: str = "."):
        self.output_path = output_path
        self.interval_s = interval_s
        self._disk_path = disk_path

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._samples: list = []
        self._warned: set = set()
        self._fh = None

    def _warn_once(self, name: str, reason: str) -> None:
        if name not in self._warned:
            warnings.warn(
                f"telemetry: metric '{name}' unavailable ({reason}) -- recording null for the "
                f"rest of this run.",
                stacklevel=3,
            )
            self._warned.add(name)

    def _safe(self, name: str, fn) -> Any:
        try:
            value = fn()
        except Exception as e:  # noqa: BLE001 -- any sensor failure degrades to null, never crashes the sampler
            self._warn_once(name, f"{type(e).__name__}: {e}")
            return None
        if value is None:
            self._warn_once(name, "no data available")
        return value

    def _read_package_temp(self) -> Optional[float]:
        return read_cpu_package_temp_c()

    def _read_throttle_count(self) -> Optional[int]:
        return read_thermal_throttle_count()

    def _take_sample(self) -> dict:
        import psutil

        now = datetime.now(timezone.utc)
        sample: dict = {"ts": now.isoformat(), "ts_epoch": now.timestamp()}

        sample["cpu_percent_overall"] = self._safe("cpu_percent_overall", lambda: psutil.cpu_percent(interval=None))
        sample["cpu_percent_per_core"] = self._safe(
            "cpu_percent_per_core", lambda: psutil.cpu_percent(interval=None, percpu=True)
        )
        sample["cpu_freq_per_core_mhz"] = self._safe(
            "cpu_freq_per_core_mhz",
            lambda: [f.current for f in (psutil.cpu_freq(percpu=True) or [])] or None,
        )

        load = self._safe("load_avg", lambda: psutil.getloadavg())
        sample["load_avg_1m"], sample["load_avg_5m"], sample["load_avg_15m"] = load if load else (None, None, None)

        vmem = self._safe("virtual_memory", lambda: psutil.virtual_memory())
        sample["ram_used_mb"] = round(vmem.used / 1e6, 1) if vmem else None
        sample["ram_available_mb"] = round(vmem.available / 1e6, 1) if vmem else None

        swap = self._safe("swap_memory", lambda: psutil.swap_memory())
        sample["swap_used_mb"] = round(swap.used / 1e6, 1) if swap else None

        disk = self._safe("disk_usage", lambda: psutil.disk_usage(self._disk_path))
        sample["disk_free_gb"] = round(disk.free / 1e9, 2) if disk else None

        sample["process_count"] = self._safe("process_count", lambda: len(psutil.pids()))
        sample["cpu_package_temp_c"] = self._safe("cpu_package_temp_c", self._read_package_temp)
        sample["thermal_throttle_count"] = self._safe("thermal_throttle_count", self._read_throttle_count)

        return sample

    def _run(self) -> None:
        # Prime psutil.cpu_percent's internal baseline -- its first
        # non-blocking (interval=None) call is meaningless (no prior call
        # to compare against), so the first WRITTEN sample should not be it.
        import psutil

        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

        while not self._stop_event.is_set():
            sample = self._take_sample()
            with self._lock:
                self._samples.append(sample)
            if self._fh is not None:
                self._fh.write(json.dumps(sample, sort_keys=True, default=str) + "\n")
                self._fh.flush()
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TelemetrySampler.start() called twice")
        self._fh = open(self.output_path, "a")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry-sampler", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def latest_sample(self) -> Optional[dict]:
        """The single most recent sample, or None before the first one has
        been taken -- for a live display (dashboard.py) that wants "what is
        the machine doing right now", not an aggregate over the run."""
        with self._lock:
            return self._samples[-1] if self._samples else None

    def summarise(self, window: Optional[Tuple[float, float]] = None) -> dict:
        """Mean/max/p95 for every metric, either over the whole run (window=
        None) or over [start_epoch, end_epoch] (matching each sample's
        "ts_epoch" field) -- what a stage calls with its own (start, end)
        to attach machine conditions to its result. Per-core list metrics
        (cpu_percent_per_core, cpu_freq_per_core_mhz) are pooled across
        cores AND samples into one distribution, not broken out per-core --
        this answers "how loaded were the cores in general", not "which
        specific core", which is enough for a stage's own provenance."""
        with self._lock:
            samples = list(self._samples)

        if window is not None:
            start, end = window
            samples = [s for s in samples if start <= s.get("ts_epoch", 0) <= end]

        result: dict = {"n_samples": len(samples)}
        for metric in _SCALAR_METRICS:
            values = [s[metric] for s in samples if s.get(metric) is not None]
            result[metric] = stats(values)
        for metric in _LIST_METRICS:
            pooled: list = []
            for s in samples:
                v = s.get(metric)
                if v:
                    pooled.extend(x for x in v if x is not None)
            result[metric] = stats(pooled)
        return result

    def __enter__(self) -> "TelemetrySampler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
