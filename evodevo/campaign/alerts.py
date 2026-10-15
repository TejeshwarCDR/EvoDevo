"""Durable alert decisions and optional out-of-band notifications."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shlex
import statistics
import subprocess
import time
import urllib.request
from collections import deque
from pathlib import Path

import yaml

from evodevo.preflight.telemetry import thermal_throttle_is_genuine

REN_NOTIFY_S = 6 * 3600
HEARTBEAT_S = 24 * 3600
TEMPERATURE_SUSTAIN_S = 10 * 60
PARTIAL_STALL_S = 3600
LOW_UTILIZATION_S = 2 * 3600
WORKER_RSS_LIMIT_BYTES = 40 * 1024 ** 3
# Step 5 steady-state measurements, minus the fixed 15 percentage-point
# production guard margin.  These are operational thresholds, not estimates.
UTILIZATION_FLOOR = {"Walker-v0": 0.996922 - 0.15, "UpStepper-v0": 0.984949 - 0.15}
# Tasks without their own Step-5 measurement use the lower of the two
# measured floors, so the low-utilization alert is never silently disabled.
DEFAULT_UTILIZATION_FLOOR = min(UTILIZATION_FLOOR.values())


def utilization_floor(task):
    return UTILIZATION_FLOOR.get(task, DEFAULT_UTILIZATION_FLOOR) if task else None


def format_display_message(name, details, status, *, label="alert"):
    """The only operator-facing alert formatter; journal machine text remains raw."""
    from .status import CampaignStatus, status_footer
    if status is None:
        status = CampaignStatus.from_snapshot({"phase": (details or {}).get("phase", "unknown")})
    heading = f"{label}: {name}"
    body = json.dumps(details or {}, sort_keys=True, indent=2, default=str)
    return f"{heading}\n{body}\n{status_footer(status)}"


DEFAULT_NOTIFY_CONFIG = "campaign.local.yaml"
NOTIFY_CONFIG_ENV = "EVODEVO_NOTIFY_CONFIG"
NO_NOTIFIER_MESSAGE = "NO NOTIFIER CONFIGURED - alerts go to alerts.jsonl only"


class NotifierNotConfigured(RuntimeError):
    """No delivery channel is configured; nothing can leave the machine."""


def notify_config_path(path=None):
    """Explicit path, else $EVODEVO_NOTIFY_CONFIG, else ./campaign.local.yaml."""
    return path or os.environ.get(NOTIFY_CONFIG_ENV) or DEFAULT_NOTIFY_CONFIG


def load_local_config(path=None):
    try:
        with open(notify_config_path(path), encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


class Notifier:
    def __init__(self, config=None):
        self.config = config if config is not None else load_local_config()

    @property
    def channels(self):
        return [name for name in ("ntfy_topic_url", "shell_command") if self.config.get(name)]

    @property
    def configured(self) -> bool:
        return bool(self.channels)

    def send(self, message: str):
        """Deliver on every configured channel; raise unless each one
        verifiably succeeded (HTTP 2xx / shell exit 0). Returns channels."""
        if not self.configured:
            raise NotifierNotConfigured(NO_NOTIFIER_MESSAGE)
        url = self.config.get("ntfy_topic_url")
        command = self.config.get("shell_command")
        if url:
            req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST")
            with urllib.request.urlopen(req, timeout=15) as response:
                status = getattr(response, "status", None)
            if status is None or not 200 <= status < 300:
                raise RuntimeError(f"ntfy delivery returned HTTP {status}")
        if command:
            # Message is supplied as one argument, never interpolated into a
            # shell script.  The operator may configure a command and args.
            argv = shlex.split(command) if isinstance(command, str) else list(command)
            subprocess.run([*argv, message], check=True, timeout=30)
        return self.channels


class AlertManager:
    def __init__(self, path, notifier=None, now=None, sleep=None):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.notifier = notifier or Notifier()
        self.now = now or time.time
        self.sleep = sleep or time.sleep
        self.status = None
        self._active = {}
        self._last_notice = {}
        self._temp_since = None
        self._low_rate_since = None
        self._low_utilization_since = None
        self._last_heartbeat = None
        self._next_heartbeat_at = self._next_1800(self.now())
        self._last_throttle_count = None
        self._temp_24h = deque()
        self._journal_offset = 0
        self._restore()

    def _restore(self):
        """Keep six-hour deduplication and heartbeat timing across restarts."""
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._refresh_locked(handle, restore_timers=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _consume(self, row, *, restore_timers=False):
        cutoff = self.now() - HEARTBEAT_S
        name = row.get("condition")
        stamp = row.get("timestamp", 0)
        if not name:
            return
        self._active[name] = row.get("status") == "active"
        # "journal_only" (no notifier configured, i.e. --no-notify) still
        # counts for de-duplication timing: the notice WAS emitted, to
        # alerts.jsonl, which is the only channel that exists.
        emitted = row.get("notified") or row.get("delivery") == "journal_only"
        if emitted:
            self._last_notice[name] = stamp
        if name == "daily_heartbeat" and emitted:
            self._last_heartbeat = stamp
            self._next_heartbeat_at = self._next_1800(stamp)
        if not restore_timers:
            return
        if name == "temperature" and stamp >= cutoff:
            temp = row.get("details", {}).get("temperature_c")
            if temp is not None:
                self._temp_24h.append((stamp, temp))
            if temp is not None and temp >= 95:
                self._temp_since = self._temp_since or stamp
            else:
                self._temp_since = None
        if name == "partial_stall" and stamp >= cutoff:
            details = row.get("details", {})
            rate, median = details.get("current_per_h"), details.get("median_24h_per_h")
            if isinstance(rate, (float, int)) and isinstance(median, (float, int)) and median > 0 and rate < 0.25 * median:
                self._low_rate_since = self._low_rate_since or stamp
            else:
                self._low_rate_since = None
        if name == "worker_utilization" and stamp >= cutoff:
            details = row.get("details", {})
            if details.get("below_floor"):
                self._low_utilization_since = self._low_utilization_since or stamp
            else:
                self._low_utilization_since = None

    def _refresh_locked(self, handle, *, restore_timers=False):
        handle.seek(self._journal_offset)
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith("\n"):
                # A process killed during its final write can leave only an
                # incomplete tail. It was never a completed alert record;
                # trim just that suffix so future JSONL readers remain safe.
                handle.seek(position)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            self._consume(json.loads(line), restore_timers=restore_timers)
        self._journal_offset = handle.tell()

    def evaluate(self, name: str, triggered: bool, details=None, *, informational=False,
                 status=None, force=False):
        """Journal every check; notify first, every 6 h, and once on clear."""
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._refresh_locked(handle)
            stamp = self.now()
            previous = self._active.get(name, False)
            state = "active" if triggered else "clear"
            event = {"timestamp": stamp, "condition": name, "status": state,
                     "details": details or {}, "notified": False}
            should_notify = (
                force or (triggered and (not previous or stamp - self._last_notice.get(name, float("-inf")) >= REN_NOTIFY_S))
            ) or (not triggered and previous)
            if should_notify:
                label = "cleared" if not triggered else ("information" if informational else "alert")
                message = f"[{label}] {name}: {json.dumps(details or {}, sort_keys=True)}"
                event["message"] = message
                event["display_message"] = format_display_message(name, details, status or self.status, label=label)
                if not getattr(self.notifier, "configured", True):
                    event["delivery"] = "journal_only"
                else:
                    try:
                        self.notifier.send(event["display_message"])
                        event["notified"] = True
                    except Exception as exc:  # notifier failure must be visible in journal
                        event["notification_error"] = f"{type(exc).__name__}: {exc}"
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            self._journal_offset = handle.tell()
            self._consume(event)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return event

    def inspect(self, snapshot: dict, *, p95_ecore_s: float, mirror_ok=True, status=None):
        self.status = status or snapshot.get("campaign_status_object")
        stamp = self.now()
        last = snapshot.get("seconds_since_last_task")
        self.evaluate("stall", last is not None and last > 2 * p95_ecore_s,
                      {"seconds_since_task": last, "threshold_s": 2 * p95_ecore_s})
        temp = snapshot.get("temperature_c")
        if temp is not None:
            self._temp_24h.append((stamp, temp))
        while self._temp_24h and self._temp_24h[0][0] < stamp - HEARTBEAT_S:
            self._temp_24h.popleft()
        if temp is not None and temp >= 95:
            self._temp_since = self._temp_since or stamp
        else:
            self._temp_since = None
        count = snapshot.get("thermal_throttle_count")
        incremented = count is not None and self._last_throttle_count is not None and count > self._last_throttle_count
        if count is not None:
            self._last_throttle_count = count
        throttled = thermal_throttle_is_genuine(incremented, temp, snapshot.get("critical_temp_c"))
        self.evaluate("temperature", self._temp_since is not None and stamp - self._temp_since >= TEMPERATURE_SUSTAIN_S,
                      {"temperature_c": temp, "throttled": throttled})
        main = snapshot.get("main_disk_free_gb")
        mirror = snapshot.get("mirror_disk_free_gb")
        self.evaluate("disk_main", main is not None and main < 50, {"free_gb": main})
        self.evaluate("disk_mirror", mirror is not None and mirror < 50, {"free_gb": mirror})
        count_ppo = snapshot.get("completed_ppo", 0)
        failures = snapshot.get("infra_failure_count", 0)
        self.evaluate("infra_failure_rate", count_ppo > 0 and failures / count_ppo > 0.005,
                      {"failures": failures, "tasks": count_ppo})
        swap = snapshot.get("swap_used_gb")
        self.evaluate("swap", swap is not None and swap > 1, {"swap_used_gb": swap})
        self.evaluate("mirror_failed", not mirror_ok, {"last_mirror_success": snapshot.get("last_mirror_success")})
        rate = snapshot.get("throughput_now_per_h", 0)
        history = snapshot.get("hourly_rates_24h", [])
        median = statistics.median(history) if history else 0
        if median > 0 and rate < 0.25 * median:
            self._low_rate_since = self._low_rate_since or stamp
        else:
            self._low_rate_since = None
        self.evaluate("partial_stall", self._low_rate_since is not None and stamp - self._low_rate_since >= PARTIAL_STALL_S,
                      {"current_per_h": rate, "median_24h_per_h": median})
        task = snapshot.get("phase")
        floor = utilization_floor(task) if task not in (None, "unknown", "COMPLETE") else None
        utilisation = snapshot.get("worker_utilization_last_hour")
        final_tail = snapshot.get("phase_completion_fraction", 0.0) >= 0.95
        below = (isinstance(utilisation, (int, float)) and floor is not None and
                 utilisation < floor and not final_tail)
        self._low_utilization_since = self._low_utilization_since or stamp if below else None
        self.evaluate("worker_utilization", self._low_utilization_since is not None and
                      stamp - self._low_utilization_since >= LOW_UTILIZATION_S,
                      {"last_hour_busy_fraction": utilisation, "floor": floor,
                       "phase": task, "below_floor": below, "excluded_final_5pct": final_tail})
        rss = snapshot.get("worker_rss_bytes")
        self.evaluate("worker_rss", isinstance(rss, (int, float)) and rss > WORKER_RSS_LIMIT_BYTES,
                      {"rss_bytes": rss, "limit_bytes": WORKER_RSS_LIMIT_BYTES})
        if self._heartbeat_due(stamp):
            self.heartbeat(snapshot)

    @staticmethod
    def _next_1800(stamp):
        local = dt.datetime.fromtimestamp(stamp).astimezone()
        today = local.replace(hour=18, minute=0, second=0, microsecond=0)
        return today.timestamp() if local <= today else (today + dt.timedelta(days=1)).timestamp()

    def _heartbeat_due(self, stamp):
        # No catch-up loop: one inspection after downtime emits at most once.
        if self._last_heartbeat is None:
            return stamp >= self._next_heartbeat_at
        return stamp >= self._next_1800(self._last_heartbeat)

    def heartbeat(self, snapshot):
        stamp = self.now()
        grid = snapshot.get("generation", {})
        arms = {arm for arm, _seed in grid}
        means = {arm: statistics.mean(g for (a, _seed), g in grid.items() if a == arm) for arm in arms}
        details = {"phase": snapshot.get("phase"), "per_arm_mean_generation": means,
                   "completed_evaluations": snapshot.get("completed_evaluations"),
                   "eta_range_s": snapshot.get("campaign_eta_s"),
                   "infra_failure_count": snapshot.get("infra_failure_count"),
                   "max_temperature_24h_c": max((t for _, t in self._temp_24h), default=None),
                   "main_disk_free_gb": snapshot.get("main_disk_free_gb"),
                   "mirror_disk_free_gb": snapshot.get("mirror_disk_free_gb"),
                   "last_mirror_success": snapshot.get("last_mirror_success")}
        event = self.evaluate("daily_heartbeat", True, details, informational=True)
        if event["notified"] or event.get("delivery") == "journal_only":
            self._last_heartbeat = stamp

    def phase_complete(self, task):
        # One durable condition per phase: restarting after a phase boundary
        # cannot emit another notice, while the next phase still gets its own.
        return self.evaluate(f"phase_complete:{task}", True, {"task": task}, informational=True)

    def campaign_started(self, details):
        return self.evaluate("campaign_started", True, details, informational=True)

    def phase_started(self, task):
        return self.evaluate(f"phase_started:{task}", True, {"task": task}, informational=True)

    def campaign_complete(self, details, marker_path):
        """Three durable attempts, then a marker that survives channel failure."""
        event = None
        for attempt in range(1, 4):
            event = self.evaluate("campaign_complete", True, {**details, "attempt": attempt},
                                  informational=True, force=True)
            if event.get("notified") or event.get("delivery") == "journal_only":
                break
            if attempt < 3:
                self.sleep(10 * (2 ** (attempt - 1)))
        Path(marker_path).write_text(details["marker_text"], encoding="utf-8")
        return event

    def milestone_failure(self, task, missing, error=None):
        return self.evaluate(f"milestone_failure:{task}", True,
                             {"task": task, "missing_artifacts": missing, "error": error})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="send one test notification; exit 0 only after verified delivery")
    parser.add_argument("--config", default=None,
                        help=f"notifier config (default ${NOTIFY_CONFIG_ENV} or {DEFAULT_NOTIFY_CONFIG})")
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("use --test")
    notifier = Notifier(load_local_config(args.config))
    if not notifier.configured:
        print(NO_NOTIFIER_MESSAGE)
        return 2
    try:
        channels = notifier.send("[test] EvoDevo campaign notification")
    except Exception as exc:
        print(f"NOTIFICATION DELIVERY FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"Test notification delivered via {', '.join(channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
