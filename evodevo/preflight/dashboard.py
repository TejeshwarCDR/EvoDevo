"""Live operator-facing status display for a running pre-flight.

Two renderers share one state-gathering core (_gather_snapshot /
_EventTailer): Dashboard, a full-screen ANSI redraw for an interactive TTY,
and PlainLogger, one line per tick with no cursor control -- used whenever
stdout is not a TTY or --no-dashboard was passed. Neither renderer ever
prints a worker's own stdout/stderr into its output: WorkerResult captures
that per-task (see evodevo/parallel/evaluator_entry.py's _capture_stdio)
and neither renderer reads WorkerResult at all, only manifest.json,
events.jsonl and the telemetry sampler's own in-memory samples -- so there
is no code path by which worker output could reach this display even by
accident.

This module is numpy/torch-free and imports nothing from evodevo.preflight
.__main__ (the dependency runs the other way: __main__ imports this).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from evodevo.preflight.events import read_events
from evodevo.preflight.stages import load_manifest

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_STATUS_SYMBOLS = {"pending": ".", "running": ">", "done": "✓", "failed": "!", "skipped": "-"}


def _spark(values: list) -> str:
    if not values:
        return ""
    return "".join(_SPARK_CHARS[min(7, max(0, int(v / 100.0 * 7)))] if v is not None else " " for v in values)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _EventTailer:
    """Incrementally reads events.jsonl, keeping only the small rolling
    aggregate a status display needs -- never re-parses the whole file on
    every tick, which matters over a run that can run for hours and
    accumulate many thousands of records."""

    def __init__(self, events_path: str):
        self.events_path = events_path
        self._offset = 0
        self.failure_count = 0
        self.last_failure: Optional[dict] = None
        self.last_measurement: Optional[dict] = None
        self.last_measurement_ts_epoch: Optional[float] = None
        self.recent_measurement_ts_epochs: list = []

    def poll(self) -> None:
        if not os.path.exists(self.events_path):
            return
        with open(self.events_path) as f:
            f.seek(self._offset)
            new_lines = f.readlines()
            self._offset = f.tell()
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("kind")
            if kind == "failure":
                self.failure_count += 1
                self.last_failure = event
            elif kind == "measurement":
                self.last_measurement = event
                try:
                    ts_epoch = datetime.fromisoformat(event["ts"]).timestamp()
                except (KeyError, ValueError):
                    ts_epoch = time.time()
                self.last_measurement_ts_epoch = ts_epoch
                self.recent_measurement_ts_epochs.append(ts_epoch)
                cutoff = time.time() - 300.0
                self.recent_measurement_ts_epochs = [t for t in self.recent_measurement_ts_epochs if t >= cutoff]


class _StatusCore:
    """Shared state-gathering + lifecycle for both renderers."""

    def __init__(self, ctx, registry, interval_s: float = 2.0):
        self.ctx = ctx
        self.registry = registry
        self.interval_s = interval_s
        self._tailer = _EventTailer(ctx.run_dir.events_path)
        self._start_epoch = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _snapshot(self) -> dict:
        self._tailer.poll()
        manifest = load_manifest(self.ctx.run_dir)
        order = self.registry.ordered()
        stage_rows = []
        n_done = 0
        current_stage = None
        for name in order:
            entry = manifest.get("stages", {}).get(name, {})
            status = entry.get("status", "pending")
            if status == "done":
                n_done += 1
            if status == "running":
                current_stage = name
            stage_rows.append(
                {
                    "name": name,
                    "status": status,
                    "duration_s": entry.get("duration_s"),
                    "start_time": entry.get("start_time"),
                    "error": entry.get("error"),
                }
            )

        elapsed = time.time() - self._start_epoch
        completed_fraction = n_done / len(order) if order else 0.0
        if completed_fraction > 0:
            point_estimate = elapsed / completed_fraction - elapsed
            eta_low, eta_high = max(0.0, point_estimate * 0.5), point_estimate * 1.5
        else:
            eta_low, eta_high = None, None

        n_recent = len(self._tailer.recent_measurement_ts_epochs)
        throughput_now = (n_recent / 300.0) * 60.0 if n_recent else 0.0

        time_since_last_eval = (
            time.time() - self._tailer.last_measurement_ts_epoch
            if self._tailer.last_measurement_ts_epoch is not None
            else None
        )

        latest_sample = self.ctx.telemetry.latest_sample() if self.ctx.telemetry is not None else None

        return {
            "stage_rows": stage_rows,
            "n_done": n_done,
            "n_total": len(order),
            "current_stage": current_stage,
            "elapsed_s": elapsed,
            "eta_low_s": eta_low,
            "eta_high_s": eta_high,
            "failure_count": self._tailer.failure_count,
            "last_failure": self._tailer.last_failure,
            "last_measurement": self._tailer.last_measurement,
            "time_since_last_eval_s": time_since_last_eval,
            "throughput_now_per_min": throughput_now,
            "telemetry": latest_sample,
        }

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"{type(self).__name__}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
            self._thread = None
        self._final_render()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._render(self._snapshot())
            except Exception:  # noqa: BLE001 -- a display bug must never take down the pre-flight itself
                pass
            self._stop_event.wait(self.interval_s)

    def _render(self, snapshot: dict) -> None:
        raise NotImplementedError

    def _final_render(self) -> None:
        pass


class Dashboard(_StatusCore):
    """Full-screen ANSI redraw for an interactive TTY."""

    def _render(self, snapshot: dict) -> None:
        lines = []
        lines.append(f"EvoDevo pre-flight -- run {self.ctx.run_dir.run_id}")
        lines.append(f"elapsed {_fmt_hms(snapshot['elapsed_s'])}  " + self._eta_str(snapshot))
        lines.append("")
        lines.append(f"stages ({snapshot['n_done']}/{snapshot['n_total']} done):")
        for row in snapshot["stage_rows"]:
            sym = _STATUS_SYMBOLS.get(row["status"], "?")
            dur = f" ({_fmt_hms(row['duration_s'])})" if row.get("duration_s") else ""
            marker = " <-- current" if row["status"] == "running" else ""
            lines.append(f"  [{sym}] {row['name']:<22} {row['status']:<9}{dur}{marker}")
        lines.append("")
        lm = snapshot["last_measurement"]
        if lm:
            lines.append(f"current measurement: stage={lm.get('stage')} {lm.get('payload')}")
        else:
            lines.append("current measurement: (none yet)")
        watchdog = snapshot["time_since_last_eval_s"]
        watchdog_str = f"{watchdog:.0f}s ago" if watchdog is not None else "n/a"
        lines.append(f"time since last successful evaluation: {watchdog_str}")
        lines.append(
            f"throughput: now={snapshot['throughput_now_per_min']:.1f}/min  "
            f"avg=n/a (see report.py for per-stage rates once a stage completes)"
        )
        lines.append("")
        t = snapshot["telemetry"]
        if t:
            spark = _spark(t.get("cpu_percent_per_core") or [])
            cpu = t.get("cpu_percent_overall")
            ram_used = t.get("ram_used_mb")
            disk_free = t.get("disk_free_gb")
            temp = t.get("cpu_package_temp_c")
            lines.append(
                f"cpu={cpu if cpu is not None else 'n/a'}%  cores=[{spark}]  "
                f"ram_used={ram_used if ram_used is not None else 'n/a'}MB  "
                f"disk_free={disk_free if disk_free is not None else 'n/a'}GB  "
                f"pkg_temp={temp if temp is not None else 'n/a'}C"
            )
        else:
            lines.append("resource telemetry: (no sample yet)")
        lines.append("")
        fc = snapshot["failure_count"]
        if fc:
            lf = snapshot["last_failure"] or {}
            lines.append(f"** FAILURES: {fc} -- most recent: [{lf.get('stage')}] {lf.get('payload', {}).get('error')}")
        else:
            lines.append("failures: 0")

        # Clear screen, home cursor, print. Never writes a worker's own
        # stdout/stderr here -- see module docstring.
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def _eta_str(self, snapshot: dict) -> str:
        if snapshot["eta_low_s"] is None:
            return "ETA: n/a (no stage completed yet)"
        return f"ETA: {_fmt_hms(snapshot['eta_low_s'])}-{_fmt_hms(snapshot['eta_high_s'])} (rough range, not a commitment)"

    def _final_render(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class PlainLogger(_StatusCore):
    """One line per tick, no cursor control -- used under --no-dashboard or
    whenever stdout is not a TTY (e.g. piped into `tee`, or under a
    subprocess-based test)."""

    def _render(self, snapshot: dict) -> None:
        fc = snapshot["failure_count"]
        watchdog = snapshot["time_since_last_eval_s"]
        watchdog_str = f"{watchdog:.0f}s" if watchdog is not None else "n/a"
        line = (
            f"[preflight t={_fmt_hms(snapshot['elapsed_s'])}] "
            f"stage={snapshot['current_stage'] or '-'} "
            f"done={snapshot['n_done']}/{snapshot['n_total']} "
            f"failures={fc} "
            f"last_eval={watchdog_str}_ago"
        )
        print(line, file=sys.stderr, flush=True)
