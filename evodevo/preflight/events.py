"""Append-only event log: the file that makes an interrupted pre-flight
still useful.

Every record is flushed AND fsync'd before the call returns, so a SIGKILL
(a killed tmux pane, an OOM kill, `kill -9`) the instant after a record was
logged still leaves that record durably on disk -- see run_dir.py's
docstring for the same D8 reasoning applied to JSON artifacts. This is
slower than buffered writes, but events are logged at "stage started",
"stage finished", "one measurement taken", "one anomaly" granularity, not
per-millisecond, so the cost is negligible against what it buys: a run
killed at hour 3 of 6 still has every event from those first 3 hours,
readable and parseable, with nothing to recover or repair.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterator, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """One schema, every record: {"ts", "stage", "kind", "payload"}.

    kind is a free-form short label (e.g. "stage_start", "stage_end",
    "measurement", "failure", "anomaly") -- this module does not enumerate
    or validate the set, since new kinds will be added by every future
    B1-B7 stage without needing to touch this module.
    """

    def __init__(self, path: str):
        self.path = path
        # Append mode: resuming a run must never truncate prior events.
        self._fh = open(path, "a")

    def record(self, stage: Optional[str], kind: str, payload: Optional[dict] = None) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "stage": stage,
            "kind": kind,
            "payload": payload or {},
        }
        self._fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def read_events(path: str) -> list:
    """Reads back every record in an events.jsonl file. Tolerant of a
    truncated final line (e.g. a write that was mid-flush at the moment of
    a kill) -- skips it rather than raising, since "everything flushed
    before the kill is present and parseable" is the guarantee, not "the
    file as a whole is valid JSON"."""
    events: list = []
    if not os.path.exists(path):
        return events
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def iter_events(path: str) -> Iterator[dict]:
    for event in read_events(path):
        yield event
