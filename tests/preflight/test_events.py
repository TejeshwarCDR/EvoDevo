from __future__ import annotations

import os
import signal
import subprocess
import sys

from evodevo.preflight.events import EventLog, read_events

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_record_writes_the_fixed_schema(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.record("stage_a", "stage_start", {"foo": "bar"})
    log.close()

    events = read_events(path)
    assert len(events) == 1
    event = events[0]
    assert set(event.keys()) == {"ts", "stage", "kind", "payload"}
    assert event["stage"] == "stage_a"
    assert event["kind"] == "stage_start"
    assert event["payload"] == {"foo": "bar"}


def test_record_defaults_payload_to_empty_dict(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.record(None, "anomaly")
    log.close()

    events = read_events(path)
    assert events[0]["payload"] == {}
    assert events[0]["stage"] is None


def test_resume_appends_rather_than_truncates(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log1 = EventLog(path)
    log1.record("a", "stage_start")
    log1.close()

    log2 = EventLog(path)
    log2.record("a", "stage_end")
    log2.close()

    events = read_events(path)
    assert [e["kind"] for e in events] == ["stage_start", "stage_end"]


def test_read_events_skips_a_truncated_trailing_line(tmp_path):
    path = str(tmp_path / "events.jsonl")
    log = EventLog(path)
    log.record("a", "stage_start")
    log.close()

    with open(path, "a") as f:
        f.write('{"ts": "2026-01-01T00:00:00Z", "stage": "a", "kind": "trunc')  # deliberately truncated, no newline

    events = read_events(path)
    assert len(events) == 1
    assert events[0]["kind"] == "stage_start"


_DRIVER_SCRIPT = f"""
import sys
sys.path.insert(0, {REPO_ROOT!r})
import time
from evodevo.preflight.events import EventLog

path = sys.argv[1]
n_before_kill = int(sys.argv[2])

log = EventLog(path)
for i in range(n_before_kill):
    log.record("driver", "measurement", {{"i": i}})
print("logged", flush=True)
time.sleep(60)  # killed before this returns
"""


def test_event_log_records_survive_a_sigkill_of_the_writing_process(tmp_path):
    script_path = tmp_path / "driver.py"
    script_path.write_text(_DRIVER_SCRIPT)
    events_path = str(tmp_path / "events.jsonl")

    proc = subprocess.Popen(
        [sys.executable, str(script_path), events_path, "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.strip() == "logged", f"driver did not confirm logging (stderr: {proc.stderr.read()})"
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    events = read_events(events_path)
    assert len(events) == 5
    assert [e["payload"]["i"] for e in events] == [0, 1, 2, 3, 4]
    for e in events:
        assert e["stage"] == "driver"
        assert e["kind"] == "measurement"
