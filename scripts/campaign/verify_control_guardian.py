#!/usr/bin/env python3
"""Fail-closed identity check for the root production control guardian."""

from __future__ import annotations

import argparse
import enum
import os
from pathlib import Path, PurePosixPath


class Probe(enum.Enum):
    DEAD = "dead"
    SIGNALABLE = "signalable"
    INACCESSIBLE = "exists-but-signal-inaccessible"


def signal_probe(pid: int) -> Probe:
    """Distinguish ESRCH from EPERM instead of treating every error as death."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return Probe.DEAD
    except PermissionError:
        return Probe.INACCESSIBLE
    return Probe.SIGNALABLE


def _status(proc_root: Path, pid: int) -> dict[str, str]:
    fields = {}
    with (proc_root / str(pid) / "status").open(encoding="ascii") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
    return fields


def _wrapper_identity(argv: list[str], expected_script: Path) -> bool:
    expected = PurePosixPath(expected_script.as_posix())
    for argument in argv:
        candidate = PurePosixPath(argument)
        if candidate == expected:
            return True
        if not candidate.is_absolute() and candidate.parts[-3:] == expected.parts[-3:]:
            return True
    return False


def verify_guardian(*, pid: int, expected_uid: int, descendant_pid: int,
                    expected_script: Path, proc_root: Path = Path("/proc")) -> tuple[bool, str]:
    probe = signal_probe(pid) if proc_root == Path("/proc") else Probe.INACCESSIBLE
    if probe is Probe.DEAD:
        return False, f"guardian PID {pid} does not exist"
    try:
        status = _status(proc_root, pid)
        uids = [int(value) for value in status["Uid"].split()]
        argv = (proc_root / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        decoded_argv = [value.decode("utf-8", "surrogateescape") for value in argv]
    except (FileNotFoundError, ProcessLookupError):
        return False, f"guardian PID {pid} disappeared during identity verification"
    except (KeyError, OSError, ValueError) as exc:
        return False, f"cannot verify guardian PID {pid}: {exc}"
    if len(uids) != 4 or any(uid != expected_uid for uid in uids):
        return False, f"guardian PID {pid} has UIDs {uids}, expected only {expected_uid}"
    if not _wrapper_identity(decoded_argv, expected_script):
        return False, f"guardian PID {pid} is not {expected_script}"

    current = descendant_pid
    visited = set()
    try:
        while current > 1 and current not in visited:
            if current == pid:
                return True, f"guardian PID {pid} verified ({probe.value})"
            visited.add(current)
            current = int(_status(proc_root, current)["PPid"])
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        return False, f"cannot verify guardian ancestry for PID {pid}: {exc}"
    return False, f"guardian PID {pid} is not an ancestor of launcher PID {descendant_pid}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--expected-uid", required=True, type=int)
    parser.add_argument("--descendant-pid", required=True, type=int)
    parser.add_argument("--expected-script", required=True, type=Path)
    args = parser.parse_args()
    ok, message = verify_guardian(pid=args.pid, expected_uid=args.expected_uid,
                                  descendant_pid=args.descendant_pid,
                                  expected_script=args.expected_script)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
