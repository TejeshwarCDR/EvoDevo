from __future__ import annotations

import errno
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/campaign/verify_control_guardian.py"
SPEC = importlib.util.spec_from_file_location("verify_control_guardian", SCRIPT)
guardian = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guardian)


def _process(root: Path, pid: int, *, uid: int, ppid: int, argv: list[str]):
    folder = root / str(pid)
    folder.mkdir()
    (folder / "status").write_text(
        f"Name:\ttest\nPid:\t{pid}\nPPid:\t{ppid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
    )
    (folder / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")


def test_permission_denied_probe_means_exists_but_inaccessible(monkeypatch):
    def denied(_pid, _signal):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(guardian.os, "kill", denied)
    assert guardian.signal_probe(123) is guardian.Probe.INACCESSIBLE


def test_nonexistent_probe_means_dead(monkeypatch):
    def missing(_pid, _signal):
        raise ProcessLookupError(errno.ESRCH, "No such process")

    monkeypatch.setattr(guardian.os, "kill", missing)
    assert guardian.signal_probe(123) is guardian.Probe.DEAD


def test_root_wrapper_identity_and_ancestry_are_required(tmp_path):
    expected = REPO_ROOT / "scripts/campaign/with_cpu_power_cap.sh"
    _process(tmp_path, 100, uid=0, ppid=1,
             argv=["bash", "scripts/campaign/with_cpu_power_cap.sh", "--turbo"])
    _process(tmp_path, 200, uid=0, ppid=100, argv=["runuser", "-u", "ubuntu"])
    _process(tmp_path, 300, uid=1000, ppid=200, argv=["bash", "run_campaign.sh"])
    ok, message = guardian.verify_guardian(
        pid=100, expected_uid=0, descendant_pid=300,
        expected_script=expected, proc_root=tmp_path)
    assert ok, message
    assert "exists-but-signal-inaccessible" in message


@pytest.mark.parametrize(
    ("uid", "argv", "ppid", "reason"),
    [
        (1000, ["bash", "scripts/campaign/with_cpu_power_cap.sh"], 100, "UIDs"),
        (0, ["bash", "scripts/campaign/not-the-guardian.sh"], 100, "is not"),
        (0, ["bash", "scripts/campaign/with_cpu_power_cap.sh"], 1, "not an ancestor"),
    ],
)
def test_wrong_owner_process_or_ancestry_fails(tmp_path, uid, argv, ppid, reason):
    expected = REPO_ROOT / "scripts/campaign/with_cpu_power_cap.sh"
    _process(tmp_path, 100, uid=uid, ppid=1, argv=argv)
    _process(tmp_path, 300, uid=1000, ppid=ppid, argv=["bash", "run_campaign.sh"])
    ok, message = guardian.verify_guardian(
        pid=100, expected_uid=0, descendant_pid=300,
        expected_script=expected, proc_root=tmp_path)
    assert not ok
    assert reason in message


def test_disappeared_guardian_fails_closed(tmp_path):
    ok, message = guardian.verify_guardian(
        pid=404, expected_uid=0, descendant_pid=300,
        expected_script=REPO_ROOT / "scripts/campaign/with_cpu_power_cap.sh",
        proc_root=tmp_path)
    assert not ok
    assert "disappeared" in message
