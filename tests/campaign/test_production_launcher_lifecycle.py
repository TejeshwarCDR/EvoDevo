"""Regression coverage for the root-held production-control lifecycle."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
POWER_WRAPPER = REPO_ROOT / "scripts/campaign/with_cpu_power_cap.sh"
CAMPAIGN_LAUNCHER = REPO_ROOT / "scripts/campaign/run_campaign.sh"


def _wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition was not reached before timeout")


def _power_tree(root: Path):
    pstate = root / "sys/devices/system/cpu/intel_pstate"
    rapl = root / "sys/class/powercap/intel-rapl:0"
    pstate.mkdir(parents=True)
    rapl.mkdir(parents=True)
    (pstate / "no_turbo").write_text("1")
    (rapl / "constraint_0_power_limit_uw").write_text("253000000")
    (rapl / "constraint_1_power_limit_uw").write_text("253000000")
    for cpu in range(16):
        policy = root / f"sys/devices/system/cpu/cpufreq/policy{cpu}"
        policy.mkdir(parents=True)
        (policy / "scaling_max_freq").write_text(str(5_500_000 + (cpu % 2) * 100_000))


def _assert_applied(root: Path):
    assert (root / "sys/devices/system/cpu/intel_pstate/no_turbo").read_text() == "0"
    assert (root / "sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw").read_text() == "125000000"
    assert (root / "sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw").read_text() == "160000000"
    assert all((root / f"sys/devices/system/cpu/cpufreq/policy{cpu}/scaling_max_freq").read_text()
               == "4200000" for cpu in range(16))


def _assert_restored(root: Path):
    assert (root / "sys/devices/system/cpu/intel_pstate/no_turbo").read_text() == "1"
    assert (root / "sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw").read_text() == "253000000"
    assert (root / "sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw").read_text() == "253000000"
    assert [(root / f"sys/devices/system/cpu/cpufreq/policy{cpu}/scaling_max_freq").read_text()
            for cpu in range(16)] == [str(5_500_000 + (cpu % 2) * 100_000) for cpu in range(16)]


@pytest.fixture
def lifecycle(tmp_path):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is required")
    power = tmp_path / "power"
    _power_tree(power)
    events = tmp_path / "events.jsonl"
    release = tmp_path / "release"
    trace = tmp_path / "restore.trace"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "systemd-inhibit").write_text("#!/bin/sh\nexit 1\n")
    (fake_bin / "systemd-inhibit").chmod(0o755)
    fake_python = tmp_path / "campaign-python"
    fake_python.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, pathlib, signal, sys, time

        args = sys.argv[1:]
        root = pathlib.Path(os.environ["FAKE_POWER_ROOT"])
        events = pathlib.Path(os.environ["FAKE_EVENTS"])
        release = pathlib.Path(os.environ["FAKE_RELEASE"])

        def record(kind):
            values = {{
                "kind": kind,
                "no_turbo": (root / "sys/devices/system/cpu/intel_pstate/no_turbo").read_text(),
                "pl1": (root / "sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw").read_text(),
                "pl2": (root / "sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw").read_text(),
                "freqs": [(root / f"sys/devices/system/cpu/cpufreq/policy{{cpu}}/scaling_max_freq").read_text()
                          for cpu in range(16)],
            }}
            with events.open("a") as handle:
                handle.write(json.dumps(values) + "\\n")

        module = args[args.index("-m") + 1]
        if module.endswith(".monitor"):
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
            while True: time.sleep(0.2)

        record("dry" if "--dry-run" in args else "check" if "--check-only" in args else "scheduler")
        assert "--production-power" in args
        assert (root / "sys/devices/system/cpu/intel_pstate/no_turbo").read_text() == "0"
        assert (root / "sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw").read_text() == "125000000"
        assert (root / "sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw").read_text() == "160000000"
        assert all((root / f"sys/devices/system/cpu/cpufreq/policy{{cpu}}/scaling_max_freq").read_text()
                   == "4200000" for cpu in range(16))
        if "--dry-run" in args or "--check-only" in args:
            sys.exit(0)
        if os.environ.get("FAKE_STARTUP_FAIL") == "1":
            sys.exit(9)
        time.sleep(float(os.environ.get("FAKE_STARTUP_DELAY", "0.8")))
        output = pathlib.Path(args[args.index("--output-root") + 1])
        campaign = args[args.index("--campaign-id") + 1]
        run = output / campaign
        run.mkdir(parents=True, exist_ok=True)
        (run / "manifest.json").write_text(json.dumps({{"campaign_id": campaign, "status": "running"}}))
        (run / "scheduler.pgid").write_text(str(os.getpgrp()) + "\\n")
        def stopped(*_):
            (run / "manifest.json").write_text(json.dumps({{"campaign_id": campaign, "status": "stopped"}}))
            sys.exit(130)
        signal.signal(signal.SIGINT, stopped)
        signal.signal(signal.SIGTERM, stopped)
        while not release.exists(): time.sleep(0.05)
        (run / "manifest.json").write_text(json.dumps({{"campaign_id": campaign, "status": "complete"}}))
    """))
    fake_python.chmod(0o755)
    tmux_tmp = tmp_path / "tmux"
    tmux_tmp.mkdir(mode=0o700)
    output = tmp_path / "campaigns"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "EVODEVO_POWER_CONTROL_TEST_ROOT": str(power),
        "EVODEVO_ALLOW_UNPRIVILEGED_POWER_TEST": "1",
        "EVODEVO_POWER_CONTROL_TRACE": str(trace),
        "EVODEVO_CAMPAIGN_PYTHON": str(fake_python),
        "EVODEVO_SCHEDULER_START_TIMEOUT_S": "5",
        "FAKE_POWER_ROOT": str(power),
        "FAKE_EVENTS": str(events),
        "FAKE_RELEASE": str(release),
        "TMUX_TMPDIR": str(tmux_tmp),
    }

    def start(campaign="lifecycle", *extra):
        command = ["bash", str(POWER_WRAPPER), "--turbo", "--p-max-mhz", "4200",
                   "--p-cpus", "0-15", "--pl1-w", "125", "--pl2-w", "160", "--run-as-root", "--",
                   "bash", str(CAMPAIGN_LAUNCHER), "--campaign-id", campaign, "--output-root", str(output),
                   "--no-mirror", "--mode", "real", "--workers", "18",
                   "--cpu-profile", "thermal-validated", "--wait-for-campaign", *extra]
        return subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    yield {"power": power, "events": events, "release": release, "trace": trace,
           "output": output, "env": env, "start": start}
    subprocess.run(["tmux", "kill-server"], env=env, capture_output=True)


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_controls_wait_for_delayed_scheduler_and_normal_completion(lifecycle):
    process = lifecycle["start"]()
    _wait_for(lambda: any(row["kind"] == "scheduler" for row in _records(lifecycle["events"])))
    # This is the original failure window: tmux exists, but scheduler.pgid has
    # not been written.  The guardian must still be alive and controls held.
    assert process.poll() is None
    _assert_applied(lifecycle["power"])
    run = lifecycle["output"] / "lifecycle"
    assert not (run / "scheduler.pgid").exists()
    _wait_for(lambda: (run / "control_guardian.pid").exists())
    assert int((run / "control_guardian.pid").read_text()) == process.pid
    assert subprocess.run(["tmux", "has-session", "-t", "evodevo-campaign"],
                          env=lifecycle["env"], capture_output=True).returncode == 0
    assert "evodevo-campaign" in subprocess.run(["tmux", "ls"], env=lifecycle["env"],
                                                  capture_output=True, text=True, check=True).stdout
    assert process.poll() is None
    lifecycle["release"].touch()
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, stdout + stderr
    _assert_restored(lifecycle["power"])
    assert lifecycle["trace"].read_text().splitlines() == ["restore"]
    assert not (run / "control_guardian.pid").exists()
    assert not (run / "scheduler.pgid").exists()
    assert subprocess.run(["tmux", "has-session", "-t", "evodevo-campaign"],
                          env=lifecycle["env"], capture_output=True).returncode != 0
    scheduler = next(row for row in _records(lifecycle["events"]) if row["kind"] == "scheduler")
    assert (scheduler["no_turbo"], scheduler["pl1"], scheduler["pl2"]) == (
        "0", "125000000", "160000000")
    assert scheduler["freqs"] == ["4200000"] * 16


def test_ssh_hangup_does_not_release_controls(lifecycle):
    process = lifecycle["start"]("hangup")
    run = lifecycle["output"] / "hangup"
    _wait_for(lambda: (run / "control_guardian.pid").exists())
    process.send_signal(signal.SIGHUP)
    time.sleep(0.3)
    assert process.poll() is None
    _assert_applied(lifecycle["power"])
    lifecycle["release"].touch()
    process.communicate(timeout=15)
    assert process.returncode == 0
    _assert_restored(lifecycle["power"])


def test_scheduler_startup_failure_restores_once_without_orphan(lifecycle):
    lifecycle["env"]["FAKE_STARTUP_FAIL"] = "1"
    process = lifecycle["start"]("startup-fail")
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode != 0
    assert "failed before its lifecycle acknowledgement" in stderr
    _assert_restored(lifecycle["power"])
    assert lifecycle["trace"].read_text().splitlines() == ["restore"]
    assert not (lifecycle["output"] / "startup-fail/control_guardian.pid").exists()
    assert not (lifecycle["output"] / "startup-fail/scheduler.pgid").exists()
    assert subprocess.run(["tmux", "has-session", "-t", "evodevo-campaign"],
                          env=lifecycle["env"], capture_output=True).returncode != 0


def test_controlled_stop_restores_and_reaps_guardian(lifecycle):
    process = lifecycle["start"]("controlled-stop")
    run = lifecycle["output"] / "controlled-stop"
    _wait_for(lambda: (run / "control_guardian.pid").exists())
    stopped = subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign", "controlled-stop",
                              "--output-root", str(lifecycle["output"]), "--grace-s", "2"],
                             cwd=REPO_ROOT, env=lifecycle["env"], capture_output=True, text=True, timeout=15)
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    process.communicate(timeout=15)
    assert process.returncode == 0
    _assert_restored(lifecycle["power"])
    assert lifecycle["trace"].read_text().splitlines() == ["restore"]
    assert not (run / "control_guardian.pid").exists()
    assert not (run / "scheduler.pgid").exists()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_guardian_signal_restores_exactly_once(lifecycle, sig):
    process = lifecycle["start"](f"signal-{sig.name.lower()}")
    run = lifecycle["output"] / f"signal-{sig.name.lower()}"
    _wait_for(lambda: (run / "control_guardian.pid").exists())
    process.send_signal(sig)
    process.communicate(timeout=15)
    assert process.returncode != 0
    _assert_restored(lifecycle["power"])
    assert lifecycle["trace"].read_text().splitlines() == ["restore"]
    assert not (run / "control_guardian.pid").exists()
    assert not (run / "scheduler.pgid").exists()


def test_production_dry_run_applies_verifies_restores_without_campaign(lifecycle):
    command = ["bash", str(POWER_WRAPPER), "--turbo", "--p-max-mhz", "4200", "--p-cpus", "0-15",
               "--pl1-w", "125", "--pl2-w", "160", "--run-as-root", "--",
               "bash", str(CAMPAIGN_LAUNCHER), "--dry-run", "--campaign-id", "dry",
               "--output-root", str(lifecycle["output"]), "--no-mirror", "--mode", "real",
               "--workers", "18", "--cpu-profile", "thermal-validated"]
    done = subprocess.run(command, cwd=REPO_ROOT, env=lifecycle["env"], capture_output=True, text=True, timeout=15)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [row["kind"] for row in _records(lifecycle["events"])] == ["dry"]
    _assert_restored(lifecycle["power"])
    assert lifecycle["trace"].read_text().splitlines() == ["restore"]
    assert not lifecycle["output"].exists()
    assert subprocess.run(["tmux", "has-session", "-t", "evodevo-campaign"],
                          env=lifecycle["env"], capture_output=True).returncode != 0


def test_production_wrapper_forces_user_default_tmux_and_strips_test_seams():
    text = (REPO_ROOT / "scripts/campaign/run_production_campaign.sh").read_text()
    assert '[[ "$SUDO_USER" == ubuntu ]]' in text
    assert '--run-as-user "$SUDO_USER"' in text
    assert "unset EVODEVO_POWER_CONTROL_TEST_ROOT" in text
    assert "EVODEVO_CAMPAIGN_PYTHON" in text
    assert "EVODEVO_TMUX_SOCKET TMUX TMUX_TMPDIR" in text
    assert "env -u TMUX -u TMUX_TMPDIR -u EVODEVO_TMUX_SOCKET" in text
