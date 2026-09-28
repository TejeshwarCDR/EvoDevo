import json
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

from evodevo.campaign.alerts import AlertManager, REN_NOTIFY_S
from evodevo.campaign.dashboard import render
from evodevo.campaign.mirror import sync, verify_mirror
from evodevo.campaign.milestone import build, REQUIRED, run_subprocess


class SpyNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def test_alert_rate_limit_and_clear_are_durable(tmp_path):
    clock = [1000.0]
    spy = SpyNotifier()
    path = tmp_path / "alerts.jsonl"
    alerts = AlertManager(path, notifier=spy, now=lambda: clock[0])
    alerts.evaluate("disk_main", True, {"free_gb": 40})
    clock[0] += 100
    alerts.evaluate("disk_main", True, {"free_gb": 39})
    clock[0] += REN_NOTIFY_S
    alerts.evaluate("disk_main", True, {"free_gb": 38})
    alerts.evaluate("disk_main", False, {"free_gb": 80})
    alerts.evaluate("disk_main", False, {"free_gb": 81})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 5
    assert [r["notified"] for r in rows] == [True, False, True, True, False]
    assert len(spy.messages) == 3
    assert "cleared" in spy.messages[-1]
    restarted = AlertManager(path, notifier=spy, now=lambda: clock[0])
    restarted.evaluate("disk_main", True, {"free_gb": 40})
    restarted.evaluate("disk_main", True, {"free_gb": 39})
    assert len(spy.messages) == 4  # new trigger after a cleared condition


def test_all_alert_conditions_and_daily_heartbeat(tmp_path):
    clock = [10000.0]
    spy = SpyNotifier()
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=spy, now=lambda: clock[0])
    data = {"seconds_since_last_task": 500, "temperature_c": 99,
            "thermal_throttle_count": 100, "critical_temp_c": 100,
            "main_disk_free_gb": 40, "mirror_disk_free_gb": 40,
            "completed_ppo": 100, "infra_failure_count": 2,
            "swap_used_gb": 2, "throughput_now_per_h": 1,
            "hourly_rates_24h": [20] * 24, "phase": "Walker-v0",
            "generation": {("direct", 0): 3}, "completed_evaluations": 50,
            "campaign_eta_s": (100, 200), "last_mirror_success": 9000}
    alerts.inspect(data, p95_ecore_s=100, mirror_ok=False)
    clock[0] += 3601
    data["thermal_throttle_count"] = 101
    alerts.inspect(data, p95_ecore_s=100, mirror_ok=False)
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    active = {r["condition"] for r in rows if r["status"] == "active"}
    assert {"stall", "temperature", "disk_main", "disk_mirror", "infra_failure_rate",
            "swap", "mirror_failed", "partial_stall", "daily_heartbeat"} <= active
    temp = next(r for r in reversed(rows) if r["condition"] == "temperature")
    assert temp["details"]["throttled"] is True


def test_alert_test_cli_calls_configured_mock_target(tmp_path):
    target = tmp_path / "notifier.py"
    output = tmp_path / "received.txt"
    target.write_text("import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.argv[2])\n")
    config = tmp_path / "campaign.local.yaml"
    config.write_text(f"shell_command: '{sys.executable} {target} {output}'\n")
    done = subprocess.run([sys.executable, "-m", "evodevo.campaign.alerts", "--test",
                           "--config", str(config)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "[test]" in output.read_text()


def test_dashboard_plain_and_tty_render_eta_range():
    snapshot = {"campaign_id": "x", "phase": "Walker-v0", "elapsed_s": 3600,
                "phase_completed": 10, "phase_total": 300, "completed_evaluations": 10,
                "total_evaluations": 600, "completed_ppo": 10,
                "generation": {("direct", 0): 2}, "tasks_running": 1, "tasks_queued": 5,
                "throughput_now_per_h": 10, "throughput_average_per_h": 10,
                "phase_eta_s": (100, 200), "campaign_eta_s": (300, 400),
                "eta_source": "prior", "walker_milestone_epoch": None,
                "p2_mean_h": 403.2, "p2_p95_h": 406, "projection_exceeds_432h": False,
                "infra_failure_count": 0, "last_error": None, "seconds_since_last_task": 5}
    assert "average 10.0" in render(snapshot, tty=False)
    assert "phase 0.0 h–0.1 h" in render(snapshot, tty=True)


def test_mirror_checksum_and_failure_alert(tmp_path):
    source = tmp_path / "campaign"
    source.mkdir()
    (source / "journal.jsonl").write_text('{"completed": true}\n')
    mirror = tmp_path / "mirror"
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=SpyNotifier())
    result = sync(source, mirror, alert_manager=alerts)
    assert result["ok"]
    assert verify_mirror(source, mirror / "campaign")["ok"]
    (mirror / "campaign" / "journal.jsonl").write_text("corrupted")
    assert not verify_mirror(source, mirror / "campaign")["ok"]
    failed = sync(source, source / "nested", alert_manager=alerts)
    assert not failed["ok"]


def test_milestone_package_and_subprocess_failure_isolated(tmp_path, monkeypatch):
    task = "Walker-v0"
    arms = ["direct", "nca", "rule_based", "rule_based_monotone"]
    body = [[0] * 5 for _ in range(5)]
    body[2][2] = 3
    for arm_index, arm in enumerate(arms):
        for seed in (0, 1):
            directory = tmp_path / "runs" / arm / task / f"seed_{seed}"
            directory.mkdir(parents=True)
            row = {"eval_index": 0, "generation": 0, "fitness": arm_index + seed / 10,
                   "valid": True, "body_hash": f"{arm}-{seed}", "cache_hit": False,
                   "wall_clock_s": 1.0, "body": body, "per_seed_fitness": [arm_index + seed / 10]}
            (directory / "run_log.jsonl").write_text(json.dumps(row) + "\n")
    plan = {"phases": [{"task": task, "runs": [{"config": {"eval_budget": 1}} for _ in range(8)]}]}
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": plan}))
    outcome = build(tmp_path, task)
    folder = Path(outcome["output"])
    assert all((folder / name).exists() for name in REQUIRED)
    assert len(list((folder / "best_bodies").glob("*.npz"))) == 4
    assert "figures/fitness_curves.pdf" in outcome["files"]
    pairwise = list(__import__("csv").DictReader((folder / "stats/pairwise.csv").open()))
    direct_nca = next(r for r in pairwise if r["arm_a"] == "direct" and r["arm_b"] == "nca")
    assert float(direct_nca["a12"]) == 0.0
    spy = SpyNotifier()
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=spy)
    class Failed:
        returncode = 1
        stderr = "injected crash"
        stdout = ""
    monkeypatch.setattr("evodevo.campaign.milestone.subprocess.run", lambda *a, **k: Failed())
    result = run_subprocess(tmp_path, "UpStepper-v0", alerts)
    assert not result["ok"]
    assert any("milestone_failure" in message for message in spy.messages)


def test_stop_script_kills_process_group_and_tmux_session(tmp_path):
    campaign = "surrogate-stop-test"
    folder = tmp_path / campaign
    folder.mkdir()
    socket = f"evodevo-test-{os.getpid()}"
    subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", "evodevo-campaign",
                    "sleep 300"], check=True)
    process = subprocess.Popen(["setsid", "sleep", "300"])
    (folder / "scheduler.pgid").write_text(str(process.pid))
    environment = {**os.environ, "EVODEVO_TMUX_SOCKET": socket}
    try:
        stopped = subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign",
                                  campaign, "--output-root", str(tmp_path), "--grace-s", "0"],
                                 env=environment, capture_output=True, text=True, timeout=20)
        assert stopped.returncode == 0, stopped.stderr + stopped.stdout
        process.wait(timeout=5)
        assert process.returncode != 0
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        subprocess.run(["tmux", "-L", socket, "kill-server"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
