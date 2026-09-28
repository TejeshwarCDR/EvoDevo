import json
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

from evodevo.campaign.alerts import AlertManager, REN_NOTIFY_S
from evodevo.campaign.dashboard import collect, render
from evodevo.campaign.monitor import SNAPSHOT_S, monitor_once, slowest_ecore_p95_s
from evodevo.campaign.mirror import sync, verify_mirror, verify_checksum_manifest
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


def test_heartbeat_and_phase_notice_deduplication(tmp_path):
    clock = [1000.0]
    spy = SpyNotifier()
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=spy, now=lambda: clock[0])
    snapshot = {"phase": "Walker-v0", "generation": {("direct", 0): 2, ("direct", 1): 4},
                "completed_evaluations": 50, "campaign_eta_s": (100, 200),
                "infra_failure_count": 0, "main_disk_free_gb": 200,
                "mirror_disk_free_gb": 300, "last_mirror_success": 900,
                "temperature_c": 90, "critical_temp_c": 100}
    alerts.heartbeat(snapshot)
    assert len(spy.messages) == 1
    row = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[-1])
    assert row["details"]["per_arm_mean_generation"] == {"direct": 3}
    assert row["details"]["max_temperature_24h_c"] is None  # no inspect sample yet
    alerts.phase_complete("Walker-v0")
    alerts.phase_complete("Walker-v0")
    alerts.phase_complete("UpStepper-v0")
    assert len(spy.messages) == 3
    clock[0] += 23 * 3600
    alerts.inspect(snapshot, p95_ecore_s=100)
    assert len(spy.messages) == 3
    clock[0] += 3600
    alerts.inspect(snapshot, p95_ecore_s=100)
    assert len(spy.messages) == 4
    restarted = AlertManager(tmp_path / "alerts.jsonl", notifier=spy, now=lambda: clock[0])
    restarted.inspect(snapshot, p95_ecore_s=100)
    assert len(spy.messages) == 4


def test_failed_heartbeat_is_journaled_and_retried(tmp_path):
    class BrokenNotifier:
        def send(self, _message):
            raise RuntimeError("offline")
    clock = [1000.0]
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=BrokenNotifier(), now=lambda: clock[0])
    alerts.heartbeat({"phase": "Walker-v0", "generation": {}})
    clock[0] += 60
    alerts.heartbeat({"phase": "Walker-v0", "generation": {}})
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(not row["notified"] and "offline" in row["notification_error"] for row in rows)


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


def test_temperature_throttle_requires_tjmax_proximity(tmp_path):
    clock = [1000.0]
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=SpyNotifier(), now=lambda: clock[0])
    snapshot = {"temperature_c": 60, "critical_temp_c": 100,
                "thermal_throttle_count": 100, "main_disk_free_gb": 200,
                "mirror_disk_free_gb": 200}
    alerts.inspect(snapshot, p95_ecore_s=100)
    clock[0] += 60
    snapshot["thermal_throttle_count"] = 101
    alerts.inspect(snapshot, p95_ecore_s=100)
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    temperature = [row for row in rows if row["condition"] == "temperature"]
    assert temperature[-1]["details"]["throttled"] is False


def test_partial_stall_one_hour_window_survives_monitor_restart(tmp_path):
    clock = [1000.0]
    spy = SpyNotifier()
    path = tmp_path / "alerts.jsonl"
    snapshot = {"throughput_now_per_h": 4, "hourly_rates_24h": [20] * 24,
                "main_disk_free_gb": 200, "mirror_disk_free_gb": 200,
                "swap_used_gb": 0, "completed_ppo": 100, "infra_failure_count": 0}
    alerts = AlertManager(path, notifier=spy, now=lambda: clock[0])
    alerts.inspect(snapshot, p95_ecore_s=100)
    clock[0] += 1800
    restarted = AlertManager(path, notifier=spy, now=lambda: clock[0])
    restarted.inspect(snapshot, p95_ecore_s=100)
    assert not any("partial_stall" in message for message in spy.messages)
    clock[0] += 1801
    restarted.inspect(snapshot, p95_ecore_s=100)
    assert sum("partial_stall" in message for message in spy.messages) == 1


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


def test_run_script_help_and_existing_session_attach(tmp_path):
    import shutil
    if shutil.which("tmux") is None:
        return
    script = Path("scripts/campaign/run_campaign.sh").resolve()
    socket = f"evodevo-test-{os.getpid()}-{time.time_ns()}"
    env = {**os.environ, "EVODEVO_TMUX_SOCKET": socket}
    try:
        help_result = subprocess.run(["bash", str(script), "--help"], env=env,
                                     capture_output=True, text=True, check=True)
        assert "Usage:" in help_result.stdout
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        refused = subprocess.run(["bash", str(script), "--campaign-id", "no-mirror",
                                  "--output-root", str(tmp_path)], env=env,
                                 capture_output=True, text=True)
        assert refused.returncode != 0
        assert "--mirror-dir is required" in refused.stderr
        conflict = subprocess.run(["bash", str(script), "--campaign-id", "conflict",
                                   "--mirror-dir", str(tmp_path), "--no-mirror"], env=env,
                                  capture_output=True, text=True)
        assert conflict.returncode != 0
        assert "mutually exclusive" in conflict.stderr
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", "evodevo-campaign", "sleep 30"],
                       check=True)
        attached = subprocess.run(["bash", str(script), "--no-mirror"], env=env,
                                  capture_output=True, text=True, check=True)
        assert "already running" in attached.stdout
    finally:
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)


def test_dashboard_plain_and_tty_render_eta_range():
    snapshot = {"campaign_id": "x", "phase": "Walker-v0", "elapsed_s": 3600,
                "phase_completed": 10, "phase_total": 300, "completed_evaluations": 10,
                "total_evaluations": 600, "completed_ppo": 10,
                "generation": {("direct", 0): 2}, "tasks_running": 1, "tasks_queued": 5,
                "throughput_now_per_h": 10, "throughput_average_per_h": 10,
                "phase_eta_s": (100, 200), "campaign_eta_s": (300, 400),
                "eta_source": "prior", "walker_milestone_epoch": None,
                "p2_mean_h": 403.2, "p2_p95_h": 406, "projection_exceeds_432h": False,
                "infra_failure_count": 0, "last_error": None, "seconds_since_last_task": 5,
                "projection_total_h": (402, 407), "mirror_disabled": True}
    assert "average 10.0" in render(snapshot, tty=False)
    assert "phase 0.0 h–0.1 h" in render(snapshot, tty=True)
    assert "projected total 402.0–407.0 h" in render(snapshot)
    assert "mirror explicitly disabled" in render(snapshot)


def test_dashboard_prior_then_per_arm_core_class_measured_eta(tmp_path):
    task = "Walker-v0"
    runs = [{"arm": arm, "seed": 0, "config": {"eval_budget": 100, "ppo_seeds": 1}}
            for arm in ("direct", "nca")]
    manifest = {"campaign_id": "synthetic", "started_at": 1000,
                "status": "running", "phase_status": {task: {"status": "running", "started_at": 1000}},
                "plan": {"phases": [{"task": task, "runs": runs}]},
                "worker_slots": {"slots": [[0], [1]]},
                "topology": {"cores": [{"logical_id": 0, "kind": "P"},
                                        {"logical_id": 1, "kind": "E"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for arm in ("direct", "nca"):
        directory = tmp_path / "runs" / arm / task / "seed_0"
        directory.mkdir(parents=True)
        (directory / "run_log.jsonl").write_text("".join(json.dumps(
            {"eval_index": index, "generation": index // 5, "valid": True}) + "\n"
            for index in range(49)))
    journal = tmp_path / "journal.jsonl"
    rows = []
    for arm, base in (("direct", 100), ("nca", 200)):
        for index in range(49):
            rows.append({"run_key": f"{arm}/{task}/seed_0", "eval_index": index,
                         "ppo_k": 0, "completed_at": 1900,
                         "result": {"wall_clock_s": base * (2 if index % 2 else 1),
                                    "cpu_ids": [index % 2], "completed": True}})
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prior = collect(str(tmp_path), now=2000, telemetry={})
    assert prior["eta_source"] == "prior"
    assert prior["seconds_since_last_task"] == 100
    assert abs(prior["throughput_average_per_h"] - 98 * 3.6) < 1e-9
    for arm in ("direct", "nca"):
        path = tmp_path / "runs" / arm / task / "seed_0" / "run_log.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps({"eval_index": 49, "generation": 9, "valid": True}) + "\n")
        rows.append({"run_key": f"{arm}/{task}/seed_0", "eval_index": 49,
                     "ppo_k": 0, "completed_at": 1900,
                     "result": {"wall_clock_s": (100 if arm == "direct" else 200) * 2,
                                "cpu_ids": [1], "completed": True}})
        # A prior phase's journal entries must never be mistaken for current
        # UpStepper/Walker work already completed for this run.
        rows.append({"run_key": f"{arm}/UpStepper-v0/seed_0", "eval_index": 0,
                     "ppo_k": 0, "completed_at": 900,
                     "result": {"wall_clock_s": 300, "cpu_ids": [0], "completed": True}})
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "dispatch_status.json").write_text(json.dumps(
        {"task": task, "run_generations": {f"direct/{task}/seed_0": 10},
         "tasks_running": 1, "tasks_queued": 3}))
    measured = collect(str(tmp_path), now=2000, telemetry={})
    assert measured["eta_source"] == "measured"
    assert measured["phase_eta_s"] == (10000.0, 10000.0)
    assert measured["projection_total_h"][0] == measured["projection_total_h"][1]
    assert measured["generation"][("direct", 0)] == 10
    assert measured["tasks_running"] == 1 and measured["tasks_queued"] == 3


def test_stall_timer_starts_before_first_completed_task_and_ecore_threshold(tmp_path):
    manifest = {"campaign_id": "empty", "started_at": 1000, "status": "running",
                "phase_status": {"UpStepper-v0": {"started_at": 1000, "status": "running"}},
                "plan": {"phases": [{"task": "UpStepper-v0", "runs": []}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    snapshot = collect(str(tmp_path), now=2000, telemetry={})
    assert snapshot["seconds_since_last_task"] == 1000
    expected = max((mean + 1.645 * sd) for key, (mean, sd) in
                   json.loads((Path("results/campaign_prep/final_estimate/final_estimate.json")).read_text())
                   ["inputs"]["t_ppo_mean_sd_at_b3_footing"].items()
                   if key.endswith("|UpStepper-v0")) * 2.3172
    assert abs(slowest_ecore_p95_s("UpStepper-v0") - expected) < 1e-9


def test_snapshot_failure_retries_without_waiting_another_twelve_hours(tmp_path, monkeypatch):
    task = "Walker-v0"
    manifest = {"campaign_id": "snapshot", "started_at": 1000, "status": "running",
                "no_mirror": True, "phase_status": {task: {"started_at": 1000, "status": "running"}},
                "plan": {"phases": [{"task": task, "runs": []}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    outcomes = iter(({"ok": False}, {"ok": True}))
    called = []

    def snapshot_runner(*args, **kwargs):
        called.append((args, kwargs))
        return next(outcomes)

    monkeypatch.setattr("evodevo.campaign.monitor.run_subprocess", snapshot_runner)
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=SpyNotifier(), now=lambda: 1000 + SNAPSHOT_S)
    first = monitor_once(tmp_path, alerts, now=1000 + SNAPSHOT_S, p95_ecore_s=100)
    assert first["last_snapshot"] == 1000
    second = monitor_once(tmp_path, alerts, now=1000 + SNAPSHOT_S + 60,
                          p95_ecore_s=100, last_snapshot=first["last_snapshot"])
    assert second["last_snapshot"] == 1000 + SNAPSHOT_S + 60
    assert len(called) == 2
    # Between D7 phases the next task is visible in the plan but has not
    # started; it must not trigger a premature snapshot of empty run logs.
    manifest["phase_status"] = {task: {"started_at": 1000, "status": "complete"}}
    manifest["plan"]["phases"].append({"task": "UpStepper-v0", "runs": []})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    transition = monitor_once(tmp_path, alerts, now=1000 + 2 * SNAPSHOT_S,
                              p95_ecore_s=100, last_snapshot=second["last_snapshot"])
    assert transition["last_snapshot"] == 1000 + 2 * SNAPSHOT_S
    assert len(called) == 2


def test_missing_mirror_alerts_before_next_six_hour_sync(tmp_path):
    task = "Walker-v0"
    manifest = {"campaign_id": "missing-mirror", "started_at": 1000, "status": "running",
                "mirror_dir": str(tmp_path / "unmounted"), "no_mirror": False,
                "phase_status": {task: {"started_at": 1000, "status": "running"}},
                "plan": {"phases": [{"task": task, "runs": []}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=SpyNotifier(), now=lambda: 1100)
    monitor_once(tmp_path, alerts, now=1100, p95_ecore_s=100, last_mirror=1099)
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert any(row["condition"] == "mirror_failed" and row["status"] == "active" for row in rows)


def test_mirror_checksum_and_failure_alert(tmp_path, monkeypatch):
    source = tmp_path / "campaign"
    source.mkdir()
    (source / "journal.jsonl").write_text('{"completed": true}\n')
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=SpyNotifier())
    absent = sync(source, tmp_path / "not-mounted", alert_manager=alerts)
    assert not absent["ok"] and "mirror directory absent" in absent["error"]
    assert not (tmp_path / "not-mounted").exists()
    result = sync(source, mirror, alert_manager=alerts)
    assert result["ok"]
    assert result["manifest_verified"]
    assert verify_mirror(source, mirror / "campaign")["ok"]
    assert verify_checksum_manifest(mirror / "campaign")["ok"]
    (mirror / "campaign" / "journal.jsonl").write_text("corrupted")
    assert not verify_mirror(source, mirror / "campaign")["ok"]
    assert not verify_checksum_manifest(mirror / "campaign")["ok"]
    import evodevo.campaign.mirror as mirror_module
    actual_run = mirror_module.subprocess.run
    copies = [0]

    def advancing_source(command, *args, **kwargs):
        result = actual_run(command, *args, **kwargs)
        if command[0] == "rsync" and "--delete" in command:
            copies[0] += 1
            if copies[0] == 1:
                with (source / "journal.jsonl").open("a") as handle:
                    handle.write('{"completed": false}\n')
        return result

    monkeypatch.setattr(mirror_module.subprocess, "run", advancing_source)
    recovered = sync(source, mirror, alert_manager=alerts)
    assert recovered["ok"] and copies[0] == 2
    assert verify_checksum_manifest(mirror / "campaign")["ok"]
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
    omnibus = list(__import__("csv").DictReader((folder / "stats/omnibus.csv").open()))
    assert abs(float(omnibus[0]["statistic"]) - 20 / 3) < 1e-10
    direct_nca = next(r for r in pairwise if r["arm_a"] == "direct" and r["arm_b"] == "nca")
    assert float(direct_nca["a12"]) == 0.0
    assert abs(float(direct_nca["p_raw"]) - 1 / 3) < 1e-10
    spy = SpyNotifier()
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=spy)
    class Stale:
        returncode = 0
        stderr = ""
        stdout = ""
    monkeypatch.setattr("evodevo.campaign.milestone.subprocess.run", lambda *a, **k: Stale())
    stale = run_subprocess(tmp_path, task, alerts)
    assert not stale["ok"]
    assert "fresh package_manifest.json for this invocation" in stale["missing"]
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
    subprocess.run(["tmux", "-L", socket, "set-environment", "-t", "evodevo-campaign",
                    "EVODEVO_CAMPAIGN_ID", campaign], check=True)
    process = subprocess.Popen(["setsid", sys.executable, "-m", "evodevo.campaign.monitor",
                                "--campaign", campaign, "--output-root", str(tmp_path),
                                "--interval-s", "1"])
    (folder / "scheduler.pgid").write_text(str(process.pid))
    environment = {**os.environ, "EVODEVO_TMUX_SOCKET": socket}
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if subprocess.run(["pgrep", "-af", f"evodevo.campaign.monitor.*{campaign}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("surrogate monitor did not start")
        stopped = subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign",
                                  campaign, "--output-root", str(tmp_path), "--grace-s", "0"],
                                 env=environment, capture_output=True, text=True, timeout=20)
        assert stopped.returncode == 0, stopped.stderr + stopped.stdout
        process.wait(timeout=5)
        assert process.returncode != 0
        assert subprocess.run(["pgrep", "-af", f"evodevo.campaign.monitor.*{campaign}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        subprocess.run(["tmux", "-L", socket, "kill-server"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_stop_does_not_remove_another_campaigns_tmux_session(tmp_path):
    socket = f"evodevo-owner-test-{os.getpid()}-{time.time_ns()}"
    env = {**os.environ, "EVODEVO_TMUX_SOCKET": socket}
    try:
        subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", "evodevo-campaign",
                        "sleep 30"], check=True)
        subprocess.run(["tmux", "-L", socket, "set-environment", "-t", "evodevo-campaign",
                        "EVODEVO_CAMPAIGN_ID", "current"], check=True)
        stopped = subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign",
                                  "old", "--output-root", str(tmp_path), "--grace-s", "0"],
                                 env=env, capture_output=True, text=True, timeout=10)
        assert stopped.returncode == 0
        assert "Leaving tmux session intact" in stopped.stderr
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    finally:
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)


def test_launcher_starts_and_cleans_tiny_surrogate_campaign():
    import shutil
    import tempfile
    if shutil.which("tmux") is None:
        return
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="evodevo-launch-") as directory:
        root = Path(directory)
        config = root / "benchmark.yaml"
        config.write_text(
            "profile: campaign\n"
            "encoders: [direct]\n"
            "tasks: [Walker-v0]\n"
            "seeds: [0]\n"
            f"configs_root: {repo / 'evodevo/experiments/configs'}\n"
            "output_dir: results/runs\n"
            "surrogate: true\n"
            "factors: {dev_steps: 100, init_condition: center_seed}\n"
            "profile_overrides: {population_size: 2, offspring_size: 2, eval_budget: 2, "
            "ppo_timesteps: {Walker-v0: 2000}, ppo_seeds: {Walker-v0: 1}}\n")
        socket = f"evodevo-launch-test-{os.getpid()}-{time.time_ns()}"
        env = {**os.environ, "EVODEVO_TMUX_SOCKET": socket}
        run_root = root / "campaigns"
        try:
            started = subprocess.run(["bash", "scripts/campaign/run_campaign.sh", "--campaign-id", "smoke",
                "--config", str(config), "--output-root", str(run_root), "--no-mirror",
                "--workers", "1", "--mode", "surrogate"], cwd=repo, env=env,
                capture_output=True, text=True, timeout=20)
            assert started.returncode == 0, started.stderr + started.stdout
            deadline = time.monotonic() + 30
            manifest = run_root / "smoke" / "manifest.json"
            while time.monotonic() < deadline:
                if manifest.exists() and json.loads(manifest.read_text()).get("status") == "complete":
                    break
                time.sleep(0.1)
            else:
                transcript = run_root / "smoke" / "transcript.log"
                detail = transcript.read_text() if transcript.exists() else "no transcript"
                raise AssertionError(f"surrogate launcher did not finish: {started.stdout}\n{detail}")
            assert (run_root / "smoke" / "runs" / "direct" / "Walker-v0" / "seed_0" / "run_log.jsonl").exists()
            completed_manifest = json.loads(manifest.read_text())
            assert completed_manifest["no_mirror"] is True
            assert completed_manifest["milestone_status"]["Walker-v0"]["ok"] is True
            assert (run_root / "smoke" / "milestones" / "Walker-v0" / "MILESTONE_REPORT.md").is_file()
            assert collect(str(run_root / "smoke"), telemetry={})["mirror_disabled"] is True
        finally:
            subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign", "smoke",
                "--output-root", str(run_root), "--grace-s", "0"], cwd=repo, env=env,
                capture_output=True, text=True, timeout=20)
            subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        config.write_text(config.read_text().replace("eval_budget: 2", "eval_budget: 3"))
        mismatch = subprocess.run(["bash", "scripts/campaign/run_campaign.sh", "--resume", "smoke",
            "--config", str(config), "--output-root", str(run_root), "--no-mirror",
            "--workers", "1", "--mode", "surrogate"], cwd=repo, env=env,
            capture_output=True, text=True, timeout=20)
        assert mismatch.returncode != 0
        assert "config-hash mismatch" in mismatch.stderr
