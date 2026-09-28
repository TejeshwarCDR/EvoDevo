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
    omnibus = list(__import__("csv").DictReader((folder / "stats/omnibus.csv").open()))
    assert abs(float(omnibus[0]["statistic"]) - 20 / 3) < 1e-10
    direct_nca = next(r for r in pairwise if r["arm_a"] == "direct" and r["arm_b"] == "nca")
    assert float(direct_nca["a12"]) == 0.0
    assert abs(float(direct_nca["p_raw"]) - 1 / 3) < 1e-10
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
        finally:
            subprocess.run(["bash", "scripts/campaign/stop_campaign.sh", "--campaign", "smoke",
                "--output-root", str(run_root), "--grace-s", "0"], cwd=repo, env=env,
                capture_output=True, text=True, timeout=20)
            subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
