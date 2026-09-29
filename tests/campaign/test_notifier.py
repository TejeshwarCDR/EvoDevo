"""Notifier truthfulness: `alerts --test` exits 0 only after verified
delivery, and a REAL launch is refused unless a notifier is configured or
--no-notify is passed explicitly (recorded, permanent dashboard warning)."""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from evodevo.campaign.alerts import NO_NOTIFIER_MESSAGE, AlertManager, Notifier
from evodevo.campaign.dashboard import collect, render
from evodevo.preflight.dirty_check import project_is_dirty

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _alerts_test(config, env_config=None):
    env = dict(os.environ)
    env.pop("EVODEVO_NOTIFY_CONFIG", None)
    if env_config is not None:
        env["EVODEVO_NOTIFY_CONFIG"] = str(env_config)
    argv = [sys.executable, "-m", "evodevo.campaign.alerts", "--test"]
    if config is not None:
        argv += ["--config", str(config)]
    return subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, env=env)


def test_unconfigured_test_exits_nonzero_with_exact_message(tmp_path):
    result = _alerts_test(tmp_path / "missing.yaml")
    assert result.returncode != 0
    assert NO_NOTIFIER_MESSAGE in result.stdout
    assert NO_NOTIFIER_MESSAGE == "NO NOTIFIER CONFIGURED - alerts go to alerts.jsonl only"
    empty = tmp_path / "empty.yaml"
    empty.write_text("ntfy_topic_url: ''\n")
    assert _alerts_test(empty).returncode != 0


def test_shell_command_delivery_exit_zero_only_on_success(tmp_path):
    received = tmp_path / "received.log"
    ok = tmp_path / "ok.yaml"
    ok.write_text(f"shell_command: [sh, -c, 'printf \"%s\\n\" \"$0\" >> {received}']\n")
    result = _alerts_test(None, env_config=ok)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[test] EvoDevo campaign notification" in received.read_text()
    failing = tmp_path / "fail.yaml"
    failing.write_text("shell_command: [sh, -c, 'exit 3']\n")
    result = _alerts_test(failing)
    assert result.returncode != 0
    assert "DELIVERY FAILED" in result.stdout


@pytest.mark.parametrize("status,expect_ok", [(200, True), (500, False)])
def test_ntfy_delivery_requires_http_2xx(tmp_path, status, expect_ok):
    bodies = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = tmp_path / "ntfy.yaml"
        config.write_text(f"ntfy_topic_url: http://127.0.0.1:{server.server_port}/evodevo\n")
        result = _alerts_test(config)
    finally:
        server.shutdown()
    assert (result.returncode == 0) is expect_ok, result.stdout + result.stderr
    assert bodies and b"[test]" in bodies[0]


def test_unconfigured_alerts_are_journal_only_and_still_deduplicated(tmp_path):
    clock = [1000.0]
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=Notifier({}), now=lambda: clock[0])
    first = alerts.evaluate("disk_main", True, {"free_gb": 10})
    assert first["notified"] is False and first["delivery"] == "journal_only"
    clock[0] += 60
    second = alerts.evaluate("disk_main", True, {"free_gb": 10})
    assert "delivery" not in second  # de-duplicated: no second notice within 6 h
    alerts.heartbeat({})
    clock[0] += 60
    alerts.inspect({"completed_ppo": 0}, p95_ecore_s=100)
    beats = [json.loads(l) for l in (tmp_path / "alerts.jsonl").read_text().splitlines()
             if json.loads(l)["condition"] == "daily_heartbeat"]
    assert len(beats) == 1  # journal-only heartbeat counts; no per-loop spam


def _check_only(tmp_path, *extra, notify_config):
    config = tmp_path / "smoke.yaml"
    config.write_text(
        "profile: campaign\nencoders: [direct]\ntasks: [Walker-v0]\nseeds: [0]\n"
        f"configs_root: {REPO_ROOT}/evodevo/experiments/configs\noutput_dir: results/runs\n"
        "surrogate: false\nfactors: {dev_steps: 100, init_condition: center_seed}\n"
        "profile_overrides: {population_size: 2, offspring_size: 2, eval_budget: 2, "
        "ppo_timesteps: {Walker-v0: 500}, ppo_seeds: {Walker-v0: 1}}\n")
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(notify_config))
    return subprocess.run([sys.executable, "-m", "evodevo.campaign", "--check-only", "--config", str(config),
                           "--output-root", str(tmp_path / "out"), "--campaign-id", "gate", "--mode", "real",
                           "--cpu-profile", "standard", "--no-mirror", *extra], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=180, env=env)


def test_real_launch_refused_without_notifier(tmp_path):
    result = _check_only(tmp_path, notify_config=tmp_path / "absent.yaml")
    assert result.returncode != 0
    assert NO_NOTIFIER_MESSAGE in result.stderr
    assert not (tmp_path / "out").exists()


def test_real_launch_allowed_with_configured_notifier(tmp_path):
    notify = tmp_path / "n.yaml"
    notify.write_text("shell_command: 'true'\n")
    result = _check_only(tmp_path, notify_config=notify)
    assert NO_NOTIFIER_MESSAGE not in result.stderr
    if project_is_dirty(REPO_ROOT):
        assert result.returncode != 0 and "dirty" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert '"configured": true' in result.stdout


def test_explicit_no_notify_is_allowed_recorded_and_warned(tmp_path):
    result = _check_only(tmp_path, "--no-notify", notify_config=tmp_path / "absent.yaml")
    assert NO_NOTIFIER_MESSAGE not in result.stderr
    if project_is_dirty(REPO_ROOT):
        assert result.returncode != 0 and "dirty" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert '"no_notify": true' in result.stdout
    root = tmp_path / "c"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"campaign_id": "c", "no_notify": True}))
    assert "WARNING: --no-notify" in render(collect(str(root)))


def test_run_script_refuses_without_notifier_before_creating_tmux(tmp_path):
    socket = f"evodevo-notify-{os.getpid()}"
    env = dict(os.environ, EVODEVO_TMUX_SOCKET=socket, EVODEVO_NOTIFY_CONFIG=str(tmp_path / "absent.yaml"))
    try:
        result = subprocess.run(["bash", "scripts/campaign/run_campaign.sh", "--campaign-id", "nonotify",
                                 "--output-root", str(tmp_path / "out"), "--no-mirror", "--cpu-profile", "standard", "--mode", "real"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=180, env=env)
        assert result.returncode != 0
        if not project_is_dirty(REPO_ROOT):
            assert NO_NOTIFIER_MESSAGE in result.stderr
        assert subprocess.run(["tmux", "-L", socket, "has-session", "-t", "evodevo-campaign"],
                              capture_output=True).returncode != 0
        assert not (tmp_path / "out" / "nonotify").exists()
    finally:
        subprocess.run(["tmux", "-L", socket, "kill-server"], capture_output=True)


def test_explicit_local_config_is_used_for_launch_checks(tmp_path):
    notify = tmp_path / "local.yaml"
    notify.write_text("shell_command: 'true'\n")
    config = tmp_path / "smoke.yaml"
    config.write_text(
        "profile: campaign\nencoders: [direct]\ntasks: [Walker-v0]\nseeds: [0]\n"
        f"configs_root: {REPO_ROOT}/evodevo/experiments/configs\noutput_dir: results/runs\n"
        "surrogate: false\nfactors: {dev_steps: 100, init_condition: center_seed}\n"
        "profile_overrides: {population_size: 2, offspring_size: 2, eval_budget: 2, "
        "ppo_timesteps: {Walker-v0: 500}, ppo_seeds: {Walker-v0: 1}}\n")
    result = subprocess.run([sys.executable, "-m", "evodevo.campaign", "--check-only", "--config", str(config),
                             "--output-root", str(tmp_path / "out"), "--campaign-id", "local", "--mode", "real",
                             "--cpu-profile", "standard", "--no-mirror", "--local-config", str(notify)], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=180)
    if project_is_dirty(REPO_ROOT):
        assert "dirty" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert str(notify) in result.stdout
