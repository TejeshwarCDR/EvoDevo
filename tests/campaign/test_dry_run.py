"""`--dry-run` writes NOTHING and prints the plan, calendar, disk, mirror,
notifier, placement and launch-check status (python entry point and
run_campaign.sh)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _listing(*roots):
    entries = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            stat = path.lstat()
            entries[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return entries


def _git_status():
    return subprocess.run(["git", "status", "--porcelain", "--ignored"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout


def test_run_script_dry_run_writes_nothing_and_prints_everything(tmp_path):
    output_root, mirror = tmp_path / "campaigns", tmp_path / "mirror"
    output_root.mkdir()
    mirror.mkdir()
    notify = tmp_path / "notify.yaml"
    notify.write_text("shell_command: 'true'\n")
    socket = f"evodevo-dry-{os.getpid()}"
    env = dict(os.environ, EVODEVO_TMUX_SOCKET=socket, EVODEVO_NOTIFY_CONFIG=str(notify))
    before, git_before = _listing(tmp_path, REPO_ROOT / "evodevo", REPO_ROOT / "scripts"), _git_status()
    result = subprocess.run(["bash", "scripts/campaign/run_campaign.sh", "--dry-run", "--output-root",
                             str(output_root), "--mirror-dir", str(mirror), "--launch-at", "2026-10-01T00:00"],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr
    assert _listing(tmp_path, REPO_ROOT / "evodevo", REPO_ROOT / "scripts") == before
    assert _git_status() == git_before
    assert subprocess.run(["tmux", "-L", socket, "has-session"], capture_output=True).returncode != 0
    out = result.stdout
    for needle in ("PLAN (D7 phase order) -- guard outcome: all four approved campaign arms",
                   "phase 1: Walker-v0: 20 runs", "ppo_timesteps=1,000,000; K (ppo_seeds)=1",
                   "phase 2: UpStepper-v0: 20 runs", "ppo_timesteps=1,200,000; K (ppo_seeds)=2",
                   "CALENDAR from launch 2026-10-01 00:00", "Walker milestone", "Walker-v0 phase end",
                   "campaign end", "2026-10-06", "2026-10-17", "config hash: ", "git commit: ", "dirty tree: ",
                   "DISK (", "main (", "mirror (", "MIRROR PHYSICAL DISK", "same physical disk: True",
                   "NOTIFIER:", "configured (shell_command)", "CPU PLACEMENT", "worker  0: cpu", "scheduler: cpus",
                   "LAUNCH CHECKS: WOULD BE REFUSED", "share physical disk"):
        assert needle in out, (needle, out)


def test_dry_run_of_a_non_campaign_design_shows_no_calendar(tmp_path):
    config = tmp_path / "small.yaml"
    config.write_text(
        "profile: campaign\nencoders: [direct]\ntasks: [Walker-v0]\nseeds: [0]\n"
        f"configs_root: {REPO_ROOT}/evodevo/experiments/configs\noutput_dir: results/runs\n"
        "surrogate: false\nfactors: {dev_steps: 100, init_condition: center_seed}\n"
        "profile_overrides: {population_size: 2, offspring_size: 2, eval_budget: 2, "
        "ppo_timesteps: {Walker-v0: 500}, ppo_seeds: {Walker-v0: 1}}\n")
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(tmp_path / "absent.yaml"), PYTHONDONTWRITEBYTECODE="1")
    before = _listing(tmp_path)
    result = subprocess.run([sys.executable, "-m", "evodevo.campaign", "--dry-run", "--config", str(config),
                             "--output-root", str(tmp_path / "out"), "--no-mirror"], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr
    assert _listing(tmp_path) == before and not (tmp_path / "out").exists()
    assert "NOT APPLICABLE to this plan" in result.stdout and "CALENDAR" not in result.stdout
    assert "NOT CONFIGURED" in result.stdout and "mirror: DISABLED" in result.stdout
