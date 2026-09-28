"""Guards for the campaign entry point (`python -m evodevo.campaign`).

Regression tests for the spawn/import bug found in the ab8f73d readiness
validation: `python -m evodevo.campaign.scheduler` made every spawned PPO
worker re-import evodevo.campaign/__init__ -> plan -> benchmark -> numpy
BEFORE its initializer's configure_threading(1, strict=True), so every
worker raised ThreadingConfiguredTooLateError and the pool died with
BrokenProcessPool. The earlier suite missed it because the only real-tmux
launch test ran --mode surrogate (no pool) and the real-PPO pool tests built
EvaluationPool directly under pytest's own __main__.

Every check runs in a FRESH subprocess: pytest itself has numpy loaded.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[2])
HEAVY = ("numpy", "torch", "evogym", "stable_baselines3")
HAS_SIMULATOR = importlib.util.find_spec("evogym.simulator_cpp") is not None


def _python(code, timeout=60):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.pop(var, None)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=timeout, env=env)


def test_campaign_package_and_pre_threading_imports_are_heavy_free():
    """Importing evodevo.campaign and every module __main__ imports before
    configure_threading leaves numpy/torch/evogym/SB3 out of sys.modules."""
    result = _python(f"""
        import sys
        import evodevo, evodevo.parallel, evodevo.parallel.threading_control
        import evodevo.campaign
        heavy = [m for m in {HEAVY!r} if m in sys.modules]
        assert not heavy, heavy
        print("PURE")
    """)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PURE" in result.stdout


def test_lazy_reexports_still_work():
    result = _python("""
        from evodevo.campaign import CampaignPlan, CampaignScheduler
        print(CampaignPlan.__name__, CampaignScheduler.__name__)
    """)
    assert result.returncode == 0, result.stderr
    assert "CampaignPlan CampaignScheduler" in result.stdout


def test_entry_point_import_order_regression():
    """Mirror of tests/preflight/test_cli.py's regression: trace every import
    made by importing evodevo.campaign.__main__ (which runs only
    configure_threading at import; main() is guarded by __name__)."""
    result = _python(f"""
        import builtins, sys
        order = []
        real_import = builtins.__import__
        def tracing(name, *args, **kwargs):
            if name not in order:
                order.append(name)
            return real_import(name, *args, **kwargs)
        builtins.__import__ = tracing
        import evodevo.campaign.__main__ as entry
        builtins.__import__ = real_import
        heavy = [m for m in {HEAVY!r} if m in sys.modules or any(n == m or n.startswith(m + ".") for n in order)]
        assert not heavy, (heavy, order[:60])
        assert "evodevo.parallel.threading_control" in order, order
        report = entry._THREADING_REPORT
        assert report.guard_passed and report.n_threads == 1, report
        import os
        from evodevo.parallel.threading_control import THREAD_ENV_VARS
        assert all(os.environ[v] in ("1", "FALSE") for v in THREAD_ENV_VARS), dict(os.environ)
        print("IMPORT_ORDER_OK")
    """)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_ORDER_OK" in result.stdout


def test_entry_point_refuses_when_numpy_already_loaded():
    result = _python("""
        import numpy
        import evodevo.campaign.__main__
    """)
    assert result.returncode != 0
    assert "ThreadingConfiguredTooLateError" in result.stderr


def test_scheduler_module_cannot_be_the_entry_point():
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    result = subprocess.run([sys.executable, "-m", "evodevo.campaign.scheduler", "--help"], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode != 0
    assert "python -m evodevo.campaign" in result.stderr


def test_launch_scripts_use_the_package_entry_point():
    for name in ("run_campaign.sh", "stop_campaign.sh", "status_campaign.sh"):
        text = (Path(REPO_ROOT) / "scripts" / "campaign" / name).read_text()
        assert "evodevo.campaign.scheduler" not in text, name
        assert "evodevo.campaign.dashboard" not in text, name
    assert "-m evodevo.campaign --config" in (Path(REPO_ROOT) / "scripts/campaign/run_campaign.sh").read_text()
    assert "-m evodevo.campaign status" in (Path(REPO_ROOT) / "scripts/campaign/status_campaign.sh").read_text()
    # plain `setsid` under `script` forks and exits, racing the pane's SIGHUP
    assert "-c 'setsid --wait " in (Path(REPO_ROOT) / "scripts/campaign/run_campaign.sh").read_text()


def smoke_config(path: Path, *, arm="nca", population=15, eval_budget=15, timesteps=500) -> Path:
    path.write_text(
        "profile: campaign\n"
        f"encoders: [{arm}]\n"
        "tasks: [Walker-v0]\n"
        "seeds: [0]\n"
        f"configs_root: {REPO_ROOT}/evodevo/experiments/configs\n"
        "output_dir: results/runs\n"
        "surrogate: false\n"
        "factors: {dev_steps: 100, init_condition: center_seed}\n"
        f"profile_overrides: {{population_size: {population}, offspring_size: {population}, "
        f"eval_budget: {eval_budget}, ppo_timesteps: {{Walker-v0: {timesteps}}}, "
        "ppo_seeds: {Walker-v0: 1}, ppo_seed_aggregation: max}\n")
    return path


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_ppo_cli_smoke(tmp_path):
    """Real PPO THROUGH `python -m evodevo.campaign` (never by constructing
    EvaluationPool): 1 arm x 1 seed x Walker-v0, population 15, eval_budget
    15, ppo_timesteps 500, 4 workers."""
    config = smoke_config(tmp_path / "smoke.yaml")
    root = tmp_path / "campaigns"
    delivered = tmp_path / "notifications.log"
    notify = tmp_path / "notify.yaml"
    notify.write_text(f"shell_command: [sh, -c, 'printf \"%s\\n\" \"$0\" >> {delivered}']\n")
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(notify))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.pop(var, None)
    command = [sys.executable, "-m", "evodevo.campaign", "--config", str(config), "--output-root", str(root),
               "--campaign-id", "cli-smoke", "--workers", "4", "--mode", "real", "--no-mirror"]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200, env=env)
    output = result.stdout + result.stderr
    assert "BrokenProcessPool" not in output
    assert "ThreadingConfiguredTooLateError" not in output
    assert result.returncode == 0, output[-4000:]
    campaign = root / "cli-smoke"
    assert json.loads((campaign / "manifest.json").read_text())["status"] == "complete"
    records = [json.loads(line) for line in (campaign / "journal.jsonl").read_text().splitlines()]
    assert records, "no real PPO task reached the journal"
    completed = [r for r in records if r["result"]["completed"] and r["result"]["fitness"] is not None]
    assert completed, "no real evaluation completed"
    # Invariant is per worker SLOT: a retried task rebuilds the executor, and
    # the fresh processes (new pids) legitimately reuse the same slots/CPUs.
    cpus_by_slot, slots_by_pid = {}, {}
    for record in records:
        res = record["result"]
        assert res["torch_num_threads"] == 1, res
        for var, value in res["thread_env"].items():
            assert value == ("FALSE" if var.endswith("_DYNAMIC") else "1"), (var, value)
        cpus_by_slot.setdefault(res["worker_slot"], set()).add(tuple(res["cpu_ids"]))
        slots_by_pid.setdefault(res["worker_pid"], set()).add(res["worker_slot"])
    assert all(len(slots) == 1 for slots in slots_by_pid.values()), slots_by_pid
    assert all(len(sets) == 1 for sets in cpus_by_slot.values()), cpus_by_slot
    flat = [cpu for sets in cpus_by_slot.values() for cpu in next(iter(sets))]
    assert len(flat) == len(set(flat)), cpus_by_slot
    assert len(cpus_by_slot) >= 2, cpus_by_slot


@pytest.mark.slow
@pytest.mark.skipif(not HAS_SIMULATOR, reason="evogym.simulator_cpp is not built")
def test_real_cli_graceful_group_sigint_exits_promptly(tmp_path):
    """stop_campaign.sh sends SIGINT to the scheduler's WHOLE process group,
    workers included. Workers used to die inside call_queue.get() holding the
    queue lock; the survivors then blocked forever and the scheduler hung at
    interpreter exit until SIGTERM. Now workers ignore SIGINT and the
    scheduler drains in-flight tasks and exits cleanly."""
    import signal
    import time
    config = smoke_config(tmp_path / "stop.yaml", eval_budget=90)
    notify = tmp_path / "notify.yaml"
    notify.write_text("shell_command: 'true'\n")
    env = dict(os.environ, EVODEVO_NOTIFY_CONFIG=str(notify))
    root = tmp_path / "campaigns"
    process = subprocess.Popen([sys.executable, "-m", "evodevo.campaign", "--config", str(config),
                                "--output-root", str(root), "--campaign-id", "stop", "--workers", "20",
                                "--mode", "real", "--no-mirror", "--grace-s", "120"],
                               cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    journal = root / "stop" / "journal.jsonl"
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and not (journal.exists() and len(journal.read_text().splitlines()) >= 20):
            assert process.poll() is None, process.stdout.read()
            time.sleep(0.2)
        os.killpg(process.pid, signal.SIGINT)
        stopped_at = time.monotonic()
        output, _ = process.communicate(timeout=30)
        elapsed = time.monotonic() - stopped_at
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    assert process.returncode == 0, output[-3000:]
    assert elapsed < 30
    assert "Resume: scripts/campaign/run_campaign.sh --resume stop" in output
    assert json.loads((root / "stop" / "manifest.json").read_text())["status"] == "stopped"
    group = subprocess.run(["ps", "-eo", "pgid=,stat="], capture_output=True, text=True).stdout.split("\n")
    assert not [line for line in group if line.split() and line.split()[0] == str(process.pid)
                and not line.split()[1].startswith("Z")]
