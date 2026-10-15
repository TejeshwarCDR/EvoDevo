"""Campaign machinery works for any task set, using the campaign's own estimate."""
import csv
import json
from pathlib import Path

import pytest

from evodevo.campaign.alerts import DEFAULT_UTILIZATION_FLOOR, UTILIZATION_FLOOR, utilization_floor
from evodevo.campaign.dashboard import collect, render
from evodevo.campaign.estimate import is_v2, load, phase_band, slowest_ecore_p95_s
from evodevo.campaign.milestone import REQUIRED, build, phase_ppo_seeds, sensitivity_files
from evodevo.campaign.monitor import slowest_ecore_p95_s as monitor_p95
from evodevo.campaign.status import CampaignStatus

V2 = Path("evodevo/experiments/configs/ext_validation_estimate.json").resolve()
TASKS = ["ObstacleTraverser-v1", "Climber-v0", "CaveCrawler-v0"]


def _manifest(tmp_path, current="ObstacleTraverser-v1", estimate=V2):
    phases = [{"task": t, "ppo_seeds": 2 if t == "Climber-v0" else 1,
               "runs": [{"arm": a, "seed": 0, "config": {"eval_budget": 300, "ppo_seeds": 1}}
                        for a in ("direct", "nca", "rule_based")]} for t in TASKS]
    status = {}
    for task in TASKS:
        if task == current:
            status[task] = {"status": "running", "started_at": 1000}
            break
        status[task] = {"status": "complete", "started_at": 0, "ended_at": 1000}
    manifest = {"campaign_id": "ext", "started_at": 1000, "status": "running", "phase_status": status,
                "plan": {"phases": phases}, "estimate_path": str(estimate)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_v2_estimate_is_recognised_and_read_per_task():
    estimate = load(V2)
    assert is_v2(estimate)
    assert phase_band(estimate, "Climber-v0") == {"p5": 1.0, "mean": 1.5, "p95": 2.0}
    assert slowest_ecore_p95_s(estimate, "CaveCrawler-v0") == pytest.approx((1800 + 1.645 * 60) * 2.3172)


def test_dashboard_prior_uses_the_campaign_estimate_for_every_phase(tmp_path):
    _manifest(tmp_path)
    snapshot = collect(str(tmp_path), now=1000, telemetry={})
    assert snapshot["eta_source"] == "prior"
    # current phase 1.0-2.0 h, plus two future phases at 1.0-2.0 h each.
    assert snapshot["campaign_eta_s"][0] / 3600 == pytest.approx(3.0)
    assert snapshot["campaign_eta_s"][1] / 3600 == pytest.approx(6.0)
    assert snapshot["guard_h"] == 360.0 and snapshot["p2_p95_h"] == 6.0
    assert ">360 h" in render(snapshot)


def test_dashboard_legacy_campaign_without_estimate_path_is_unchanged(tmp_path):
    walker = {"task": "Walker-v0", "runs": [{"arm": "direct", "seed": 0, "config": {"eval_budget": 300}}]}
    manifest = {"campaign_id": "legacy", "started_at": 1000, "status": "running",
                "phase_status": {"Walker-v0": {"status": "running", "started_at": 1000}},
                "plan": {"phases": [walker]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    snapshot = collect(str(tmp_path), now=1000, telemetry={})
    assert snapshot["guard_h"] == 432.0 and snapshot["p2_mean_h"] == 403.2


def test_monitor_stall_threshold_uses_the_campaign_estimate():
    assert monitor_p95("Climber-v0", V2) == pytest.approx((1800 + 1.645 * 60) * 2.3172)
    with pytest.raises(ValueError, match="no measured t_PPO prior"):
        monitor_p95("Walker-v0", V2)


def test_new_tasks_get_a_utilization_floor_and_status_target():
    assert UTILIZATION_FLOOR["Walker-v0"] == pytest.approx(0.846922)
    assert utilization_floor("Climber-v0") == DEFAULT_UTILIZATION_FLOOR == pytest.approx(0.834949)
    status = CampaignStatus.from_snapshot({"phase": "Climber-v0", "as_of_epoch": 0})
    assert status.worker_utilization_steady_state_target == DEFAULT_UTILIZATION_FLOOR
    assert CampaignStatus.from_snapshot({"phase": "unknown", "as_of_epoch": 0}).worker_utilization_steady_state_target is None


def _runs(tmp_path, task, k):
    body = [[0] * 5 for _ in range(5)]
    body[2][2] = 3
    for arm_index, arm in enumerate(("direct", "nca", "rule_based")):
        for seed in (0, 1):
            directory = tmp_path / "runs" / arm / task / f"seed_{seed}"
            directory.mkdir(parents=True)
            primary = arm_index + seed / 10
            row = {"eval_index": 0, "generation": 0, "fitness": primary, "valid": True,
                   "body_hash": f"{arm}-{seed}", "cache_hit": False, "wall_clock_s": 1.0, "body": body,
                   "per_seed_fitness": [primary] + ([primary + 2] if k == 2 else [])}
            (directory / "run_log.jsonl").write_text(json.dumps(row) + "\n")


def test_milestone_for_a_k2_extension_task_has_sensitivity_and_no_monotone_contrast(tmp_path):
    _runs(tmp_path, "Climber-v0", 2)
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"phases": [
        {"task": "Climber-v0", "ppo_seeds": 2, "runs": [{"config": {"eval_budget": 1}}] * 6}]}}))
    folder = Path(build(tmp_path, "Climber-v0")["output"])
    assert all((folder / name).is_file() for name in REQUIRED)
    assert all((folder / name).is_file() for name in sensitivity_files(2))
    pairwise = list(csv.DictReader((folder / "stats/pairwise.csv").open()))
    assert len(pairwise) == 3  # Holm family = the 3 core-arm pairs on this task
    assert list(csv.DictReader((folder / "stats/rule_based_vs_monotone.csv").open())) == []
    report = (folder / "MILESTONE_REPORT.md").read_text()
    assert "Climber-v0 mean-of-2 sensitivity" in report and "not part of this campaign" in report


def test_milestone_for_a_k1_extension_task_has_no_sensitivity(tmp_path):
    _runs(tmp_path, "CaveCrawler-v0", 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"phases": [
        {"task": "CaveCrawler-v0", "ppo_seeds": 1, "runs": [{"config": {"eval_budget": 1}}] * 6}]}}))
    folder = Path(build(tmp_path, "CaveCrawler-v0")["output"])
    assert not any((folder / name).exists() for name in sensitivity_files(2))


def test_phase_ppo_seeds_reads_the_plan_and_keeps_the_legacy_default():
    manifest = {"plan": {"phases": [{"task": "Climber-v0", "ppo_seeds": 2}, {"task": "Walker-v0", "ppo_seeds": 1}]}}
    assert phase_ppo_seeds(manifest, "Climber-v0") == 2 and phase_ppo_seeds(manifest, "Walker-v0") == 1
    assert phase_ppo_seeds({}, "UpStepper-v0") == 2 and phase_ppo_seeds({}, "CaveCrawler-v0") == 1
    assert sensitivity_files(2)[0] == "stats/mean_of_2_sensitivity.csv"


def test_floor_effect_check_flags_a_task_where_no_arm_beats_random_bodies(tmp_path):
    from evodevo.campaign.milestone import floor_effect_rows
    import pandas as pd
    rows = []
    for arm in ("direct", "nca", "rule_based"):
        for seed in range(5):
            for index, fitness in enumerate((0.5, 0.1, 0.2)):
                rows.append({"encoder": arm, "seed": seed, "eval_index": index,
                             "generation": 0 if index < 2 else 1, "valid": True, "fitness": fitness})
    data = pd.DataFrame(rows)
    finals = {arm: [0.5] * 5 for arm in ("direct", "nca", "rule_based")}
    limited = floor_effect_rows(data, finals, "ObstacleTraverser-v1")
    assert all(r["task_floor_limited"] for r in limited) and limited[0]["q75_generation0_valid"] == 0.5
    finals["nca"] = [0.9] * 5
    assert not any(r["task_floor_limited"] for r in floor_effect_rows(data, finals, "ObstacleTraverser-v1"))


def test_milestone_writes_the_floor_effect_check(tmp_path):
    _runs(tmp_path, "CaveCrawler-v0", 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"phases": [
        {"task": "CaveCrawler-v0", "ppo_seeds": 1, "runs": [{"config": {"eval_budget": 1}}] * 6}]}}))
    folder = Path(build(tmp_path, "CaveCrawler-v0")["output"])
    rows = list(csv.DictReader((folder / "stats/floor_effect_check.csv").open()))
    assert len(rows) == 3 and {r["task_floor_limited"] for r in rows} == {"False"}
    assert "Floor-effect check" in (folder / "MILESTONE_REPORT.md").read_text()


# --- one source for the legacy estimate path ----------------------------------

def test_legacy_estimate_path_has_one_definition():
    from evodevo.campaign import dashboard, dry_run, estimate, plan
    assert dashboard.PRIOR_PATH is estimate.LEGACY_ESTIMATE_PATH
    assert plan.LEGACY_ESTIMATE_PATH is estimate.LEGACY_ESTIMATE_PATH
    assert dry_run.FINAL_ESTIMATE is estimate.LEGACY_ESTIMATE_PATH


def test_call_time_readers_honour_a_patched_legacy_path(tmp_path, monkeypatch):
    from evodevo.campaign import dashboard, estimate, monitor, plan
    from evodevo.experiments.benchmark import BenchmarkConfig
    moved = tmp_path / "estimate.json"
    moved.write_text(V2.read_text())
    monkeypatch.setattr(estimate, "LEGACY_ESTIMATE_PATH", moved)
    assert estimate.estimate_path_for_manifest({}) == moved
    assert estimate.estimate_path_for_manifest(None) == moved
    benchmark = BenchmarkConfig.from_yaml("evodevo/experiments/configs/benchmark.yaml")
    assert plan.resolve_estimate_path(benchmark) == moved
    assert is_v2(load(moved)) and dashboard._prior() == {}  # v2 file has no legacy row
    assert monitor.slowest_ecore_p95_s("Climber-v0") == pytest.approx((1800 + 1.645 * 60) * 2.3172)


# --- no silent fallback bands ---------------------------------------------------

def _estimate_without(tmp_path, task):
    estimate = json.loads(V2.read_text())
    del estimate["result"]["phase_h"][task]
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps(estimate))
    return path


def test_dashboard_refuses_a_v2_estimate_missing_a_phase_band(tmp_path):
    _manifest(tmp_path, estimate=_estimate_without(tmp_path, "CaveCrawler-v0"))
    with pytest.raises(ValueError, match="no phase band for task 'CaveCrawler-v0'"):
        collect(str(tmp_path), now=1000, telemetry={})
    _manifest(tmp_path, current="CaveCrawler-v0", estimate=_estimate_without(tmp_path, "CaveCrawler-v0"))
    with pytest.raises(ValueError, match="no phase band"):
        collect(str(tmp_path), now=1000, telemetry={})


def test_plan_refuses_a_v2_estimate_missing_a_phase_band(tmp_path):
    from evodevo.campaign.plan import CampaignPlan
    from evodevo.experiments.benchmark import BenchmarkConfig
    benchmark = BenchmarkConfig.from_yaml("evodevo/experiments/configs/campaign_ext_mini_validation.yaml")
    path = _estimate_without(tmp_path, benchmark.tasks[-1])
    with pytest.raises(ValueError, match="has no phase band"):
        CampaignPlan.from_benchmark(benchmark, estimate_path=path)
    assert CampaignPlan.from_benchmark(benchmark)  # its own estimate is complete


def test_original_task_milestone_packages_have_no_floor_effect_output(tmp_path):
    _runs(tmp_path, "Walker-v0", 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"plan": {"phases": [
        {"task": "Walker-v0", "ppo_seeds": 1, "runs": [{"config": {"eval_budget": 1}}] * 6}]}}))
    folder = Path(build(tmp_path, "Walker-v0")["output"])
    assert not (folder / "stats/floor_effect_check.csv").exists()
    assert "Floor-effect check" not in (folder / "MILESTONE_REPORT.md").read_text()
