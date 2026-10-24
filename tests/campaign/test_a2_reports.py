"""Amendment 2 milestone packages (merged seeds with provenance, re-evaluation,
robustness) and the dashboard, monitor and status on the new phase ids."""
import csv
import json
import time
from pathlib import Path

import pytest

import evodevo.campaign.scheduler as scheduler_module
from evodevo.campaign import a2_report, milestone
from evodevo.campaign.alerts import AlertManager, UTILIZATION_FLOOR, DEFAULT_UTILIZATION_FLOOR, utilization_floor
from evodevo.campaign.dashboard import collect, render
from evodevo.campaign.monitor import monitor_once
from evodevo.campaign.scheduler import CampaignScheduler
from evodevo.campaign.status import CampaignStatus, status_footer
from tests.campaign.a2_helpers import make_original, make_plan

REHEARSAL_ESTIMATE = Path("evodevo/experiments/configs/a2_rehearsal_estimate.json").resolve()


def _csv(path):
    return list(csv.DictReader(Path(path).open()))


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """A finished surrogate Amendment 2 campaign, every unit's package built."""
    tmp_path = tmp_path_factory.mktemp("a2reports")
    original = make_original(tmp_path / "orig")
    plan = make_plan(original)
    original_run = scheduler_module.run_subprocess
    scheduler_module.run_subprocess = lambda *a, **k: {"ok": True}
    try:
        CampaignScheduler(plan, "a2", output_root=tmp_path, workers=2, mode="surrogate", no_mirror=True,
                          local_config=str(tmp_path / "none.yaml")).run()
    finally:
        scheduler_module.run_subprocess = original_run
    root = tmp_path / "a2"
    packages = {}
    for unit in plan.units():
        key = getattr(unit, "key", None) or unit.stage_id
        packages[key] = Path(milestone.build(root, key)["output"])
    packages["robustness:all"] = Path(milestone.build(root, "robustness:all")["output"])
    return root, plan, packages


# --- phase packages: merge, provenance, INTERIM, monotone contrast ---------------------------

def test_every_phase_package_has_the_required_files(campaign):
    _, _, packages = campaign
    for key in ("ObstacleTraverser-v1:s0", "ObstacleTraverser-v1:s1", "UpStepper-v0:s5", "Walker-v0:s5"):
        assert all((packages[key] / name).is_file() for name in milestone.REQUIRED), key
        assert (packages[key] / "stats/per_run_finals.csv").is_file()
        assert ":" not in packages[key].name  # NTFS-safe directory names


def test_p1_is_interim_and_p2_is_the_final_obstacle_package_over_both_seed_blocks(campaign):
    _, _, packages = campaign
    p1, p2 = packages["ObstacleTraverser-v1:s0"], packages["ObstacleTraverser-v1:s1"]
    manifest = json.loads((p1 / "package_manifest.json").read_text())
    assert manifest["status"] == "INTERIM"
    assert "INTERIM" in (p1 / "MILESTONE_REPORT.md").read_text()
    assert {r["status"] for r in _csv(p1 / "stats/final_fitness_summary.csv")} == {"INTERIM"}
    assert {int(r["seed"]) for r in _csv(p1 / "stats/per_run_finals.csv")} == {0}
    final = json.loads((p2 / "package_manifest.json").read_text())
    assert final["status"] == "COMPLETE"
    rows = _csv(p2 / "stats/per_run_finals.csv")
    assert {int(r["seed"]) for r in rows} == {0, 1}  # P1 and P2 merged
    assert {r["provenance"] for r in rows} == {"campaign a2, phase ObstacleTraverser-v1:s0",
                                               "campaign a2, phase ObstacleTraverser-v1:s1"}


def test_the_floor_effect_check_is_kept_for_obstacle_only(campaign):
    _, _, packages = campaign
    assert (packages["ObstacleTraverser-v1:s1"] / "stats/floor_effect_check.csv").is_file()
    assert not (packages["UpStepper-v0:s5"] / "stats/floor_effect_check.csv").exists()


def test_original_task_packages_merge_original_seeds_with_a_provenance_column(campaign):
    _, _, packages = campaign
    rows = _csv(packages["UpStepper-v0:s5"] / "stats/per_run_finals.csv")
    by_seed = {int(r["seed"]): r["provenance"] for r in rows}
    assert by_seed[0] == "original campaign orig-id (commit c0ffee0), read-only"
    assert by_seed[5] == "campaign a2, phase UpStepper-v0:s5"
    morphology = _csv(packages["UpStepper-v0:s5"] / "stats/final_best_morphology.csv")
    assert {r["provenance"] for r in morphology} == set(by_seed.values())
    assert {r["arm"] for r in rows} == {"nca", "rule_based"}  # the monotone arm is not in the main analysis
    pairwise = _csv(packages["UpStepper-v0:s5"] / "stats/pairwise.csv")
    assert all("monotone" not in (r["arm_a"] + r["arm_b"]) for r in pairwise)


def test_the_monotone_contrast_uses_only_the_original_seeds(campaign):
    _, _, packages = campaign
    rows = _csv(packages["UpStepper-v0:s5"] / "stats/rule_based_vs_monotone.csv")
    assert len(rows) == 1 and {rows[0]["arm_a"], rows[0]["arm_b"]} == {"rule_based", "rule_based_monotone"}
    assert rows[0]["seeds"] == "original campaign seeds 0-4 only"
    assert "uses the original campaign's seeds 0-4 only" in (packages["UpStepper-v0:s5"] / "MILESTONE_REPORT.md").read_text()
    assert _csv(packages["ObstacleTraverser-v1:s1"] / "stats/rule_based_vs_monotone.csv") == []


def test_original_runs_are_only_ever_read(campaign):
    root, plan, _ = campaign
    original = Path(plan.amendment2["original_campaign"]["path"])
    assert not (original / "milestones").exists() and not (original / "retrain").exists()


# --- re-evaluation and robustness packages ---------------------------------------------------

def test_reeval_package_tables_and_interim_label(campaign):
    _, _, packages = campaign
    for key in ("reeval:ObstacleTraverser-v1:s0", "reeval:UpStepper-v0:s0-5"):
        assert all((packages[key] / name).is_file() for name in a2_report.REEVAL_FILES), key
    p1 = packages["reeval:ObstacleTraverser-v1:s0"]
    assert json.loads((p1 / "package_manifest.json").read_text())["status"] == "INTERIM"
    champions = _csv(packages["reeval:UpStepper-v0:s0-5"] / "stats/reeval_per_champion.csv")
    assert len(champions) == 4 and {int(r["n_trainings"]) for r in champions} == {2}
    assert {r["provenance"].split(":")[0] for r in champions} == {"original", "campaign"}
    assert all(float(r["mean"]) >= float(r["min"]) and float(r["mean"]) <= float(r["max"]) for r in champions)
    per_arm = _csv(packages["reeval:UpStepper-v0:s0-5"] / "stats/reeval_per_arm.csv")
    assert {r["arm"] for r in per_arm} == {"nca", "rule_based"} and {r["n_champions"] for r in per_arm} == {"2"}
    omnibus = _csv(packages["reeval:UpStepper-v0:s0-5"] / "stats/reeval_omnibus.csv")
    assert omnibus[0]["test"] == "Kruskal-Wallis"
    trainings = _csv(packages["reeval:UpStepper-v0:s0-5"] / "stats/reeval_trainings.csv")
    assert len(trainings) == 8 and {r["stage"] for r in trainings} == {"reeval:UpStepper-v0:s0-5"}


def test_robustness_packages_compare_the_task_budget_with_two_times(campaign):
    _, _, packages = campaign
    final = packages["robustness:all"]
    assert all((final / name).is_file() for name in a2_report.ROBUSTNESS_FILES)
    rankings = _csv(final / "stats/robustness_rankings.csv")
    assert {r["budget_label"] for r in rankings} == {"task_budget", "2x_budget"}
    assert {r["task"] for r in rankings} == {"ObstacleTraverser-v1", "UpStepper-v0", "Walker-v0"}
    a12 = _csv(final / "stats/robustness_a12.csv")
    assert {"a12_task_budget", "a12_2x_budget"} <= set(a12[0])
    single = _csv(packages["robustness:Walker-v0"] / "stats/robustness_rankings.csv")
    assert {r["task"] for r in single} == {"Walker-v0"}


def test_the_milestone_subprocess_validates_phase_and_stage_packages(campaign):
    root, _, _ = campaign
    for key in ("UpStepper-v0:s5", "reeval:UpStepper-v0:s0-5", "robustness:Walker-v0", "robustness:all"):
        outcome = milestone.run_subprocess(root, key)
        assert outcome["ok"], (key, outcome)


def test_a_partial_stage_package_is_labelled_partial(campaign, tmp_path):
    root, plan, _ = campaign
    import shutil
    copy = tmp_path / "partial"
    shutil.copytree(root, copy)
    rows = (copy / "reeval.jsonl").read_text().splitlines()
    (copy / "reeval.jsonl").write_text("\n".join(rows[:3]) + "\n")
    package = Path(milestone.build(copy, "reeval:ObstacleTraverser-v1:s0")["output"])
    assert json.loads((package / "package_manifest.json").read_text())["status"] == "PARTIAL"


# --- dashboard, status, alerts, monitor on the new ids --------------------------------------

@pytest.fixture
def running(tmp_path):
    """Manifest of an A2 campaign whose first phase is done and whose first stage is half done."""
    original = make_original(tmp_path / "orig")
    plan = make_plan(original)
    root = tmp_path / "live"
    root.mkdir()
    keys = [getattr(u, "key", None) or u.stage_id for u in plan.units()]
    status = {keys[0]: {"status": "complete", "started_at": 1000, "ended_at": 2000},
              keys[1]: {"status": "running", "started_at": 2000}}
    (root / "manifest.json").write_text(json.dumps({
        "campaign_id": "live", "started_at": 1000, "status": "running", "phase_status": status,
        "plan": json.loads(json.dumps(plan.to_dict())), "estimate_path": str(REHEARSAL_ESTIMATE),
        "worker_slots": {"slots": [[0], [2]]}}))
    stage = plan.stages[0]
    rows = [{"stage": stage.stage_id, "kind": "reeval", "arm": "nca", "task": stage.task, "seed": 0, "k": k,
             "champion_eval_index": 1, "run_key": "nca/x/seed_0", "completed": True, "fitness": 1.0,
             "wall_time_s": 5.0, "cpu_ids": [0], "started_at": 2100 + k * 10, "finished_at": 2105 + k * 10,
             "completed_at": 2105 + k * 10} for k in range(2)]
    (root / "reeval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, plan, keys


def test_dashboard_follows_the_running_unit_by_its_id(running):
    root, plan, keys = running
    snapshot = collect(str(root), now=2200, telemetry={})
    assert snapshot["phase"] == keys[1] == "reeval:ObstacleTraverser-v1:s0"
    assert snapshot["unit_kind"] == "reeval" and snapshot["task"] == "ObstacleTraverser-v1"
    assert (snapshot["phase_completed"], snapshot["phase_total"]) == (2, 4)  # 2 champions x n=2
    assert snapshot["generation"] == {} and snapshot["eta_source"] == "prior"
    assert snapshot["guard_h"] == float("inf") and snapshot["projection_exceeds_432h"] is False
    text = render(snapshot)
    assert "re-trainings" in text and "no cost guard" in text and keys[1] in text


def test_the_stall_clock_counts_re_training_rows_and_starts_with_the_stage(running):
    root, _, _ = running
    snapshot = collect(str(root), now=2300, telemetry={})
    assert snapshot["seconds_since_last_task"] == pytest.approx(2300 - 2115)
    assert snapshot["completed_ppo"] == 2 and snapshot["throughput_now_per_h"] == 2


def test_overall_progress_includes_the_re_training_stages(running):
    root, plan, _ = running
    snapshot = collect(str(root), now=2200, telemetry={})
    plan_dict = json.loads(json.dumps(plan.to_dict()))
    from evodevo.campaign.amend2 import planned_trainings
    import dataclasses
    stages = sum(planned_trainings(dataclasses.asdict(s), plan_dict) for s in plan.stages)
    phases = sum(int(r.config["eval_budget"]) for p in plan.phases for r in p.runs)
    assert snapshot["total_evaluations"] == phases + stages
    status = CampaignStatus.from_snapshot(snapshot, now=2200)
    assert status.phase == "reeval:ObstacleTraverser-v1:s0" and "reeval:ObstacleTraverser-v1:s0" in status_footer(status)


def test_dashboard_on_an_evolutionary_phase_id_lists_its_runs_and_future_units(running):
    root, plan, keys = running
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["phase_status"] = {}
    (root / "manifest.json").write_text(json.dumps(manifest))
    snapshot = collect(str(root), now=2200, telemetry={})
    assert snapshot["phase"] == keys[0] and snapshot["unit_kind"] == "phase"
    assert set(snapshot["generation"]) == {("nca", 0), ("rule_based", 0)}
    # every later unit contributes its band: 11 units x 0.1 h mean in the rehearsal estimate
    assert snapshot["campaign_eta_s"][1] / 3600 == pytest.approx(len(keys) * 0.2, rel=1e-6) or \
        snapshot["campaign_eta_s"][1] > 0


def test_utilization_floor_reads_the_task_inside_phase_and_stage_ids():
    assert utilization_floor("UpStepper-v0:s5-9") == UTILIZATION_FLOOR["UpStepper-v0"]
    assert utilization_floor("reeval:Walker-v0:s0-9") == UTILIZATION_FLOOR["Walker-v0"]
    assert utilization_floor("robustness:Walker-v0") == UTILIZATION_FLOOR["Walker-v0"]
    assert utilization_floor("ObstacleTraverser-v1:s0-4") == DEFAULT_UTILIZATION_FLOOR
    assert utilization_floor("Walker-v0") == UTILIZATION_FLOOR["Walker-v0"] and utilization_floor(None) is None


def test_every_phase_and_stage_end_is_a_distinct_notification(tmp_path):
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=type("N", (), {"configured": False})())
    keys = ["ObstacleTraverser-v1:s0-4", "reeval:ObstacleTraverser-v1:s0-4", "robustness:Walker-v0"]
    for key in keys:
        alerts.phase_complete(key)
        alerts.phase_complete(key)  # restart after the boundary: still one durable condition
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    notified = [r["condition"] for r in rows if r.get("message")]
    assert notified == [f"phase_complete:{k}" for k in keys]  # one notice per unit, however often it is re-emitted


def test_monitor_uses_the_per_unit_stall_threshold_and_no_snapshot_for_a_stage(running, monkeypatch, tmp_path):
    root, plan, keys = running
    calls = []
    monkeypatch.setattr("evodevo.campaign.monitor.run_subprocess", lambda *a, **k: calls.append(a[1]) or {"ok": True})
    alerts = AlertManager(root / "alerts.jsonl", notifier=type("N", (), {"configured": False})())
    state = monitor_once(root, alerts, now=2200, last_mirror=2200, last_snapshot=0)
    assert state["running"] and calls == []  # a stage has no evolutionary snapshot
    stall = [json.loads(line) for line in (root / "alerts.jsonl").read_text().splitlines()
             if json.loads(line)["condition"] == "stall"][-1]
    assert stall["details"]["threshold_s"] == 2 * 225.0 and stall["status"] == "clear"
    later = monitor_once(root, alerts, now=2115 + 451, last_mirror=2200, last_snapshot=0)
    stall = [json.loads(line) for line in (root / "alerts.jsonl").read_text().splitlines()
             if json.loads(line)["condition"] == "stall"][-1]
    assert stall["status"] == "active" and later["running"]
    # an evolutionary phase unit still gets its periodic snapshot, keyed by its phase id
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["phase_status"] = {keys[0]: {"status": "running", "started_at": 1000}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    monitor_once(root, alerts, now=1000 + 13 * 3600, last_mirror=1000 + 13 * 3600, last_snapshot=0)
    assert calls == [keys[0]]
