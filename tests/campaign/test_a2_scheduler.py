"""The scheduler runs an Amendment 2 plan: phases keyed by (task, seed block), the
re-training stages between them, strict order, resume, and the notifications."""
import dataclasses
import json
from pathlib import Path

import pytest

import evodevo.campaign.scheduler as scheduler_module
from evodevo.campaign import amend2
from evodevo.campaign.plan import CampaignPlan
from evodevo.campaign.scheduler import CampaignScheduler
from tests.campaign.a2_helpers import make_original, make_plan

KEYS = ["ObstacleTraverser-v1:s0", "reeval:ObstacleTraverser-v1:s0", "ObstacleTraverser-v1:s1",
        "reeval:ObstacleTraverser-v1:s1", "UpStepper-v0:s5", "reeval:UpStepper-v0:s0-5", "Walker-v0:s5",
        "reeval:Walker-v0:s0-5", "robustness:ObstacleTraverser-v1", "robustness:UpStepper-v0",
        "robustness:Walker-v0"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    original = make_original(tmp_path / "orig")
    plan = make_plan(original)
    calls = []
    monkeypatch.setattr(scheduler_module, "run_subprocess",
                        lambda root, key, alerts=None, **kw: calls.append(key) or {"ok": True})
    return tmp_path, plan, calls


def _run(tmp_path, plan, name="a2"):
    scheduler = CampaignScheduler(plan, name, output_root=tmp_path, workers=2, mode="surrogate", no_mirror=True,
                                  local_config=str(tmp_path / "no-notifier.yaml"))
    scheduler.run()
    return scheduler, tmp_path / name


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_the_plan_sequence_is_phase_stage_alternating_in_strict_order(world):
    _, plan, _ = world
    assert [u.key if hasattr(u, "key") else u.stage_id for u in plan.units()] == KEYS


def test_a_whole_amendment_2_plan_runs_in_order_and_keys_everything_by_phase_id(world):
    tmp_path, plan, calls = world
    scheduler, root = _run(tmp_path, plan)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    status = manifest["phase_status"]
    assert set(status) == set(KEYS) and all(v["status"] == "complete" for v in status.values())
    assert sorted(KEYS, key=lambda key: status[key]["started_at"]) == KEYS
    ends = [status[key]["ended_at"] for key in KEYS]
    starts = [status[key]["started_at"] for key in KEYS]
    assert all(starts[i + 1] >= ends[i] for i in range(len(KEYS) - 1))  # no interleaving
    metrics = json.loads((root / "phase_metrics.json").read_text())
    assert set(metrics) == set(KEYS)  # two ObstacleTraverser-v1 phases do not collide
    assert metrics["ObstacleTraverser-v1:s0"]["phase_id"] == "ObstacleTraverser-v1:s0"
    # one milestone package per unit, in order, and the final robustness package last
    assert calls == KEYS + ["robustness:all"]
    assert set(manifest["milestone_status"]) == set(KEYS) | {"robustness:all"}


def test_run_directories_and_journal_keep_the_two_obstacle_blocks_apart(world):
    tmp_path, plan, _ = world
    _, root = _run(tmp_path, plan)
    for seed in (0, 1):
        for arm in ("nca", "rule_based"):
            assert (root / "runs" / arm / "ObstacleTraverser-v1" / f"seed_{seed}" / "run_log.jsonl").is_file()
    run_keys = {r["run_key"] for r in _rows(root / "journal.jsonl")}
    assert not run_keys  # surrogate mode journals nothing; run keys stay arm/task/seed
    assert {p.name for p in (root / "runs" / "nca" / "ObstacleTraverser-v1").iterdir()} == {"seed_0", "seed_1"}


def test_stage_outputs_have_the_planned_counts(world):
    tmp_path, plan, _ = world
    _, root = _run(tmp_path, plan)
    reeval, robust = _rows(root / "reeval.jsonl"), _rows(root / "robustness.jsonl")
    counts = {}
    for row in reeval + robust:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    plan_dict = json.loads(json.dumps(plan.to_dict()))
    for stage in plan.stages:
        assert counts[stage.stage_id] == amend2.planned_trainings(dataclasses.asdict(stage), plan_dict)
    assert {r["kind"] for r in reeval} == {"reeval"} and {r["kind"] for r in robust} == {"robustness"}
    assert {r["budget"] for r in robust} == {4000} or {r["budget"] for r in robust} == {2000 * 2}


def test_every_phase_and_stage_end_is_notified_once(world):
    tmp_path, plan, _ = world
    _, root = _run(tmp_path, plan)
    alerts = _rows(root / "alerts.jsonl")
    for key in KEYS:
        assert sum(a["condition"] == f"phase_complete:{key}" for a in alerts) == 1
        assert sum(a["condition"] == f"phase_started:{key}" for a in alerts) == 1


def test_rerunning_a_finished_campaign_changes_nothing(world):
    tmp_path, plan, calls = world
    _, root = _run(tmp_path, plan)
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and p.name not in
                {"alerts.jsonl", "manifest.json", "dispatch_status.json", "transcript.log"}}
    calls.clear()
    _run(tmp_path, plan)
    assert calls == []
    for path, content in snapshot.items():
        assert path.read_bytes() == content, path


def test_a_crash_inside_a_stage_resumes_to_the_same_outputs_as_an_uninterrupted_run(world, monkeypatch):
    tmp_path, plan, _ = world
    _, clean = _run(tmp_path, plan, "clean")
    real = amend2.run_stage
    boom = {"armed": True}

    def crashing(root, campaign_id, plan_dict, stage, pool, **kw):
        if boom["armed"] and stage["stage_id"] == "reeval:UpStepper-v0:s0-5":
            boom["armed"] = False
            stops = iter([False])
            real(root, campaign_id, plan_dict, stage, pool, **{**kw, "stop_requested": lambda: next(stops, True)})
            raise RuntimeError("injected crash mid-stage")
        return real(root, campaign_id, plan_dict, stage, pool, **kw)

    monkeypatch.setattr(amend2, "run_stage", crashing)
    with pytest.raises(RuntimeError, match="injected crash"):
        _run(tmp_path, plan, "crashed")
    monkeypatch.setattr(amend2, "run_stage", real)
    _run(tmp_path, plan, "crashed")  # resume: same campaign id, same plan
    resumed = tmp_path / "crashed"

    def stable(path):
        return sorted(json.dumps({k: v for k, v in row.items() if k not in {"completed_at", "started_at", "finished_at", "campaign_id", "source_campaign"}},
                                 sort_keys=True) for row in _rows(path))
    for name in ("reeval.jsonl", "robustness.jsonl"):
        assert stable(resumed / name) == stable(clean / name), name
    for rel in ("runs/nca/Walker-v0/seed_5/run_log.jsonl", "runs/rule_based/ObstacleTraverser-v1/seed_1/run_log.jsonl"):
        assert [dict(r, wall_clock_s=0) for r in _rows(resumed / rel)] == [dict(r, wall_clock_s=0) for r in _rows(clean / rel)]


def test_resume_refuses_a_plan_whose_phase_list_changed(world):
    tmp_path, plan, _ = world
    _run(tmp_path, plan)
    changed = dataclasses.replace(plan, phases=tuple(reversed(plan.phases[:2])) + plan.phases[2:])
    assert changed.config_hash != plan.config_hash
    with pytest.raises(ValueError, match="config-hash mismatch"):
        CampaignScheduler(changed, "a2", output_root=tmp_path, workers=2, mode="surrogate", no_mirror=True,
                          local_config=str(tmp_path / "no-notifier.yaml")).run()


def test_the_plan_hash_covers_the_phase_list_and_the_stages(world):
    _, plan, _ = world
    dropped_stage = dataclasses.replace(plan, stages=plan.stages[:-1])
    other_seed = dataclasses.replace(plan.phases[0], phase_id="ObstacleTraverser-v1:s9")
    renamed = dataclasses.replace(plan, phases=(other_seed,) + plan.phases[1:])
    assert len({plan.config_hash, dropped_stage.config_hash, renamed.config_hash}) == 3


def test_a_plan_round_trips_through_its_manifest_form_with_the_same_hash(world):
    _, plan, _ = world
    again = CampaignPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again.config_hash == plan.config_hash and again == dataclasses.replace(
        plan, phases=again.phases, stages=again.stages, amendment2=again.amendment2)
