"""Amendment 2 re-training stages: champions, seeds, unique output paths, resume."""
import json
import dataclasses
from pathlib import Path

import pytest

from evodevo.campaign import amend2
from evodevo.campaign.amend2 import (Champion, ChampionError, champions_for_stage, retrain_artifact_dir,
                                     retrain_file_paths, retrain_seed, run_stage, select_champion, stage_trainings)
from tests.campaign.a2_helpers import ARMS, BODY, BODY_B, EVALS, ORIGINAL_EVALS, make_original, make_plan, write_run_log


def _plan_dict(plan):
    return json.loads(json.dumps(plan.to_dict()))


def _world(tmp_path):
    original = make_original(tmp_path / "orig")
    plan = make_plan(original)
    root = tmp_path / "campaign"
    for phase in plan.phases:
        for run in phase.runs:
            write_run_log(root / "runs" / run.arm / run.task / f"seed_{run.seed}",
                          [0.2, 0.7, 0.4 + 0.01 * run.seed, 0.7, 0.1, None] + [0.05] * (EVALS - 6),
                          {1: BODY_B, 3: BODY})
    (root / "manifest.json").write_text(json.dumps({"campaign_id": "this-id", "git_commit": "a" * 40}))
    return original, plan, root


# --- champion selection ---------------------------------------------------------

def test_champion_is_the_maximum_recorded_fitness():
    rows = [{"eval_index": i, "fitness": f} for i, f in enumerate([0.1, 0.8, 0.3])]
    assert select_champion(rows)["eval_index"] == 1


def test_champion_ties_go_to_the_lowest_evaluation_index_whatever_the_row_order():
    rows = [{"eval_index": i, "fitness": f} for i, f in enumerate([0.1, 0.9, 0.3, 0.9, 0.9])]
    assert select_champion(rows)["eval_index"] == 1
    assert select_champion(list(reversed(rows)))["eval_index"] == 1


def test_champion_ignores_invalid_sentinel_rows_unless_everything_is_invalid(tmp_path):
    write_run_log(tmp_path / "run", [None, 0.2, None, None, None, None])
    champion = amend2._champion_from_run(tmp_path / "run", "Walker-v0", "nca", 0, 6, "c", "x", "campaign", "p")
    assert champion.eval_index == 1 and champion.fitness == 0.2
    write_run_log(tmp_path / "bad", [None] * 6)
    with pytest.raises(ChampionError, match="not a valid body"):
        amend2._champion_from_run(tmp_path / "bad", "Walker-v0", "direct", 0, 6, "c", "x", "campaign", "p")


def test_champion_is_read_only_from_a_completed_run(tmp_path):
    write_run_log(tmp_path / "run", [0.1, 0.2])
    with pytest.raises(ChampionError, match="2 logged evaluations, expected 6"):
        amend2._champion_from_run(tmp_path / "run", "Walker-v0", "direct", 0, 6, "c", "x", "campaign", "p")


def test_stage_champions_cover_this_campaign_and_the_original_with_provenance(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = next(s for s in plan.stages if s.stage_id == "reeval:UpStepper-v0:s0-5")
    champions = champions_for_stage(dataclasses.asdict(stage), _plan_dict(plan), root)
    assert [(c.arm, c.seed, c.source_kind) for c in champions] == [
        ("nca", 0, "original"), ("rule_based", 0, "original"), ("nca", 5, "campaign"), ("rule_based", 5, "campaign")]
    assert {c.source_campaign for c in champions if c.source_kind == "original"} == {"orig-id"}
    assert {c.source_commit for c in champions if c.source_kind == "campaign"} == {"a" * 40}
    assert all(c.eval_index == 1 for c in champions)  # first of the tied 0.9s / the maximum


def test_the_original_monotone_arm_is_never_a_champion_source(tmp_path):
    original, plan, root = _world(tmp_path)
    for stage in plan.stages:
        arms = {c.arm for c in champions_for_stage(dataclasses.asdict(stage), _plan_dict(plan), root)}
        assert "rule_based_monotone" not in arms


# --- seeds ---------------------------------------------------------------------------

def test_seed_bases_are_the_registered_ones_and_distinct_from_every_run_and_preflight_seed():
    from evodevo.evaluation.ppo_runner import ppo_seed
    reeval = [retrain_seed(amend2.SEED_BASE_REEVAL, k) for k in range(10)]
    robust = [retrain_seed(amend2.SEED_BASE_ROBUSTNESS, k) for k in range(3)]
    assert reeval[0] == 20261101 and robust[0] == 20261201
    assert reeval == [ppo_seed(20261101, k) for k in range(10)]
    run_seeds = {s for seed in range(10) for s in (ppo_seed(seed, 0), ppo_seed(seed, 1), seed)}
    preflight = {20261015, *[ppo_seed(20261016, r) for r in range(4)]}
    assert len(set(reeval + robust)) == 13
    assert not (set(reeval + robust) & (run_seeds | preflight))


# --- paths ---------------------------------------------------------------------------

def test_every_retraining_in_the_whole_plan_has_a_unique_output_path(tmp_path):
    original, plan, root = _world(tmp_path)
    paths, keys = [], []
    for stage in plan.stages:
        champions = champions_for_stage(dataclasses.asdict(stage), _plan_dict(plan), root)
        for champion, k, seed, task in stage_trainings(dataclasses.asdict(stage), champions, root):
            policy, log = retrain_file_paths(task.artifact_dir, task.eval_index, task.ppo_k)
            paths += [policy, log]
            keys.append((task.run_key, task.eval_index, task.ppo_k))
            assert stage.stage_id.replace(":", "__") in policy
            assert f"seed_{champion.seed}" in policy and champion.task in policy and champion.arm in policy
    assert len(paths) == len(set(paths)) and len(keys) == len(set(keys))
    assert len(paths) == 2 * sum(s.n * len(champions_for_stage(dataclasses.asdict(s), _plan_dict(plan), root))
                                 for s in plan.stages)


def test_the_worker_writes_exactly_where_the_unique_path_says(tmp_path):
    # PPOTask.artifact_dir + (eval_index, ppo_k) is how evaluate_ppo_task names its files.
    directory = retrain_artifact_dir(tmp_path, "reeval:Walker-v0:s0-9", "Walker-v0", "nca", 7, 123)
    policy, log = retrain_file_paths(directory, 123, 4)
    assert policy == f"{directory}/policies/e00123_k4.zip" and log == f"{directory}/training_logs/e00123_k4.json"
    assert "__" in directory and ":" not in directory


# --- resume / idempotency ---------------------------------------------------------------

def _stage(plan, stage_id):
    return dataclasses.asdict(next(s for s in plan.stages if s.stage_id == stage_id))


def _rows(root, name):
    path = Path(root) / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_surrogate_stage_writes_one_row_per_training_with_every_recorded_field(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = _stage(plan, "reeval:ObstacleTraverser-v1:s0")
    metrics = run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    rows = _rows(root, "reeval.jsonl")
    assert metrics["status"] == "complete" and len(rows) == 2 * 2 == 4  # 2 champions x n=2
    row = rows[0]
    for field in ("campaign_id", "run_key", "source_campaign", "source_commit", "seed", "champion_eval_index",
                  "body_hash", "budget", "ppo_seed", "fitness", "wall_time_s", "core_kind"):
        assert field in row
    assert row["campaign_id"] == "this-id" and row["budget"] == 2000 and row["source_commit"] == "a" * 40
    assert sorted((r["arm"], r["k"], r["ppo_seed"]) for r in rows) == sorted(
        (arm, k, retrain_seed(amend2.SEED_BASE_REEVAL, k)) for arm in ARMS for k in range(2))


def test_rerunning_a_finished_stage_adds_nothing(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = _stage(plan, "robustness:Walker-v0")
    run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    before = (root / "robustness.jsonl").read_bytes()
    metrics = run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    assert (root / "robustness.jsonl").read_bytes() == before
    assert metrics["trainings_skipped_as_done"] == 8 and metrics["completed_timed_tasks"] == 0


def test_a_stopped_stage_resumes_without_repeating_or_losing_trainings(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = _stage(plan, "reeval:UpStepper-v0:s0-5")
    stops = iter([False, False, False, True])
    first = run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2, stop_requested=lambda: next(stops, True))
    assert first["status"] == "stopped" and len(_rows(root, "reeval.jsonl")) == 3
    second = run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    assert second["status"] == "complete"
    rows = _rows(root, "reeval.jsonl")
    keys = [amend2.row_key(r) for r in rows]
    assert len(rows) == 4 * 2 == 8 and len(set(keys)) == 8


def test_infrastructure_failures_are_retried_on_resume_but_training_failures_are_not(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = _stage(plan, "robustness:Walker-v0")
    run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    rows = _rows(root, "robustness.jsonl")
    rows[0].update(completed=False, fitness=None, reason="infra_failure: worker lost")
    rows[1].update(completed=False, fitness=None, reason="simulation_instability: boom")
    (root / "robustness.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metrics = run_stage(root, "this-id", _plan_dict(plan), stage, None, workers=2)
    assert metrics["completed_timed_tasks"] == 1  # only the infra failure re-ran
    latest = amend2.latest_rows(root / "robustness.jsonl")
    assert latest[amend2.row_key(rows[0])]["completed"] is True
    assert latest[amend2.row_key(rows[1])]["reason"].startswith("simulation_instability")


class FakePool:
    """submit/next like EvaluationPool, completing in submission order."""

    def __init__(self):
        self.queue, self.submitted = [], []

    def submit_ppo_task(self, task, timeout_s=None):
        self.queue.append(task)
        self.submitted.append((task, timeout_s))

    def next_ppo_result(self, should_stop=None, grace_s=120):
        from evodevo.evaluation.ppo_runner import PPOTaskResult
        if not self.queue:
            return None
        task = self.queue.pop(0)
        return task, PPOTaskResult(1.0 + task.ppo_k, True, None, 3.0, None, 5, [4], None, None, None, None,
                                   started_at=10.0, finished_at=13.0)


def test_pool_path_uses_the_stage_timeout_core_kind_and_the_real_task_fields(tmp_path):
    original, plan, root = _world(tmp_path)
    stage = _stage(plan, "robustness:ObstacleTraverser-v1")
    pool = FakePool()
    run_stage(root, "this-id", _plan_dict(plan), stage, pool, workers=2, prefetch=1, core_kinds={4: "P"})
    assert {timeout for _task, timeout in pool.submitted} == {stage["timeout_s"]}
    assert all(task.timesteps == stage["budget"] and task.generation == 0 for task, _ in pool.submitted)
    rows = _rows(root, "robustness.jsonl")
    assert len(rows) == 4 * 2 and {r["core_kind"] for r in rows} == {"P"} and {r["wall_time_s"] for r in rows} == {3.0}
