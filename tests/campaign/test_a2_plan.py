"""The Amendment 2 plan: phases by (task, seed block), counts, order, hashes, and
the two launch checks (config equality and library versions)."""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from evodevo.campaign import amend2
from evodevo.campaign.plan import CampaignPlan, DecisionRecordError
from evodevo.campaign.units import plan_units
from evodevo.experiments.benchmark import BenchmarkConfig

CONFIGS = Path("evodevo/experiments/configs")
ORIGINAL = Path("results/campaign/20260930T044002Z")
PRODUCTION_IDS = ["ObstacleTraverser-v1:s0-4", "reeval:ObstacleTraverser-v1:s0-4", "ObstacleTraverser-v1:s5-9",
                  "reeval:ObstacleTraverser-v1:s5-9", "UpStepper-v0:s5-9", "reeval:UpStepper-v0:s0-9",
                  "Walker-v0:s5-9", "reeval:Walker-v0:s0-9", "robustness:ObstacleTraverser-v1",
                  "robustness:UpStepper-v0", "robustness:Walker-v0"]


@pytest.fixture(scope="module")
def production():
    benchmark = BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_a2.yaml"))
    return benchmark, CampaignPlan.from_benchmark(benchmark)


def _by_id(plan):
    return {p.key: p for p in plan.phases}


class Args:
    def __init__(self, skip=False):
        self.skip_config_equality_check = skip


# --- phases: parsing, order, counts, seeds, arms ------------------------------------

def test_phase_ids_order_and_unit_sequence(production):
    _, plan = production
    assert [p.key for p in plan.phases] == ["ObstacleTraverser-v1:s0-4", "ObstacleTraverser-v1:s5-9",
                                            "UpStepper-v0:s5-9", "Walker-v0:s5-9"]
    assert [u.key if hasattr(u, "key") else u.stage_id for u in plan.units()] == PRODUCTION_IDS
    assert [u["key"] for u in plan_units(json.loads(json.dumps(plan.to_dict())))] == PRODUCTION_IDS


def test_seeds_arms_and_budgets_per_phase(production):
    _, plan = production
    phases = _by_id(plan)
    for key, seeds in (("ObstacleTraverser-v1:s0-4", range(0, 5)), ("ObstacleTraverser-v1:s5-9", range(5, 10)),
                       ("UpStepper-v0:s5-9", range(5, 10)), ("Walker-v0:s5-9", range(5, 10))):
        phase = phases[key]
        assert sorted({r.seed for r in phase.runs}) == list(seeds)
        assert sorted({r.arm for r in phase.runs}) == ["direct", "nca", "rule_based"]
        assert len(phase.runs) == 15 and phase.timeout_s > 0
        assert all(r.task == phase.task for r in phase.runs)
    assert [(p.key, p.ppo_timesteps, p.ppo_seeds) for p in plan.phases] == [
        ("ObstacleTraverser-v1:s0-4", 1_500_000, 1), ("ObstacleTraverser-v1:s5-9", 1_500_000, 1),
        ("UpStepper-v0:s5-9", 1_200_000, 2), ("Walker-v0:s5-9", 1_000_000, 1)]
    assert sum(len(p.runs) for p in plan.phases) == 60
    assert all(r.arm != "rule_based_monotone" for p in plan.phases for r in p.runs)


def test_re_training_counts_and_budgets(production):
    _, plan = production
    plan_dict = json.loads(json.dumps(plan.to_dict()))
    counts = {s.stage_id: amend2.planned_trainings(dataclasses.asdict(s), plan_dict) for s in plan.stages}
    assert [counts[s.stage_id] for s in plan.stages if s.kind == "reeval"] == [150, 150, 300, 300]
    robust = [s for s in plan.stages if s.kind == "robustness"]
    assert [(s.task, s.budget, counts[s.stage_id]) for s in robust] == [
        ("ObstacleTraverser-v1", 3_000_000, 90), ("UpStepper-v0", 2_400_000, 90), ("Walker-v0", 2_000_000, 90)]
    assert sum(counts[s.stage_id] for s in robust) == 270
    assert {s.seed_base for s in plan.stages if s.kind == "reeval"} == {20261101}
    assert {s.seed_base for s in robust} == {20261201}
    assert {s.n for s in plan.stages if s.kind == "reeval"} == {10} and {s.n for s in robust} == {3}
    for s in plan.stages:
        phase_budget = next(p.ppo_timesteps for p in plan.phases if p.task == s.task)
        assert s.budget == phase_budget * (2 if s.kind == "robustness" else 1)


def test_timeouts_walker_and_upstepper_are_the_originals_and_obstacle_scales_with_budget(production):
    _, plan = production
    phases = _by_id(plan)
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    originals = {p["task"]: p["timeout_s"] for p in manifest["plan"]["phases"]}
    assert phases["Walker-v0:s5-9"].timeout_s == originals["Walker-v0"]
    assert phases["UpStepper-v0:s5-9"].timeout_s == originals["UpStepper-v0"]
    stages = {s.stage_id: s for s in plan.stages}
    assert stages["robustness:Walker-v0"].timeout_s == pytest.approx(2 * originals["Walker-v0"])
    assert stages["robustness:UpStepper-v0"].timeout_s == pytest.approx(2 * originals["UpStepper-v0"])
    assert stages["robustness:ObstacleTraverser-v1"].timeout_s == pytest.approx(
        2 * stages["reeval:ObstacleTraverser-v1:s0-4"].timeout_s)
    assert phases["ObstacleTraverser-v1:s0-4"].timeout_s == stages["reeval:ObstacleTraverser-v1:s0-4"].timeout_s


# --- the plan hash ------------------------------------------------------------------

def test_plan_hash_covers_the_phase_list_and_is_stable_through_the_manifest(production):
    _, plan = production
    again = CampaignPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again.config_hash == plan.config_hash
    shifted = dataclasses.replace(plan.phases[1], runs=tuple(dataclasses.replace(r, seed=r.seed + 10)
                                                           for r in plan.phases[1].runs))
    assert dataclasses.replace(plan, phases=(plan.phases[0], shifted) + plan.phases[2:]).config_hash != plan.config_hash
    assert dataclasses.replace(plan, phases=plan.phases[:3]).config_hash != plan.config_hash


def test_original_campaign_plan_hash_is_unchanged_on_the_real_manifest():
    """benchmark.yaml with seeds 0-4 still hashes to the original manifest's config_hash."""
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    benchmark = dataclasses.replace(BenchmarkConfig.from_yaml(str(CONFIGS / "benchmark.yaml")), seeds=[0, 1, 2, 3, 4])
    plan = CampaignPlan.from_benchmark(benchmark)
    assert plan.config_hash == manifest["config_hash"]
    assert "phase_id" not in json.dumps(plan.to_dict()) and "stages" not in plan.to_dict()
    assert "amendment2" not in plan.to_dict()


def test_a_production_config_must_name_its_record_and_a_test_config_cannot_borrow_it(production):
    benchmark, _ = production
    with pytest.raises(DecisionRecordError, match="must name its decision record"):
        CampaignPlan.from_benchmark(dataclasses.replace(benchmark, decisions_path=None))
    test = BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_a2_rehearsal.yaml"))
    with pytest.raises(DecisionRecordError, match="amendment2 block differs"):
        CampaignPlan.from_benchmark(dataclasses.replace(test, decisions_path="docs/campaign_prep/ext/A2_DECISIONS.json"))


def test_the_record_must_agree_with_the_yaml(production):
    benchmark, _ = production
    changed = copy.deepcopy(benchmark.amendment2)
    changed["phases"][0]["seeds"] = [0, 1, 2, 3]
    with pytest.raises(DecisionRecordError, match="amendment2 block differs"):
        CampaignPlan.from_benchmark(dataclasses.replace(benchmark, amendment2=changed))
    overrides = dict(benchmark.profile_overrides)
    overrides["ppo_timesteps"] = {**overrides["ppo_timesteps"], "ObstacleTraverser-v1": 1_200_000}
    with pytest.raises(DecisionRecordError, match="ppo_timesteps"):
        CampaignPlan.from_benchmark(dataclasses.replace(benchmark, profile_overrides=overrides))


def test_a_test_labelled_record_never_launches_a_production_config(tmp_path, production):
    benchmark, _ = production
    record = json.loads(Path("docs/campaign_prep/ext/A2_DECISIONS.json").read_text())
    record["label"] = "TEST"
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    with pytest.raises(DecisionRecordError, match="not 'PRODUCTION'"):
        CampaignPlan.from_benchmark(dataclasses.replace(benchmark, decisions_path=str(path)))


# --- config equality against the original manifest ---------------------------------

def test_config_equality_passes_against_the_real_original_manifest(production):
    benchmark, plan = production
    original = amend2.original_manifest(plan.amendment2["original_campaign"])
    assert amend2.config_equality_problems(benchmark, plan, original) == []


def test_new_seed_configs_differ_from_the_original_only_in_the_seed(production):
    _, plan = production
    original = amend2.original_manifest(plan.amendment2["original_campaign"])
    reference = {(r["arm"], p["task"]): r for p in original["plan"]["phases"] for r in p["runs"]}
    for key in ("UpStepper-v0:s5-9", "Walker-v0:s5-9"):
        for run in _by_id(plan)[key].runs:
            ref = reference[(run.arm, run.task)]["config"]
            assert {k for k in set(ref) | set(run.config) if ref.get(k) != run.config.get(k)} == {"seed"}
            assert run.config["seed"] == run.seed and 5 <= run.seed <= 9


@pytest.mark.parametrize("mutate, expected", [
    (lambda m: m["plan"]["phases"][0]["runs"][0]["config"].update(population_size=14), "population_size"),
    (lambda m: m["plan"]["phases"][1]["runs"][0]["config"]["campaign_encoder_kwargs"].update(mutation_rate=0.5),
     "campaign_encoder_kwargs.mutation_rate"),
    (lambda m: m["plan"]["phases"][0].update(timeout_s=1.0), "timeout_s"),
    (lambda m: m["plan"]["phases"][1].update(ppo_seeds=1), "ppo_seeds"),
    (lambda m: m["plan"]["phases"][0].update(ppo_timesteps=900000), "ppo_timesteps"),
])
def test_config_equality_fails_on_any_difference_but_the_seed(production, mutate, expected):
    benchmark, plan = production
    original = copy.deepcopy(amend2.original_manifest(plan.amendment2["original_campaign"]))
    mutate(original)
    problems = amend2.config_equality_problems(benchmark, plan, original)
    assert problems and any(expected in p for p in problems), problems


def test_a_seed_only_difference_is_not_a_failure(production):
    benchmark, plan = production
    original = copy.deepcopy(amend2.original_manifest(plan.amendment2["original_campaign"]))
    for phase in original["plan"]["phases"]:
        for run in phase["runs"]:
            run["config"]["seed"] += 100
    assert amend2.config_equality_problems(benchmark, plan, original) == []


def test_only_a_test_labelled_config_may_skip_the_config_check(production):
    benchmark, plan = production
    refusals, record = amend2.launch_problems(Args(skip=True), benchmark, plan)
    assert any("only accepted for a TEST-labelled config" in r for r in refusals)
    assert record["config_equality"] != "skipped (TEST-labelled config, explicit flag)"
    test_benchmark = BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_a2_rehearsal.yaml"))
    test_plan = CampaignPlan.from_benchmark(test_benchmark)
    refusals, record = amend2.launch_problems(Args(skip=True), test_benchmark, test_plan)
    assert refusals == [] and record["config_equality"].startswith("skipped")


def test_launch_checks_pass_for_production_on_this_machine(production):
    benchmark, plan = production
    refusals, record = amend2.launch_problems(Args(), benchmark, plan)
    assert refusals == [] and record["version_check"] == "passed" and record["config_equality"] == "passed"
    assert amend2.launch_problems(Args(), None, dataclasses.replace(plan, amendment2=None)) == ([], {})


# --- library versions --------------------------------------------------------------

def test_version_check_passes_on_identical_versions_and_names_every_difference():
    recorded = dict(amend2.RECORDED_VERSIONS)
    assert amend2.version_problems(recorded, dict(recorded)) == []
    installed = dict(recorded, torch="2.15.0+cpu", evogym_commit="deadbeef", python="3.11.0")
    problems = amend2.version_problems(recorded, installed)
    assert len(problems) == 3 and any("torch" in p for p in problems) and any("evogym_commit" in p for p in problems)
    assert amend2.version_problems(recorded, {**recorded, "numpy": None})


def test_recorded_versions_name_every_library_the_amendment_requires():
    assert set(amend2.RECORDED_VERSIONS) == {"python", "torch", "stable-baselines3", "gymnasium", "numpy",
                                             "cloudpickle", "evogym", "evogym_commit"}
    assert amend2.RECORDED_VERSIONS["evogym_commit"].startswith("f63c649c")


def test_installed_versions_equal_the_recorded_ones_on_this_machine():
    assert amend2.version_problems(amend2.RECORDED_VERSIONS) == []


def test_a_version_mismatch_refuses_the_launch_even_for_a_test_rehearsal(monkeypatch, production):
    benchmark, plan = production
    monkeypatch.setattr(amend2, "installed_versions", lambda repo_root=None: {
        **amend2.RECORDED_VERSIONS, "stable-baselines3": "2.10.0"})
    refusals, record = amend2.launch_problems(Args(), benchmark, plan)
    assert any("stable-baselines3" in r for r in refusals) and record["version_check"] == "FAILED"
    test_benchmark = BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_a2_rehearsal.yaml"))
    refusals, _ = amend2.launch_problems(Args(skip=True), test_benchmark, CampaignPlan.from_benchmark(test_benchmark))
    assert any("stable-baselines3" in r for r in refusals)


def test_the_rehearsal_has_the_production_phase_structure_at_tiny_budgets():
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_a2_rehearsal.yaml")))
    assert plan.amendment2["label"] == "TEST"
    assert [p.key for p in plan.phases] == ["ObstacleTraverser-v1:s0", "ObstacleTraverser-v1:s1", "UpStepper-v0:s5",
                                            "Walker-v0:s5"]
    assert {p.ppo_timesteps for p in plan.phases} == {2048}
    assert [s.n for s in plan.stages if s.kind == "reeval"] == [2, 2, 2, 2]
    assert [s.budget for s in plan.stages if s.kind == "robustness"] == [4096] * 3
    assert [s.stage_id for s in plan.stages if s.kind == "robustness"][0] == "robustness:ObstacleTraverser-v1"
    assert any("original" in source for s in plan.stages if s.task == "UpStepper-v0" for source in s.sources)
