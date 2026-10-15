"""The extension must not change anything about the original campaign.

The pinned hashes were computed at commit 7774f76 (before any extension
code) with tests/fixtures/legacy_final_estimate_synthetic.json as the
estimate. Any change here means a resume of the original campaign would be
refused with "campaign config-hash mismatch".
"""
import json
import shutil
from pathlib import Path

import pytest

from evodevo.campaign.plan import (CampaignPlan, DecisionRecordError, LEGACY_GUARD_OUTCOME,
                                   resolve_estimate_path, LEGACY_ESTIMATE_PATH)
from evodevo.experiments.benchmark import BenchmarkConfig

FIXTURE = Path("tests/fixtures/legacy_final_estimate_synthetic.json")
CONFIGS = Path("evodevo/experiments/configs")
PINNED_AT_7774F76 = {
    "benchmark.yaml": "aba9fb3882d427ff9844b4477e5ac2677b0583e355dfb07e855aa94d1a15b411",
    "campaign_mini_validation.yaml": "098b870d402933657ca46eca80e3114b17f5c356c9fe3f4c807f79b5a71428dd",
    "campaign_resilience_validation.yaml": "20537f817abbcfe12c7377b1d397c0025748848ace1ce1ad05c78f0b20e09c65",
}


@pytest.mark.parametrize("name", sorted(PINNED_AT_7774F76))
def test_original_campaign_plan_hash_is_unchanged(name):
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / name)), estimate_path=FIXTURE)
    assert plan.config_hash == PINNED_AT_7774F76[name]
    assert plan.guard_outcome == LEGACY_GUARD_OUTCOME


def test_original_benchmark_still_uses_the_legacy_estimate_path():
    assert resolve_estimate_path(BenchmarkConfig.from_yaml(str(CONFIGS / "benchmark.yaml"))) == LEGACY_ESTIMATE_PATH


def test_original_validation_profile_keeps_its_time_scales():
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_validation.yaml")),
                                       estimate_path=FIXTURE)
    assert plan.validation_profile["time_scale"] == {"Walker-v0": 1 / 100, "UpStepper-v0": 1 / 250}
    assert plan.validation_profile["e_over_p_ratio"] == 2.3172


def test_original_benchmark_yaml_and_profile_are_untouched():
    bconfig = BenchmarkConfig.from_yaml(str(CONFIGS / "benchmark.yaml"))
    assert bconfig.tasks == ["Walker-v0", "UpStepper-v0"]
    assert [a.arm_id for a in bconfig.resolved_arms()] == ["direct", "nca", "rule_based", "rule_based_monotone"]
    assert bconfig.estimate_path is None and bconfig.decisions_path is None and bconfig.guard_outcome is None


def test_per_method_config_must_declare_the_requested_task(tmp_path):
    root = tmp_path / "configs"
    shutil.copytree(CONFIGS, root)
    (root / "nca" / "climber.yaml").write_text("seed: 0\nencoder: nca\ntask: Climber-v1\noutput_dir: results/runs\n")
    text = (CONFIGS / "campaign_ext_mini_validation.yaml").read_text()
    text = text.replace("configs_root: evodevo/experiments/configs", f"configs_root: {root}")
    path = tmp_path / "bench.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="declares task 'Climber-v1'"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(path)))


def test_every_extension_method_config_names_its_exact_task():
    for encoder in ("direct", "nca", "rule_based"):
        for stem, task in (("obstacletraverser", "ObstacleTraverser-v1"), ("climber", "Climber-v0"),
                           ("cavecrawler", "CaveCrawler-v0")):
            text = (CONFIGS / encoder / f"{stem}.yaml").read_text()
            assert f"task: {task}\n" in text and f"encoder: {encoder}\n" in text


def test_extension_campaign_refuses_without_a_final_decision_record():
    with pytest.raises(DecisionRecordError, match="does not exist"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_ext.yaml")))


def _record(tmp_path, **changes):
    record = {"status": "FINAL", "task_order": ["ObstacleTraverser-v1", "Climber-v0", "CaveCrawler-v0"],
              "arms": ["direct", "nca", "rule_based"], "seeds": [0, 1],
              "shared": {"population_size": 15, "offspring_size": 15, "eval_budget": 30,
                         "ppo_seed_aggregation": "max"},
              "tasks": {"ObstacleTraverser-v1": {"ppo_timesteps": 2048, "ppo_seeds": 1},
                        "Climber-v0": {"ppo_timesteps": 2048, "ppo_seeds": 2},
                        "CaveCrawler-v0": {"ppo_timesteps": 2048, "ppo_seeds": 1}}}
    record.update(changes)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    text = (CONFIGS / "campaign_ext_mini_validation.yaml").read_text() + f"decisions_path: {path}\n"
    bench = tmp_path / "bench.yaml"
    bench.write_text(text)
    return BenchmarkConfig.from_yaml(str(bench))


def test_matching_final_record_builds_the_plan(tmp_path):
    plan = CampaignPlan.from_benchmark(_record(tmp_path))
    assert [(p.task, p.ppo_timesteps, p.ppo_seeds) for p in plan.phases] == [
        ("ObstacleTraverser-v1", 2048, 1), ("Climber-v0", 2048, 2), ("CaveCrawler-v0", 2048, 1)]
    assert {r.arm for p in plan.phases for r in p.runs} == {"direct", "nca", "rule_based"}


@pytest.mark.parametrize("changes, message", [
    ({"status": "DRAFT"}, "not 'FINAL'"),
    ({"tasks": {"ObstacleTraverser-v1": {"ppo_timesteps": 4096, "ppo_seeds": 1},
                "Climber-v0": {"ppo_timesteps": 2048, "ppo_seeds": 2},
                "CaveCrawler-v0": {"ppo_timesteps": 2048, "ppo_seeds": 1}}}, "ppo_timesteps"),
    ({"tasks": {"ObstacleTraverser-v1": {"ppo_timesteps": 2048, "ppo_seeds": 1},
                "Climber-v0": {"ppo_timesteps": 2048, "ppo_seeds": 1},
                "CaveCrawler-v0": {"ppo_timesteps": 2048, "ppo_seeds": 1}}}, "ppo_seeds"),
    ({"task_order": ["Climber-v0", "ObstacleTraverser-v1", "CaveCrawler-v0"]}, "task order"),
    ({"arms": ["direct", "nca", "rule_based", "rule_based_monotone"]}, "arms"),
    ({"seeds": [0, 1, 2]}, "seeds"),
    ({"shared": {"eval_budget": 300}}, "eval_budget"),
])
def test_any_disagreement_with_the_record_refuses(tmp_path, changes, message):
    with pytest.raises(DecisionRecordError, match=message):
        CampaignPlan.from_benchmark(_record(tmp_path, **changes))


def test_missing_t_ppo_prior_for_a_task_is_a_clear_error(tmp_path):
    estimate = json.loads((CONFIGS / "ext_validation_estimate.json").read_text())
    estimate["inputs"]["t_ppo_mean_sd_at_b3_footing"] = {
        k: v for k, v in estimate["inputs"]["t_ppo_mean_sd_at_b3_footing"].items() if "Climber" not in k}
    path = tmp_path / "estimate.json"
    path.write_text(json.dumps(estimate))
    with pytest.raises(ValueError, match="no t_PPO prior for task 'Climber-v0'"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_ext_mini_validation.yaml")),
                                    estimate_path=path)


def test_extension_validation_configs_build_with_three_arms_and_their_own_estimate():
    for name, seeds in (("campaign_ext_mini_validation.yaml", [0, 1]), ("campaign_ext_validation.yaml",
                                                                         [0, 1, 2, 3, 4])):
        plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / name)))
        assert [p.task for p in plan.phases] == ["ObstacleTraverser-v1", "Climber-v0", "CaveCrawler-v0"]
        for phase in plan.phases:
            assert sorted({r.arm for r in phase.runs}) == ["direct", "nca", "rule_based"]
            assert sorted({r.seed for r in phase.runs}) == seeds
            assert phase.timeout_s > 0
