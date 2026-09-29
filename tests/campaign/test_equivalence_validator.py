import importlib.util
from pathlib import Path

from evodevo.campaign.plan import CampaignPlan
from evodevo.experiments.benchmark import BenchmarkConfig


ROOT = Path(__file__).resolve().parents[2]


def _validator_module():
    path = ROOT / "scripts/campaign/validate_equivalence.py"
    spec = importlib.util.spec_from_file_location("validate_equivalence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equivalence_normalization_removes_only_operational_fields():
    module = _validator_module()
    value = {
        "fitness": 1.25,
        "wall_clock_s": 4.0,
        "policy_paths": ["somewhere.zip"],
        "nested": {"worker_slot": 3, "genome": [1, 2, 3]},
    }
    assert module.normalized(value) == {
        "fitness": 1.25,
        "nested": {"genome": [1, 2, 3], "worker_slot": 3},
    }


def test_real_ppo_mini_profile_is_the_required_step6_and_step9_matrix():
    config = BenchmarkConfig.from_yaml(
        str(ROOT / "evodevo/experiments/configs/campaign_mini_validation.yaml")
    )
    plan = CampaignPlan.from_benchmark(config)
    assert [phase.task for phase in plan.phases] == ["Walker-v0", "UpStepper-v0"]
    assert all(len(phase.runs) == 8 for phase in plan.phases)
    assert {run.arm for run in plan.phases[0].runs} == {
        "direct", "nca", "rule_based", "rule_based_monotone"
    }
    assert all(run.config["population_size"] == 15 for phase in plan.phases for run in phase.runs)
    assert all(run.config["eval_budget"] == 30 for phase in plan.phases for run in phase.runs)
    assert [phase.ppo_timesteps for phase in plan.phases] == [2000, 2000]
    assert [phase.ppo_seeds for phase in plan.phases] == [1, 2]
    assert all(phase.aggregation == "max" for phase in plan.phases)


def test_resilience_profile_preserves_arms_phases_and_k():
    config = BenchmarkConfig.from_yaml(
        str(ROOT / "evodevo/experiments/configs/campaign_resilience_validation.yaml")
    )
    plan = CampaignPlan.from_benchmark(config)
    assert [phase.task for phase in plan.phases] == ["Walker-v0", "UpStepper-v0"]
    assert [phase.ppo_seeds for phase in plan.phases] == [1, 2]
    assert all(len(phase.runs) == 4 for phase in plan.phases)
    assert all(run.config["population_size"] == 3 for phase in plan.phases for run in phase.runs)
    assert all(run.config["eval_budget"] == 12 for phase in plan.phases for run in phase.runs)
