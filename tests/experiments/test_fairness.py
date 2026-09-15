import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.evolution.ea import resolved_policy_for
from evodevo.experiments.fairness import FairnessViolation, assert_comparable, assert_identical_policy


def _make_config(**overrides):
    defaults = dict(
        seed=0,
        encoder="direct",
        task="Walker-v0",
        population_size=10,
        offspring_size=10,
        eval_budget=100,
        ppo_timesteps=5000,
        ppo_eval_episodes=1,
        output_dir="results/runs",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_passes_when_only_encoder_differs():
    config_a = _make_config(encoder="direct")
    config_b = _make_config(encoder="nca")
    assert_comparable(config_a, config_b)  # must not raise


def test_passes_when_only_seed_and_encoder_differ():
    config_a = _make_config(encoder="direct", seed=0)
    config_b = _make_config(encoder="nca", seed=1)
    assert_comparable(config_a, config_b)  # must not raise


def test_raises_on_ppo_timesteps_mismatch():
    config_a = _make_config(ppo_timesteps=5000)
    config_b = _make_config(ppo_timesteps=10000)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_eval_budget_mismatch():
    config_a = _make_config(eval_budget=100)
    config_b = _make_config(eval_budget=200)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_sentinel_fitness_mismatch():
    config_a = _make_config(sentinel_fitness=-10.0)
    config_b = _make_config(sentinel_fitness=-5.0)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_task_mismatch():
    config_a = _make_config(task="Walker-v0")
    config_b = _make_config(task="UpStepper-v0")
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_population_size_mismatch():
    config_a = _make_config(population_size=10)
    config_b = _make_config(population_size=20)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_offspring_size_mismatch():
    config_a = _make_config(offspring_size=10)
    config_b = _make_config(offspring_size=20)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_ppo_eval_episodes_mismatch():
    config_a = _make_config(ppo_eval_episodes=1)
    config_b = _make_config(ppo_eval_episodes=5)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_grid_shape_mismatch():
    config_a = _make_config(grid_shape=(5, 5))
    config_b = _make_config(grid_shape=(6, 6))
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_raises_on_cache_enabled_mismatch():
    config_a = _make_config(cache_enabled=False)
    config_b = _make_config(cache_enabled=True)
    with pytest.raises(FairnessViolation):
        assert_comparable(config_a, config_b)


def test_violation_message_lists_all_mismatches():
    config_a = _make_config(ppo_timesteps=5000, eval_budget=100)
    config_b = _make_config(ppo_timesteps=10000, eval_budget=200)
    with pytest.raises(FairnessViolation) as exc_info:
        assert_comparable(config_a, config_b)
    message = str(exc_info.value)
    assert "ppo_timesteps" in message
    assert "eval_budget" in message


def test_assert_identical_policy_passes_for_identical_configs():
    config_a = _make_config(encoder="direct", sentinel_fitness=-10.0)
    config_b = _make_config(encoder="nca", sentinel_fitness=-10.0)
    policies = {
        "direct": resolved_policy_for(config_a),
        "nca": resolved_policy_for(config_b),
    }
    assert_identical_policy(policies)  # must not raise


def test_assert_identical_policy_raises_on_sentinel_fitness_mismatch():
    config_a = _make_config(encoder="direct", sentinel_fitness=-10.0)
    config_b = _make_config(encoder="nca", sentinel_fitness=-5.0)
    policies = {
        "direct": resolved_policy_for(config_a),
        "nca": resolved_policy_for(config_b),
    }
    with pytest.raises(FairnessViolation):
        assert_identical_policy(policies)


def test_assert_identical_policy_noop_for_a_single_arm():
    config_a = _make_config(encoder="direct")
    assert_identical_policy({"direct": resolved_policy_for(config_a)})  # must not raise


def test_assert_identical_policy_detects_tampered_policy():
    config_a = _make_config(encoder="direct")
    config_b = _make_config(encoder="nca")
    policy_a = resolved_policy_for(config_a)
    policy_b = dict(resolved_policy_for(config_b))
    policy_b["cache_charging_policy"] = "cache_hit_is_free"  # simulate drift
    with pytest.raises(FairnessViolation):
        assert_identical_policy({"direct": policy_a, "nca": policy_b})
