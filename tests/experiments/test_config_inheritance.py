import os
import tempfile

import pytest
import yaml

from evodevo.core.config import ConfigOverrideError, ExperimentConfig, SHARED_BASE_KEYS
from evodevo.experiments.fairness import assert_comparable

BASE_YAML = "evodevo/experiments/configs/shared/base.yaml"


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_per_method_config_overriding_a_shared_key_raises():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "base.yaml")
        override_path = os.path.join(tmp, "override.yaml")
        _write_yaml(base_path, {"population_size": 15, "eval_budget": 300})
        _write_yaml(
            override_path,
            {"seed": 0, "encoder": "direct", "task": "Walker-v0", "output_dir": "results/runs",
             "population_size": 999},  # illegally overrides a shared-base key
        )

        with pytest.raises(ConfigOverrideError, match="population_size"):
            ExperimentConfig.from_base_and_override(base_path, override_path)


def test_valid_per_method_config_loads_and_equals_expected_merged_dict():
    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "base.yaml")
        override_path = os.path.join(tmp, "override.yaml")
        _write_yaml(
            base_path,
            {
                "population_size": 4,
                "offspring_size": 4,
                "eval_budget": 20,
                "ppo_timesteps": 10,
                "ppo_eval_episodes": 1,
                "grid_shape": [5, 5],
                "sentinel_fitness": -10.0,
                "cache_enabled": False,
                "min_voxels": 3,
                "min_actuators": 1,
                "use_crossover": False,
            },
        )
        _write_yaml(
            override_path,
            {"seed": 3, "encoder": "nca", "task": "Walker-v0", "output_dir": "results/runs"},
        )

        config = ExperimentConfig.from_base_and_override(base_path, override_path)
        expected = ExperimentConfig(
            seed=3,
            encoder="nca",
            task="Walker-v0",
            population_size=4,
            offspring_size=4,
            eval_budget=20,
            ppo_timesteps=10,
            ppo_eval_episodes=1,
            output_dir="results/runs",
            grid_shape=(5, 5),
            sentinel_fitness=-10.0,
            cache_enabled=False,
            min_voxels=3,
            min_actuators=1,
            use_crossover=False,
        )
        assert config == expected


def test_all_three_methods_configs_load_and_pass_assert_comparable_pairwise():
    direct = ExperimentConfig.from_method_yaml("evodevo/experiments/configs/direct/walker.yaml")
    nca = ExperimentConfig.from_method_yaml("evodevo/experiments/configs/nca/walker.yaml")
    rule_based = ExperimentConfig.from_method_yaml("evodevo/experiments/configs/rule_based/walker.yaml")

    assert_comparable(direct, nca)
    assert_comparable(direct, rule_based)
    assert_comparable(nca, rule_based)


def test_shared_base_yaml_only_contains_shared_base_keys():
    with open(BASE_YAML) as f:
        data = yaml.safe_load(f)
    assert set(data.keys()) == SHARED_BASE_KEYS


def test_setting_a_different_ppo_timesteps_in_a_method_config_fails_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        # simulate a hand-edited configs/nca/walker.yaml that (incorrectly)
        # tries to set its own ppo_timesteps, diverging from the shared base.
        bad_override_path = os.path.join(tmp, "walker.yaml")
        _write_yaml(
            bad_override_path,
            {"seed": 0, "encoder": "nca", "task": "Walker-v0", "output_dir": "results/runs", "ppo_timesteps": 999999},
        )
        with pytest.raises(ConfigOverrideError, match="ppo_timesteps"):
            ExperimentConfig.from_base_and_override(BASE_YAML, bad_override_path)


def test_real_nca_config_directory_rejects_a_ppo_timesteps_override():
    # Same as above but exercised through the real configs/nca/ directory
    # layout and from_method_yaml's auto-discovery of the sibling base.yaml,
    # to prove this is enforced for the actual project configs, not just in
    # a synthetic setup.
    with tempfile.TemporaryDirectory() as tmp_configs_root:
        method_dir = os.path.join(tmp_configs_root, "nca")
        shared_dir = os.path.join(tmp_configs_root, "shared")
        os.makedirs(method_dir)
        os.makedirs(shared_dir)
        with open("evodevo/experiments/configs/shared/base.yaml") as f:
            base_contents = f.read()
        with open(os.path.join(shared_dir, "base.yaml"), "w") as f:
            f.write(base_contents)

        divergent_config_path = os.path.join(method_dir, "walker.yaml")
        _write_yaml(
            divergent_config_path,
            {"seed": 0, "encoder": "nca", "task": "Walker-v0", "output_dir": "results/runs", "ppo_timesteps": 1},
        )

        with pytest.raises(ConfigOverrideError, match="ppo_timesteps"):
            ExperimentConfig.from_method_yaml(divergent_config_path)
