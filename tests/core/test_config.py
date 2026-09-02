import os
import tempfile

from evodevo.core.config import ExperimentConfig


def _make_config(**overrides):
    defaults = dict(
        seed=42,
        encoder="direct",
        task="Walker-v0",
        population_size=10,
        offspring_size=10,
        eval_budget=100,
        ppo_timesteps=1000,
        ppo_eval_episodes=3,
        output_dir="results/runs",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_config_is_frozen():
    config = _make_config()
    try:
        config.seed = 1
        assert False, "ExperimentConfig should be frozen"
    except Exception:
        pass


def test_config_to_dict():
    config = _make_config()
    d = config.to_dict()
    assert d["seed"] == 42
    assert d["grid_shape"] == [5, 5]


def test_config_yaml_roundtrip():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        config.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded == config


def test_example_configs_differ_only_in_encoder():
    direct = ExperimentConfig.from_method_yaml("evodevo/experiments/configs/direct/walker.yaml")
    nca = ExperimentConfig.from_method_yaml("evodevo/experiments/configs/nca/walker.yaml")
    direct_dict = direct.to_dict()
    nca_dict = nca.to_dict()
    del direct_dict["encoder"]
    del nca_dict["encoder"]
    assert direct_dict == nca_dict
    assert direct.encoder != nca.encoder
