import json
import os
import tempfile

import pytest
import yaml

from evodevo.analysis.aggregate import load_runs
from evodevo.experiments.run_experiment import run_experiment


def _write_tiny_config(path: str, output_dir: str, eval_budget: int = 12, encoder: str = "direct"):
    config = dict(
        seed=0,
        encoder=encoder,
        task="dummy-task",
        grid_shape=[5, 5],
        population_size=4,
        offspring_size=4,
        eval_budget=eval_budget,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir=output_dir,
    )
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


def test_end_to_end_surrogate_run_produces_expected_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_root = os.path.join(tmp, "results")
        _write_tiny_config(config_path, output_root, eval_budget=12)

        output_dir = run_experiment(config_path, seed=0, surrogate=True, overwrite=False)

        assert output_dir == os.path.join(output_root, "direct", "dummy-task", "seed_0")
        assert os.path.exists(os.path.join(output_dir, "run_meta.json"))

        log_path = os.path.join(output_dir, "run_log.jsonl")
        with open(log_path, "r") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 12

        with open(os.path.join(output_dir, "run_meta.json"), "r") as f:
            meta = json.load(f)
        assert meta["config"]["eval_budget"] == 12
        assert meta["encoder"]["name"] == "direct"


def test_aggregate_loads_run():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_root = os.path.join(tmp, "results")
        _write_tiny_config(config_path, output_root, eval_budget=8)

        output_dir = run_experiment(config_path, seed=0, surrogate=True, overwrite=False)
        df = load_runs([output_dir])

        assert len(df) == 8
        assert set(["encoder", "task", "seed", "fitness", "eval_index"]) <= set(df.columns)
        assert (df["encoder"] == "direct").all()
        assert (df["seed"] == 0).all()


def test_rerun_without_overwrite_raises():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_root = os.path.join(tmp, "results")
        _write_tiny_config(config_path, output_root, eval_budget=8)

        run_experiment(config_path, seed=0, surrogate=True, overwrite=False)

        with pytest.raises(FileExistsError):
            run_experiment(config_path, seed=0, surrogate=True, overwrite=False)


def test_rerun_with_overwrite_succeeds():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_root = os.path.join(tmp, "results")
        _write_tiny_config(config_path, output_root, eval_budget=8)

        run_experiment(config_path, seed=0, surrogate=True, overwrite=False)
        # Should not raise.
        output_dir = run_experiment(config_path, seed=0, surrogate=True, overwrite=True)
        assert os.path.exists(os.path.join(output_dir, "run_meta.json"))


def test_encoder_kwargs_and_tag_disambiguate_output_and_reach_the_encoder():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.yaml")
        output_root = os.path.join(tmp, "results")
        _write_tiny_config(config_path, output_root, eval_budget=6, encoder="nca")

        output_dir_a = run_experiment(
            config_path, seed=0, surrogate=True, overwrite=False,
            encoder_kwargs={"dev_steps": 1}, tag="T1",
        )
        output_dir_b = run_experiment(
            config_path, seed=0, surrogate=True, overwrite=False,
            encoder_kwargs={"dev_steps": 20}, tag="T20",
        )

        assert output_dir_a != output_dir_b
        assert output_dir_a.endswith(os.path.join("nca", "dummy-task", "T1", "seed_0"))

        with open(os.path.join(output_dir_a, "run_meta.json")) as f:
            meta_a = json.load(f)
        with open(os.path.join(output_dir_b, "run_meta.json")) as f:
            meta_b = json.load(f)

        assert meta_a["encoder"]["dev_steps"] == 1
        assert meta_b["encoder"]["dev_steps"] == 20
        assert meta_a["config"]["encoder_kwargs"] == {"dev_steps": 1}
        assert meta_a["config"]["tag"] == "T1"
