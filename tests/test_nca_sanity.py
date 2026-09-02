import json
import os
import tempfile

from evodevo.experiments.nca_sanity import run_sanity


def test_rerun_same_seed_reproduces_identical_statistics():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        stats1 = run_sanity(seed=7, n=50, output_dir=tmp1, n_example_trajectories=2)
        stats2 = run_sanity(seed=7, n=50, output_dir=tmp2, n_example_trajectories=2)
        assert stats1 == stats2

        with open(os.path.join(tmp1, "nca_sanity.json")) as f:
            json1 = json.load(f)
        with open(os.path.join(tmp2, "nca_sanity.json")) as f:
            json2 = json.load(f)
        assert json1 == json2


def test_different_seeds_can_differ():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        stats1 = run_sanity(seed=1, n=50, output_dir=tmp1, n_example_trajectories=1)
        stats2 = run_sanity(seed=2, n=50, output_dir=tmp2, n_example_trajectories=1)
        assert stats1 != stats2


def test_outputs_written():
    with tempfile.TemporaryDirectory() as tmp:
        run_sanity(seed=0, n=30, output_dir=tmp, n_example_trajectories=3)
        assert os.path.exists(os.path.join(tmp, "nca_sanity.md"))
        assert os.path.exists(os.path.join(tmp, "nca_sanity.json"))
        assert os.path.exists(os.path.join(tmp, "nca_sanity_trajectories.png"))


def test_stats_shape_and_ranges():
    with tempfile.TemporaryDirectory() as tmp:
        stats = run_sanity(seed=0, n=100, output_dir=tmp, n_example_trajectories=3)

        assert stats["n"] == 100
        for key in ("nca", "direct_reference"):
            block = stats[key]
            assert 0.0 <= block["valid_rate"] <= 1.0
            assert 0.0 <= block["degenerate_fraction"] <= 1.0
            assert 0 <= block["unique_body_count"] <= 100

        assert 0.0 <= stats["nca"]["never_settled_fraction"] <= 1.0
        assert len(stats["nca"]["settle_step_distribution"]) == stats["nca_config"]["dev_steps"] + 1
        assert sum(stats["nca"]["settle_step_distribution"]) == 100

        mag = stats["nca"]["state_magnitude_at_T"]
        assert mag["min"] <= mag["mean"] <= mag["max"]
