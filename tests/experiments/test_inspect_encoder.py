import json
import os
import tempfile
import warnings

from evodevo.experiments.inspect_encoder import inspect_encoder
from evodevo.experiments.nca_sanity import run_sanity


def test_works_for_direct_encoder_alone():
    with tempfile.TemporaryDirectory() as tmp:
        stats = inspect_encoder("direct", seed=0, n=50, output_dir=tmp, n_example_trajectories=0)
        assert "direct" in stats
        assert 0.0 <= stats["direct"]["valid_rate"] <= 1.0
        assert os.path.exists(os.path.join(tmp, "direct_sanity.json"))
        assert os.path.exists(os.path.join(tmp, "direct_sanity.md"))
        # direct has no develop() trajectory hook -- no settling stats.
        assert "settle_step_distribution" not in stats["direct"]


def test_works_for_rule_based_encoder_alone():
    with tempfile.TemporaryDirectory() as tmp:
        stats = inspect_encoder("rule_based", seed=0, n=50, output_dir=tmp, n_example_trajectories=0)
        assert "rule_based" in stats
        assert 0.0 <= stats["rule_based"]["valid_rate"] <= 1.0
        assert os.path.exists(os.path.join(tmp, "rule_based_sanity.json"))


def test_nca_with_direct_reference_matches_the_original_shim_shape():
    with tempfile.TemporaryDirectory() as tmp:
        stats = inspect_encoder(
            "nca", seed=0, n=50, output_dir=tmp, n_example_trajectories=2, compare_to="direct"
        )
        for key in ("nca", "direct_reference", "nca_config", "direct_config"):
            assert key in stats
        assert "settle_step_distribution" in stats["nca"]
        assert "settle_step_distribution" not in stats["direct_reference"]


def test_deprecated_shim_reproduces_inspect_encoder_directly():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        direct_stats = inspect_encoder(
            "nca", seed=3, n=40, output_dir=tmp1, n_example_trajectories=1, compare_to="direct", report_prefix="nca"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            shim_stats = run_sanity(seed=3, n=40, output_dir=tmp2, n_example_trajectories=1)
        assert direct_stats == shim_stats

        with open(os.path.join(tmp1, "nca_sanity.json")) as f:
            json1 = json.load(f)
        with open(os.path.join(tmp2, "nca_sanity.json")) as f:
            json2 = json.load(f)
        assert json1 == json2


def test_encoder_kwargs_are_applied():
    with tempfile.TemporaryDirectory() as tmp:
        stats = inspect_encoder(
            "nca", seed=0, n=30, output_dir=tmp, n_example_trajectories=0, encoder_kwargs={"dev_steps": 1}
        )
        assert stats["nca_config"]["dev_steps"] == 1
        assert len(stats["nca"]["settle_step_distribution"]) == 2  # dev_steps + 1
