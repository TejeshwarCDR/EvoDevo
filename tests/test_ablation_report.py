import os
import tempfile

from evodevo.analysis.ablation_report import run_ablation_report


def test_report_runs_with_no_ppo_results_yet():
    with tempfile.TemporaryDirectory() as tmp:
        empty_root = os.path.join(tmp, "runs")
        os.makedirs(empty_root)
        stats = run_ablation_report(root_dir=empty_root, output_dir=tmp, prior_seed=0)

        assert os.path.exists(os.path.join(tmp, "ablations.md"))

        for row in stats["a1"]:
            assert row["status"] == "not yet run"
            assert "prior_validity_rate" in row
        for row in stats["a2"]:
            assert row["status"] == "not yet run"
        # E3's direct rows have no prior stats; nca rows do.
        e3_by_arm = {row["arm"]: row for row in stats["e3"]}
        assert any("direct" in arm for arm in e3_by_arm)
        assert any("nca" in arm for arm in e3_by_arm)


def test_report_content_mentions_all_axes():
    with tempfile.TemporaryDirectory() as tmp:
        empty_root = os.path.join(tmp, "runs")
        os.makedirs(empty_root)
        run_ablation_report(root_dir=empty_root, output_dir=tmp, prior_seed=1)

        with open(os.path.join(tmp, "ablations.md")) as f:
            content = f.read()

        assert "dev_steps" in content
        assert "hidden_channels" in content
        assert "UpStepper-v0" in content
        assert "hypothesis" in content.lower()
