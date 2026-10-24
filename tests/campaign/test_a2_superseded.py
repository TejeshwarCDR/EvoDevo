"""The 3-task extension design is retired by Amendment 2: its preflight record
is evidence only, and no plan or launch may be built from it."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evodevo.campaign import plan as plan_module
from evodevo.campaign.plan import CampaignPlan, DecisionRecordError
from evodevo.experiments.benchmark import BenchmarkConfig

CONFIGS = Path("evodevo/experiments/configs")
DOCS = Path("docs/campaign_prep/ext")


def test_superseded_record_holds_the_sha256_of_the_preflight_record():
    retired = json.loads((DOCS / "SUPERSEDED_BY_AMENDMENT_2.json").read_text())
    assert retired["superseded_path"] == "docs/campaign_prep/ext/EXT_DECISIONS.json"
    assert retired["sha256"] == hashlib.sha256((DOCS / "EXT_DECISIONS.json").read_bytes()).hexdigest()
    assert "Amendment 2" in retired["message"]


def test_plan_builder_refuses_the_three_task_extension_and_names_amendment_2():
    with pytest.raises(DecisionRecordError, match="Amendment 2"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(CONFIGS / "campaign_ext.yaml")))


def test_refusal_holds_for_a_copy_of_the_record_with_the_same_content(tmp_path):
    copy = tmp_path / "renamed_record.json"
    shutil.copy(DOCS / "EXT_DECISIONS.json", copy)
    text = (CONFIGS / "campaign_ext.yaml").read_text().replace(
        "decisions_path: docs/campaign_prep/ext/EXT_DECISIONS.json", f"decisions_path: {copy}")
    path = tmp_path / "bench.yaml"
    path.write_text(text)
    with pytest.raises(DecisionRecordError, match="Amendment 2"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(path)))


def test_a_different_record_is_not_refused_as_superseded(tmp_path):
    record = tmp_path / "other.json"
    record.write_text(json.dumps({"status": "DRAFT"}))
    text = (CONFIGS / "campaign_ext.yaml").read_text().replace(
        "decisions_path: docs/campaign_prep/ext/EXT_DECISIONS.json", f"decisions_path: {record}")
    path = tmp_path / "bench.yaml"
    path.write_text(text)
    with pytest.raises(DecisionRecordError, match="not 'FINAL'"):
        CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(str(path)))


def test_the_check_command_can_still_read_the_preflight_record():
    from evodevo.extension import derive
    assert derive.check(None, None) == []


def test_the_launcher_refuses_it_too(capsys):
    from evodevo.campaign.scheduler import cli_main
    for extra in ([], ["--dry-run"], ["--check-only"]):
        with pytest.raises(SystemExit) as exit_info:
            cli_main(["--config", str(CONFIGS / "campaign_ext.yaml"), "--no-mirror", "--no-notify",
                      "--campaign-id", "x", "--workers", "18", *extra])
        assert exit_info.value.code == 2
        assert "Amendment 2" in capsys.readouterr().err
