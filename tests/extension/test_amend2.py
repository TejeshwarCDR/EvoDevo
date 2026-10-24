"""`python -m evodevo.extension amend2 derive|check`: deterministic, byte-for-byte, no hand edits."""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from evodevo.extension import amend2, spec
from evodevo.campaign import amend2 as runtime

DOCS = Path("docs/campaign_prep/ext")
ORIGINAL = Path("results/campaign/20260930T044002Z")
PREFLIGHT = Path("results/extension_preflight/ext-20261023")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """The committed Amendment 2 files copied to tmp and the module paths pointed at them."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("A2_DECISIONS.json", "A2_DECISIONS.md", "final_estimate_a2.json", "A2_GATES_REPORT.md",
                 "SUPERSEDED_BY_AMENDMENT_2.json", "EXT_DECISIONS.json"):
        shutil.copy(DOCS / name, docs / name)
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("campaign_a2.yaml", "campaign_a2_rehearsal.yaml", "a2_rehearsal_estimate.json"):
        shutil.copy(Path("evodevo/experiments/configs") / name, configs / name)
    for attr, path in (("DECISIONS_JSON", docs / "A2_DECISIONS.json"), ("DECISIONS_MD", docs / "A2_DECISIONS.md"),
                       ("ESTIMATE_JSON", docs / "final_estimate_a2.json"), ("GATES_REPORT", docs / "A2_GATES_REPORT.md"),
                       ("SUPERSEDED", docs / "SUPERSEDED_BY_AMENDMENT_2.json"),
                       ("PREFLIGHT_RECORD", docs / "EXT_DECISIONS.json"),
                       ("CAMPAIGN_YAML", configs / "campaign_a2.yaml"),
                       ("REHEARSAL_YAML", configs / "campaign_a2_rehearsal.yaml"),
                       ("REHEARSAL_ESTIMATE", configs / "a2_rehearsal_estimate.json")):
        monkeypatch.setattr(amend2, attr, path)
    return docs, configs


def _problems(**kw):
    # the plan is built from the real campaign_a2.yaml path inside check(); only text comparisons are exercised here
    return amend2.check(None, None, **kw)


# --- the committed files are consistent -------------------------------------------------

def test_committed_files_pass_check_without_inputs():
    assert amend2.check(None, None) == []


def test_committed_files_equal_a_fresh_derivation_from_the_preflight_and_original_campaign():
    assert amend2.check(PREFLIGHT, ORIGINAL) == []


def test_record_states_the_registered_design():
    record = json.loads((DOCS / "A2_DECISIONS.json").read_text())
    assert record["label"] == "PRODUCTION" and record["status"] == "FINAL" and record["format"] == runtime.RECORD_FORMAT
    assert len(record["git_commit"]) == 40
    assert record["tasks"]["ObstacleTraverser-v1"]["ppo_timesteps"] == 1_500_000
    assert [(t, record["tasks"][t]["ppo_timesteps"], record["tasks"][t]["ppo_seeds"]) for t in record["task_order"]] == [
        ("ObstacleTraverser-v1", 1_500_000, 1), ("UpStepper-v0", 1_200_000, 2), ("Walker-v0", 1_000_000, 1)]
    ids = [p["id"] for p in record["campaign_spec"]["phases"]]
    assert ids == ["ObstacleTraverser-v1:s0-4", "ObstacleTraverser-v1:s5-9", "UpStepper-v0:s5-9", "Walker-v0:s5-9"]
    assert record["gates"]["T"]["t_obstacle"] == 1_500_000 and record["gates"]["R"]["outcome"] == "PASS"
    assert record["reevaluation"]["seed_base"] == 20261101 and record["robustness"]["seed_base"] == 20261201
    assert record["robustness"]["budgets"] == {"ObstacleTraverser-v1": 3_000_000, "UpStepper-v0": 2_400_000,
                                               "Walker-v0": 2_000_000}
    assert record["robustness"]["order"][0] == "ObstacleTraverser-v1" and record["robustness"]["trainings"] == 270
    assert sum(record["reevaluation"]["trainings"].values()) == 900
    assert record["inputs"]["gates_report_sha256"] == runtime.sha256_file(DOCS / "A2_GATES_REPORT.md")
    assert record["inputs"]["preflight_record_sha256"] == runtime.sha256_file(DOCS / "EXT_DECISIONS.json")


def test_the_estimate_reproduces_the_gate_report_projection():
    estimate = json.loads((DOCS / "final_estimate_a2.json").read_text())
    bands = estimate["result"]["phase_h"]

    def rounded(key):
        return tuple(round(bands[key][q], 1) for q in ("p5", "mean", "p95"))
    assert rounded("ObstacleTraverser-v1:s0-4") == rounded("ObstacleTraverser-v1:s5-9") == (293.1, 325.7, 358.3)
    assert rounded("UpStepper-v0:s5-9") == (261.2, 290.2, 319.2)
    assert rounded("Walker-v0:s5-9") == (107.7, 119.7, 131.6)
    # the re-evaluation of 30 champions is the gate's 32.9 h split over two stages (each adds its own tail)
    obstacle = sum(bands[f"reeval:ObstacleTraverser-v1:{s}"]["mean"] for s in ("s0-4", "s5-9"))
    assert 32.5 < obstacle < 34.5
    assert round(bands["reeval:UpStepper-v0:s0-9"]["mean"], 1) == 14.6
    assert round(bands["reeval:Walker-v0:s0-9"]["mean"], 1) == 12.1
    assert round(estimate["result"]["total_h"]["mean"], -1) == 1160
    assert estimate["guard"]["applies"] is False and "360 h" in estimate["guard"]["outcome"]
    assert set(estimate["a2"]["timeout_s"]) == set(bands) == set(estimate["result"]["task_order"])


def test_yaml_is_the_rendering_of_the_record_and_parses_to_the_same_spec():
    record = json.loads((DOCS / "A2_DECISIONS.json").read_text())
    text = (Path("evodevo/experiments/configs/campaign_a2.yaml")).read_text()
    assert text == amend2.render_yaml(record)
    assert yaml.safe_load(text)["amendment2"] == record["campaign_spec"]
    assert text.startswith("# GENERATED by `python -m evodevo.extension amend2 derive` -- do not edit by hand.")


def test_the_rehearsal_files_are_generated_too():
    assert Path("evodevo/experiments/configs/campaign_a2_rehearsal.yaml").read_text() == amend2.render_rehearsal_yaml()
    assert Path("evodevo/experiments/configs/a2_rehearsal_estimate.json").read_text() == amend2.render_rehearsal_estimate()
    loaded = yaml.safe_load(amend2.render_rehearsal_yaml())
    assert loaded["amendment2"]["label"] == "TEST" and "decisions_path" not in loaded


# --- check refuses hand edits ------------------------------------------------------------

def test_check_passes_on_the_sandbox_copy_and_names_each_hand_edit(sandbox):
    docs, configs = sandbox
    assert _problems() == []
    edits = [
        (docs / "A2_DECISIONS.md", lambda t: t.replace("1,500,000", "1,400,000", 1), "A2_DECISIONS.md is not the rendering"),
        (configs / "campaign_a2.yaml", lambda t: t.replace("eval_budget: 300", "eval_budget: 299"),
         "campaign_a2.yaml is not the rendering"),
        (docs / "final_estimate_a2.json", lambda t: t.replace('"workers": 18', '"workers": 17'),
         "A2_DECISIONS.md is not the rendering" if False else "final_estimate_a2.json"),
        (docs / "A2_GATES_REPORT.md", lambda t: t + "\nedited\n", "A2_GATES_REPORT.md differs"),
        (docs / "EXT_DECISIONS.json", lambda t: t + " ", "differs from the version the record was derived from"),
        (configs / "campaign_a2_rehearsal.yaml", lambda t: t.replace("eval_budget: 30", "eval_budget: 31"),
         "campaign_a2_rehearsal.yaml is not the generated"),
        (configs / "a2_rehearsal_estimate.json", lambda t: t.replace("900.0", "901.0"),
         "a2_rehearsal_estimate.json is not the generated"),
    ]
    for path, edit, expected in edits:
        original = path.read_text()
        path.write_text(edit(original))
        problems = _problems()
        assert any(expected in problem for problem in problems), (path.name, problems)
        path.write_text(original)
        assert _problems() == [], path.name


def test_a_non_canonical_or_changed_record_is_refused(sandbox):
    docs, _ = sandbox
    path = docs / "A2_DECISIONS.json"
    original = path.read_text()
    record = json.loads(original)
    path.write_text(json.dumps(record, indent=4, sort_keys=True))
    assert any("not the canonical dump" in p for p in _problems())
    record["label"] = "TEST"
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    assert any("not 'PRODUCTION'" in p for p in _problems())
    record["label"] = "PRODUCTION"
    record["inputs"]["t_obstacle"] = 1_200_000
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    assert any("T_obstacle differs from the gates report" in p for p in _problems())


def test_a_missing_file_is_a_problem(sandbox):
    docs, _ = sandbox
    (docs / "final_estimate_a2.json").unlink()
    assert any("missing" in p and "final_estimate_a2.json" in p for p in _problems())


def test_superseded_record_must_hold_the_preflight_records_hash(sandbox):
    docs, _ = sandbox
    path = docs / "SUPERSEDED_BY_AMENDMENT_2.json"
    data = json.loads(path.read_text())
    data["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    assert any("does not hold the sha256" in p for p in _problems())


# --- derive -------------------------------------------------------------------------------

def test_the_gates_line_must_be_the_pass_line_for_the_same_t():
    assert amend2.parse_gates_first_line("GATE T: 1500000 | GATE R: PASS | GATE S: saving possible yes") == {
        "t_obstacle": 1500000}
    for line in ("GATE T: 1500000 | GATE R: FAIL | GATE S: saving possible yes",
                 "GATE T: 1500000 | GATE R: PASS | GATE S: saving possible no",
                 "GATE T: 1500000 | GATE R: PASS", "T = 1500000"):
        with pytest.raises(ValueError, match="not the expected pass line"):
            amend2.parse_gates_first_line(line)
    with pytest.raises(ValueError, match="Gate T"):
        amend2.derive(PREFLIGHT, ORIGINAL, 1_200_000)


def test_derive_is_deterministic_and_matches_the_committed_files_apart_from_stamp_and_commit():
    first = amend2.derive(PREFLIGHT, ORIGINAL, 1_500_000)
    second = amend2.derive(PREFLIGHT, ORIGINAL, 1_500_000)
    for result in (first, second):
        result["record"]["derived_at"] = "fixed"
    assert amend2._dump(first["record"]) == amend2._dump(second["record"])
    assert amend2._dump(first["estimate"]) == amend2._dump(second["estimate"])
    committed = json.loads((DOCS / "A2_DECISIONS.json").read_text())
    committed["derived_at"] = "fixed"
    committed["git_commit"] = first["record"]["git_commit"]
    assert amend2._dump(committed) == amend2._dump(first["record"])


def test_derive_refuses_a_dirty_tree(monkeypatch, capsys):
    monkeypatch.setattr(amend2, "_dirty", lambda: [" M evodevo/campaign/scheduler.py"])
    with pytest.raises(SystemExit):
        amend2.main(["derive", "--t-obstacle", "1500000"])
    assert "dirty working tree" in capsys.readouterr().err


def test_derive_requires_t_obstacle():
    with pytest.raises(SystemExit):
        amend2.main(["derive"])


def test_derive_refuses_a_preflight_record_that_is_no_longer_the_superseded_one(tmp_path, monkeypatch):
    changed = tmp_path / "EXT_DECISIONS.json"
    changed.write_text((DOCS / "EXT_DECISIONS.json").read_text() + " ")
    with pytest.raises(ValueError, match="no longer matches"):
        amend2.derive(PREFLIGHT, ORIGINAL, 1_500_000, preflight_record=changed)


def test_campaign_spec_encodes_the_design_exactly():
    spec_ = amend2.campaign_spec(1_500_000)
    assert spec_["sequence"][0] == "ObstacleTraverser-v1:s0-4" and spec_["sequence"][-1] == "robustness:Walker-v0"
    stages = {s["id"]: s for s in spec_["stages"]}
    assert stages["reeval:UpStepper-v0:s0-9"]["sources"] == [
        {"original": True, "arms": ["direct", "nca", "rule_based"], "seeds": [0, 1, 2, 3, 4]},
        {"phase": "UpStepper-v0:s5-9"}]
    assert stages["robustness:ObstacleTraverser-v1"]["sources"] == [
        {"phase": "ObstacleTraverser-v1:s0-4"}, {"phase": "ObstacleTraverser-v1:s5-9"}]
    packages = {p["id"]: p["package"] for p in spec_["phases"]}
    assert packages["ObstacleTraverser-v1:s0-4"]["label"] == "INTERIM"
    assert packages["ObstacleTraverser-v1:s5-9"]["label"] == "FINAL" and len(packages["ObstacleTraverser-v1:s5-9"]["merge"]) == 2
    assert packages["Walker-v0:s5-9"]["monotone_contrast"] and not packages["ObstacleTraverser-v1:s5-9"]["monotone_contrast"]
    assert [p["config_source"] for p in spec_["phases"]] == ["benchmark", "benchmark", "original", "original"]
    assert spec_["recorded_versions"]["evogym_commit"].startswith("f63c649c")


def test_the_cli_routes_amend2_through_the_extension_entry_point():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "evodevo.extension", "amend2", "check"], capture_output=True,
                            text=True)
    assert result.returncode == 0 and "amendment 2 check passed" in result.stdout
