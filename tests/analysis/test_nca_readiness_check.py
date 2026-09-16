"""Smoke/correctness tests for evodevo/analysis/nca_readiness_check.py.
Full-report generation (python -m evodevo.analysis.nca_readiness_check) is
the acceptance run and takes ~15s reading/measuring against real committed
artifacts; these tests check the individual pieces cheaply."""
import numpy as np
import pytest

from evodevo.analysis import nca_readiness_check as N


def test_table_rows_extracts_rows_between_header_and_blank_line():
    text = (
        "intro text\n\n"
        "| a | b | c |\n"
        "| --- | --- | --- |\n"
        "| 1 | 2 | 3 |\n"
        "| 4 | 5 | 6 |\n"
        "\n"
        "more text\n"
    )
    rows = N._table_rows(text, "| a | b | c |")
    assert rows == [["1", "2", "3"], ["4", "5", "6"]]


def test_table_rows_returns_empty_for_missing_header():
    assert N._table_rows("no tables here", "| a | b |") == []


def test_table_rows_handles_none_text():
    assert N._table_rows(None, "| a |") == []


def test_first_match_returns_group_or_none():
    assert N._first_match("D* = 2.5000", r"D\* = ([\d.]+)") == "2.5000"
    assert N._first_match("nothing here", r"D\* = ([\d.]+)") is None
    assert N._first_match(None, r"x") is None


def test_gather_provenance_config_values_are_read_off_a_live_encoder():
    prov = N.gather_provenance()
    assert prov["config"]["update_rule"][0] == "bounded_leaky"
    assert prov["config"]["state_bound (B*)"][0] == N.PRIMARY.state_bound
    assert prov["config"]["decay (lambda*)"][0] == N.PRIMARY.decay
    assert prov["config"]["hidden_channels (H)"][0] == N.PRIMARY.hidden_channels
    assert "resolved_encoder" in prov


def test_run_section1_check_small_grid_reports_zero_violations():
    result = N.run_section1_check(n_genomes=10, t_steps=15, h_values=(0, 3), seed=0)
    assert result["violations"] == 0
    assert result["passed"] is True
    assert result["total_checks"] == 10 * 15 * len(result["boundaries"]) * 2


def test_run_section9_checks_grep_excludes_own_descriptive_strings():
    prov = N.gather_provenance()
    s8 = {"table_rows": [["T_primary implied", "39", "39", "unchanged"]]}
    # Patch DEPTH_STUDY_RAW read indirectly isn't needed here -- just check
    # the grep check specifically doesn't flag itself on its own name string.
    result = N.run_section9_checks(prov, s8)
    grep_check = next(c for c in result["checks"] if "hardcoded literal" in c["name"])
    assert grep_check["passed"] is True, grep_check["detail"]


def test_regression_check_reports_ok_for_a_file_unchanged_since_head(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import subprocess

    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "t"], check=True)
    f = tmp_path / "report.md"
    f.write_text("| direct | 1 | 2 |\n| rule_based | 3 | 4 |\n| nca | 5 | 6 |\n")
    subprocess.run(["git", "add", "report.md"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], check=True)

    monkeypatch.setattr(N, "REPO_ROOT", str(tmp_path))
    result = N._regression_check(str(f))
    assert result["ok"] is True


def test_regression_check_detects_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import subprocess

    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "t"], check=True)
    f = tmp_path / "report.md"
    f.write_text("| direct | 1 | 2 |\n| rule_based | 3 | 4 |\n| nca | 5 | 6 |\n")
    subprocess.run(["git", "add", "report.md"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], check=True)

    f.write_text("| direct | 999 | 2 |\n| rule_based | 3 | 4 |\n| nca | 5 | 6 |\n")

    monkeypatch.setattr(N, "REPO_ROOT", str(tmp_path))
    result = N._regression_check(str(f))
    assert result["ok"] is False
    assert result["mismatches"][0]["arm"] == "direct"


def test_full_report_generation_smoke(tmp_path):
    """End-to-end run against the real committed artifacts (fast: no
    N=2000 sweep is re-run, only cheap fresh checks + artifact reads)."""
    output_path = str(tmp_path / "NCA_READINESS.md")
    result_path = N.run_nca_readiness_check(output_path=output_path)
    assert result_path == output_path

    with open(output_path) as f:
        text = f.read()

    for section in [
        "Section 0 -- Provenance", "Section 1 -- Proposition 1", "Section 2 -- Proposition 2",
        "Section 3 -- Timescale calibration", "Section 4 -- Genome space",
        "Section 5 -- Readout and hidden channels", "Section 6 -- Mutation re-calibration",
        "Section 7 -- Regeneration", "Section 8 -- Before/after delta",
        "Section 9 -- Self-consistency", "Section 10 -- Risk register", "Section 11 -- Verdict",
    ]:
        assert section in text, f"missing {section}"
