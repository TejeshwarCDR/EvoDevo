"""Tests for evodevo/analysis/rule_based_readiness_check.py -- the rule_based
architecture readiness report generator. Focuses on the trickiest parts:
the tokenize-based literal grep, the OPEN-row scanner not false-positiving
on prose mentioning "OPEN", and an end-to-end check that two sections'
printed values match byte-for-byte where they should."""
import os
import re
import tempfile

from evodevo.analysis.readiness_check import scan_decisions_open_rows
from evodevo.analysis.rule_based_readiness_check import (
    gather_provenance,
    render_section0,
    run_rule_based_readiness_check,
    run_section10_checks,
    run_section8_check,
)


def test_gather_provenance_reads_from_resolved_config_not_a_literal():
    """Every Section 0 config value must trace back to
    resolved_config.py::PRIMARY or an encoder constructor default -- not a
    retyped literal in this module."""
    from evodevo.encoders.rule_based.resolved_config import PRIMARY

    prov = gather_provenance()
    assert prov["config"]["conflict_resolution"][0] == PRIMARY.conflict_resolution
    assert prov["config"]["position_predicate"][0] == PRIMARY.position_predicate
    assert prov["config"]["action_monotonicity"][0] == PRIMARY.action_monotonicity


def test_section0_renders_without_error():
    prov = gather_provenance()
    text = render_section0(prov)
    assert "conflict_resolution" in text
    assert "resolved_config.py" in text


def test_section10_self_consistency_all_pass_on_the_real_repo():
    """End-to-end regression check: on this repo's actual committed state,
    every self-consistency check must PASS -- if this ever fails, it is a
    genuine regression, not a test artifact."""
    prov = gather_provenance()
    s8 = run_section8_check()
    s10 = run_section10_checks(prov, s8)
    for c in s10["checks"]:
        assert c["passed"], f"{c['name']}: {c['detail']}"
    assert s10["all_passed"]


def test_no_hardcoded_literal_check_has_a_real_grep_pattern():
    prov = gather_provenance()
    s8 = run_section8_check()
    s10 = run_section10_checks(prov, s8)
    grep_check = next(c for c in s10["checks"] if "hardcoded literal" in c["name"])
    assert grep_check["grep_pattern"]
    assert grep_check["passed"] is True


def test_scan_decisions_open_rows_ignores_the_word_open_in_resolved_rationale(tmp_path):
    """Regression test reusing readiness_check.py's own OPEN-row scanner:
    a resolved row's Chosen column must not say bare 'OPEN', but its
    Rationale is free to discuss the word 'OPEN' in prose without being
    misdetected as a still-open row."""
    import evodevo.analysis.readiness_check as rc

    fake_decisions = tmp_path / "decisions.md"
    fake_decisions.write_text(
        "| Date | Decision | Options considered | Chosen | Rationale |\n"
        "|---|---|---|---|---|\n"
        "| 2026-09-16 | rule_based R4 status | a vs b | monotone proved | "
        "This proposition was previously left OPEN in the spec; established here. |\n"
        "| | Some other rule_based row | x vs y | OPEN | |\n"
    )
    original = rc.DECISIONS_MD
    rc.DECISIONS_MD = str(fake_decisions)
    try:
        open_rows = scan_decisions_open_rows()
    finally:
        rc.DECISIONS_MD = original

    assert open_rows == ["Some other rule_based row"]


def test_end_to_end_report_generation_t_primary_consistency_and_no_forbidden_literals():
    """Runs the real pipeline once (it re-runs t_primary_unified's N=2000
    sweep internally, so this is deliberately the only end-to-end
    invocation in this file) and checks three things against that single
    generated report: (1) Section 8's T_primary printed value matches
    benchmark.yaml's dev_steps byte-for-byte, not merely 'both look right
    by luck'; (2) it does not contain H_ACT=/N_MATERIALS= style substrings,
    which would false-positive tests/core/test_materials.py's own grep;
    (3) the risk register and verdict sections are non-empty."""
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "RULE_BASED_READINESS.md")
        run_rule_based_readiness_check(output_path=output_path)
        with open(output_path) as f:
            text = f.read()

    assert "## Section 0" in text
    assert "## Section 12" in text

    yaml_match = re.search(r"benchmark\.yaml's own `dev_steps` = \*\*(\d+)\*\*", text)
    rerun_match = re.search(r"re-derived shared T_primary = \*\*(\d+)\*\*", text)
    assert yaml_match is not None
    assert rerun_match is not None
    assert yaml_match.group(1) == rerun_match.group(1)

    assert "H_ACT=" not in text
    assert "N_MATERIALS=" not in text

    assert "Section 11 -- Risk register" in text
    assert text.count("\n- ") > 3
