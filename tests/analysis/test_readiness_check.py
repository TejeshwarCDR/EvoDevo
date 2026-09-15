"""Tests for evodevo/analysis/readiness_check.py -- the P1-P7 verification
report generator. Focuses on the parts most prone to silent bugs (grep
comment/docstring filtering, risk-register construction), not on re-running
the full report end-to-end (that's covered by manually inspecting the
committed results/COMPARISON_READINESS.md)."""
import os

from evodevo.analysis.readiness_check import (
    _line_is_comment_or_string,
    _string_literal_line_ranges,
    build_risk_register,
    run_p1_checks,
)


def test_p1_docstring_mention_is_not_counted_as_code():
    """evodevo/encoders/rule_based/_rules/interpreter.py has two grep hits
    for the dev_steps/iterations pattern, both inside comments/docstrings
    (documented, not a hardcoded literal) -- this must be classified as
    PASS, not FAIL, and is the exact regression this test guards against
    (an earlier version of this check mismatched an absolute/relative path
    key and misclassified a docstring line as code)."""
    result = run_p1_checks()
    assert result["grep_clean"] is True
    assert result["grep_code_lines"] == []
    assert len(result["grep_raw_lines"]) >= 1  # the raw matches do exist


def test_string_literal_line_ranges_covers_a_known_docstring():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "evodevo", "encoders", "rule_based", "_rules", "interpreter.py")
    ranges = _string_literal_line_ranges(path)
    assert any(start <= 282 <= end for start, end in ranges)


def test_line_is_comment_or_string_detects_both_forms():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(repo_root, "evodevo", "encoders", "rule_based", "_rules", "interpreter.py")
    string_ranges = {path: _string_literal_line_ranges(path)}
    assert _line_is_comment_or_string(path, 180, string_ranges) is True  # '#' comment
    assert _line_is_comment_or_string(path, 282, string_ranges) is True  # docstring


def test_build_risk_register_flags_nca_never_settling():
    fake_depth = {
        "t_primary": 20,
        "layer2": {
            "nca": {
                "never_settled_fraction": 0.209, "never_settled_count": 418,
                "median_settle_step": 37.0, "p95_settle_step": 39.0,
                "stabilize_by_step_2": 0.0038,
            },
            "rule_based": {"stabilize_by_step_2": 0.71},
        },
    }
    risks = build_risk_register(None, fake_depth, None, None, [], {"exists": False})
    issues = [r[0] for r in risks]
    assert any("does not settle" in i for i in issues)
    assert any("BELOW NCA's own median settle_step" in i for i in issues)


def test_build_risk_register_flags_open_decisions_rows():
    risks = build_risk_register(None, None, None, None, ["Neighbourhood type", "Hidden channel count"], {"exists": False})
    assert any("still marked OPEN" in r[0] for r in risks)
    assert any(r[1] == "MAJOR" for r in risks if "OPEN" in r[0])


def test_build_risk_register_flags_stale_primary_results():
    staleness = {
        "exists": True, "any_run_meta": True, "n_runs": 5,
        "sample_path": "/fake/path/run_meta.json",
        "has_resolved_policy": False, "has_strategy_field": False,
        "factors_json_present": False, "predates_workstream": True,
    }
    risks = build_risk_register(None, None, None, None, [], staleness)
    assert any(r[1] == "BLOCKER" for r in risks)
    assert any("predate the ENTIRE P1-P7" in r[0] for r in risks)


def test_build_risk_register_only_the_always_present_finding_when_everything_else_clean():
    # The hardcoded-encoder-parameters finding (empty_prob/hidden_width/etc.)
    # is unconditional -- it is always true and always reported, regardless
    # of the other checks' outcomes. With no other data supplied and a
    # non-stale primary run, it must be the ONLY entry in the register.
    staleness = {"exists": True, "any_run_meta": True, "predates_workstream": False}
    risks = build_risk_register(None, None, None, None, [], staleness)
    assert len(risks) == 1
    assert "hardcoded" in risks[0][0]
