"""Smoke/correctness tests for evodevo/analysis/rule_based_settling_audit.py
-- small N, not the full N=2000 acceptance run (that is the committed
results/rule_based/SETTLING_AUDIT.md)."""
from evodevo.analysis import rule_based_settling_audit as A


def test_check_t_shift_implementation_finds_no_bug_small_probe():
    result = A.check_t_shift_implementation(n_probe=10, seed=0)
    assert result["n_probe"] == 10
    assert result["n_cross_check_mismatch"] == 0
    assert len(result["rows"]) == 10


def test_run_settling_audit_small_n_smoke():
    result = A.run_settling_audit(n=50, seed=0)
    assert result["n"] == 50
    assert 0.0 <= result["never_settled_fraction"] <= 1.0
    assert result["never_settled_count"] + result["n_settled"] == 50
    assert len(result["never_settled_genomes"]) == result["never_settled_count"]


def test_trace_example_genomes_returns_short_period_cycles():
    result = A.run_settling_audit(n=200, seed=0)
    examples = A.trace_example_genomes(result["never_settled_genomes"], n_examples=3)
    for ex in examples:
        assert ex["period"] <= 4
        assert len(ex["rules"]) >= 1
        assert len(ex["last_snapshots"]) == ex["period"] + 2


def test_render_report_contains_bug_check_and_period_sections():
    bug_check = A.check_t_shift_implementation(n_probe=10, seed=0)
    audit = A.run_settling_audit(n=100, seed=0)
    examples = A.trace_example_genomes(audit["never_settled_genomes"], n_examples=3)
    report = A.render_report(bug_check, audit, examples)
    assert "implementation-bug check" in report
    assert "never_settled_fraction" in report
    assert "Period-cycle breakdown" in report
