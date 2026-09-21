"""Tests for evodevo.preflight.readiness.

Unlike evodevo.preflight.__main__, this module has no import-order
constraint of its own (it does not configure process-wide threading) and
is safe to import directly in this process -- see its own module
docstring for why, and test_cli.py's docstring for the contrast.
"""
from __future__ import annotations

from evodevo.preflight.readiness import _find_pytest_summary_line, _section_8, generate_readiness_markdown

_EXPECTED_HEADINGS = [
    "## 1. Git state and config lock",
    "## 2. File inventory",
    "## 3. Test results",
    "## 4. Stage list",
    "## 5. `python -m evodevo.preflight --dry-run`",
    "## 6. B1 machine profile",
    "## 7. Self-check results",
    "## 8. Environment provenance",
    "## 9. Known gaps",
    "## 10. Estimated pre-flight wall clock",
    "## 11. Open questions",
]


def test_generates_all_eleven_sections_by_heading():
    markdown = generate_readiness_markdown(skip_full_suite=True, skip_machine_profile=True)
    for heading in _EXPECTED_HEADINGS:
        assert heading in markdown, f"missing section heading: {heading!r}"


def test_states_the_real_preflight_was_not_executed():
    markdown = generate_readiness_markdown(skip_full_suite=True, skip_machine_profile=True)
    assert "has NOT been executed" in markdown
    assert "python -m evodevo.preflight" in markdown


def test_section_6_collects_a_live_b1_profile_when_not_skipped():
    markdown = generate_readiness_markdown(skip_full_suite=True, skip_machine_profile=False)
    assert "not collected" not in markdown.split("## 6.")[1].split("## 7.")[0]
    assert '"topology"' in markdown


def test_section_6_states_not_collected_when_vetoed():
    markdown = generate_readiness_markdown(skip_full_suite=True, skip_machine_profile=True)
    section_6 = markdown.split("## 6.")[1].split("## 7.")[0]
    assert "not collected" in section_6


def test_section_7_reports_not_found_for_both_kinds_when_none_exist(tmp_path, monkeypatch):
    import evodevo.preflight.readiness as readiness_module

    monkeypatch.setattr(readiness_module, "REPO_ROOT", str(tmp_path))
    markdown = readiness_module._section_7()
    assert markdown.count("**Not found.**") == 2


def test_section_7_distinguishes_full_from_plain_self_check_runs(tmp_path, monkeypatch):
    import evodevo.preflight.readiness as readiness_module

    monkeypatch.setattr(readiness_module, "REPO_ROOT", str(tmp_path))
    base = tmp_path / "results" / "preflight"

    for run_id in ("selfcheck_20260101T000000Z", "selfcheck_full_20260101T000000Z"):
        run_dir = base / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            '{"run_id": "%s", "project_dirty": false, "stages": {"machine_profile": '
            '{"status": "done", "duration_s": 1.0}}}' % run_id
        )

    markdown = readiness_module._section_7()
    assert "**Not found.**" not in markdown
    assert "selfcheck_full_20260101T000000Z" in markdown
    assert "selfcheck_20260101T000000Z" in markdown


def test_find_pytest_summary_line_ignores_diagnostic_noise_printed_after_it():
    """Regression test: some tests (visualization/EvoGym ones) print
    diagnostic text to stdout AFTER pytest's own summary line -- this must
    not be mistaken for the real result (caught by hand-inspecting an
    early generated readiness doc, where Section 3 showed "Status: Using
    GLEW 2.2.0" instead of the actual pass/fail count)."""
    stdout = (
        "........................................ [100%]\n"
        "876 passed, 33 warnings in 686.84s (0:11:26)\n"
        "Using Evolution Gym Simulator v2.2.5\n"
        "Status: Using GLEW 2.2.0\n"
    )
    assert _find_pytest_summary_line(stdout) == "876 passed, 33 warnings in 686.84s (0:11:26)"


def test_find_pytest_summary_line_handles_failures_too():
    stdout = "F.. [100%]\n2 failed, 5 passed in 1.23s\n"
    assert _find_pytest_summary_line(stdout) == "2 failed, 5 passed in 1.23s"


def test_section_8_evogym_version_matches_b1s_own_report():
    """Regression test: readiness.py's Section 8 must report evogym's
    version via the same method B1 (b1_machine.py's _evogym_version())
    uses -- importlib.metadata, not module.__version__, which evogym does
    not define. They previously disagreed (Section 8 said "unknown" while
    a live B1 profile correctly said "2.0.0")."""
    from evodevo.preflight.stages.b1_machine import _evogym_version

    section_8_text = _section_8()
    b1_reported = _evogym_version()

    assert b1_reported is not None
    assert f"- evogym: `{b1_reported}`" in section_8_text
    assert "unknown" not in section_8_text


def test_main_writes_output_file(tmp_path):
    import evodevo.preflight.readiness as readiness_module

    out_path = tmp_path / "PREFLIGHT_READINESS.md"
    exit_code = readiness_module.main(["--skip-tests", "--skip-machine-profile", "--output", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()
    content = out_path.read_text()
    for heading in _EXPECTED_HEADINGS:
        assert heading in content
