"""Tests for evodevo.preflight.dirty_check against real, disposable git
repositories -- this module has no import-order constraint of its own
(unlike evodevo.preflight.__main__) and is safe to test in-process.
"""
from __future__ import annotations

import subprocess

from evodevo.preflight.dirty_check import dirty_paths_outside_results, project_is_dirty


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def _commit_all(path, message="initial"):
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)


def test_clean_tree_is_accepted(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n")
    _commit_all(tmp_path)

    assert project_is_dirty(str(tmp_path)) is False
    assert dirty_paths_outside_results(str(tmp_path)) == []


def test_dirty_tree_is_detected(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n")
    _commit_all(tmp_path)

    (tmp_path / "file.txt").write_text("modified\n")

    assert project_is_dirty(str(tmp_path)) is True
    offending = dirty_paths_outside_results(str(tmp_path))
    assert len(offending) == 1
    assert "file.txt" in offending[0]


def test_untracked_file_is_detected(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n")
    _commit_all(tmp_path)

    (tmp_path / "new_file.txt").write_text("new\n")

    assert project_is_dirty(str(tmp_path)) is True
    offending = dirty_paths_outside_results(str(tmp_path))
    assert any("new_file.txt" in line for line in offending)


def test_changes_under_results_are_ignored(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "existing.json").write_text("{}\n")
    _commit_all(tmp_path)

    # New untracked files AND modifications under results/ must not count.
    (tmp_path / "results" / "new_run_output.json").write_text('{"a": 1}\n')
    (tmp_path / "results" / "existing.json").write_text('{"b": 2}\n')

    assert project_is_dirty(str(tmp_path)) is False
    assert dirty_paths_outside_results(str(tmp_path)) == []


def test_results_changes_do_not_mask_a_real_source_change(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "file.txt").write_text("hello\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "existing.json").write_text("{}\n")
    _commit_all(tmp_path)

    (tmp_path / "results" / "new_run_output.json").write_text('{"a": 1}\n')
    (tmp_path / "file.txt").write_text("modified\n")

    assert project_is_dirty(str(tmp_path)) is True
    offending = dirty_paths_outside_results(str(tmp_path))
    assert len(offending) == 1
    assert "file.txt" in offending[0]


def test_not_a_git_repository_returns_clean_rather_than_raising(tmp_path):
    # No git init at all -- `git status` fails; the function must degrade
    # to "nothing to report" rather than raise, since a launch-gate check
    # must never itself crash the CLI.
    assert dirty_paths_outside_results(str(tmp_path)) == []
    assert project_is_dirty(str(tmp_path)) is False
