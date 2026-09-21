"""Working-tree cleanliness check gating a real pre-flight launch.

A real pre-flight run's own manifest records a git commit hash per stage
(stages/__init__.py's run_stages() -> git_short_hash()), which is only a
trustworthy description of "what code produced this measurement" if the
tree was actually clean (or the operator explicitly accepted otherwise via
--allow-dirty) when the run started. results/ is excluded from the check:
it holds pre-flight run output (including this very run's own directory,
which may not exist yet when this check runs), never source that gates a
launch.

Split into its own module, separate from evodevo.preflight.__main__, so it
can be unit-tested directly (against a real temporary git repo, with a
controlled clean/dirty state) without going through a subprocess -- unlike
__main__ itself, this module has no import-order requirement of its own
and is safe to import anywhere.
"""
from __future__ import annotations

import subprocess
from typing import List


def dirty_paths_outside_results(repo_root: str) -> List[str]:
    """Each `git status --porcelain` line for `repo_root`, excluding any
    path under results/. Returns [] both when the tree is clean AND when
    `git status` itself fails (e.g. not a git repository) -- the latter is
    deliberately not distinguished here, since a caller that wants "may I
    launch" only needs "is there something to refuse", not why a git
    command failed; see _project_is_dirty()'s own docstring in __main__.py
    for how a launch decision uses this."""
    result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    offending = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if path.startswith("results/") or path.startswith('"results/'):
            continue
        offending.append(line)
    return offending


def project_is_dirty(repo_root: str) -> bool:
    return bool(dirty_paths_outside_results(repo_root))
