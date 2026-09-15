"""Minimal run provenance: git commit hash, working-tree dirty status, and
the resolved ExperimentFactors, written to results/<run_id>/factors.json.

This is a STANDALONE wrapper around run_experiment.py/benchmark.py -- it
does not touch ExperimentConfig's shape, SHARED_BASE_KEYS, or
evodevo/experiments/lock_hash.py, and has no effect on the lock hash.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import List, Tuple


class DirtyTreeError(RuntimeError):
    """Raised when the working tree is dirty and allow_dirty was not set."""


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_dirty_files() -> List[str]:
    """The paths reported by `git status --porcelain` (empty if the tree is
    clean, or if this isn't a git checkout at all)."""
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    return [line[3:] for line in output.splitlines() if line.strip()]


def check_git_state(allow_dirty: bool) -> Tuple[str, List[str]]:
    """Captures (commit_hash, dirty_files) and enforces the dirty-tree gate:
    raises DirtyTreeError (printing the dirty file list to stderr) unless
    the tree is clean or `allow_dirty` is set. `allow_dirty` is for local
    iteration only -- never for a run whose output is cited in a report or
    a docs/decisions.md row."""
    commit_hash = git_commit_hash()
    dirty_files = git_dirty_files()

    if dirty_files and not allow_dirty:
        print("error: working tree is dirty -- refusing to run without --allow-dirty:", file=sys.stderr)
        for f in dirty_files:
            print(f"  {f}", file=sys.stderr)
        print(
            "--allow-dirty is for local iteration only -- never use it for a run whose "
            "output is cited in a report or a docs/decisions.md row.",
            file=sys.stderr,
        )
        raise DirtyTreeError(f"working tree is dirty: {dirty_files}")

    if dirty_files:
        print("warning: running with a dirty working tree (--allow-dirty was passed):", file=sys.stderr)
        for f in dirty_files:
            print(f"  {f}", file=sys.stderr)

    return commit_hash, dirty_files


def write_factors_json(output_dir: str, factors, commit_hash: str, dirty_files: List[str]) -> str:
    """Serializes the resolved ExperimentFactors plus commit hash and dirty
    status to <output_dir>/factors.json. Independent of RunLogger's own
    run_meta.json -- this file exists purely for provenance, even for
    callers (e.g. dry runs, benchmark-level bookkeeping) that never
    instantiate a RunLogger."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "factors.json")
    data = {
        "factors": dataclasses.asdict(factors),
        "git_commit": commit_hash,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path
