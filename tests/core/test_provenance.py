import json
import os

import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.core.provenance import (
    DirtyTreeError,
    check_git_state,
    git_commit_hash,
    git_dirty_files,
    write_factors_json,
)


def test_git_commit_hash_is_a_real_hash():
    commit = git_commit_hash()
    assert commit != "unknown"
    assert len(commit) == 40


def test_check_git_state_raises_on_dirty_without_allow_dirty(monkeypatch):
    monkeypatch.setattr("evodevo.core.provenance.git_dirty_files", lambda: ["fake_dirty_file.py"])
    with pytest.raises(DirtyTreeError):
        check_git_state(allow_dirty=False)


def test_check_git_state_proceeds_on_dirty_with_allow_dirty(monkeypatch):
    monkeypatch.setattr("evodevo.core.provenance.git_dirty_files", lambda: ["fake_dirty_file.py"])
    commit_hash, dirty_files = check_git_state(allow_dirty=True)
    assert dirty_files == ["fake_dirty_file.py"]
    assert commit_hash != "unknown"


def test_check_git_state_never_raises_on_a_clean_tree(monkeypatch):
    monkeypatch.setattr("evodevo.core.provenance.git_dirty_files", lambda: [])
    commit_hash, dirty_files = check_git_state(allow_dirty=False)
    assert dirty_files == []


def test_write_factors_json_populates_all_three_fields(tmp_path):
    factors = ExperimentFactors(dev_steps=5, init_condition="center_seed", seed=0)
    path = write_factors_json(str(tmp_path), factors, "abc123", ["some_file.py"])

    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)

    assert data["factors"]["dev_steps"] == 5
    assert data["factors"]["init_condition"] == "center_seed"
    assert data["git_commit"] == "abc123"
    assert data["dirty"] is True
    assert data["dirty_files"] == ["some_file.py"]


def test_write_factors_json_clean_tree(tmp_path):
    factors = ExperimentFactors(dev_steps=5, init_condition="center_seed", seed=0)
    path = write_factors_json(str(tmp_path), factors, "abc123", [])

    with open(path) as f:
        data = json.load(f)
    assert data["dirty"] is False
    assert data["dirty_files"] == []
