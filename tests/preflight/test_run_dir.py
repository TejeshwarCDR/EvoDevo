from __future__ import annotations

import json
import os

import pytest

from evodevo.preflight.run_dir import (
    RunDirNotFoundError,
    atomic_write_json,
    create_or_resume_run_dir,
    read_json,
)


def test_run_dir_creation_produces_every_expected_path(tmp_path):
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))

    assert os.path.isdir(run_dir.path)
    assert os.path.isfile(run_dir.manifest_path)
    assert os.path.isfile(run_dir.machine_path)
    assert os.path.isdir(run_dir.stages_dir)
    assert os.path.isfile(run_dir.events_path)
    assert os.path.isfile(run_dir.telemetry_path)
    assert os.path.isdir(run_dir.logs_dir)
    assert os.path.isdir(run_dir.report_dir)

    manifest = read_json(run_dir.manifest_path)
    assert manifest["stages"] == {}

    machine = read_json(run_dir.machine_path)
    assert "topology" in machine
    assert machine["os_cpu_count"] == os.cpu_count()


def test_run_dir_name_is_timestamp_and_git_hash(tmp_path):
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))
    parts = run_dir.run_id.split("_")
    assert len(parts) == 2
    timestamp, git_hash = parts
    assert timestamp.endswith("Z")
    assert len(git_hash) > 0


def test_resume_with_run_id_returns_the_same_directory(tmp_path):
    run_dir = create_or_resume_run_dir(base_dir=str(tmp_path))
    atomic_write_json(run_dir.manifest_path, {"run_id": run_dir.run_id, "stages": {"x": {"status": "done"}}})

    resumed = create_or_resume_run_dir(base_dir=str(tmp_path), run_id=run_dir.run_id)

    assert resumed.path == run_dir.path
    manifest = read_json(resumed.manifest_path)
    assert manifest["stages"]["x"]["status"] == "done"


def test_resume_with_unknown_run_id_raises(tmp_path):
    with pytest.raises(RunDirNotFoundError):
        create_or_resume_run_dir(base_dir=str(tmp_path), run_id="does-not-exist")


def test_atomic_write_leaves_no_tmp_file_behind_on_success(tmp_path):
    path = str(tmp_path / "artifact.json")
    atomic_write_json(path, {"a": 1})

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    assert read_json(path) == {"a": 1}


def test_atomic_write_never_leaves_a_half_written_file(tmp_path):
    """Writes go to <path>.tmp then os.replace -- a reader can only ever
    see the old complete content or the new complete content."""
    path = str(tmp_path / "artifact.json")
    atomic_write_json(path, {"version": 1})
    assert read_json(path) == {"version": 1}

    atomic_write_json(path, {"version": 2})
    assert read_json(path) == {"version": 2}
    assert not os.path.exists(path + ".tmp")


def test_creating_over_an_existing_path_raises(tmp_path, monkeypatch):
    import evodevo.preflight.run_dir as run_dir_module

    monkeypatch.setattr(run_dir_module, "_utc_timestamp", lambda: "20260101T000000Z")
    run_dir_module.create_or_resume_run_dir(base_dir=str(tmp_path))
    with pytest.raises(FileExistsError):
        run_dir_module.create_or_resume_run_dir(base_dir=str(tmp_path))
