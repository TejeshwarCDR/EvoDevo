"""The mirror must be real redundancy: a dedicated mountpoint on a different
PHYSICAL disk from the results (findmnt -> lsblk PKNAME walk, handling
partitions and LVM), unless --allow-same-disk-mirror is passed explicitly."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evodevo.campaign import mirror as mirror_module
from evodevo.campaign.alerts import AlertManager
from evodevo.campaign.dashboard import collect, render
from evodevo.campaign.mirror import DeviceResolutionError, check_mirror_location, physical_disks, sync

REPO_ROOT = str(Path(__file__).resolve().parents[2])

BLOCK = {
    "/dev/sda": {"TYPE": "disk", "PKNAME": "", "KNAME": "sda"},
    "/dev/sda2": {"TYPE": "part", "PKNAME": "sda", "KNAME": "sda2"},
    "/dev/sda3": {"TYPE": "part", "PKNAME": "sda", "KNAME": "sda3"},
    "/dev/sda7": {"TYPE": "part", "PKNAME": "sda", "KNAME": "sda7"},
    "/dev/sdb": {"TYPE": "disk", "PKNAME": "", "KNAME": "sdb"},
    "/dev/sdb1": {"TYPE": "part", "PKNAME": "sdb", "KNAME": "sdb1"},
    "/dev/nvme1n1": {"TYPE": "disk", "PKNAME": "", "KNAME": "nvme1n1"},
    "/dev/nvme1n1p1": {"TYPE": "part", "PKNAME": "nvme1n1", "KNAME": "nvme1n1p1"},
    "/dev/mapper/vg0-root": {"TYPE": "lvm", "PKNAME": "sda2", "KNAME": "dm-0"},
    "/dev/mapper/vg0-backup": {"TYPE": "lvm", "PKNAME": "sda3", "KNAME": "dm-1"},
    "/dev/mapper/vg1-backup": {"TYPE": "lvm", "PKNAME": "nvme1n1p1", "KNAME": "dm-2"},
    "/dev/loop3": {"TYPE": "loop", "PKNAME": "", "KNAME": "loop3"},
}


def fake(mounts):
    """runner answering findmnt from {path: (source, target)} and lsblk from BLOCK."""
    def run(argv):
        if argv[0] == "findmnt":
            field, path = argv[2], argv[4]
            source, target = mounts[path]
            return (source if field == "SOURCE" else target) + "\n"
        if argv[0] == "lsblk":
            field, device = argv[2], argv[3]
            return BLOCK[device][field] + "\n"
        raise AssertionError(argv)
    return run


@pytest.fixture
def dirs(tmp_path):
    results, backup = tmp_path / "results", tmp_path / "backup"
    results.mkdir()
    backup.mkdir()
    return str(results), str(backup)


def test_separate_physical_disks_pass(dirs):
    results, backup = dirs
    refusals, record = check_mirror_location(results, backup, run=fake(
        {results: ("/dev/sda7", "/"), backup: ("/dev/sdb1", "/mnt/evodevo-backup")}))
    assert refusals == []
    assert record["results_disks"] == ["sda"] and record["mirror_disks"] == ["sdb"]
    assert record["same_physical_disk"] is False


def test_other_partition_of_same_disk_is_refused_unless_explicitly_allowed(dirs):
    results, backup = dirs
    run = fake({results: ("/dev/sda7", "/"), backup: ("/dev/sda3", "/media/ubuntu/A27C")})
    refusals, record = check_mirror_location(results, backup, run=run)
    assert len(refusals) == 1 and "share physical disk(s) ['sda']" in refusals[0]
    assert "--allow-same-disk-mirror" in refusals[0]
    allowed, record = check_mirror_location(results, backup, allow_same_disk=True, run=run)
    assert allowed == [] and record["same_physical_disk"] is True and record["allow_same_disk_mirror"]


def test_lvm_is_walked_to_physical_disks(dirs):
    results, backup = dirs
    same = fake({results: ("/dev/mapper/vg0-root", "/"), backup: ("/dev/mapper/vg0-backup", "/backup")})
    refusals, record = check_mirror_location(results, backup, run=same)
    assert refusals and record["results_disks"] == ["sda"] == record["mirror_disks"]
    other = fake({results: ("/dev/mapper/vg0-root", "/"), backup: ("/dev/mapper/vg1-backup", "/backup")})
    refusals, record = check_mirror_location(results, backup, run=other)
    assert refusals == [] and record["mirror_disks"] == ["nvme1n1"]


def test_unmounted_backup_directory_on_root_is_refused_even_with_override(dirs):
    results, backup = dirs
    run = fake({results: ("/dev/sdb1", "/data"), backup: ("/dev/sda7", "/")})
    refusals, _ = check_mirror_location(results, backup, allow_same_disk=True, run=run)
    assert any("not on a dedicated mountpoint" in r for r in refusals)


def test_mirror_inside_the_results_filesystem_is_refused(dirs):
    results, backup = dirs
    run = fake({results: ("/dev/sdb1", "/data"), backup: ("/dev/sdb1", "/data")})
    refusals, _ = check_mirror_location(results, backup, allow_same_disk=True, run=run)
    assert any("not on a dedicated mountpoint" in r for r in refusals)


def test_unresolvable_sources_are_refused_unless_overridden(dirs):
    results, backup = dirs
    run = fake({results: ("/dev/sda7", "/"), backup: ("nas:/export", "/mnt/nas")})
    refusals, record = check_mirror_location(results, backup, run=run)
    assert any("cannot verify" in r for r in refusals)
    assert check_mirror_location(results, backup, allow_same_disk=True, run=run)[0] == []
    with pytest.raises(DeviceResolutionError):
        physical_disks("/dev/loop3", run=fake({}))


def test_missing_mirror_directory_is_refused(tmp_path):
    refusals, _ = check_mirror_location(tmp_path, tmp_path / "absent", run=fake({}))
    assert refusals == [f"mirror directory does not exist: {tmp_path / 'absent'}"]


def test_sync_rechecks_location_and_alerts_when_backup_disk_vanishes(tmp_path, monkeypatch):
    source = tmp_path / "results" / "campaign"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}")
    backup = tmp_path / "backup"
    backup.mkdir()
    monkeypatch.setattr(mirror_module, "mount_of",
                        lambda path, run=None: ("/dev/sda7", "/"))
    monkeypatch.setattr(mirror_module, "physical_disks", lambda device, run=None: {"sda"})
    alerts = AlertManager(tmp_path / "alerts.jsonl", notifier=type("N", (), {"send": lambda s, m: None})())
    result = sync(source, backup, alert_manager=alerts, location_policy={"allow_same_disk": False})
    assert result["ok"] is False and "mirror location refused" in result["error"]
    assert not (backup / "campaign").exists(), "nothing may be written to a refused mirror"
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert rows[-1]["condition"] == "mirror_failed" and rows[-1]["status"] == "active"


def test_same_disk_override_is_a_permanent_dashboard_warning(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"campaign_id": "c", "allow_same_disk_mirror": True}))
    assert "WARNING: --allow-same-disk-mirror" in render(collect(str(tmp_path)))


def test_cli_refuses_same_disk_mirror_and_override_is_recorded(tmp_path):
    """Refusal path (the only permitted use of --allow-same-disk-mirror)."""
    config = tmp_path / "c.yaml"
    config.write_text(
        "profile: campaign\nencoders: [direct]\ntasks: [Walker-v0]\nseeds: [0]\n"
        f"configs_root: {REPO_ROOT}/evodevo/experiments/configs\noutput_dir: results/runs\n"
        "surrogate: true\nfactors: {dev_steps: 100, init_condition: center_seed}\n"
        "profile_overrides: {population_size: 2, offspring_size: 2, eval_budget: 2, "
        "ppo_timesteps: {Walker-v0: 500}, ppo_seeds: {Walker-v0: 1}}\n")
    same_disk_mirror = tmp_path / "mirror"
    same_disk_mirror.mkdir()
    base = [sys.executable, "-m", "evodevo.campaign", "--check-only", "--config", str(config),
            "--output-root", str(tmp_path / "out"), "--campaign-id", "m", "--mode", "surrogate",
            "--mirror-dir", str(same_disk_mirror)]
    refused = subprocess.run(base, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert refused.returncode != 0
    assert "share physical disk" in refused.stderr and "dedicated mountpoint" in refused.stderr
    overridden = subprocess.run(base + ["--allow-same-disk-mirror"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=180)
    assert overridden.returncode != 0  # /tmp is still not a dedicated mountpoint
    assert "share physical disk" not in overridden.stderr and "dedicated mountpoint" in overridden.stderr


@pytest.mark.skipif(not os.path.ismount("/mnt/evodevo-backup"), reason="backup disk not mounted")
def test_this_machines_backup_disk_is_separate_physical_disk():
    refusals, record = check_mirror_location(os.path.join(REPO_ROOT, "results"), "/mnt/evodevo-backup")
    assert refusals == [], record
    assert not set(record["results_disks"]) & set(record["mirror_disks"])
