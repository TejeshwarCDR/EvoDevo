"""Six-hour and phase-end rsync mirror with content verification."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

INTERVAL_S = 6 * 3600
MANIFEST_NAME = "mirror_checksums.json"


def checksum_manifest(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME or path.name.endswith(".tmp"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path.relative_to(directory))] = digest.hexdigest()
    return result


def verify_mirror(source, destination):
    expected = checksum_manifest(source)
    actual = checksum_manifest(destination)
    missing = sorted(set(expected) - set(actual))
    altered = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    extra = sorted(set(actual) - set(expected))
    return {"ok": not (missing or altered or extra), "missing": missing,
            "altered": altered, "extra": extra, "files": len(expected)}


def verify_checksum_manifest(destination):
    """Verify the stable copied snapshot against its own saved SHA-256 list."""
    destination = Path(destination)
    try:
        with (destination / MANIFEST_NAME).open(encoding="utf-8") as handle:
            expected = json.load(handle)["checksums"]
    except (OSError, ValueError, KeyError) as exc:
        return {"ok": False, "error": f"checksum manifest unavailable: {exc}"}
    actual = checksum_manifest(destination)
    missing = sorted(set(expected) - set(actual))
    altered = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    extra = sorted(set(actual) - set(expected))
    return {"ok": not (missing or altered or extra), "missing": missing,
            "altered": altered, "extra": extra, "files": len(expected)}


class DeviceResolutionError(RuntimeError):
    """A mount source that cannot be traced to physical disks."""


def _run_cmd(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


def _existing(path):
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def mount_of(path, run=_run_cmd):
    """(source device, mount target) of the filesystem holding `path` (or its
    nearest existing ancestor): `findmnt -no SOURCE -T` / `-no TARGET -T`."""
    where = str(_existing(path))
    source = run(["findmnt", "-no", "SOURCE", "-T", where]).strip().splitlines()[0]
    target = run(["findmnt", "-no", "TARGET", "-T", where]).strip().splitlines()[0]
    return source.split("[", 1)[0], target  # strip btrfs/bind "[/subvol]" suffix


def physical_disks(device, run=_run_cmd):
    """Every whole disk under `device`, walking `lsblk -dno PKNAME` upward
    through partitions, LVM/dm, crypt and md layers (plus sysfs `slaves` for
    multi-PV volumes). Raises DeviceResolutionError for non-block sources
    (network, tmpfs, loop-backed files) that cannot be traced to a disk."""
    if not device.startswith("/dev/"):
        raise DeviceResolutionError(f"mount source {device!r} is not a block device")
    disks, seen, stack = set(), set(), [device]
    while stack:
        dev = stack.pop()
        name = dev.rsplit("/", 1)[-1]
        if name in seen:
            continue
        seen.add(name)
        kind = run(["lsblk", "-dno", "TYPE", dev]).strip()
        if kind == "disk":
            disks.add(name)
            continue
        if kind == "loop":
            raise DeviceResolutionError(f"{dev} is a loop device; its backing disk cannot be verified")
        parents = [line.strip() for line in run(["lsblk", "-dno", "PKNAME", dev]).splitlines() if line.strip()]
        kname = run(["lsblk", "-dno", "KNAME", dev]).strip()
        slaves = Path("/sys/class/block") / kname / "slaves"
        if run is _run_cmd and slaves.is_dir():
            parents += [entry.name for entry in slaves.iterdir()]
        if not parents:
            raise DeviceResolutionError(f"{dev} (type {kind or 'unknown'}) has no parent disk")
        stack.extend(f"/dev/{parent}" for parent in parents)
    return disks


def check_mirror_location(results_path, mirror_dir, *, allow_same_disk=False, run=_run_cmd):
    """Refusals (list of strings) plus the resolved devices for the record.

    The mirror must (1) exist, (2) sit on a DEDICATED mounted filesystem --
    its mount target may be neither `/` nor the results filesystem's own
    mountpoint, because an unmounted backup directory silently lands on the
    root disk -- and (3) be on a different physical disk from the results,
    unless allow_same_disk is passed explicitly."""
    record = {"results_path": str(Path(results_path).absolute()), "mirror_dir": str(mirror_dir),
              "allow_same_disk_mirror": bool(allow_same_disk)}
    refusals = []
    if not Path(mirror_dir).is_dir():
        return [f"mirror directory does not exist: {mirror_dir}"], record
    results_source, results_target = mount_of(results_path, run)
    mirror_source, mirror_target = mount_of(mirror_dir, run)
    record.update(results_source=results_source, results_mount=results_target,
                  mirror_source=mirror_source, mirror_mount=mirror_target)
    if mirror_target == "/" or mirror_target == results_target:
        refusals.append(f"mirror {mirror_dir} is not on a dedicated mountpoint (it resolves to the "
                        f"{mirror_target!r} filesystem, {mirror_source}); is the backup disk mounted?")
    try:
        results_disks = physical_disks(results_source, run)
        mirror_disks = physical_disks(mirror_source, run)
    except (DeviceResolutionError, subprocess.CalledProcessError) as exc:
        record["resolution_error"] = str(exc)
        if not allow_same_disk:
            refusals.append(f"cannot verify the mirror is on a separate physical disk: {exc}")
        return refusals, record
    shared = sorted(results_disks & mirror_disks)
    record.update(results_disks=sorted(results_disks), mirror_disks=sorted(mirror_disks),
                  same_physical_disk=bool(shared))
    if shared and not allow_same_disk:
        refusals.append(f"mirror {mirror_dir} ({mirror_source}) and results ({results_source}) share "
                        f"physical disk(s) {shared}: no redundancy. Use a separate disk, or pass "
                        "--allow-same-disk-mirror explicitly.")
    return refusals, record


def sync(source, mirror_dir, *, alert_manager=None, location_policy=None):
    """location_policy (the scheduler/monitor always pass one): re-verify the
    mirror's mount and physical disk before EVERY copy, so a backup disk that
    is unmounted mid-campaign fails loudly instead of filling the root disk."""
    source = Path(source).resolve()
    if mirror_dir is None:
        return {"ok": False, "warning": "mirror directory absent"}
    target = Path(mirror_dir).resolve() / source.name
    lock_fd = None
    try:
        if source == target or source in target.parents:
            raise ValueError("mirror target must be outside the campaign directory")
        if not Path(mirror_dir).is_dir():
            raise FileNotFoundError(f"mirror directory absent: {mirror_dir}")
        if location_policy is not None:
            refusals, _record = check_mirror_location(
                source.parent, mirror_dir, allow_same_disk=location_policy.get("allow_same_disk", False))
            if refusals:
                raise ValueError("mirror location refused: " + "; ".join(refusals))
        # Phase-end and six-hour syncs can coincide. Lock the source
        # directory itself, leaving no extra campaign artifact behind.
        lock_fd = os.open(source, os.O_RDONLY)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        target.mkdir(parents=True, exist_ok=True)
        # The campaign is live: a journal append or new policy can happen
        # during the first rsync. Re-copy the delta and compare again rather
        # than declaring a healthy mirror broken because the source advanced.
        for attempt in range(3):
            # --checksum is essential for same-size rewrites within rsync's
            # timestamp resolution (notably manifest.json/checkpoint files).
            subprocess.run(["rsync", "-a", "--checksum", "--delete", str(source) + "/", str(target) + "/"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            result = verify_mirror(source, target)
            if result["ok"]:
                break
        result["destination"] = str(target)
        result["timestamp"] = time.time()
        if not result["ok"]:
            raise RuntimeError(f"checksum verification failed after 3 copies: {result}")
        payload = {"timestamp": result["timestamp"], "destination": str(target),
                   "checksums": checksum_manifest(target)}
        temporary = source / (MANIFEST_NAME + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, source / MANIFEST_NAME)
        subprocess.run(["rsync", "-a", "--checksum", str(source / MANIFEST_NAME),
                        str(target / MANIFEST_NAME)], check=True)
        manifest_check = verify_checksum_manifest(target)
        if not manifest_check["ok"]:
            raise RuntimeError(f"stored checksum manifest failed: {manifest_check}")
        result["manifest_verified"] = True
        if alert_manager:
            alert_manager.evaluate("mirror_failed", False, result)
        return result
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "destination": str(target)}
        if alert_manager:
            alert_manager.evaluate("mirror_failed", True, result)
        return result
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--mirror-dir", required=True)
    args = parser.parse_args(argv)
    result = sync(args.campaign_dir, args.mirror_dir)
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
