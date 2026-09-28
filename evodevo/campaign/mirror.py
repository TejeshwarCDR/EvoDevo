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


def sync(source, mirror_dir, *, alert_manager=None):
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
