"""Six-hour and phase-end rsync mirror with content verification."""
from __future__ import annotations

import argparse
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


def sync(source, mirror_dir, *, alert_manager=None):
    source = Path(source).resolve()
    if mirror_dir is None:
        return {"ok": False, "warning": "mirror directory absent"}
    target = Path(mirror_dir).resolve() / source.name
    try:
        if source == target or source in target.parents:
            raise ValueError("mirror target must be outside the campaign directory")
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", str(source) + "/", str(target) + "/"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result = verify_mirror(source, target)
        result["destination"] = str(target)
        result["timestamp"] = time.time()
        if not result["ok"]:
            raise RuntimeError(f"checksum verification failed: {result}")
        with (source / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump({"timestamp": result["timestamp"], "destination": str(target),
                       "checksums": checksum_manifest(source)}, handle, indent=2, sort_keys=True)
        # Copy the checksum manifest as well, after computing the set it lists.
        subprocess.run(["rsync", "-a", str(source / MANIFEST_NAME), str(target / MANIFEST_NAME)], check=True)
        if alert_manager:
            alert_manager.evaluate("mirror_failed", False, result)
        return result
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "destination": str(target)}
        if alert_manager:
            alert_manager.evaluate("mirror_failed", True, result)
        return result


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
