"""Read-only campaign sidecar: alerts, snapshots, and verified mirror cadence."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

from .alerts import AlertManager
from .dashboard import collect, render
from .milestone import REQUIRED, run_subprocess
from .mirror import INTERVAL_S, sync

SNAPSHOT_S = 12 * 3600


def slowest_ecore_p95_s(task, estimate_path=None):
    """B3/P1 P-core mean+1.645 SD, scaled by the measured 2.3172 E-core factor.

    ``estimate_path`` is the campaign's own estimate (recorded in its
    manifest); the default is the original campaign's estimate.
    """
    from . import estimate as estimates
    data = json.loads(Path(estimate_path or estimates.LEGACY_ESTIMATE_PATH).read_text())
    return estimates.unit_p95_ecore_s(data, task)


def monitor_once(root, alerts, *, now=None, p95_ecore_s=None, last_mirror=0,
                 last_snapshot=0):
    now = time.time() if now is None else now
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        return {"running": True, "last_mirror": last_mirror, "last_snapshot": last_snapshot}
    manifest = json.loads(manifest_path.read_text())
    snapshot = collect(str(root), now=now)
    phase_state = manifest.get("phase_status", {}).get(snapshot["phase"], {})
    phase_started = phase_state.get("started_at", now)
    last_snapshot = max(last_snapshot, phase_started)
    if p95_ecore_s is None:
        from .estimate import estimate_path_for_manifest
        p95_ecore_s = slowest_ecore_p95_s(snapshot["phase"], estimate_path_for_manifest(manifest))
    mirror_dir = manifest.get("mirror_dir")
    mirror_ok = manifest.get("no_mirror", False) or bool(mirror_dir and Path(mirror_dir).is_dir())
    if mirror_dir and now - last_mirror >= INTERVAL_S:
        result = sync(root, mirror_dir, alert_manager=alerts,
                      location_policy={"allow_same_disk": manifest.get("allow_same_disk_mirror", False)})
        mirror_ok = result["ok"]
        if mirror_ok:
            last_mirror = now
            # The scheduler owns the manifest; a sidecar state file prevents
            # concurrent manifest rewrites from losing phase transitions.
            (Path(root) / "mirror_status.json").write_text(json.dumps(
                {"last_success": now, "destination": result["destination"]}))
    from . import units as unit_keys
    running_unit = unit_keys.unit_by_key(manifest.get("plan", {}), snapshot["phase"])
    is_phase_unit = running_unit is None or running_unit["kind"] == unit_keys.PHASE
    if now - last_snapshot >= SNAPSHOT_S and phase_state.get("status") == "running" and is_phase_unit:
        outcome = run_subprocess(root, snapshot["phase"], alerts, snapshot=True)
        if outcome["ok"]:
            last_snapshot = now
    alerts.inspect(snapshot, p95_ecore_s=p95_ecore_s, mirror_ok=mirror_ok)
    return {"running": manifest.get("status") not in {"complete", "stopped"}, "last_mirror": last_mirror,
            "last_snapshot": last_snapshot, "snapshot": snapshot}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", default="results/campaign")
    parser.add_argument("--p95-ecore-s", type=float,
                        help="test override; default derives B3/P1 E-core p95 for the active task")
    parser.add_argument("--interval-s", type=float, default=60)
    args = parser.parse_args(argv)
    root = Path(args.output_root) / args.campaign
    alerts = AlertManager(root / "alerts.jsonl")
    try:
        last_mirror = float(json.loads((root / "mirror_status.json").read_text())["last_success"])
    except (OSError, ValueError, KeyError):
        last_mirror = 0
    last_snapshot = 0
    for path in (root / "milestones" / "snapshots").glob("*"):
        try:
            if not (path / "package_manifest.json").is_file() or not all((path / name).is_file() for name in REQUIRED):
                continue
            stamp = dt.datetime.strptime(path.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            last_snapshot = max(last_snapshot, stamp.timestamp())
        except ValueError:
            continue
    while True:
        state = monitor_once(root, alerts, p95_ecore_s=args.p95_ecore_s,
                             last_mirror=last_mirror, last_snapshot=last_snapshot)
        last_mirror, last_snapshot = state["last_mirror"], state["last_snapshot"]
        if "snapshot" in state:
            print(render(state["snapshot"]), flush=True)
        if not state["running"]:
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
