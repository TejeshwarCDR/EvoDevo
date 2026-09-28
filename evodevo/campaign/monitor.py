"""Read-only campaign sidecar: alerts, snapshots, and verified mirror cadence."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .alerts import AlertManager
from .dashboard import collect, render
from .dashboard import PRIOR_PATH
from .milestone import run_subprocess
from .mirror import INTERVAL_S, sync

SNAPSHOT_S = 12 * 3600


def slowest_ecore_p95_s(task):
    """B3/P1 P-core mean+1.645 SD, scaled by the measured 2.3172 E-core factor."""
    data = json.loads(PRIOR_PATH.read_text())
    pairs = data["inputs"]["t_ppo_mean_sd_at_b3_footing"]
    values = [mean + 1.645 * sd for key, (mean, sd) in pairs.items() if key.endswith("|" + task)]
    if not values:
        raise ValueError(f"no measured t_PPO prior for {task}")
    return max(values) * 2.3172


def monitor_once(root, alerts, *, now=None, p95_ecore_s=None, last_mirror=0,
                 last_snapshot=0):
    now = time.time() if now is None else now
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        return {"running": True, "last_mirror": last_mirror, "last_snapshot": last_snapshot}
    manifest = json.loads(manifest_path.read_text())
    snapshot = collect(str(root), now=now)
    if p95_ecore_s is None:
        p95_ecore_s = slowest_ecore_p95_s(snapshot["phase"])
    mirror_dir = manifest.get("mirror_dir")
    mirror_ok = manifest.get("no_mirror", False) or bool(mirror_dir)
    if mirror_dir and now - last_mirror >= INTERVAL_S:
        result = sync(root, mirror_dir, alert_manager=alerts)
        mirror_ok = result["ok"]
        if mirror_ok:
            last_mirror = now
            # The scheduler owns the manifest; a sidecar state file prevents
            # concurrent manifest rewrites from losing phase transitions.
            (Path(root) / "mirror_status.json").write_text(json.dumps(
                {"last_success": now, "destination": result["destination"]}))
    if now - last_snapshot >= SNAPSHOT_S and manifest.get("status") == "running":
        run_subprocess(root, snapshot["phase"], alerts, snapshot=True)
        last_snapshot = now
    alerts.inspect(snapshot, p95_ecore_s=p95_ecore_s, mirror_ok=mirror_ok)
    return {"running": manifest.get("status") != "complete", "last_mirror": last_mirror,
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
    last_mirror = last_snapshot = 0
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
