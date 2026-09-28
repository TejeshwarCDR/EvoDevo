"""Read-only campaign dashboard for terminals and SSH status checks."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIOR_PATH = Path(__file__).resolve().parents[2] / "results/campaign_prep/final_estimate/final_estimate.json"
P2_MEAN_H = 403.2
P2_P95_H = 406.0
# Exact registered joint-total quantiles used by the initial dashboard prior.
# Keep these independent of PRIOR_PATH: Step 4 legitimately regenerates that
# artifact, while the dashboard explicitly reports the locked P2 prior until
# enough campaign measurements exist to replace it.
P2_TOTAL_P5_H = 400.5636689293408
P2_TOTAL_P95_H = 405.9615436943582
GUARD_H = 432.0
E_CORE_FACTOR = 2.3172


def _json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _rows(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return []


def _prior():
    data = _json(PRIOR_PATH, {})
    # The versioned P2 simulator output has one final-campaign row with
    # result.walker_phase_h/upstepper_phase_h/total_h. Search by shape so a
    # harmless top-level metadata addition cannot change this calculation.
    def locate(obj):
        if isinstance(obj, dict):
            value = obj.get("result")
            if isinstance(value, dict) and "walker_phase_h" in value and "total_h" in value:
                return value
            for child in obj.values():
                found = locate(child)
                if found:
                    return found
        if isinstance(obj, list):
            for child in obj:
                found = locate(child)
                if found:
                    return found
        return None
    return locate(data) or {}


def _duration(seconds):
    if seconds is None:
        return "n/a"
    return f"{seconds / 3600:.1f} h"


def _hours_range(hours):
    return "n/a" if hours is None else f"{hours[0]:.1f}–{hours[1]:.1f} h"


def _when(epoch):
    if epoch is None:
        return "n/a"
    return dt.datetime.fromtimestamp(epoch).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _telemetry(main_path, mirror_path):
    try:
        import psutil
        from evodevo.preflight.telemetry import (read_cpu_package_temp_c,
            read_cpu_package_temp_critical_c, read_thermal_throttle_count)
        cpu = psutil.cpu_percent(percpu=True, interval=None)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "cpu_percent": round(statistics.mean(cpu), 1) if cpu else None,
            "per_core_percent": cpu,
            "temperature_c": read_cpu_package_temp_c(),
            "critical_temp_c": read_cpu_package_temp_critical_c(),
            "thermal_throttle_count": read_thermal_throttle_count(),
            "ram_used_gb": memory.used / 1e9,
            "swap_used_gb": swap.used / 1e9,
            "main_disk_free_gb": psutil.disk_usage(main_path).free / 1e9,
            "mirror_disk_free_gb": psutil.disk_usage(mirror_path).free / 1e9 if mirror_path and os.path.exists(mirror_path) else None,
        }
    except (ImportError, OSError):
        return {}


def _measured_phase_eta(runs, task, logged, journaled, durations, worker_slots, cpu_kind):
    """Estimate remaining wall time from each arm's measured P/E service rate.

    Each arm's remaining valid PPO count is estimated from its own observed
    validity fraction. For a given arm the pool's P- and E-core capacities
    add; arm workloads then add as server-time. The 5th/95th per-task
    duration quantiles form the ETA range. A class without a sample yet uses
    the measured other class scaled by the B3/P1 E-core factor.
    """
    slots = Counter(cpu_kind.get(slot[0], "unknown") for slot in worker_slots if slot)
    if not slots:
        return None
    low_total = high_total = 0.0
    for run in runs:
        arm = run["arm"]
        spent, valid = logged[(arm, run["seed"])]
        budget = int(run["config"]["eval_budget"])
        k = int(run["config"].get("ppo_seeds", 1))
        if spent == 0:
            return None
        valid_fraction = valid / spent
        pending = (budget - spent) * valid_fraction * k
        # Completed seed tasks for a generation not yet told are already
        # durable in the journal and should not be projected a second time.
        pending = max(0, pending - max(0, journaled[(arm, task, run["seed"])] - valid * k))
        samples = {core: durations[(arm, task, core)] for core in ("P", "E", "unknown")}
        if not any(samples.values()):
            return None
        rate_slow = rate_fast = 0.0
        for core, count in slots.items():
            values = samples.get(core) or []
            scale = 1.0
            if not values:
                if core == "E" and samples["P"]:
                    values, scale = samples["P"], E_CORE_FACTOR
                elif core == "P" and samples["E"]:
                    values, scale = samples["E"], 1 / E_CORE_FACTOR
                else:
                    values = next((v for v in samples.values() if v), [])
            if not values:
                return None
            fast = max(1e-6, float(np.quantile(values, 0.05)) * scale)
            slow = max(1e-6, float(np.quantile(values, 0.95)) * scale)
            rate_fast += count / fast
            rate_slow += count / slow
        low_total += pending / rate_fast
        high_total += pending / rate_slow
    return low_total, high_total


def collect(root: str, now: float | None = None, telemetry: dict | None = None) -> dict:
    now = time.time() if now is None else now
    manifest = _json(os.path.join(root, "manifest.json"), {})
    phases = manifest.get("plan", {}).get("phases", [])
    status = manifest.get("phase_status", {})
    current = next((p for p in phases if status.get(p["task"], {}).get("status") != "complete"), None)
    if current is None and phases:
        current = phases[-1]
    task = current["task"] if current else manifest.get("phase", "unknown")
    runs = current.get("runs", []) if current else []
    generation = {}
    logged = {}
    completed_evals = 0
    total_evals = 0
    for phase in phases:
        for run in phase.get("runs", []):
            budget = int(run["config"]["eval_budget"])
            total_evals += budget
            path = os.path.join(root, "runs", run["arm"], phase["task"], f"seed_{run['seed']}", "run_log.jsonl")
            rows = _rows(path)
            completed_evals += len(rows)
            if phase["task"] == task:
                generation[(run["arm"], run["seed"])] = max((r["generation"] for r in rows), default=0)
                logged[(run["arm"], run["seed"])] = (len(rows), sum(bool(r.get("valid")) for r in rows))
    journal = _rows(os.path.join(root, "journal.jsonl"))
    completed_ppo = len(journal)
    completed_by_pair = Counter((r.get("run_key", "").split("/")[0], r.get("run_key", "").split("/")[1])
                                for r in journal if len(r.get("run_key", "").split("/")) >= 2)
    durations = defaultdict(list)
    journaled_by_run = Counter()
    cpu_kind = {core["logical_id"]: core["kind"] for core in manifest.get("topology", {}).get("cores", [])}
    for row in journal:
        parts = row.get("run_key", "").split("/")
        if len(parts) >= 2:
            if len(parts) >= 3:
                try:
                    journaled_by_run[(parts[0], parts[1], int(parts[2].removeprefix("seed_")))] += 1
                except ValueError:
                    pass
            duration = row.get("result", {}).get("wall_clock_s")
            if isinstance(duration, (float, int)) and duration > 0:
                ids = row.get("result", {}).get("cpu_ids") or []
                core_class = cpu_kind.get(ids[0], "unknown") if ids else "unknown"
                durations[(parts[0], parts[1], core_class)].append(duration)
    arms = {r["arm"] for r in runs}
    measured = bool(runs) and all(completed_by_pair[(arm, task)] >= 50 for arm in arms)
    prior = _prior()
    elapsed = max(0.0, now - float(manifest.get("started_at", now)))
    completed_phase = sum(len(_rows(os.path.join(root, "runs", r["arm"], task,
                                            f"seed_{r['seed']}", "run_log.jsonl"))) for r in runs)
    phase_total = sum(int(r["config"]["eval_budget"]) for r in runs)
    remaining_fraction = max(0.0, 1 - completed_phase / phase_total) if phase_total else 0.0
    prior_key = "walker_phase_h" if task == "Walker-v0" else "upstepper_phase_h"
    prior_band = prior.get(prior_key, {})
    prior_phase_low = float(prior_band.get("p5", 121.9 if task == "Walker-v0" else 277.8)) * 3600
    prior_phase_high = float(prior_band.get("p95", 124.1 if task == "Walker-v0" else 282.9)) * 3600
    if measured:
        phase_eta = _measured_phase_eta(runs, task, logged, journaled_by_run, durations,
            manifest.get("worker_slots", {}).get("slots", []), cpu_kind)
        if phase_eta is None:
            phase_eta = (remaining_fraction * prior_phase_low, remaining_fraction * prior_phase_high)
            eta_source = "prior"
        else:
            eta_source = "measured"
    else:
        phase_eta = (remaining_fraction * prior_phase_low, remaining_fraction * prior_phase_high)
        eta_source = "prior"
    future_low = future_high = 0.0
    if current and phases.index(current) + 1 < len(phases):
        for phase in phases[phases.index(current) + 1:]:
            key = "walker_phase_h" if phase["task"] == "Walker-v0" else "upstepper_phase_h"
            band = prior.get(key, {})
            future_low += float(band.get("p5", band.get("mean", 123 if key == "walker_phase_h" else 280.2))) * 3600
            future_high += float(band.get("p95", band.get("mean", 123 if key == "walker_phase_h" else 280.2))) * 3600
        # Preserve the P2 simulator's *joint* total distribution at campaign
        # start. Summing marginal phase quantiles would discard its observed
        # phase correlation and misstate the registered 400.6–406.0 h band.
        if eta_source == "prior" and task == "Walker-v0":
            future_low = P2_TOTAL_P5_H * 3600 - prior_phase_low
            future_high = P2_TOTAL_P95_H * 3600 - prior_phase_high
    campaign_eta = (phase_eta[0] + future_low, phase_eta[1] + future_high)
    last_task = max((r.get("completed_at", r.get("timestamp", 0)) for r in journal), default=None)
    if last_task is None and manifest.get("status") == "running":
        last_task = status.get(task, {}).get("started_at", manifest.get("started_at"))
    recent = sum(1 for r in journal if r.get("completed_at", r.get("timestamp", 0)) >= now - 3600)
    hourly_rates = [sum(1 for r in journal if now - (hour + 1) * 3600 <=
                         r.get("completed_at", r.get("timestamp", 0)) < now - hour * 3600)
                    for hour in range(min(24, max(1, math.ceil(elapsed / 3600))))]
    alerts = _rows(os.path.join(root, "alerts.jsonl"))
    dispatch = _json(os.path.join(root, "dispatch_status.json"), {})
    if dispatch.get("task") == task:
        for run_key, proposed_generation in dispatch.get("run_generations", {}).items():
            parts = run_key.split("/")
            if len(parts) == 3 and parts[1] == task:
                try:
                    key = (parts[0], int(parts[2].removeprefix("seed_")))
                except ValueError:
                    continue
                if key in generation:
                    generation[key] = max(generation[key], int(proposed_generation))
    else:
        dispatch = {}
    failures = sum(1 for r in journal if not r.get("result", {}).get("completed", True))
    mirror = manifest.get("mirror_dir")
    mirror_status = _json(os.path.join(root, "mirror_status.json"), {})
    info = telemetry if telemetry is not None else _telemetry(root, mirror)
    return {
        "campaign_id": manifest.get("campaign_id", os.path.basename(root)), "phase": task,
        "elapsed_s": elapsed, "phase_completed": completed_phase, "phase_total": phase_total,
        "generation": generation, "tasks_running": dispatch.get("tasks_running", 0),
        "tasks_queued": dispatch.get("tasks_queued", 0), "completed_evaluations": completed_evals,
        "total_evaluations": total_evals, "completed_ppo": completed_ppo,
        "throughput_now_per_h": recent, "throughput_average_per_h": completed_ppo * 3600 / elapsed if elapsed else 0.0,
        "hourly_rates_24h": hourly_rates,
        "phase_eta_s": phase_eta, "campaign_eta_s": campaign_eta, "eta_source": eta_source,
        "future_eta_source": "prior" if future_high else None,
        # Latest-bound ETA while Walker-v0 runs; the actual time once reached.
        "walker_milestone_epoch": (now + phase_eta[1] if task == "Walker-v0"
                                   else status.get("Walker-v0", {}).get("ended_at")),
        "walker_milestone_reached": status.get("Walker-v0", {}).get("status") == "complete",
        "p2_mean_h": P2_MEAN_H, "p2_p95_h": P2_P95_H,
        "projection_total_h": ((elapsed + campaign_eta[0]) / 3600,
                               (elapsed + campaign_eta[1]) / 3600),
        "projection_exceeds_432h": (elapsed + campaign_eta[1]) / 3600 > GUARD_H,
        "infra_failure_count": failures, "last_error": next((r.get("result", {}).get("error") for r in reversed(journal)
                                                           if r.get("result", {}).get("error")), None),
        "seconds_since_last_task": now - last_task if last_task else None,
        "last_alert": alerts[-1] if alerts else None,
        "mirror_disabled": manifest.get("no_mirror", False),
        "notify_disabled": manifest.get("no_notify", False),
        "same_disk_mirror": manifest.get("allow_same_disk_mirror", False),
        "last_mirror_success": mirror_status.get("last_success", manifest.get("last_mirror_success")),
        **info,
    }


def render(snapshot: dict, tty: bool = False) -> str:
    lines = [f"Campaign {snapshot['campaign_id']} | phase {snapshot['phase']} | elapsed {_duration(snapshot['elapsed_s'])}",
             f"Phase {snapshot['phase_completed']}/{snapshot['phase_total']} evaluations; total {snapshot['completed_evaluations']}/{snapshot['total_evaluations']}; PPO {snapshot['completed_ppo']}"]
    grid = snapshot.get("generation", {})
    for arm in sorted({a for a, _ in grid}):
        cells = "  ".join(f"s{seed}:g{grid[(arm, seed)]}" for seed in sorted(s for a, s in grid if a == arm))
        lines.append(f"{arm:<22} {cells}")
    lines.extend([
        f"Running {snapshot['tasks_running']} queued {snapshot['tasks_queued']} | PPO/h now {snapshot['throughput_now_per_h']:.1f}, average {snapshot['throughput_average_per_h']:.1f}",
        f"ETA ({snapshot['eta_source']}): phase {_duration(snapshot['phase_eta_s'][0])}–{_duration(snapshot['phase_eta_s'][1])}; campaign {_duration(snapshot['campaign_eta_s'][0])}–{_duration(snapshot['campaign_eta_s'][1])}" + (" (future phase: prior)" if snapshot.get("future_eta_source") else ""),
        f"Walker milestone ({'reached' if snapshot.get('walker_milestone_reached') else 'latest'}): {_when(snapshot['walker_milestone_epoch'])}; P2 {snapshot['p2_mean_h']} h mean / {snapshot['p2_p95_h']} h p95; projected total {_hours_range(snapshot.get('projection_total_h'))}; >432 h: {snapshot['projection_exceeds_432h']}",
        f"CPU {snapshot.get('cpu_percent', 'n/a')}% per-core {snapshot.get('per_core_percent', [])}; temp {snapshot.get('temperature_c', 'n/a')} °C; RAM {snapshot.get('ram_used_gb', 'n/a')} GB; swap {snapshot.get('swap_used_gb', 'n/a')} GB",
        f"Disk main {snapshot.get('main_disk_free_gb', 'n/a')} GB mirror {snapshot.get('mirror_disk_free_gb', 'n/a')} GB; last mirror {_when(snapshot.get('last_mirror_success'))}",
        f"Infra failures {snapshot['infra_failure_count']}; last error {snapshot['last_error'] or 'none'}; since task {_duration(snapshot['seconds_since_last_task'])}",
        f"Last alert: {snapshot.get('last_alert', {}).get('condition', 'none') if snapshot.get('last_alert') else 'none'}",
    ])
    if snapshot.get("mirror_disabled"):
        lines.append("WARNING: mirror explicitly disabled; results have no second copy")
    if snapshot.get("notify_disabled"):
        lines.append("WARNING: --no-notify: no out-of-band notifier; alerts go to alerts.jsonl only")
    if snapshot.get("same_disk_mirror"):
        lines.append("WARNING: --allow-same-disk-mirror: the mirror shares a physical disk with the results")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", default="results/campaign")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    root = os.path.join(args.output_root, args.campaign)
    while True:
        frame = render(collect(root), sys.stdout.isatty())
        if sys.stdout.isatty():
            print("\x1b[H\x1b[J" + frame, end="", flush=True)
        else:
            print(frame, end="", flush=True)
        if not args.watch:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
