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
from . import units as unit_keys
from .status import CampaignStatus

# Alias of the single definition; readers use ``estimate.LEGACY_ESTIMATE_PATH`` at call time.
from .estimate import LEGACY_ESTIMATE_PATH as PRIOR_PATH
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


def worker_utilization(rows, worker_count, start, end):
    """Aggregate completed PPO wall time over an observation window.

    The journal's worker start/finish timestamps are authoritative for
    completed work.  Clipping each interval makes this useful for both the
    current hour and a phase-to-date view without double-counting time outside
    the requested window.
    """
    span = max(0.0, end - start)
    if worker_count <= 0 or span <= 0:
        return 0.0
    busy = 0.0
    for row in rows:
        result = row.get("result", {})
        began, finished = result.get("started_at"), result.get("finished_at")
        if not isinstance(began, (int, float)) or not isinstance(finished, (int, float)):
            continue
        busy += max(0.0, min(end, finished) - max(start, began))
    return min(1.0, busy / (worker_count * span))


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


RETRAIN_FILES = ("reeval.jsonl", "robustness.jsonl")


def _retrain_rows(root):
    """Amendment 2 re-training rows (reeval.jsonl, robustness.jsonl), shaped
    like journal rows so utilisation, throughput and the stall clock count them."""
    out = []
    for name in RETRAIN_FILES:
        for row in _rows(os.path.join(root, name)):
            out.append({"run_key": f"{row.get('arm')}/{row.get('task')}/seed_{row.get('seed')}",
                        "stage": row.get("stage"), "completed_at": row.get("completed_at", 0),
                        "result": {"started_at": row.get("started_at"), "finished_at": row.get("finished_at"),
                                   "wall_clock_s": row.get("wall_time_s"), "cpu_ids": row.get("cpu_ids") or [],
                                   "completed": row.get("completed", True),
                                   "error": None if row.get("completed", True) else row.get("reason")}})
    return out


def _settled_stage_rows(root, stage_id):
    from .amend2 import OUTPUT_FILES, latest_rows, settled
    kind = stage_id.split(":", 1)[0]
    path = os.path.join(root, OUTPUT_FILES[kind])
    return sum(1 for key, row in latest_rows(path).items() if key[0] == stage_id and settled(row))


def _prior(path=None):
    """The legacy final-campaign row (kept for callers of the original API)."""
    from . import estimate as estimates
    return estimates._legacy_result(estimates.load(estimates.LEGACY_ESTIMATE_PATH if path is None else path))


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
    plan = manifest.get("plan", {})
    phases = plan.get("phases", [])
    status = manifest.get("phase_status", {})
    units = unit_keys.plan_units(plan) if (phases or plan.get("stages")) else []
    current_unit = next((u for u in units if status.get(u["key"], {}).get("status") != "complete"), None)
    if current_unit is None and units:
        current_unit = units[-1]
    key = current_unit["key"] if current_unit else manifest.get("phase", "unknown")
    task = current_unit["task"] if current_unit else key
    stage = current_unit["stage"] if current_unit and unit_keys.is_stage(current_unit) else None
    current = current_unit.get("phase") if current_unit else None
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
            if unit_keys.phase_key(phase) == key:
                generation[(run["arm"], run["seed"])] = max((r["generation"] for r in rows), default=0)
                logged[(run["arm"], run["seed"])] = (len(rows), sum(bool(r.get("valid")) for r in rows))
    stage_totals = {}
    if plan.get("stages"):
        from .amend2 import planned_trainings
        for item in plan["stages"]:
            stage_totals[item["stage_id"]] = (planned_trainings(item, plan),
                                               _settled_stage_rows(root, item["stage_id"]))
        total_evals += sum(total for total, _ in stage_totals.values())
        completed_evals += sum(done for _, done in stage_totals.values())
    journal = _rows(os.path.join(root, "journal.jsonl"))
    retrain = _retrain_rows(root)
    work = journal + retrain
    completed_ppo = len(work)
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
    measured = bool(runs) and stage is None and all(completed_by_pair[(arm, task)] >= 50 for arm in arms)
    from . import estimate as estimates
    estimate = estimates.load(estimates.estimate_path_for_manifest(manifest))
    v2 = estimates.is_v2(estimate)
    guard_h = estimates.guard_threshold_h(estimate)
    elapsed = max(0.0, now - float(manifest.get("started_at", now)))
    if stage is not None:
        phase_total, completed_phase = stage_totals[stage["stage_id"]]
    else:
        completed_phase = sum(len(_rows(os.path.join(root, "runs", r["arm"], task,
                                                f"seed_{r['seed']}", "run_log.jsonl"))) for r in runs)
        phase_total = sum(int(r["config"]["eval_budget"]) for r in runs)
    worker_count = len(manifest.get("worker_slots", {}).get("slots", []))
    phase_started = float(status.get(key, {}).get("started_at", manifest.get("started_at", now)))
    phase_utilization = worker_utilization(work, worker_count, phase_started, now)
    last_hour_utilization = worker_utilization(work, worker_count, max(phase_started, now - 3600), now)
    remaining_fraction = max(0.0, 1 - completed_phase / phase_total) if phase_total else 0.0
    def require_band(name):
        # A v2 estimate states every phase's band; a missing one is an error,
        # never a silent fall back to the original campaign's hours.
        band = estimates.phase_band(estimate, name)
        if v2 and name in {u["key"] for u in units} and not {"p5", "p95"} <= set(band):
            raise ValueError(f"the campaign's estimate has no phase band for task {name!r}")
        return band

    prior_band = require_band(key)
    prior_phase_low = float(prior_band.get("p5", 121.9 if task == "Walker-v0" else 277.8)) * 3600
    prior_phase_high = float(prior_band.get("p95", 124.1 if task == "Walker-v0" else 282.9)) * 3600
    stage_rows = [r for r in retrain if r.get("stage") == key]
    if stage is not None and len(stage_rows) >= 20 and phase_total:
        # Re-trainings are independent: the stage's own completion rate is the estimate.
        rate = len(stage_rows) / max(1.0, now - phase_started)
        remaining = max(0.0, (phase_total - completed_phase) / rate)
        phase_eta = (0.9 * remaining, 1.1 * remaining)
        eta_source = "measured"
    elif measured:
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
    if current_unit and units.index(current_unit) + 1 < len(units):
        for later in units[units.index(current_unit) + 1:]:
            phase = {"task": later["task"], "key": later["key"]}
            walker = phase["task"] == "Walker-v0"
            band = require_band(phase["key"])
            future_low += float(band.get("p5", band.get("mean", 123 if walker else 280.2))) * 3600
            future_high += float(band.get("p95", band.get("mean", 123 if walker else 280.2))) * 3600
        # Preserve the P2 simulator's *joint* total distribution at campaign
        # start. Summing marginal phase quantiles would discard its observed
        # phase correlation and misstate the registered 400.6–406.0 h band.
        # (Original campaign only; a v2 estimate's phases are summed.)
        if not v2 and eta_source == "prior" and key == "Walker-v0":
            future_low = P2_TOTAL_P5_H * 3600 - prior_phase_low
            future_high = P2_TOTAL_P95_H * 3600 - prior_phase_high
    campaign_eta = (phase_eta[0] + future_low, phase_eta[1] + future_high)
    last_task = max((r.get("completed_at", r.get("timestamp", 0)) for r in work), default=None)
    if last_task is None and manifest.get("status") == "running":
        last_task = status.get(key, {}).get("started_at", manifest.get("started_at"))
    if last_task is not None and stage is not None:
        # A stage's clock starts when the stage does, not at the previous
        # phase's last evolutionary result.
        last_task = max(last_task, phase_started)
    recent = sum(1 for r in work if r.get("completed_at", r.get("timestamp", 0)) >= now - 3600)
    hourly_rates = [sum(1 for r in work if now - (hour + 1) * 3600 <=
                         r.get("completed_at", r.get("timestamp", 0)) < now - hour * 3600)
                    for hour in range(min(24, max(1, math.ceil(elapsed / 3600))))]
    alerts = _rows(os.path.join(root, "alerts.jsonl"))
    alert_states = {}
    for alert in alerts:
        if alert.get("condition"):
            alert_states[alert["condition"]] = alert.get("status")
    dispatch = _json(os.path.join(root, "dispatch_status.json"), {})
    if dispatch.get("task") == key:
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
    failures = sum(1 for r in work if not r.get("result", {}).get("completed", True))
    mirror = manifest.get("mirror_dir")
    mirror_status = _json(os.path.join(root, "mirror_status.json"), {})
    info = telemetry if telemetry is not None else _telemetry(root, mirror)
    snapshot = {
        "campaign_id": manifest.get("campaign_id", os.path.basename(root)), "phase": key,
        "unit_kind": current_unit["kind"] if current_unit else "phase", "task": task,
        "elapsed_s": elapsed, "phase_completed": completed_phase, "phase_total": phase_total,
        "generation": generation, "tasks_running": dispatch.get("tasks_running", 0),
        "tasks_queued": dispatch.get("tasks_queued", 0), "completed_evaluations": completed_evals,
        "worker_rss_bytes": dispatch.get("worker_rss_bytes"),
        "total_evaluations": total_evals, "completed_ppo": completed_ppo,
        "throughput_now_per_h": recent, "throughput_average_per_h": completed_ppo * 3600 / elapsed if elapsed else 0.0,
        "hourly_rates_24h": hourly_rates,
        "phase_eta_s": phase_eta, "campaign_eta_s": campaign_eta, "eta_source": eta_source,
        "worker_count": worker_count, "worker_utilization_since_phase_start": phase_utilization,
        "worker_utilization_last_hour": last_hour_utilization,
        "phase_completion_fraction": (completed_phase / phase_total if phase_total else 0.0),
        "phase_started_at": phase_started,
        "future_eta_source": "prior" if future_high else None,
        # Latest-bound ETA while Walker-v0 runs; the actual time once reached.
        "walker_milestone_epoch": (now + phase_eta[1] if key == "Walker-v0"
                                   else status.get("Walker-v0", {}).get("ended_at")),
        "walker_milestone_reached": status.get("Walker-v0", {}).get("status") == "complete",
        "p2_mean_h": (round(float(estimates.total_band(estimate).get("mean", 0.0)), 1) if v2 else P2_MEAN_H),
        "p2_p95_h": (round(float(estimates.total_band(estimate).get("p95", 0.0)), 1) if v2 else P2_P95_H),
        "guard_h": guard_h, "estimate_v2": v2,
        "projection_total_h": ((elapsed + campaign_eta[0]) / 3600,
                               (elapsed + campaign_eta[1]) / 3600),
        # Key name kept for compatibility; the threshold is the campaign's own guard.
        "projection_exceeds_432h": (elapsed + campaign_eta[1]) / 3600 > guard_h,
        "infra_failure_count": failures, "last_error": next((r.get("result", {}).get("error") for r in reversed(journal)
                                                           if r.get("result", {}).get("error")), None),
        "seconds_since_last_task": now - last_task if last_task else None,
        "last_alert": alerts[-1] if alerts else None,
        "alert_states": alert_states,
        "mirror_disabled": manifest.get("no_mirror", False),
        "notify_disabled": manifest.get("no_notify", False),
        "same_disk_mirror": manifest.get("allow_same_disk_mirror", False),
        "last_mirror_success": mirror_status.get("last_success", manifest.get("last_mirror_success")),
        "as_of_epoch": now,
        "campaign_status": manifest.get("status"),
        "campaign_complete": manifest.get("status") == "complete",
        "phase_transitioning": bool(status.get(key, {}).get("status") == "complete" and
                                    manifest.get("status") != "complete"),
        **info,
    }
    snapshot["campaign_status_object"] = CampaignStatus.from_snapshot(snapshot, now=now)
    return snapshot


def render(snapshot: dict, tty: bool = False, *, ascii_only: bool | None = None, now: float | None = None) -> str:
    status = snapshot.get("campaign_status_object") or CampaignStatus.from_snapshot(snapshot, now=now)
    ascii_only = (os.environ.get("EVODEVO_DASHBOARD_ASCII") == "1") if ascii_only is None else ascii_only
    banner = {"RUNNING": "RUNNING", "PHASE_TRANSITIONING": "PHASE TRANSITIONING",
              "STALLED": "STALLED - CHECK ALERTS", "CAMPAIGN_COMPLETE": "CAMPAIGN COMPLETE"}[status.banner_state]
    eta = "COMPLETE" if status.banner_state == "CAMPAIGN_COMPLETE" else (
        f"{_when(status.most_likely_finish.timestamp())} (range {_when(status.eta_range[0].timestamp()) if status.eta_range[0] else 'n/a'}–{_when(status.eta_range[2].timestamp()) if status.eta_range[2] else 'n/a'})")
    age = max(0, int((time.time() if now is None else now) - status.as_of.timestamp()))
    rule = "=" if ascii_only else "═"
    lines = [rule * 72, f"{banner}",
             f"Phase {status.phase_progress_pct}% | Overall {status.overall_progress_pct}% | ETA: {eta}",
             f"Elapsed: {status.elapsed} | utilization now: {100 * (status.worker_utilization_now or 0):.1f}% | target: {100 * (status.worker_utilization_steady_state_target or 0):.1f}%",
             f"As of {status.as_of.strftime('%Y-%m-%d %H:%M:%S %Z')} ({age}s ago)", rule * 72,
             f"Campaign {snapshot['campaign_id']} | phase {snapshot['phase']} | elapsed {_duration(snapshot['elapsed_s'])}",
             f"Phase {snapshot['phase_completed']}/{snapshot['phase_total']} "
             f"{'evaluations' if snapshot.get('unit_kind', 'phase') == 'phase' else 're-trainings'}; "
             f"total {snapshot['completed_evaluations']}/{snapshot['total_evaluations']}; PPO {snapshot['completed_ppo']}"]
    grid = snapshot.get("generation", {})
    for arm in sorted({a for a, _ in grid}):
        cells = "  ".join(f"s{seed}:g{grid[(arm, seed)]}" for seed in sorted(s for a, s in grid if a == arm))
        lines.append(f"{arm:<22} {cells}")
    lines.extend([
        f"Running {snapshot['tasks_running']} queued {snapshot['tasks_queued']} | PPO/h now {snapshot['throughput_now_per_h']:.1f}, average {snapshot['throughput_average_per_h']:.1f}",
        f"Worker utilization: {100 * snapshot.get('worker_utilization_last_hour', 0):.1f}% last 1 h; {100 * snapshot.get('worker_utilization_since_phase_start', 0):.1f}% since phase start",
        f"ETA ({snapshot['eta_source']}): phase {_duration(snapshot['phase_eta_s'][0])}–{_duration(snapshot['phase_eta_s'][1])}; campaign {_duration(snapshot['campaign_eta_s'][0])}–{_duration(snapshot['campaign_eta_s'][1])}" + (" (future phase: prior)" if snapshot.get("future_eta_source") else ""),
        (f"Estimate {snapshot['p2_mean_h']} h mean / {snapshot['p2_p95_h']} h p95; projected total {_hours_range(snapshot.get('projection_total_h'))}; "
         + (f">{snapshot.get('guard_h', GUARD_H):.0f} h: {snapshot['projection_exceeds_432h']}"
            if snapshot.get('guard_h', GUARD_H) != float('inf') else "no cost guard")
         if snapshot.get("estimate_v2") else
         f"Walker milestone ({'reached' if snapshot.get('walker_milestone_reached') else 'latest'}): {_when(snapshot['walker_milestone_epoch'])}; P2 {snapshot['p2_mean_h']} h mean / {snapshot['p2_p95_h']} h p95; projected total {_hours_range(snapshot.get('projection_total_h'))}; >{snapshot.get('guard_h', GUARD_H):.0f} h: {snapshot['projection_exceeds_432h']}"),
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


def final_screen(root: str, campaign: str, *, abnormal=False, resume_command=None) -> str:
    if abnormal:
        return ("CAMPAIGN STOPPED UNEXPECTEDLY — check alerts.jsonl and the log — resume with: "
                f"{resume_command or f'scripts/campaign/run_campaign.sh --resume {campaign}'}\n")
    return (f"CAMPAIGN COMPLETE — see {root}/CAMPAIGN_SUMMARY.md and CAMPAIGN_COMPLETE.txt — "
            "safe to close this tmux session\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", default="results/campaign")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--ascii", action="store_true", help="use plain ASCII separators")
    args = parser.parse_args(argv)
    root = os.path.join(args.output_root, args.campaign)
    while True:
        frame = render(collect(root), sys.stdout.isatty(), ascii_only=args.ascii)
        if sys.stdout.isatty():
            print("\x1b[H\x1b[J" + frame, end="", flush=True)
        else:
            print(frame, end="", flush=True)
        if not args.watch:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
