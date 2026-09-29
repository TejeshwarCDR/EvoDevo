#!/usr/bin/env python3
"""Controlled real-PPO thermal benchmark for explicit CPU-affinity mixes.

This is an investigation tool, not a campaign launcher.  Each profile runs in
its own process group so the parent can terminate the complete worker tree if
the package reaches ``--abort-temp-c``.  The output records one-second thermal,
CPU and throttle-counter telemetry alongside throughput and worker affinity.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evodevo.parallel.threading_control import configure_threading

configure_threading(1, strict=True)

import argparse
import json
import signal
import subprocess
import time

import psutil

from evodevo.parallel.topology import AffinityPlan, allocation_order, detect_topology
from evodevo.preflight.telemetry import (
    read_cpu_package_temp_c,
    read_cpu_package_temp_critical_c,
    read_thermal_throttle_count,
)


PROFILES = {
    "e8": (16, 17, 18, 19, 20, 21, 22, 23),
    "e10": (16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    "e12": (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p1e12": (0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p2e12": (0, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p3e12": (0, 2, 4, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p4e8": (0, 2, 4, 6, 16, 17, 18, 19, 20, 21, 22, 23),
    "p4e12": (0, 2, 4, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p6e6": (0, 2, 4, 6, 8, 10, 16, 17, 18, 19, 20, 21),
    "p6e10": (0, 2, 4, 6, 8, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    "p6e12": (0, 2, 4, 6, 8, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p7e12": (0, 2, 4, 6, 8, 10, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
    "p8": (0, 2, 4, 6, 8, 10, 12, 14),
    "p8e10": (0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    "p8e12": (0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
}


def _plan(cpu_ids: tuple[int, ...]) -> AffinityPlan:
    topology = detect_topology()
    available = {core.logical_id for core in topology.cores}
    missing = sorted(set(cpu_ids) - available)
    if missing:
        raise ValueError(f"profile references unavailable CPUs: {missing}")
    if len(set(cpu_ids)) != len(cpu_ids):
        raise ValueError("profile contains duplicate CPUs")
    return AffinityPlan(
        n_workers=len(cpu_ids),
        slots=tuple((cpu,) for cpu in cpu_ids),
        oversubscribed=False,
        allocation_order=tuple(allocation_order(topology)),
    )


def _tasks(n_genomes: int, timesteps: int):
    from evodevo.parallel.worker_pool import EvaluationTask
    from evodevo.preflight.stages.b2_sweep import CAMPAIGN_EVAL_EPISODES, sample_workload

    workload = sample_workload(n_genomes, seed=0)
    return [
        EvaluationTask(
            task_id=row["task_id"], arm=row["encoder"], strategy="thermal_affinity_benchmark",
            task="Walker-v0", seed=row["seed"], genome=row["genome"],
            ppo_timesteps=timesteps, eval_episodes=CAMPAIGN_EVAL_EPISODES, meta=row["meta"],
        )
        for row in workload
    ]


def child(profile: str, n_genomes: int, timesteps: int, result_path: Path) -> int:
    from evodevo.parallel.worker_pool import EvaluationPool

    cpus = PROFILES[profile]
    plan = _plan(cpus)
    tasks = _tasks(n_genomes, timesteps)
    started = time.time()
    with EvaluationPool(len(cpus), plan, "real", timeout_s=300.0, max_retries=1) as pool:
        results = list(pool.map_tasks(tasks))
    wall_s = time.time() - started
    valid = sum(result.valid for result in results)
    busy_s = sum(result.wall_clock_s or 0.0 for result in results)
    used_cpus = sorted({cpu for result in results for cpu in result.cpu_ids})
    payload = {
        "profile": profile,
        "cpu_ids": list(cpus),
        "n_workers": len(cpus),
        "n_genomes": len(results),
        "n_valid": valid,
        "worker_failures": sum(bool(result.error or result.timed_out) for result in results),
        "wall_clock_s": wall_s,
        "genomes_per_minute": len(results) / wall_s * 60.0,
        "valid_genomes_per_minute": valid / wall_s * 60.0,
        "worker_utilization": busy_s / (len(cpus) * wall_s),
        "observed_worker_cpus": used_cpus,
        "affinity_verified": used_cpus == sorted(cpus),
        "pool_stats": pool.stats(),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def _freq_mhz(cpu_ids: tuple[int, ...]) -> dict[str, float | None]:
    values = {}
    for cpu in cpu_ids:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as handle:
                values[str(cpu)] = int(handle.read().strip()) / 1000.0
        except (OSError, ValueError):
            values[str(cpu)] = None
    return values


def _sample(started: float, cpu_ids: tuple[int, ...]) -> dict:
    return {
        "elapsed_s": time.time() - started,
        "package_temp_c": read_cpu_package_temp_c(),
        "cpu_percent_overall": psutil.cpu_percent(interval=None),
        "thermal_throttle_count": read_thermal_throttle_count(),
        "cpu_freq_mhz": _freq_mhz(cpu_ids),
    }


def run_profile(args, profile: str, output_dir: Path) -> dict:
    result_path = output_dir / f".{profile}.child.json"
    if result_path.exists():
        result_path.unlink()
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child", profile,
        "--genomes", str(args.genomes), "--timesteps", str(args.timesteps),
        "--child-result", str(result_path),
    ]
    started = time.time()
    throttle_start = read_thermal_throttle_count()
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                               start_new_session=True)
    samples = []
    aborted = False
    abort_reason = None
    psutil.cpu_percent(interval=None)
    while process.poll() is None:
        sample = _sample(started, PROFILES[profile])
        samples.append(sample)
        temp = sample["package_temp_c"]
        if temp is not None and temp >= args.abort_temp_c:
            aborted = True
            abort_reason = f"package temperature {temp:.1f} C >= {args.abort_temp_c:.1f} C"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(args.sample_interval)
    return_code = process.wait()
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result_path.unlink()
    else:
        result = {}
    throttle_end = read_thermal_throttle_count()
    temps = [row["package_temp_c"] for row in samples if row["package_temp_c"] is not None]
    result.update({
        "profile": profile,
        "cpu_ids": list(PROFILES[profile]),
        "n_workers": len(PROFILES[profile]),
        "return_code": return_code,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "package_temp_critical_c": read_cpu_package_temp_critical_c(),
        "package_temp_start_c": temps[0] if temps else None,
        "package_temp_max_c": max(temps) if temps else None,
        "package_temp_end_c": temps[-1] if temps else None,
        "throttle_count_start": throttle_start,
        "throttle_count_end": throttle_end,
        "throttle_counter_incremented": (
            throttle_end > throttle_start
            if throttle_start is not None and throttle_end is not None else None
        ),
        "control_label": args.control_label,
        "telemetry": samples,
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="e12,p4e8,p4e12,p6e6,p6e10")
    parser.add_argument("--genomes", type=int, default=120)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--abort-temp-c", type=float, default=96.0)
    parser.add_argument("--cooldown-temp-c", type=float, default=55.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=300.0)
    parser.add_argument("--output", default="results/thermal_worker_study/affinity_benchmark.json")
    parser.add_argument("--control-label", default="none",
                        help="free-text reversible control recorded with the result, e.g. CPUQuota=1700%%")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--child", choices=sorted(PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child, args.genomes, args.timesteps, args.child_result)

    names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = sorted(set(names) - set(PROFILES))
    if unknown:
        parser.error(f"unknown profiles: {unknown}; choices are {sorted(PROFILES)}")
    plans = {name: _plan(PROFILES[name]).to_dict() for name in names}
    if args.dry_run:
        print(json.dumps({"profiles": plans, "real_ppo_started": False}, indent=2))
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "profiles": names, "genomes": args.genomes, "timesteps": args.timesteps,
            "sample_interval_s": args.sample_interval, "abort_temp_c": args.abort_temp_c,
            "cooldown_temp_c": args.cooldown_temp_c,
            "control_label": args.control_label,
        },
        "plans": plans,
        "results": [],
    }
    for name in names:
        deadline = time.time() + args.cooldown_timeout_s
        while (read_cpu_package_temp_c() or 0.0) > args.cooldown_temp_c:
            if time.time() >= deadline:
                raise TimeoutError(f"CPU did not cool to {args.cooldown_temp_c} C before {name}")
            time.sleep(2.0)
        result = run_profile(args, name, output.parent)
        payload["results"].append(result)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({key: result.get(key) for key in (
            "profile", "aborted", "abort_reason", "genomes_per_minute",
            "worker_utilization", "package_temp_max_c", "throttle_counter_incremented",
            "affinity_verified")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
