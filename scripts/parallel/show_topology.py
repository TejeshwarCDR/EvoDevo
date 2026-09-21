#!/usr/bin/env python3
"""Diagnostic: prints this machine's detected CPU topology, the worker
allocation order, the CPU affinity plan for a given --workers count, and
the current thread-count environment.

Usage:
    python scripts/parallel/show_topology.py [--workers N] [--json]

Run this whenever you need to explain the machine to someone else (a
supervisor, a lab notebook entry, a bug report about slow evaluations) --
it is the one place that answers "what does this machine actually look
like to the scheduler" without reading /sys by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evodevo.parallel.threading_control import THREAD_ENV_VARS, describe_threading_environment
from evodevo.parallel.topology import AffinityPlan, Topology, allocation_order, cpu_slots, detect_topology


def _core_table(topology: Topology) -> str:
    header = f"{'logical':>7}  {'core_id':>7}  {'pkg':>3}  {'kind':>4}  {'primary':>7}  {'siblings':<12}  {'max_freq_ghz':>12}"
    lines = [header, "-" * len(header)]
    for core in topology.cores:
        freq = f"{core.max_freq_khz / 1e6:.2f}" if core.max_freq_khz else "?"
        siblings = ",".join(str(s) for s in core.sibling_ids)
        lines.append(
            f"{core.logical_id:>7}  {core.core_id:>7}  {core.package_id:>3}  {core.kind:>4}  "
            f"{str(core.is_physical_primary):>7}  {siblings:<12}  {freq:>12}"
        )
    return "\n".join(lines)


def _human_report(topology: Topology, order: list[int], plan: AffinityPlan, threading_env: dict) -> str:
    parts = []
    parts.append("=== CPU topology ===")
    parts.append(
        f"logical_count={topology.logical_count}  physical_count={topology.physical_count}  "
        f"hybrid={topology.hybrid}  detection_method={topology.detection_method}"
    )
    parts.append(f"P-cores ({len(topology.p_core_ids)}): {list(topology.p_core_ids)}")
    parts.append(f"E-cores ({len(topology.e_core_ids)}): {list(topology.e_core_ids)}")
    parts.append("")
    parts.append(_core_table(topology))
    parts.append("")
    parts.append("=== allocation order ===")
    parts.append(", ".join(str(c) for c in order))
    parts.append("")
    parts.append(f"=== affinity plan (workers={plan.n_workers}) ===")
    parts.append(f"oversubscribed={plan.oversubscribed}")
    for i, slot in enumerate(plan.slots):
        parts.append(f"  worker {i:>3} -> cpu {list(slot)}")
    parts.append("")
    parts.append("=== threading environment ===")
    for var in THREAD_ENV_VARS:
        parts.append(f"  {var}={threading_env['env'][var]}")
    parts.append(f"  torch_num_threads={threading_env['torch_num_threads']}")
    parts.append(f"  torch_num_interop_threads={threading_env['torch_num_interop_threads']}")
    parts.append(f"  numpy_backend={threading_env['numpy_backend']}")
    return "\n".join(parts)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker count to build the affinity plan for (default: detected logical CPU count)",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a human table")
    args = parser.parse_args(argv)

    topology = detect_topology()
    order = allocation_order(topology)
    n_workers = args.workers if args.workers is not None else topology.logical_count
    plan = cpu_slots(topology, n_workers)
    threading_env = describe_threading_environment()

    if args.json:
        print(
            json.dumps(
                {
                    "topology": topology.to_dict(),
                    "allocation_order": order,
                    "affinity_plan": plan.to_dict(),
                    "threading_environment": threading_env,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_human_report(topology, order, plan, threading_env))


if __name__ == "__main__":
    main()
