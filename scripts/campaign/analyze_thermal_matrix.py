#!/usr/bin/env python3
"""Turn controlled thermal benchmark JSON into a launch-selection report.

This is deliberately a *selection aid*, not a launcher.  It derives the
campaign projection from measured real-PPO throughput relative to the accepted
20-worker capacity estimate and rejects results that lack sufficient duration,
thermal headroom, affinity verification, or failure evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE_GPM = 46.27
BASELINE_DAYS = 16.8


def _value(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"{path}: expected an object with a results list")
    settings = payload.get("settings", {})
    return [{**row, "source": str(path),
             "control_label": row.get("control_label", settings.get("control_label", "none"))}
            for row in payload["results"]]


def _evaluate(row: dict, args) -> dict:
    gpm = row.get("genomes_per_minute")
    wall_s = row.get("wall_clock_s")
    # A watchdog-aborted child never writes its normal result payload, but the
    # parent has already persisted one-second telemetry.  Preserve that run's
    # actual observed duration instead of presenting it as a zero-minute
    # snapshot in the selection table.
    if wall_s is None:
        elapsed = [sample.get("elapsed_s") for sample in row.get("telemetry", [])
                   if sample.get("elapsed_s") is not None]
        if elapsed:
            wall_s = max(elapsed)
    peak = row.get("package_temp_max_c")
    projection = BASELINE_DAYS * BASELINE_GPM / gpm if gpm else None
    reasons = []
    if row.get("aborted"):
        reasons.append("safety abort")
    if not gpm:
        reasons.append("no completed throughput")
    if wall_s is None or wall_s < args.min_duration_s:
        reasons.append(f"duration < {args.min_duration_s / 60:.0f} min")
    if peak is None or peak > args.accept_max_temp_c:
        reasons.append(f"peak exceeds {args.accept_max_temp_c:.0f} C")
    if row.get("throttle_counter_incremented") is not False:
        reasons.append("throttle counter incremented or unavailable")
    if row.get("worker_failures") != 0:
        reasons.append("worker failure or unavailable")
    if row.get("affinity_verified") is not True:
        reasons.append("affinity unverified")
    if projection is None or not (args.min_days <= projection <= args.max_days):
        reasons.append(f"projection outside {args.min_days:g}–{args.max_days:g} d")
    return {**row, "observed_wall_clock_s": wall_s,
            "projected_days": projection, "accepted": not reasons,
            "decision_reason": "; ".join(reasons) if reasons else "meets all measured gates"}


def _markdown(rows: list[dict], args) -> str:
    lines = [
        "# Controlled thermal matrix selection",
        "",
        "Projection: `16.8 days × 46.27 measured-baseline genomes/min ÷ measured genomes/min`.",
        f"Acceptance gates: at least {args.min_duration_s / 60:.0f} minutes, peak at most "
        f"{args.accept_max_temp_c:.0f} C, no throttle increment, zero worker failures, verified affinity, "
        f"and {args.min_days:g}–{args.max_days:g} projected days.",
        "",
        "| Profile | Control | Workers | Throughput (g/min) | Duration (min) | Peak C | Throttle | Failures | Affinity | Projection (d) | Accepted | Decision |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            _value(row.get("profile")), _value(row.get("control_label")),
            _value(row.get("n_workers")), _value(row.get("genomes_per_minute"), 2),
            _value((row.get("observed_wall_clock_s") or 0) / 60
                   if row.get("observed_wall_clock_s") is not None else None, 1),
            _value(row.get("package_temp_max_c")), _value(row.get("throttle_counter_incremented")),
            _value(row.get("worker_failures")), _value(row.get("affinity_verified")),
            _value(row.get("projected_days")), _value(row.get("accepted")),
            row["decision_reason"].replace("|", "/"),
        ]) + " |")
    accepted = [row for row in rows if row["accepted"]]
    lines.extend(["", "## Recommendation", ""])
    if accepted:
        fastest = min(accepted, key=lambda row: row["projected_days"])
        lines.append(
            f"Select `{fastest['profile']}` under `{fastest['control_label']}`: "
            f"{fastest['projected_days']:.1f} projected days with all gates satisfied."
        )
    else:
        lines.append("No configuration satisfies all launch gates; do not launch the campaign.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-duration-s", type=float, default=1_200.0)
    parser.add_argument("--accept-max-temp-c", type=float, default=90.0)
    parser.add_argument("--min-days", type=float, default=15.0)
    parser.add_argument("--max-days", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.min_days <= 0 or args.max_days < args.min_days:
        parser.error("require 0 < --min-days <= --max-days")
    rows = []
    for path in args.inputs:
        rows.extend(_evaluate(row, args) for row in _load(path))
    rows.sort(key=lambda row: (row.get("projected_days") is None, row.get("projected_days") or float("inf")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(rows, args))
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"settings": vars(args), "results": rows}, indent=2, default=str) + "\n")
    print(f"wrote {args.output} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
