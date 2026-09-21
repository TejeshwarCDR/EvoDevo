"""D5 B7: reliability aggregation.

Stage `reliability` -- cost "free" (pure aggregation over what earlier
stages already measured and already wrote to manifest.json/events.jsonl/
telemetry.jsonl; no new evaluation, no new PPO), depends_on ("projection",).
Sizes the campaign's safety margins: how often things fail, how much
headroom RAM/thermal/disk actually have, and whether B1's D13 risk
register still matches what the rest of the pre-flight run actually
observed.

WHAT THIS STAGE AGGREGATES:
  - Failures across every prior stage's events (evodevo.preflight.events),
    classified by a disclosed, best-effort text match on each recorded
    error string (see _classify_error) into invalid / timeout /
    worker_crash / oom / other -- never silently dropped or averaged away.
  - Peak RSS per worker (the max across every prior stage's own pool
    stats) and the concurrency it implies is safe against this machine's
    measured total RAM, at a stated (not assumed) margin.
  - Thermal/clock behaviour over the WHOLE pre-flight run, read directly
    from telemetry.jsonl: max package temperature, time spent above a
    threshold, and a same-run first-third-vs-last-third mean CPU frequency
    comparison as evidence of sustained-load frequency decline.
  - Disk: bytes actually written during pre-flight (measured from
    telemetry's own disk_free_gb samples), extrapolated to the campaign's
    total evaluation count (from B6's projection table) with D10's 3x
    safety margin applied, checked against this run's own measured free
    space.
  - A RISK TABLE mirroring B1's D13 risk register, with every row either a
    measured value or an explicit "not measured" plus the reason why --
    never a blank cell.

MUST NOT resolve D14/D15/D16 and MUST NOT recommend a configuration -- see
evodevo/preflight/stages/b6_projection.py's identical constraint; this
stage's whole job is reporting the measured reliability picture, not
choosing anything from it.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

from evodevo.preflight.events import read_events
from evodevo.preflight.run_dir import read_json

DEFAULT_THERMAL_THRESHOLD_C = 85.0
DEFAULT_RAM_MARGIN = 0.8  # use at most 80% of total RAM for the implied-max-concurrency estimate
DISK_SAFETY_MARGIN = 3.0  # D10's 3x safety margin

PRIOR_STAGES = ("machine_profile", "concurrency_sweep", "evaluation_cost", "learning_curves", "evaluation_noise", "projection")


def _classify_error(error_text: Optional[str]) -> str:
    """Best-effort classification of a recorded error string -- disclosed
    as such everywhere it's used, never presented as a precise measurement
    the way a directly-counted number (e.g. n_successful) is."""
    if not error_text:
        return "none"
    text = error_text.lower()
    if "oom" in text or "out of memory" in text or "memoryerror" in text or "cannot allocate memory" in text:
        return "oom"
    if "brokenprocesspool" in text or "terminated abruptly" in text or "crashed after" in text:
        return "worker_crash"
    if "timed_out" in text or "timeout" in text or "timed out" in text:
        return "timeout"
    return "other"


def _aggregate_failures(ctx) -> dict:
    """total_attempted (best-effort, from each stage's own n_successful/
    n_failed/n_requested-shaped counters) and failures_by_class (from
    events.jsonl's own recorded "failure"/"anomaly" entries)."""
    failures_by_class = {"invalid": 0, "timeout": 0, "worker_crash": 0, "oom": 0, "other": 0}
    total_attempted = 0
    total_successful = 0
    workers_restarted_total = 0
    peak_rss_mb_values = []

    for stage_name in ("concurrency_sweep", "evaluation_cost", "learning_curves", "evaluation_noise"):
        try:
            data = read_json(ctx.run_dir.stage_output_path(stage_name))
        except FileNotFoundError:
            continue

        if stage_name == "concurrency_sweep":
            for level in data.get("levels", []):
                total_attempted += level.get("n_genomes", 0)
                total_successful += level.get("n_valid", 0) + level.get("n_invalid", 0)
                workers_restarted_total += level.get("workers_restarted", 0) or 0
                rss = level.get("peak_rss_mb")
                if rss:
                    peak_rss_mb_values.append(rss)

        elif stage_name == "evaluation_cost":
            for arm_data in data.get("arms", {}).values():
                for task_data in arm_data.get("tasks", {}).values():
                    total_attempted += task_data.get("n_requested", 0)
                    total_successful += task_data.get("n_successful", 0)

        elif stage_name == "learning_curves":
            for task_data in data.get("tasks", {}).values():
                for body in task_data.get("bodies", []):
                    total_attempted += 1
                    if not body.get("failed"):
                        total_successful += 1

        elif stage_name == "evaluation_noise":
            for task_data in data.get("tasks", {}).values():
                for body in task_data.get("bodies", []):
                    total_attempted += body.get("n_requested", 0)
                    total_successful += body.get("n_successful", 0)

    for event in read_events(ctx.run_dir.events_path):
        if event.get("kind") != "failure":
            continue
        payload = event.get("payload") or {}
        errors = payload.get("errors")
        if errors is None:
            single = payload.get("error")
            errors = [single] if single else []
        for err in errors:
            cls = _classify_error(err)
            if cls != "none":
                failures_by_class[cls] = failures_by_class.get(cls, 0) + 1

    total_failed = total_attempted - total_successful
    crash_rate_per_1000 = None
    if total_attempted > 0:
        crash_count = failures_by_class["worker_crash"] + failures_by_class["oom"]
        crash_rate_per_1000 = crash_count / total_attempted * 1000.0

    return {
        "total_attempted": total_attempted,
        "total_successful": total_successful,
        "total_failed": max(0, total_failed),
        "failures_by_class": failures_by_class,
        "workers_restarted_total": workers_restarted_total,
        "crash_rate_per_1000_evals": crash_rate_per_1000,
        "peak_rss_mb_values": peak_rss_mb_values,
    }


def _implied_max_concurrency(peak_rss_mb_values: list, total_ram_mb: Optional[float], margin: float = DEFAULT_RAM_MARGIN) -> dict:
    if not peak_rss_mb_values or not total_ram_mb:
        return {"max_safe_concurrency": None, "peak_rss_mb": None, "margin": margin, "reason": "no peak_rss measurements or unknown total RAM"}
    peak = max(peak_rss_mb_values)
    usable_ram_mb = total_ram_mb * margin
    return {
        "max_safe_concurrency": int(usable_ram_mb // peak) if peak > 0 else None,
        "peak_rss_mb": peak,
        "total_ram_mb": total_ram_mb,
        "margin": margin,
    }


def _read_telemetry_samples(ctx) -> list:
    import json

    samples = []
    path = ctx.run_dir.telemetry_path
    if not os.path.exists(path):
        return samples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _thermal_summary(samples: list, threshold_c: float = DEFAULT_THERMAL_THRESHOLD_C) -> dict:
    temps = [s["cpu_package_temp_c"] for s in samples if s.get("cpu_package_temp_c") is not None]
    if not temps:
        return {
            "max_temp_c": None, "time_above_threshold_s": None, "threshold_c": threshold_c,
            "freq_decline_pct": None, "implied_projection_effect_pct": None,
            "reason": "no cpu_package_temp_c samples in telemetry.jsonl",
        }

    intervals = []
    for a, b in zip(samples, samples[1:]):
        if a.get("ts_epoch") is not None and b.get("ts_epoch") is not None:
            intervals.append(b["ts_epoch"] - a["ts_epoch"])
    typical_interval_s = (sum(intervals) / len(intervals)) if intervals else 5.0

    n_above = sum(1 for t in temps if t >= threshold_c)
    time_above_s = n_above * typical_interval_s

    freq_samples = []
    for s in samples:
        per_core = s.get("cpu_freq_per_core_mhz")
        if per_core:
            freq_samples.append(sum(per_core) / len(per_core))

    freq_decline_pct = None
    if len(freq_samples) >= 6:
        third = max(1, len(freq_samples) // 3)
        first_mean = sum(freq_samples[:third]) / third
        last_mean = sum(freq_samples[-third:]) / third
        if first_mean:
            freq_decline_pct = (first_mean - last_mean) / first_mean * 100.0

    return {
        "max_temp_c": max(temps),
        "time_above_threshold_s": time_above_s,
        "threshold_c": threshold_c,
        "freq_decline_pct": freq_decline_pct,
        # A same-run frequency decline is used directly as the implied
        # projection effect -- e.g. a 5% mean-frequency drop under
        # sustained load implies roughly a 5% throughput shortfall against
        # what B2/B3 measured over their own (shorter) windows. This is a
        # disclosed approximation, not a separately re-derived number.
        "implied_projection_effect_pct": freq_decline_pct,
    }


def _disk_projection(samples: list, run_dir_path: str, b6_output: dict, preflight_total_evals: int) -> dict:
    disk_free_values = [(s.get("ts_epoch"), s.get("disk_free_gb")) for s in samples if s.get("disk_free_gb") is not None]
    if len(disk_free_values) < 2:
        return {
            "bytes_written_gb": None, "projected_campaign_gb": None, "projected_with_margin_gb": None,
            "disk_free_gb_now": None, "fits_in_free_space": None,
            "reason": "fewer than 2 disk_free_gb telemetry samples -- cannot measure bytes written",
        }

    disk_free_values.sort(key=lambda p: (p[0] if p[0] is not None else 0))
    first_free = disk_free_values[0][1]
    last_free = disk_free_values[-1][1]
    bytes_written_gb = max(0.0, first_free - last_free)

    total_campaign_evals = sum(row.get("total_evaluations", 0) for row in b6_output.get("rows", []))

    try:
        disk_free_gb_now = shutil.disk_usage(run_dir_path).free / 1e9
    except OSError:
        disk_free_gb_now = None

    if not preflight_total_evals or bytes_written_gb <= 0:
        return {
            "bytes_written_gb": bytes_written_gb, "projected_campaign_gb": None, "projected_with_margin_gb": None,
            "disk_free_gb_now": disk_free_gb_now, "fits_in_free_space": None,
            "reason": "no pre-flight evaluation count to derive a per-evaluation disk rate from",
        }

    gb_per_eval = bytes_written_gb / preflight_total_evals
    projected_gb = gb_per_eval * total_campaign_evals
    projected_with_margin_gb = projected_gb * DISK_SAFETY_MARGIN

    return {
        "bytes_written_gb": bytes_written_gb,
        "gb_per_eval": gb_per_eval,
        "campaign_total_evaluations": total_campaign_evals,
        "projected_campaign_gb": projected_gb,
        "safety_margin": DISK_SAFETY_MARGIN,
        "projected_with_margin_gb": projected_with_margin_gb,
        "disk_free_gb_now": disk_free_gb_now,
        "fits_in_free_space": (
            (projected_with_margin_gb <= disk_free_gb_now) if disk_free_gb_now is not None else None
        ),
    }


def _add_row(rows: list, name: str, value, reason: Optional[str] = None, contradicts_plan: bool = False) -> None:
    measured = value is not None and value != {}
    rows.append(
        {
            "row": name,
            "measured": measured,
            "value": value,
            "note": (reason or "not measured") if not measured else (reason or ""),
            "contradicts_plan": contradicts_plan,
        }
    )


_FREQ_POLICY_NOTE = (
    "a measurement whose frequency policy is unrecorded cannot be reproduced -- this value must be "
    "IDENTICAL between this pre-flight and the real campaign, not just 'close enough'; if it differs, "
    "treat every B2-B5 number in this run directory as suspect until re-measured under matching policy"
)


def _build_risk_table(ctx, failures: dict, concurrency_info: dict, thermal: dict, disk: dict) -> list:
    try:
        machine = read_json(ctx.run_dir.machine_path)
        risk_state = machine.get("b1_machine_profile", {}).get("risk_state", {})
        cpu = machine.get("b1_machine_profile", {}).get("cpu", {})
        freq_policy = cpu.get("freq_policy", {})
        risk_state_reason = None
    except FileNotFoundError:
        risk_state = {}
        cpu = {}
        freq_policy = {}
        risk_state_reason = "machine.json not found -- machine_profile (B1) did not run in this run directory"

    rows = []
    swap_in_use = risk_state.get("swap_in_use")
    _add_row(
        rows, "swap_in_use", swap_in_use,
        reason=risk_state_reason or ("any swap use during the campaign is an anomaly, not a cushion" if swap_in_use is not None else None),
        contradicts_plan=bool(swap_in_use),
    )
    _add_row(rows, "ulimit_n_soft", risk_state.get("ulimit_n_soft"), reason=risk_state_reason)
    _add_row(rows, "ulimit_c_soft", risk_state.get("ulimit_c_soft"), reason=risk_state_reason)
    sdb1_mounted = risk_state.get("sdb1_mounted")
    _add_row(rows, "sdb1_mounted", sdb1_mounted, reason=risk_state_reason)
    nvidia_loaded = risk_state.get("nvidia_driver_loaded")
    _add_row(
        rows, "nvidia_driver_loaded", nvidia_loaded, reason=risk_state_reason,
        contradicts_plan=bool(nvidia_loaded),  # the campaign's plan is CPU-only (see evodevo/parallel's own decision)
    )
    _add_row(rows, "unattended_upgrades_enabled", risk_state.get("unattended_upgrades_enabled"), reason=risk_state_reason)
    coretemp_readable = risk_state.get("coretemp_readable")
    _add_row(
        rows, "coretemp_readable", coretemp_readable, reason=risk_state_reason,
        contradicts_plan=(coretemp_readable is False),
    )

    # D13 frequency policy (see PREFLIGHT_READINESS.md Section 9's own
    # subsection on why an unrecorded frequency policy is a reproducibility
    # gap) -- every field here is a VALUE this run actually observed, never
    # an assumption; see b1_machine.py's _cpu_freq_policy().
    freq_reason = risk_state_reason or _FREQ_POLICY_NOTE
    _add_row(rows, "cpufreq_governor", cpu.get("governor"), reason=freq_reason)
    _add_row(rows, "cpufreq_scaling_driver", freq_policy.get("scaling_driver"), reason=freq_reason)
    _add_row(
        rows, "energy_performance_preference_p_core", freq_policy.get("energy_performance_preference_p_core"),
        reason=freq_reason,
    )
    _add_row(
        rows, "energy_performance_preference_e_core", freq_policy.get("energy_performance_preference_e_core"),
        reason=freq_reason,
    )
    no_turbo = freq_policy.get("intel_pstate_no_turbo")
    _add_row(
        rows, "intel_pstate_no_turbo", no_turbo, reason=freq_reason,
        # Turbo ENABLED (no_turbo=False) during a measurement that will be
        # compared against another run where turbo behaved differently is
        # exactly the kind of silent confound this row exists to surface --
        # not inherently bad, but worth a reader's attention every time.
        contradicts_plan=(no_turbo is False),
    )
    _add_row(rows, "intel_pstate_status", freq_policy.get("intel_pstate_status"), reason=freq_reason)

    _add_row(
        rows, "crash_rate_per_1000_evals", failures.get("crash_rate_per_1000_evals"),
        reason=None if failures.get("crash_rate_per_1000_evals") is not None else "no evaluations attempted in this run to compute a rate from",
        contradicts_plan=bool((failures.get("crash_rate_per_1000_evals") or 0) > 10),
    )
    _add_row(
        rows, "peak_rss_mb", concurrency_info.get("peak_rss_mb"),
        reason=concurrency_info.get("reason"),
    )
    _add_row(
        rows, "implied_max_safe_concurrency", concurrency_info.get("max_safe_concurrency"),
        reason=concurrency_info.get("reason"),
    )
    _add_row(
        rows, "max_package_temp_c", thermal.get("max_temp_c"), reason=thermal.get("reason"),
        contradicts_plan=bool(thermal.get("max_temp_c") and thermal["max_temp_c"] >= 95.0),
    )
    _add_row(
        rows, "sustained_load_freq_decline_pct", thermal.get("freq_decline_pct"),
        reason=thermal.get("reason") or "fewer than 6 telemetry samples with frequency data",
    )
    _add_row(
        rows, "disk_projection_fits_free_space", disk.get("fits_in_free_space"),
        reason=disk.get("reason"),
        contradicts_plan=(disk.get("fits_in_free_space") is False),
    )

    return rows


class ReliabilityStage:
    name = "reliability"
    description = (
        "Reliability aggregation (D5 B7): failures by class, crash rate per 1000 evaluations, "
        "peak-RSS-implied max safe concurrency, whole-run thermal/frequency behaviour, disk "
        "projection with the D10 3x margin, and a risk table mirroring B1's D13 register."
    )
    estimated_cost = "free"
    depends_on = ("projection",)

    def __init__(
        self, thermal_threshold_c: float = DEFAULT_THERMAL_THRESHOLD_C, ram_margin: float = DEFAULT_RAM_MARGIN,
        self_check_stamp=None,
    ):
        self.thermal_threshold_c = thermal_threshold_c
        self.ram_margin = ram_margin
        self.self_check_stamp = self_check_stamp  # set only by --self-check

    def run(self, ctx) -> dict:
        failures = _aggregate_failures(ctx)

        try:
            machine = read_json(ctx.run_dir.machine_path)
            total_ram_mb = machine.get("ram_total_mb")
        except FileNotFoundError:
            total_ram_mb = None
        concurrency_info = _implied_max_concurrency(failures["peak_rss_mb_values"], total_ram_mb, self.ram_margin)

        telemetry_samples = _read_telemetry_samples(ctx)
        thermal = _thermal_summary(telemetry_samples, self.thermal_threshold_c)

        try:
            b6_output = read_json(ctx.run_dir.stage_output_path("projection"))
        except FileNotFoundError:
            b6_output = {"rows": []}
        disk = _disk_projection(telemetry_samples, ctx.run_dir.path, b6_output, failures["total_attempted"])

        risk_table = _build_risk_table(ctx, failures, concurrency_info, thermal, disk)

        contradictions = [row for row in risk_table if row["contradicts_plan"]]
        for row in contradictions:
            ctx.events.record(self.name, "anomaly", {"risk_row": row["row"], "value": row["value"], "note": row["note"]})

        unmeasured = [row["row"] for row in risk_table if not row["measured"]]

        ctx.events.record(
            self.name, "measurement",
            {"total_attempted": failures["total_attempted"], "total_failed": failures["total_failed"],
             "crash_rate_per_1000_evals": failures["crash_rate_per_1000_evals"]},
        )

        return {
            "failures": failures,
            "concurrency_info": concurrency_info,
            "thermal": thermal,
            "disk": disk,
            "risk_table": risk_table,
            "n_contradictions": len(contradictions),
            "n_unmeasured_rows": len(unmeasured),
            "unmeasured_rows": unmeasured,
            # "partial" tracks INCOMPLETE data (rows this run couldn't
            # measure), not contradictions -- a contradiction is a
            # complete, genuine finding worth flagging loudly (see
            # n_contradictions / the "anomaly" events just recorded above),
            # not a sign this result itself is untrustworthy.
            "partial": bool(unmeasured),
            "self_check_stamp": self.self_check_stamp,
        }
