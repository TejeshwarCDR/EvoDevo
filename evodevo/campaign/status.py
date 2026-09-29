"""One inexpensive, presentation-facing view of a campaign snapshot."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Any

from .alerts import UTILIZATION_FLOOR


def _pct(done: Any, total: Any) -> int:
    if not total:
        return 100 if done and total == 0 else 0
    return round(100 * float(done or 0) / float(total))


@dataclass(frozen=True)
class CampaignStatus:
    phase: str
    phase_progress_pct: int
    overall_progress_pct: int
    elapsed: dt.timedelta
    eta_range: tuple[dt.datetime | None, dt.datetime | None, dt.datetime | None]
    most_likely_finish: dt.datetime | None
    worker_utilization_now: float | None
    worker_utilization_steady_state_target: float | None
    banner_state: str
    as_of: dt.datetime

    @classmethod
    def from_snapshot(cls, snapshot: dict, *, now: float | None = None) -> "CampaignStatus":
        epoch = float(snapshot.get("as_of_epoch", now if now is not None else dt.datetime.now().timestamp()))
        as_of = dt.datetime.fromtimestamp(epoch).astimezone()
        complete = bool(snapshot.get("campaign_complete") or snapshot.get("campaign_status") == "complete")
        stalled = bool(snapshot.get("stalled"))
        # AlertManager adds this without recomputing the condition.
        stalled = stalled or snapshot.get("alert_states", {}).get("stall") == "active"
        transitioning = bool(snapshot.get("phase_transitioning"))
        if complete:
            banner = "CAMPAIGN_COMPLETE"
        elif stalled:
            banner = "STALLED"
        elif transitioning:
            banner = "PHASE_TRANSITIONING"
        else:
            banner = "RUNNING"
        phase = "COMPLETE" if complete else snapshot.get("phase", "unknown")
        low, high = snapshot.get("campaign_eta_s", (None, None)) or (None, None)
        mean = None if low is None or high is None else (float(low) + float(high)) / 2
        def when(seconds):
            return None if seconds is None else as_of + dt.timedelta(seconds=float(seconds))
        return cls(
            phase=phase,
            phase_progress_pct=100 if complete else _pct(snapshot.get("phase_completed"), snapshot.get("phase_total")),
            overall_progress_pct=100 if complete else _pct(snapshot.get("completed_evaluations"), snapshot.get("total_evaluations")),
            elapsed=dt.timedelta(seconds=max(0, float(snapshot.get("elapsed_s", 0)))),
            eta_range=(when(low), when(mean), when(high)),
            most_likely_finish=when(mean),
            worker_utilization_now=snapshot.get("worker_utilization_last_hour"),
            worker_utilization_steady_state_target=UTILIZATION_FLOOR.get(snapshot.get("phase")),
            banner_state=banner,
            as_of=as_of,
        )


def status_footer(status: CampaignStatus) -> str:
    if status.banner_state == "CAMPAIGN_COMPLETE":
        eta = "COMPLETE"
    else:
        fmt = lambda value: value.strftime("%Y-%m-%d %H:%M %Z") if value else "n/a"
        low, _mean, high = status.eta_range
        eta = f"{fmt(status.most_likely_finish)} (range {fmt(low)}–{fmt(high)})"
    return (f"Phase: {status.phase} ({status.phase_progress_pct}% of phase, "
            f"{status.overall_progress_pct}% overall) · ETA: {eta}")
