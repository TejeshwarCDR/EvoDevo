import datetime as dt
import json

from evodevo.campaign.alerts import AlertManager
from evodevo.campaign.dashboard import final_screen, render
from evodevo.campaign.status import CampaignStatus, status_footer


class Spy:
    configured = True
    def __init__(self, fail=False): self.messages, self.fail = [], fail
    def send(self, message):
        self.messages.append(message)
        if self.fail: raise RuntimeError("offline")


def epoch(day, hour, minute=0):
    return dt.datetime(2026, 1, day, hour, minute, tzinfo=dt.datetime.now().astimezone().tzinfo).timestamp()


def snapshot(**more):
    base = {"phase": "Walker-v0", "phase_completed": 20, "phase_total": 100,
            "completed_evaluations": 20, "total_evaluations": 200, "elapsed_s": 3600,
            "campaign_eta_s": (3600, 7200), "worker_utilization_last_hour": .9,
            "as_of_epoch": epoch(1, 14), "campaign_id": "drill", "tasks_running": 1,
            "tasks_queued": 0, "completed_ppo": 20, "throughput_now_per_h": 10,
            "throughput_average_per_h": 10, "generation": {}, "phase_eta_s": (1, 2),
            "eta_source": "prior", "worker_utilization_since_phase_start": .9,
            "walker_milestone_epoch": epoch(2, 10), "walker_milestone_reached": False,
            "p2_mean_h": 1, "p2_p95_h": 2, "projection_total_h": (1, 2),
            "projection_exceeds_432h": False, "infra_failure_count": 0, "last_error": None,
            "seconds_since_last_task": 1, "alert_states": {}}
    base.update(more)
    return base


def test_every_alert_display_message_has_shared_status_footer(tmp_path):
    clock = [epoch(1, 14)]
    spy = Spy()
    manager = AlertManager(tmp_path / "alerts.jsonl", notifier=spy, now=lambda: clock[0])
    status = CampaignStatus.from_snapshot(snapshot(), now=clock[0])
    manager.status = status
    names = ["stall", "temperature", "disk_main", "disk_mirror", "infra_failure_rate", "swap",
             "mirror_failed", "partial_stall", "worker_utilization", "worker_rss"]
    for name in names:
        manager.evaluate(name, True, {"phase": "Walker-v0"}, status=status)
    manager.heartbeat({**snapshot(), "generation": {} })
    manager.campaign_started({"plan": "locked"})
    manager.phase_started("Walker-v0")
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    displayed = [row["display_message"] for row in rows if "display_message" in row]
    assert len(displayed) == 13
    assert all(status_footer(status) in message for message in displayed)
    assert all("Phase: Walker-v0 (20% of phase, 10% overall)" in message for message in displayed)


def test_heartbeat_is_1800_anchored_and_resume_safe(tmp_path):
    clock = [epoch(1, 14)]
    manager = AlertManager(tmp_path / "alerts.jsonl", notifier=Spy(), now=lambda: clock[0])
    manager.inspect(snapshot(), p95_ecore_s=100)
    clock[0] = epoch(1, 18, 1)
    manager.inspect(snapshot(), p95_ecore_s=100)
    clock[0] = epoch(4, 18, 1)  # three missed wall-clock dates: exactly one catch-up
    manager.inspect(snapshot(), p95_ecore_s=100)
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert sum(row["condition"] == "daily_heartbeat" and row["notified"] for row in rows) == 2
    clock = [epoch(1, 20)]
    late = AlertManager(tmp_path / "late.jsonl", notifier=Spy(), now=lambda: clock[0])
    late.inspect(snapshot(), p95_ecore_s=100)
    clock[0] = epoch(2, 18)
    late.inspect(snapshot(), p95_ecore_s=100)
    assert "daily_heartbeat" in (tmp_path / "late.jsonl").read_text()


def test_complete_retries_and_marker_survives_notifier_failure(tmp_path):
    clock = [epoch(1, 18)]
    manager = AlertManager(tmp_path / "alerts.jsonl", notifier=Spy(fail=True), now=lambda: clock[0], sleep=lambda _: None)
    marker = tmp_path / "CAMPAIGN_COMPLETE.txt"
    manager.status = CampaignStatus.from_snapshot(snapshot(campaign_complete=True), now=clock[0])
    manager.campaign_complete({"marker_text": "final summary\n" + status_footer(manager.status)}, marker)
    rows = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["condition"] == "campaign_complete"]) == 3
    assert marker.exists() and "final summary" in marker.read_text()


def test_banner_transitions_freshness_ascii_and_final_screens():
    running = CampaignStatus.from_snapshot(snapshot(), now=epoch(1, 14))
    assert running.banner_state == "RUNNING"
    assert CampaignStatus.from_snapshot(snapshot(phase_transitioning=True), now=epoch(1, 14)).banner_state == "PHASE_TRANSITIONING"
    assert CampaignStatus.from_snapshot(snapshot(alert_states={"stall": "active"}), now=epoch(1, 14)).banner_state == "STALLED"
    complete = CampaignStatus.from_snapshot(snapshot(campaign_complete=True), now=epoch(1, 14))
    assert complete.banner_state == "CAMPAIGN_COMPLETE"
    frame = render(snapshot(campaign_status_object=running), ascii_only=True, now=epoch(1, 14) + 7)
    assert "RUNNING" in frame and "As of" in frame and "(7s ago)" in frame and "=" in frame
    assert "CAMPAIGN COMPLETE" in final_screen("results/campaign/drill", "drill")
    assert "STOPPED UNEXPECTEDLY" in final_screen("x", "drill", abnormal=True)
