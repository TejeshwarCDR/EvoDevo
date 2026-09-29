"""Small, deterministic scheduler checks; real campaign sizes are separate slow tests."""
import json
import argparse
import shlex
from pathlib import Path

import pytest
from evodevo.campaign.plan import CampaignPlan, PhasePlan, RunPlan
from evodevo.campaign.journal import ResultJournal
from evodevo.campaign.production import E_WORKER_CPUS, P_WORKER_CPUS, SCHEDULER_CPUS
from evodevo.campaign.scheduler import CampaignScheduler, cpu_placement, resume_command
from evodevo.core.config import ExperimentConfig
from evodevo.core.factors import extract_factors
from evodevo.encoders import get_encoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea
from evodevo.evolution.logging import RunLogger
from evodevo.parallel.topology import detect_topology


def _plan(tmp_path):
    runs = []
    for arm in ("direct", "nca"):
        for seed in (0, 1):
            config = ExperimentConfig(seed=seed, encoder=arm, task="Walker-v0",
                population_size=3, offspring_size=3, eval_budget=9,
                ppo_timesteps=2000, ppo_eval_episodes=1, output_dir=str(tmp_path))
            runs.append(RunPlan(arm, "Walker-v0", seed, config.to_dict()))
    return CampaignPlan((PhasePlan("Walker-v0", tuple(runs), 2000, 1, "max", 60),))


def _records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("wall_clock_s", None)
    return rows


def test_resume_command_preserves_every_nondefault_launch_option():
    args = argparse.Namespace(config="/tmp/custom benchmark.yaml", output_root="/tmp/campaigns",
        workers=8, mode="surrogate", grace_s=45.0, no_mirror=False, mirror_dir="/mnt/mirror disk")
    assert shlex.split(resume_command(args, "study-01")) == [
        "scripts/campaign/run_campaign.sh", "--resume", "study-01",
        "--config", "/tmp/custom benchmark.yaml", "--output-root", "/tmp/campaigns",
        "--workers", "8", "--mode", "surrogate", "--grace-s", "45.0",
        "--mirror-dir", "/mnt/mirror disk"]


def test_thermal_profile_resume_uses_the_power_held_production_launcher():
    args = argparse.Namespace(config="campaign.yaml", output_root="results/campaign",
        workers=18, cpu_profile="thermal-validated", mode="real", grace_s=120.0,
        no_mirror=False, mirror_dir="/mnt/backup", prefetch=4, no_notify=False,
        allow_same_disk_mirror=False, stamp=None, local_config=None)
    assert shlex.split(resume_command(args, "campaign-01")) == [
        "scripts/campaign/run_production_campaign.sh", "--resume", "campaign-01",
        "--config", "campaign.yaml", "--output-root", "results/campaign",
        "--grace-s", "120.0", "--mirror-dir", "/mnt/backup"]


def test_e_core_profile_reserves_two_e_cores_for_scheduler():
    topology = detect_topology()
    if len(topology.e_core_ids) < 12:
        pytest.skip("thermal campaign profile targets the 12-E-core lab machine")
    workers, scheduler_cpus = cpu_placement(topology, 10, "e-cores")
    assert [slot[0] for slot in workers.slots] == list(topology.e_core_ids[:10])
    assert scheduler_cpus == list(topology.e_core_ids[10:])
    assert not ({slot[0] for slot in workers.slots} & set(scheduler_cpus))


def test_e_core_profile_refuses_to_consume_scheduler_reserve():
    topology = detect_topology()
    if len(topology.e_core_ids) < 12:
        pytest.skip("thermal campaign profile targets the 12-E-core lab machine")
    with pytest.raises(ValueError, match="at least one E-core must remain"):
        cpu_placement(topology, len(topology.e_core_ids), "e-cores")


def test_thermal_validated_profile_is_exact_and_reserves_control_e_cores():
    topology = detect_topology()
    if topology.logical_count < 28:
        pytest.skip("thermal campaign profile targets the 28-logical-CPU lab machine")
    workers, scheduler_cpus = cpu_placement(topology, 18, "thermal-validated")
    assert [slot[0] for slot in workers.slots] == list(P_WORKER_CPUS + E_WORKER_CPUS)
    assert scheduler_cpus == list(SCHEDULER_CPUS)
    assert not ({slot[0] for slot in workers.slots} & set(scheduler_cpus))


def test_thermal_validated_profile_refuses_any_other_worker_count():
    topology = detect_topology()
    if topology.logical_count < 28:
        pytest.skip("thermal campaign profile targets the 28-logical-CPU lab machine")
    with pytest.raises(ValueError, match="exactly 18 workers"):
        cpu_placement(topology, 17, "thermal-validated")


def test_surrogate_shared_phase_matches_serial_and_resume(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    # Milestone is tested independently; a crash here must not change evolution.
    monkeypatch.setattr("evodevo.campaign.scheduler.run_subprocess", lambda *a, **k: {"ok": True})
    scheduler = CampaignScheduler(plan, "interleaved", output_root=tmp_path, workers=2, mode="surrogate", no_mirror=True)
    scheduler.run()
    for rp in plan.phases[0].runs:
        config = ExperimentConfig(**{**rp.config, "grid_shape": tuple(rp.config["grid_shape"])})
        factors, kwargs = extract_factors({}, seed=config.seed)
        encoder = get_encoder(rp.arm, grid_shape=config.grid_shape, factors=factors, **kwargs)
        serial_path = tmp_path / "serial" / rp.arm / f"seed_{rp.seed}"
        logger = RunLogger(str(serial_path))
        baseline = run_ea(encoder, evaluate_body_surrogate, config, logger=logger)
        logger.close()
        campaign_path = tmp_path / "interleaved" / "runs" / rp.arm / rp.task / f"seed_{rp.seed}"
        assert _records(campaign_path / "run_log.jsonl") == _records(serial_path / "run_log.jsonl")
        checkpoint = json.loads((campaign_path / "checkpoint" / "gen_2.json").read_text())
        assert checkpoint["counters"]["evaluations_used"] == baseline.evaluations_used
    # A completed phase must be skipped, without truncating its logs.
    before = (tmp_path / "interleaved" / "runs" / "direct" / "Walker-v0" / "seed_0" / "run_log.jsonl").read_bytes()
    CampaignScheduler(plan, "interleaved", output_root=tmp_path, workers=2, mode="surrogate", no_mirror=True).run()
    assert (tmp_path / "interleaved" / "runs" / "direct" / "Walker-v0" / "seed_0" / "run_log.jsonl").read_bytes() == before


def test_resume_repairs_log_rows_not_covered_by_checkpoint(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr("evodevo.campaign.scheduler.run_subprocess", lambda *a, **k: {"ok": True})
    baseline = CampaignScheduler(plan, "baseline", output_root=tmp_path, workers=2,
                                 mode="surrogate", no_mirror=True)
    baseline.run()
    import evodevo.campaign.scheduler as module
    original = module.write_checkpoint
    calls = [0]

    def crash_after_tell(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("injected crash between tell and checkpoint")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "write_checkpoint", crash_after_tell)
    with pytest.raises(RuntimeError, match="injected crash"):
        CampaignScheduler(plan, "resumed", output_root=tmp_path, workers=2,
                          mode="surrogate", no_mirror=True).run()
    monkeypatch.setattr(module, "write_checkpoint", original)
    CampaignScheduler(plan, "resumed", output_root=tmp_path, workers=2,
                      mode="surrogate", no_mirror=True).run()
    for rp in plan.phases[0].runs:
        relative = Path("runs") / rp.arm / rp.task / f"seed_{rp.seed}" / "run_log.jsonl"
        assert _records(tmp_path / "resumed" / relative) == _records(tmp_path / "baseline" / relative)


def test_surrogate_stop_is_observed_at_the_next_generation_boundary(tmp_path, monkeypatch):
    """A zero-worker surrogate barrier must not recursively drain the phase.

    This is the deterministic form of a SIGINT arriving during surrogate
    evaluation: the current generation is checkpointed, then the scheduler
    must return control to its stop check instead of immediately preparing
    every remaining generation in the inner barrier loop.
    """
    plan = _plan(tmp_path)
    monkeypatch.setattr("evodevo.campaign.scheduler.run_subprocess", lambda *a, **k: {"ok": True})
    import evodevo.campaign.scheduler as module
    original = module.write_checkpoint
    scheduler = CampaignScheduler(plan, "stoppable", output_root=tmp_path, workers=2,
                                  mode="surrogate", no_mirror=True)
    calls = [0]

    def request_stop_after_checkpoint(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 1:
            scheduler.stop_requested = True
        return result

    monkeypatch.setattr(module, "write_checkpoint", request_stop_after_checkpoint)
    scheduler.run()
    manifest = json.loads((tmp_path / "stoppable" / "manifest.json").read_text())
    assert manifest["status"] == "stopped"
    completed = sum(
        len(path.read_text().splitlines())
        for path in (tmp_path / "stoppable" / "runs").rglob("run_log.jsonl")
    )
    assert completed == 3
    monkeypatch.setattr(module, "write_checkpoint", original)
    CampaignScheduler(plan, "stoppable", output_root=tmp_path, workers=2,
                      mode="surrogate", no_mirror=True).run()
    CampaignScheduler(plan, "uninterrupted", output_root=tmp_path, workers=2,
                      mode="surrogate", no_mirror=True).run()
    for rp in plan.phases[0].runs:
        relative = Path("runs") / rp.arm / rp.task / f"seed_{rp.seed}" / "run_log.jsonl"
        assert _records(tmp_path / "stoppable" / relative) == _records(tmp_path / "uninterrupted" / relative)


def test_resume_aborts_loudly_on_journal_genome_hash_mismatch(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr("evodevo.campaign.scheduler.run_subprocess", lambda *a, **k: {"ok": True})
    import evodevo.campaign.scheduler as module
    original = module.write_checkpoint
    scheduler = CampaignScheduler(plan, "hash-mismatch", output_root=tmp_path, workers=2,
                                  mode="surrogate", no_mirror=True)

    def stop_after_first_checkpoint(*args, **kwargs):
        path = original(*args, **kwargs)
        scheduler.stop_requested = True
        return path

    monkeypatch.setattr(module, "write_checkpoint", stop_after_first_checkpoint)
    scheduler.run()
    monkeypatch.setattr(module, "write_checkpoint", original)
    ResultJournal(tmp_path / "hash-mismatch" / "journal.jsonl").append({
        "run_key": "direct/Walker-v0/seed_0",
        "generation": 1,
        "genome_id": 3,
        "eval_index": 3,
        "ppo_k": 0,
        "genome_hash": "deliberately-wrong",
        "result": {},
    })
    with pytest.raises(RuntimeError, match="genome-hash mismatch"):
        CampaignScheduler(plan, "hash-mismatch", output_root=tmp_path, workers=2,
                          mode="surrogate", no_mirror=True).run()


def test_milestone_failure_cannot_abort_completed_phase(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr("evodevo.campaign.scheduler.run_subprocess", lambda *a, **k:
                        {"ok": False, "missing": ["figures/fitness_curves.pdf"]})
    scheduler = CampaignScheduler(plan, "milestone-failure", output_root=tmp_path,
                                  workers=2, mode="surrogate", no_mirror=True)
    outputs = scheduler.run()
    manifest = json.loads((tmp_path / "milestone-failure" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["phase_status"]["Walker-v0"]["status"] == "complete"
    assert manifest["milestone_status"]["Walker-v0"]["ok"] is False
    assert len(outputs) == len(plan.phases[0].runs)
