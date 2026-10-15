"""The preflight's random bodies come from exactly the campaign's encoders."""
import numpy as np
import pytest

from evodevo.campaign.plan import CampaignPlan
from evodevo.campaign.scheduler import encoder_and_config_for
from evodevo.experiments.benchmark import BenchmarkConfig
from evodevo.extension import spec
import json

from evodevo.extension import preflight
from evodevo.extension.preflight import Params, campaign_encoder, noise_items, run_label


@pytest.mark.parametrize("arm", spec.ARMS)
def test_preflight_encoder_decodes_like_the_campaign_scheduler(arm):
    plan = CampaignPlan.from_benchmark(BenchmarkConfig.from_yaml(
        "evodevo/experiments/configs/campaign_ext_mini_validation.yaml"))
    run_plan = next(r for r in plan.phases[0].runs if r.arm == arm and r.seed == 0)
    scheduler_encoder, _config = encoder_and_config_for(run_plan)
    preflight_encoder = campaign_encoder(arm, spec.TASK_ORDER[0])
    for attempt in range(5):
        genome = preflight_encoder.random_genome(np.random.default_rng([spec.RANDOM_BODY_SEED, 0, attempt]))
        assert np.array_equal(preflight_encoder.decode(genome), scheduler_encoder.decode(genome))


def test_production_parameters_are_the_registered_ones():
    params = Params.production()
    assert (params.curve_max_timesteps, params.curve_eval_interval, params.noise_reps) == (2_000_000, 100_000, 4)
    assert params.reference_timesteps == {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
    assert params.is_production
    params.noise_reps = 2
    assert not params.is_production


def test_noise_stage_trains_every_body_on_every_task_plus_references():
    bodies = {"bodies": {spec.body_label(a, k): {"body": [[3] * 5] * 5} for a in spec.ARMS for k in spec.BODY_KINDS}}
    items = noise_items(bodies, {t: 1_000_000 for t in spec.TASK_ORDER}, Params.production())
    keys = [key for key, _meta, _task in items]
    assert len(keys) == len(set(keys)) == 84 == 6 * 3 * 4 + 6 * 2
    reference = [task for key, meta, task in items if meta["stage"] == "reference"]
    assert {(t.task, t.timesteps) for t in reference} == {("Walker-v0", 1_000_000), ("UpStepper-v0", 1_200_000)}
    seeds = {(meta["task"], meta["body"], meta["rep"]): task.seed for _k, meta, task in items if meta["stage"] == "noise"}
    assert len({seeds[(spec.TASK_ORDER[0], spec.body_label("direct", "evolved"), r)] for r in range(4)}) == 4


def test_run_label_requires_production_params_one_clean_commit_and_never_recovers():
    production = Params.production()
    clean = [{"git_commit": "a", "dirty": False}, {"git_commit": "a", "dirty": False}]
    assert run_label(clean, production, None) == "PRODUCTION"
    assert run_label(clean, production, "PRODUCTION") == "PRODUCTION"
    assert run_label(clean, production, "TEST") == "TEST"
    assert run_label(clean + [{"git_commit": "a", "dirty": True}], production, None) == "TEST"
    assert run_label(clean + [{"git_commit": "b", "dirty": False}], production, None) == "TEST"
    shrunk = Params.production()
    shrunk.noise_reps = 1
    assert run_label(clean, shrunk, None) == "TEST"


def test_resume_skips_a_truncated_last_line_and_repairs_it(tmp_path, capsys):
    path = tmp_path / "noise.jsonl"
    path.write_text(json.dumps({"key": "a"}) + "\n" + json.dumps({"key": "b"}) + "\n" + '{"key": "c", "fit')
    assert [row["key"] for row in preflight._rows(path)] == ["a", "b"]
    assert "truncated last line" in capsys.readouterr().out
    preflight._append(path, {"key": "d"})
    assert [row["key"] for row in preflight._rows(path)] == ["a", "b", "d"]
    assert path.read_text().endswith("\n")


def test_malformed_line_elsewhere_raises(tmp_path):
    path = tmp_path / "noise.jsonl"
    path.write_text(json.dumps({"key": "a"}) + "\n" + "{broken\n" + json.dumps({"key": "c"}) + "\n")
    with pytest.raises(ValueError, match="malformed line 2"):
        preflight._rows(path)
    path.write_text(json.dumps({"key": "a"}) + "\n" + "{broken\n")   # newline-terminated: not a crash tail
    with pytest.raises(ValueError, match="malformed line 2"):
        preflight._rows(path)


def test_complete_row_that_lost_only_its_newline_is_kept(tmp_path):
    path = tmp_path / "noise.jsonl"
    path.write_text(json.dumps({"key": "a"}))
    assert [row["key"] for row in preflight._rows(path)] == ["a"]
    preflight._append(path, {"key": "b"})
    assert [row["key"] for row in preflight._rows(path)] == ["a", "b"]


def test_preflight_training_uses_the_campaigns_eval_episodes():
    assert spec.campaign_eval_episodes() == 1
    bodies = {"bodies": {spec.body_label(a, k): {"body": [[3] * 5] * 5} for a in spec.ARMS for k in spec.BODY_KINDS}}
    tasks = [task for _k, _m, task in noise_items(bodies, {t: 1_000_000 for t in spec.TASK_ORDER}, Params.production())]
    assert {task.eval_episodes for task in tasks} == {spec.campaign_eval_episodes()}
