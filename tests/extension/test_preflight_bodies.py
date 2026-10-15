"""The preflight's random bodies come from exactly the campaign's encoders."""
import numpy as np
import pytest

from evodevo.campaign.plan import CampaignPlan
from evodevo.campaign.scheduler import encoder_and_config_for
from evodevo.experiments.benchmark import BenchmarkConfig
from evodevo.extension import spec
from evodevo.extension.preflight import Params, campaign_encoder, noise_items


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
    assert (params.curve_max_timesteps, params.curve_eval_interval, params.noise_reps) == (2_000_000, 100_000, 3)
    assert params.reference_timesteps == {"Walker-v0": 1_000_000, "UpStepper-v0": 1_200_000}
    assert params.is_production
    params.noise_reps = 2
    assert not params.is_production


def test_noise_stage_trains_every_body_on_every_task_plus_references():
    bodies = {"bodies": {spec.body_label(a, k): {"body": [[3] * 5] * 5} for a in spec.ARMS for k in spec.BODY_KINDS}}
    items = noise_items(bodies, {t: 1_000_000 for t in spec.TASK_ORDER}, Params.production())
    keys = [key for key, _meta, _task in items]
    assert len(keys) == len(set(keys)) == 6 * 3 * 3 + 6 * 2
    reference = [task for key, meta, task in items if meta["stage"] == "reference"]
    assert {(t.task, t.timesteps) for t in reference} == {("Walker-v0", 1_000_000), ("UpStepper-v0", 1_200_000)}
    seeds = {(meta["task"], meta["body"], meta["rep"]): task.seed for _k, meta, task in items if meta["stage"] == "noise"}
    assert len({seeds[(spec.TASK_ORDER[0], spec.body_label("direct", "evolved"), r)] for r in range(3)}) == 3
