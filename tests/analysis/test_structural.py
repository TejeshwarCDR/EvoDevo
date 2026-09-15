"""Correctness/invariant tests for the fitness-blind structural measurement
substrate (evodevo/analysis/structural.py). Uses small N -- the no-PPO
import guarantee and the N=2000/<10min timing budget are covered by
test_structural_no_ppo_imports.py and manual verification respectively."""
import json
import os

import numpy as np
import pytest

from evodevo.analysis import structural as S
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.direct.encoder import DirectEncoder
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

ENCODER_CLASSES = {
    "direct": DirectEncoder,
    "nca": NCAEncoder,
    "rule_based": RuleBasedEncoder,
}


def _factors(**overrides):
    params = dict(dev_steps=5, init_condition="center_seed", seed=0)
    params.update(overrides)
    return ExperimentFactors(**params)


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_validity_rate_writes_jsonl_and_reports_reasons(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    result = S.validity_rate(encoder, _factors(), 30, rng, output_root=str(tmp_path))

    assert result.n == 30
    assert 0.0 <= result.valid_rate <= 1.0

    jsonl_path = tmp_path / arm / "validity_rate.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 30
    json.loads(lines[0])  # must be valid JSON


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_settling_profile_never_settled_is_reported_explicitly_as_none(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    result = S.settling_profile(encoder, _factors(), 25, rng, t_max=8, output_root=str(tmp_path))

    assert result.settled_count + result.never_settled_count == 25
    assert result.never_settled_count == pytest.approx(round(result.never_settled_fraction * 25))

    records = [json.loads(l) for l in (tmp_path / arm / "settling_profile.jsonl").read_text().strip().splitlines()]
    never_settled_records = [r for r in records if not r["settled"]]
    assert len(never_settled_records) == result.never_settled_count
    for r in never_settled_records:
        assert r["settle_step"] is None  # never coerced to a default like -1 or 0


def test_direct_encoder_is_trivially_settled_at_step_zero(tmp_path):
    encoder = DirectEncoder(factors=_factors())
    rng = np.random.default_rng(0)
    result = S.settling_profile(encoder, _factors(), 10, rng, t_max=8, output_root=str(tmp_path))
    assert result.never_settled_count == 0
    assert result.mean_settle_step == 0.0


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_light_cone_runs_and_reports_never_reached_fraction(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    result = S.light_cone(encoder, _factors(), 10, rng, t_max=8, output_root=str(tmp_path))
    assert 0.0 <= result.never_reached_fraction <= 1.0
    assert result.perturb_cell == (2, 2)


def test_light_cone_direct_arm_has_no_propagation(tmp_path):
    # DirectEncoder's genotype-phenotype map is a pure reshape: perturbing
    # one genome cell can only ever change that one cell's readout, so
    # every OTHER cell must be "never reached".
    encoder = DirectEncoder(factors=_factors())
    rng = np.random.default_rng(0)
    result = S.light_cone(encoder, _factors(), 5, rng, t_max=8, output_root=str(tmp_path))
    assert result.never_reached_fraction == 1.0
    assert result.propagation_speed_cells_per_step is None


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_phenotypic_displacement_only_uses_valid_parents(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    # direct's random-genome valid rate is very low (~2%); use a bigger
    # attempts budget than the default so this test isn't flaky by luck.
    result = S.phenotypic_displacement(
        encoder, _factors(), 5, rng, output_root=str(tmp_path), max_attempts_multiplier=2000
    )
    assert result.n == 5
    assert result.mean_hamming_distance >= 0.0


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_redundancy_and_diversity_are_internally_consistent(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    n = 40
    red = S.redundancy(encoder, _factors(), n, rng, output_root=str(tmp_path))
    assert red.unique_phenotype_count <= n
    assert red.largest_neutral_class_size <= n

    div = S.diversity(encoder, _factors(), n, rng, output_root=str(tmp_path))
    assert div.material_entropy_bits >= 0.0
    assert div.unique_valid_body_count <= n


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_perturbation_robustness_direct_damages_the_genome(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    result = S.perturbation_robustness(encoder, _factors(), 10, rng, output_root=str(tmp_path))
    assert result.damage_fraction == 0.2
    assert result.mean_hamming_distance >= 0.0
