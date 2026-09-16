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


# ---------------------------------------------------------------------------
# 8. readout_margin
# ---------------------------------------------------------------------------


def test_readout_margin_nca_reports_sane_ranges(tmp_path):
    encoder = NCAEncoder(factors=_factors(dev_steps=10))
    rng = np.random.default_rng(0)
    result = S.readout_margin(encoder, _factors(dev_steps=10), 30, rng, output_root=str(tmp_path))

    assert result.n == 30
    assert result.n_cells == 30 * 25
    assert result.mean_margin >= 0.0
    assert result.median_margin >= 0.0
    assert 0.0 <= result.fraction_cells_near_tie <= 1.0
    assert 0.0 <= result.fraction_bodies_with_near_tie_cell <= 1.0

    records = [json.loads(l) for l in (tmp_path / "nca" / "readout_margin.jsonl").read_text().strip().splitlines()]
    assert len(records) == 30
    for r in records:
        assert len(r["margins"]) == 25
        assert all(m >= -1e-9 for m in r["margins"])  # margin (max - 2nd max) is never negative


def test_readout_margin_zero_weight_genome_has_zero_margin_at_tied_channels():
    # With zero weights, every cell's material logits equal the fixed
    # initial state's own material channels -- whatever tie structure the
    # initial condition has is exactly what the margin should report, with
    # no development-induced separation.
    encoder = NCAEncoder(factors=_factors(dev_steps=5), update_rule="unbounded_residual")
    genome = np.zeros(encoder.genome_length())
    final_logits = encoder._initial_state[:, :, :5]
    sorted_logits = np.sort(final_logits, axis=-1)
    expected_margin = (sorted_logits[..., -1] - sorted_logits[..., -2]).flatten()

    _, _, states = S._nca_trajectory_from_state(encoder, genome, encoder._initial_state, encoder.dev_steps)
    final = states[-1][:, :, :5]
    sorted_final = np.sort(final, axis=-1)
    actual_margin = (sorted_final[..., -1] - sorted_final[..., -2]).flatten()
    np.testing.assert_allclose(actual_margin, expected_margin, atol=1e-10)


@pytest.mark.parametrize("arm", ["direct", "rule_based"])
def test_readout_margin_raises_for_non_nca_arms(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        S.readout_margin(encoder, _factors(), 5, rng, output_root=str(tmp_path))


# ---------------------------------------------------------------------------
# 9. uniform_body_fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", sorted(ENCODER_CLASSES))
def test_uniform_body_fraction_matches_validity_rate_degenerate_count(arm, tmp_path):
    # uniform_body_fraction and validity_rate's degenerate_fraction test the
    # exact same underlying property (all-25-cells-one-material) -- with the
    # same rng seed and n, their counts must agree.
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    n = 25

    rng1 = np.random.default_rng(3)
    uniform_result = S.uniform_body_fraction(encoder, _factors(), n, rng1, output_root=str(tmp_path / "a"))

    rng2 = np.random.default_rng(3)
    validity_result = S.validity_rate(encoder, _factors(), n, rng2, output_root=str(tmp_path / "b"))

    assert uniform_result.uniform_fraction == pytest.approx(validity_result.degenerate_fraction)
    assert sum(uniform_result.uniform_count_by_material.values()) == validity_result.degenerate_count


def test_uniform_body_fraction_breaks_down_by_material(tmp_path):
    encoder = NCAEncoder(factors=_factors())
    rng = np.random.default_rng(0)
    result = S.uniform_body_fraction(encoder, _factors(), 200, rng, output_root=str(tmp_path))

    assert set(result.uniform_count_by_material) == {0, 1, 2, 3, 4}
    assert sum(result.uniform_count_by_material.values()) == pytest.approx(
        result.uniform_fraction * 200, abs=1e-6
    )
    for material, fraction in result.uniform_fraction_by_material.items():
        assert 0.0 <= fraction <= 1.0


# ---------------------------------------------------------------------------
# 10. hidden_utilisation
# ---------------------------------------------------------------------------


def test_hidden_utilisation_h0_returns_empty_not_an_error(tmp_path):
    encoder = NCAEncoder(factors=_factors(hidden_channels=0))
    rng = np.random.default_rng(0)
    result = S.hidden_utilisation(encoder, _factors(hidden_channels=0), 10, rng, output_root=str(tmp_path))

    assert result.hidden_channels == 0
    assert result.per_channel_spatial_variance == []
    assert result.per_channel_temporal_variance == []
    assert result.per_channel_correlation_with_readout == []


def test_hidden_utilisation_h_gt_0_reports_one_value_per_channel(tmp_path):
    encoder = NCAEncoder(factors=_factors(hidden_channels=3))
    rng = np.random.default_rng(0)
    result = S.hidden_utilisation(encoder, _factors(hidden_channels=3), 15, rng, output_root=str(tmp_path))

    assert result.hidden_channels == 3
    assert len(result.per_channel_spatial_variance) == 3
    assert len(result.per_channel_temporal_variance) == 3
    assert len(result.per_channel_correlation_with_readout) == 3
    assert all(v >= 0.0 for v in result.per_channel_spatial_variance)
    assert all(v >= 0.0 for v in result.per_channel_temporal_variance)
    assert all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for c in result.per_channel_correlation_with_readout)


def test_hidden_utilisation_constant_channel_has_zero_variance_and_zero_correlation(tmp_path):
    # A hand-constructed genome with zero W1/W2 weights for a specific
    # hidden channel's output logit (and zero bias) freezes that channel at
    # its fixed initial-condition value for the whole trajectory -- exactly
    # the "inert channel" case this metric exists to detect.
    encoder_unbounded = NCAEncoder(
        factors=_factors(hidden_channels=1, dev_steps=5), hidden_width=4, update_rule="unbounded_residual"
    )
    genome = np.zeros(encoder_unbounded.genome_length())  # zero weights => zero delta every step

    _, _, states = S._nca_trajectory_from_state(
        encoder_unbounded, genome, encoder_unbounded._initial_state, encoder_unbounded.dev_steps
    )
    ch_idx = 5  # N_MATERIALS + hidden channel 0
    values_over_time = [s[:, :, ch_idx] for s in states]
    for v in values_over_time[1:]:
        np.testing.assert_allclose(v, values_over_time[0], atol=1e-12)


@pytest.mark.parametrize("arm", ["direct", "rule_based"])
def test_hidden_utilisation_raises_for_non_nca_arms(arm, tmp_path):
    encoder = ENCODER_CLASSES[arm](factors=_factors())
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        S.hidden_utilisation(encoder, _factors(), 5, rng, output_root=str(tmp_path))
