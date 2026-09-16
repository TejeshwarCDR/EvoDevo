"""Correctness tests for evodevo/analysis/structural_report.py's
_build_encoder -- specifically the regression guarantee that direct/
rule_based are completely unaffected by NCA's resolved-architecture
override. Small N; the full N=2000 acceptance run is the committed
results/structural/STRUCTURAL.md."""
import pytest

from evodevo.analysis import structural_report as SR
from evodevo.core.factors import ExperimentFactors
from evodevo.encoders import get_encoder


def _factors():
    return ExperimentFactors(dev_steps=10, init_condition="full_nonsymmetric", seed=0)


def test_build_encoder_nca_uses_resolved_architecture():
    encoder = SR._build_encoder("nca", _factors())
    assert encoder.update_rule == SR.PRIMARY.update_rule == "bounded_leaky"
    assert encoder.decay == pytest.approx(SR.PRIMARY.decay)
    assert encoder.state_bound == pytest.approx(SR.PRIMARY.state_bound)


@pytest.mark.parametrize("arm", ["direct", "rule_based"])
def test_build_encoder_other_arms_unaffected(arm):
    encoder = SR._build_encoder(arm, _factors())
    baseline = get_encoder(arm, factors=_factors())
    assert type(encoder) is type(baseline)
    assert encoder.mutation_rate == pytest.approx(baseline.mutation_rate)
    assert encoder.grid_shape == baseline.grid_shape
