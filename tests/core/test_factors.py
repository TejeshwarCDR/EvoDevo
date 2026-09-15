import pytest

from evodevo.core.factors import ExperimentFactors


def test_missing_dev_steps_raises():
    with pytest.raises(TypeError):
        ExperimentFactors(init_condition="center_seed", seed=0)


def test_missing_init_condition_raises():
    with pytest.raises(TypeError):
        ExperimentFactors(dev_steps=5, seed=0)


def test_missing_seed_raises():
    with pytest.raises(TypeError):
        ExperimentFactors(dev_steps=5, init_condition="center_seed")


def test_invalid_init_condition_raises():
    with pytest.raises(ValueError):
        ExperimentFactors(dev_steps=5, init_condition="bogus", seed=0)


def test_invalid_neighbourhood_raises():
    with pytest.raises(ValueError):
        ExperimentFactors(dev_steps=5, init_condition="center_seed", seed=0, neighbourhood="bogus")


def test_negative_dev_steps_raises():
    with pytest.raises(ValueError):
        ExperimentFactors(dev_steps=-1, init_condition="center_seed", seed=0)


def test_defaults_apply_when_explicit_fields_are_given():
    factors = ExperimentFactors(dev_steps=5, init_condition="center_seed", seed=0)
    assert factors.mutation_scale == 1.0
    assert factors.neighbourhood == "von_neumann"
    assert factors.hidden_channels == 3


def test_is_frozen():
    factors = ExperimentFactors(dev_steps=5, init_condition="center_seed", seed=0)
    with pytest.raises(Exception):
        factors.dev_steps = 10
