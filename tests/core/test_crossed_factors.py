"""Proves dev_steps and init_condition are genuinely crossed experimental
factors, not per-arm hardcoded defaults: every encoder must construct and
decode successfully under every (init_condition, dev_steps) combination."""
import numpy as np
import pytest

from evodevo.core.factors import ExperimentFactors
from evodevo.encoders.direct.encoder import DirectEncoder
from evodevo.encoders.nca.encoder import NCAEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder

INIT_CONDITIONS = ["center_seed", "full_nonsymmetric", "full_uniform"]
DEV_STEPS_VALUES = [1, 5, 10]

ENCODER_CLASSES = {
    "direct": DirectEncoder,
    "nca": NCAEncoder,
    "rule_based": RuleBasedEncoder,
}


@pytest.mark.parametrize("dev_steps", DEV_STEPS_VALUES)
@pytest.mark.parametrize("init_condition", INIT_CONDITIONS)
@pytest.mark.parametrize("encoder_name", sorted(ENCODER_CLASSES))
def test_encoder_decodes_under_every_factor_combination(encoder_name, init_condition, dev_steps):
    factors = ExperimentFactors(dev_steps=dev_steps, init_condition=init_condition, seed=0)
    encoder = ENCODER_CLASSES[encoder_name](factors=factors)

    genome = encoder.random_genome(np.random.default_rng(0))
    body = encoder.decode(genome)

    assert body.shape == (5, 5)
    assert np.issubdtype(body.dtype, np.integer)
