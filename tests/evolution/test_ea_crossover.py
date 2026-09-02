import numpy as np
import pytest

from evodevo.core.config import ExperimentConfig
from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.rule_based.encoder import RuleBasedEncoder
from evodevo.evaluation.ppo_runner import evaluate_body_surrogate
from evodevo.evolution.ea import run_ea


def _make_config(**overrides):
    defaults = dict(
        seed=0,
        encoder="direct",
        task="dummy-task",
        population_size=4,
        offspring_size=4,
        eval_budget=16,
        ppo_timesteps=1,
        ppo_eval_episodes=1,
        output_dir="unused",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_use_crossover_defaults_to_false():
    config = _make_config()
    assert config.use_crossover is False


def test_use_crossover_true_with_non_supporting_encoder_raises_at_startup():
    encoder = DirectEncoder()
    assert encoder.supports_crossover() is False
    config = _make_config(use_crossover=True)
    with pytest.raises(ValueError, match="does not support crossover"):
        run_ea(encoder, evaluate_body_surrogate, config)


def test_use_crossover_true_with_rule_based_runs_to_exact_budget():
    encoder = RuleBasedEncoder()
    config = _make_config(encoder="rule_based", use_crossover=True, eval_budget=14, population_size=4, offspring_size=4)
    result = run_ea(encoder, evaluate_body_surrogate, config)
    assert result.evaluations_used == 14
    assert len(result.records) == 14


def test_use_crossover_false_is_byte_identical_to_omitting_the_flag():
    # Explicitly confirms the default path (use_crossover=False) is
    # unaffected by the flag's introduction -- a regression guard
    # complementing test_direct.py's/test_nca.py's pinned-behaviour tests.
    encoder_a = DirectEncoder()
    encoder_b = DirectEncoder()
    config_explicit = _make_config(use_crossover=False, seed=7)
    config_default = _make_config(seed=7)

    result_a = run_ea(encoder_a, evaluate_body_surrogate, config_explicit)
    result_b = run_ea(encoder_b, evaluate_body_surrogate, config_default)

    fitness_a = [r["fitness"] for r in result_a.records]
    fitness_b = [r["fitness"] for r in result_b.records]
    assert fitness_a == fitness_b
