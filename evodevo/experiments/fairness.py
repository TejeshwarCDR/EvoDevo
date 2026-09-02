"""Explicit enforcement of the fairness protocol (roadmap Part 4).

assert_comparable() must be called at the start of any run that puts two
arms (encodings) side by side, so that a difference in results can only be
attributed to the encoding, never to an accidental asymmetry in the shared
pipeline. It permits differences ONLY in `encoder`, `seed`, and
encoder-specific hyperparameters (which live outside ExperimentConfig
entirely -- they belong to the Encoder instance, not the run config).
"""
from __future__ import annotations

from evodevo.core.config import ExperimentConfig

# Fields that must be identical across both arms of a comparison. Everything
# NOT in this list (currently just `encoder` and `seed`) is allowed to differ.
_REQUIRED_EQUAL_FIELDS = (
    "task",
    "eval_budget",
    "ppo_timesteps",
    "ppo_eval_episodes",
    "grid_shape",
    "population_size",
    "offspring_size",
    "sentinel_fitness",
    "cache_enabled",
)


class FairnessViolation(AssertionError):
    pass


def assert_comparable(config_a: ExperimentConfig, config_b: ExperimentConfig) -> None:
    """Raise FairnessViolation if config_a and config_b are not eligible to be
    compared under the project's fairness protocol."""
    violations = []

    for field in _REQUIRED_EQUAL_FIELDS:
        value_a = getattr(config_a, field)
        value_b = getattr(config_b, field)
        if field == "grid_shape":
            value_a, value_b = tuple(value_a), tuple(value_b)
        if value_a != value_b:
            violations.append(f"{field}: {value_a!r} != {value_b!r}")

    if violations:
        raise FairnessViolation(
            "configs are not comparable under the fairness protocol; only `encoder`, `seed`, "
            "and encoder-specific hyperparameters may differ. Violations:\n  "
            + "\n  ".join(violations)
        )
