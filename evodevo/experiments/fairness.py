"""Explicit enforcement of the fairness protocol (roadmap Part 4).

assert_comparable() must be called at the start of any run that puts two
arms (encodings) side by side, so that a difference in results can only be
attributed to the encoding, never to an accidental asymmetry in the shared
pipeline. It permits differences ONLY in `encoder`, `seed`, and
encoder-specific hyperparameters (which live outside ExperimentConfig
entirely -- they belong to the Encoder instance, not the run config).
"""
from __future__ import annotations

from evodevo.core.config import SHARED_BASE_KEYS, ExperimentConfig

# Fields that must be identical across both arms of a comparison. Everything
# NOT in this list (currently just `encoder` and `seed`) is allowed to differ.
#
# Built from evodevo.core.config.SHARED_BASE_KEYS (the same single source of
# truth configs/shared/base.yaml and ExperimentConfig.from_base_and_override
# use) plus `task`, which is not itself a shared-base key (different config
# files legitimately target different tasks) but must still match for any
# *specific* pairwise comparison to be a valid head-to-head. This makes
# assert_comparable() a second, run-time line of defence rather than a
# separately maintained list that could drift from the load-time mechanism.
_REQUIRED_EQUAL_FIELDS = ("task",) + tuple(sorted(SHARED_BASE_KEYS))


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


def assert_identical_policy(policies: dict) -> None:
    """Raise FairnessViolation unless every arm's resolved invalid-genome/
    cache-charging/budget-counter policy (evodevo.evolution.ea.
    resolved_policy_for(), one dict per arm) is byte-identical.

    `policies` is {arm_label: resolved_policy_dict}. This is the run-start
    assertion the budget-accounting prompt asks for, made explicit and
    testable at the benchmark level (where every arm's config is already
    resolved and comparable) rather than only implicitly true because
    run_ea() happens to be the same function for every arm."""
    if len(policies) < 2:
        return
    items = list(policies.items())
    reference_label, reference_policy = items[0]
    violations = []
    for label, policy in items[1:]:
        if policy != reference_policy:
            violations.append(f"{label} != {reference_label}: {policy!r} vs {reference_policy!r}")
    if violations:
        raise FairnessViolation(
            "invalid-genome/cache-charging/budget-counter policy is not identical across arms "
            "(evodevo.evolution.ea.resolved_policy_for()). Violations:\n  " + "\n  ".join(violations)
        )
