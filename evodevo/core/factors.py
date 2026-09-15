"""Crossed experimental factors: initial condition, developmental depth, and
mutation strength as explicit per-run configuration instead of per-arm
hardcoded defaults (see evodevo/core/initial_conditions.py for the actual
InitialCondition implementations named by `init_condition` below).

dev_steps and init_condition have NO default -- constructing
ExperimentFactors without them is a configuration bug (an unfalsifiable,
invisible per-arm difference), not a value to silently fall back on.
"""
from __future__ import annotations

from dataclasses import dataclass

_VALID_INIT_CONDITIONS = frozenset({"center_seed", "full_nonsymmetric", "full_uniform"})
_VALID_NEIGHBOURHOODS = frozenset({"von_neumann", "moore"})


@dataclass(frozen=True)
class ExperimentFactors:
    dev_steps: int
    init_condition: str
    seed: int
    neighbourhood: str = "von_neumann"
    hidden_channels: int = 3
    mutation_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.init_condition not in _VALID_INIT_CONDITIONS:
            raise ValueError(
                f"init_condition must be one of {sorted(_VALID_INIT_CONDITIONS)}, "
                f"got {self.init_condition!r}"
            )
        if self.neighbourhood not in _VALID_NEIGHBOURHOODS:
            raise ValueError(
                f"neighbourhood must be one of {sorted(_VALID_NEIGHBOURHOODS)}, "
                f"got {self.neighbourhood!r}"
            )
        if self.dev_steps < 0:
            raise ValueError(f"dev_steps must be >= 0, got {self.dev_steps}")
        if self.hidden_channels < 0:
            raise ValueError(f"hidden_channels must be >= 0, got {self.hidden_channels}")


_FACTOR_FIELDS = frozenset({"dev_steps", "init_condition", "neighbourhood", "hidden_channels", "mutation_scale"})

# Historical default used by call sites that pre-date ExperimentFactors and
# pass flat per-arm hyperparameter overrides (e.g. {"dev_steps": 1}) rather
# than a fully-specified factors dict -- matches the NCA encoder's old
# hardcoded (dev_steps=10, fixed non-symmetric initial state) default so
# those callers' existing behaviour is unchanged unless they override it.
DEFAULT_FACTOR_OVERRIDES = {"dev_steps": 10, "init_condition": "full_nonsymmetric"}


def extract_factors(kwargs: dict, seed: int, defaults: dict = None) -> tuple:
    """Splits `kwargs` into (ExperimentFactors, remaining_kwargs): any key
    named after an ExperimentFactors field is popped out and used to build
    the factors object (layered over `defaults`, or DEFAULT_FACTOR_OVERRIDES
    if not given); everything else is passed through untouched for the
    encoder's own constructor kwargs."""
    kwargs = dict(kwargs or {})
    overrides = dict(defaults if defaults is not None else DEFAULT_FACTOR_OVERRIDES)
    for key in list(kwargs):
        if key in _FACTOR_FIELDS:
            overrides[key] = kwargs.pop(key)
    return ExperimentFactors(seed=seed, **overrides), kwargs
