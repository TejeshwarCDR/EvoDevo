"""Reproducibility spine: the single frozen config object every run is defined by."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

# The set of ExperimentConfig fields that MUST be identical across every
# encoding arm for a fair comparison. This is the single source of truth for
# that requirement: configs/shared/base.yaml holds exactly these keys, the
# base/override loader below refuses a per-method config that redefines any
# of them, and evodevo/experiments/fairness.py's assert_comparable() builds
# its required-equal field list directly from this set (as a second,
# run-time line of defence) rather than duplicating it.
SHARED_BASE_KEYS = frozenset(
    {
        "population_size",
        "offspring_size",
        "eval_budget",
        "ppo_timesteps",
        "ppo_eval_episodes",
        "grid_shape",
        "sentinel_fitness",
        "cache_enabled",
        "min_voxels",
        "min_actuators",
        "use_crossover",
    }
)


class ConfigOverrideError(ValueError):
    """Raised when a per-method config redefines a shared-base key."""


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    encoder: str
    task: str
    population_size: int
    offspring_size: int
    eval_budget: int
    ppo_timesteps: int
    ppo_eval_episodes: int
    output_dir: str
    grid_shape: tuple = (5, 5)
    sentinel_fitness: float = -10.0
    cache_enabled: bool = False
    use_crossover: bool = False
    # min_voxels/min_actuators mirror validate_body()'s own defaults (see
    # evodevo/morphology/validate.py). They are recorded here so the
    # base/override mechanism and fairness.py can treat them as a
    # fairness-critical, must-match-across-arms setting; wiring them through
    # to the actual validate_body() call inside evaluation/ppo_runner.py is
    # a follow-up (not part of this config/tooling-only change).
    min_voxels: int = 3
    min_actuators: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load a single, fully self-contained YAML file (e.g. one written
        by to_yaml()). For a per-method config that inherits shared fields
        from configs/shared/base.yaml, use from_method_yaml() instead."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if "grid_shape" in data:
            data["grid_shape"] = tuple(data["grid_shape"])
        return cls(**data)

    @classmethod
    def from_base_and_override(cls, base_path: str, override_path: str) -> "ExperimentConfig":
        """Merge a shared base YAML (configs/shared/base.yaml -- every field
        that must be identical across arms) with a per-method override YAML
        (encoder, task, seed, output_dir, and any encoder-specific keys).

        Raises ConfigOverrideError if the override file redefines any key
        already present in the base file -- this makes the fairness
        protocol structurally impossible to violate by editing a per-method
        config, rather than merely checked after configs are loaded.
        """
        with open(base_path, "r") as f:
            base_data = yaml.safe_load(f) or {}
        with open(override_path, "r") as f:
            override_data = yaml.safe_load(f) or {}

        conflicting = sorted(k for k in override_data if k in base_data)
        if conflicting:
            raise ConfigOverrideError(
                f"{override_path} attempts to override shared-base key(s) {conflicting}, "
                f"which must be identical across every encoding arm for a fair comparison "
                f"(defined in {base_path}). Remove {conflicting} from {override_path} -- if "
                "this value genuinely needs to differ, that is a fairness-protocol decision "
                "to make explicitly (e.g. a labelled ablation arm), not a config edit."
            )

        merged = {**base_data, **override_data}
        if "grid_shape" in merged:
            merged["grid_shape"] = tuple(merged["grid_shape"])
        return cls(**merged)

    @classmethod
    def from_method_yaml(cls, path: str, base_path: str = None) -> "ExperimentConfig":
        """Load a per-method config at .../configs/<method>/<file>.yaml,
        inheriting shared fields from the sibling .../configs/shared/base.yaml
        (or an explicit base_path, if given).

        If base_path is not given AND no sibling shared/base.yaml is found
        (e.g. a standalone, fully self-contained config -- such as those
        written ad hoc by tests -- that isn't part of the
        configs/<method>/ + configs/shared/ layout), falls back to plain
        from_yaml(path), treating the file as already self-contained. An
        explicitly-passed base_path that doesn't exist is NOT silently
        ignored -- that is a caller error, not a missing-inheritance case.
        """
        explicit_base = base_path is not None
        if base_path is None:
            configs_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            base_path = os.path.join(configs_dir, "shared", "base.yaml")

        if not explicit_base and not os.path.exists(base_path):
            return cls.from_yaml(path)

        return cls.from_base_and_override(base_path, path)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["grid_shape"] = list(d["grid_shape"])
        return d

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
