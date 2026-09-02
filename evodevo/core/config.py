"""Reproducibility spine: the single frozen config object every run is defined by."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml


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

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if "grid_shape" in data:
            data["grid_shape"] = tuple(data["grid_shape"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["grid_shape"] = list(d["grid_shape"])
        return d

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
