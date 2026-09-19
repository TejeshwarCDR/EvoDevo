"""Direct (non-developmental) baseline encoding: 25 categorical values, one per voxel."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.base import Encoder


class DirectEncoder(Encoder):
    def __init__(
        self,
        factors: ExperimentFactors,
        grid_shape=(5, 5),
        mutation_rate: float = 0.1,
        empty_prob: float = 0.6,
    ):
        self.grid_shape = tuple(grid_shape)
        self.factors = factors
        # DirectEncoder is not developmental: factors.dev_steps,
        # init_condition, neighbourhood and hidden_channels have no effect
        # here (explicit no-op). Only mutation_scale applies, since mutation
        # strength is meaningful for every arm.
        _scaled_mutation_rate = mutation_rate * factors.mutation_scale
        if _scaled_mutation_rate > 1.0:
            warnings.warn(
                f"DirectEncoder: mutation_rate * factors.mutation_scale = "
                f"{mutation_rate} * {factors.mutation_scale} = {_scaled_mutation_rate} exceeds 1.0 "
                "(a per-cell mutation probability cannot exceed 100%) -- clipped to 1.0. This is a "
                "real clip and must not pass silently; if unintended, lower mutation_rate or "
                "mutation_scale.",
                stacklevel=2,
            )
        self.mutation_rate = min(1.0, _scaled_mutation_rate)
        self.empty_prob = empty_prob
        remaining = (1.0 - empty_prob) / (N_MATERIALS - 1)
        self._material_probs = np.array([empty_prob] + [remaining] * (N_MATERIALS - 1))

    @property
    def name(self) -> str:
        return "direct"

    @property
    def genome_dtype(self):
        return np.int64

    def genome_length(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

    def serialize_genome(self, genome: np.ndarray) -> list:
        return np.asarray(genome, dtype=int).tolist()

    def deserialize_genome(self, obj: list) -> np.ndarray:
        return np.asarray(obj, dtype=int)

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(N_MATERIALS, size=self.genome_length(), p=self._material_probs).astype(int)

    def decode(self, genome: np.ndarray) -> np.ndarray:
        return np.asarray(genome, dtype=int).reshape(self.grid_shape)

    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        genome = np.asarray(genome, dtype=int).copy()
        mask = rng.random(genome.shape) < self.mutation_rate
        n_mutate = int(mask.sum())
        if n_mutate > 0:
            genome[mask] = rng.choice(N_MATERIALS, size=n_mutate, p=self._material_probs)
        return genome

    def describe(self) -> dict:
        return {
            "name": self.name,
            "factors": dataclasses.asdict(self.factors),
            "grid_shape": list(self.grid_shape),
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.factors.mutation_scale,
            "empty_prob": self.empty_prob,
        }
