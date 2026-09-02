"""Direct (non-developmental) baseline encoding: 25 categorical values, one per voxel."""
from __future__ import annotations

import numpy as np

from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.base import Encoder


class DirectEncoder(Encoder):
    def __init__(self, grid_shape=(5, 5), mutation_rate: float = 0.1, empty_prob: float = 0.6):
        self.grid_shape = tuple(grid_shape)
        self.mutation_rate = mutation_rate
        self.empty_prob = empty_prob
        remaining = (1.0 - empty_prob) / (N_MATERIALS - 1)
        self._material_probs = np.array([empty_prob] + [remaining] * (N_MATERIALS - 1))

    @property
    def name(self) -> str:
        return "direct"

    def genome_length(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]

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
            "grid_shape": list(self.grid_shape),
            "mutation_rate": self.mutation_rate,
            "empty_prob": self.empty_prob,
        }
