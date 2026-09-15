"""Neural Cellular Automata developmental encoder.

The genome is weights-only: a flat vector of the perception network's
parameters. Development starts from the ExperimentFactors-selected initial
cell state (see evodevo.core.initial_conditions) and runs
factors.dev_steps synchronous, residual update steps; only the network's
weights are evolved.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.initial_conditions import get_initial_condition
from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.base import Encoder

NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class NCAEncoder(Encoder):
    def __init__(
        self,
        factors: ExperimentFactors,
        grid_shape=(5, 5),
        hidden_width: int = 16,
        init_scale: float = 0.5,
        mutation_sigma: float = 0.1,
        mutation_fraction: float = 1.0,
    ):
        if factors.neighbourhood != "von_neumann":
            raise ValueError(
                f"unsupported neighbourhood '{factors.neighbourhood}'; only 'von_neumann' "
                "(4-neighbour, cardinal) is implemented"
            )

        self.grid_shape = tuple(grid_shape)
        self.factors = factors
        self.hidden_channels = factors.hidden_channels
        self.hidden_width = hidden_width
        self.dev_steps = factors.dev_steps
        self.neighbourhood = factors.neighbourhood
        self.init_scale = init_scale
        # factors.mutation_scale multiplies this arm's own perturbation
        # sigma -- it does not change the mutation operator itself (still a
        # per-weight additive Gaussian perturbation).
        self.mutation_sigma = mutation_sigma * factors.mutation_scale
        self.mutation_fraction = mutation_fraction

        self.channels = N_MATERIALS + self.hidden_channels
        self.input_dim = 5 * self.channels  # own state + 4 cardinal neighbours

        self._initial_condition = get_initial_condition(factors.init_condition)
        self._initial_state = self._initial_condition.as_state_tensor(self.grid_shape, self.channels)

        w1_size = self.input_dim * hidden_width
        b1_size = hidden_width
        w2_size = hidden_width * self.channels
        b2_size = self.channels
        self._sizes = (w1_size, b1_size, w2_size, b2_size)
        self._genome_length = w1_size + b1_size + w2_size + b2_size

    @property
    def name(self) -> str:
        return "nca"

    @property
    def genome_dtype(self):
        return np.float64

    def genome_length(self) -> int:
        return self._genome_length

    def serialize_genome(self, genome: np.ndarray) -> list:
        return np.asarray(genome, dtype=float).tolist()

    def deserialize_genome(self, obj: list) -> np.ndarray:
        return np.asarray(obj, dtype=float)

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self.init_scale, size=self._genome_length)

    def _unpack(self, genome: np.ndarray):
        genome = np.asarray(genome, dtype=float)
        w1_size, b1_size, w2_size, b2_size = self._sizes
        i = 0
        w1 = genome[i : i + w1_size].reshape(self.input_dim, self.hidden_width)
        i += w1_size
        b1 = genome[i : i + b1_size]
        i += b1_size
        w2 = genome[i : i + w2_size].reshape(self.hidden_width, self.channels)
        i += w2_size
        b2 = genome[i : i + b2_size]
        return w1, b1, w2, b2

    def _perceive(self, state: np.ndarray) -> np.ndarray:
        """state: (rows, cols, C) -> perception: (rows, cols, 5*C).

        shifted[r, c] = state[r + dr, c + dc] when in bounds, else 0 (zero padding).
        """
        rows, cols, channels = state.shape
        parts = [state]
        for dr, dc in NEIGHBOUR_OFFSETS:
            shifted = np.zeros_like(state)
            dst_r0, dst_r1 = max(0, -dr), rows - max(0, dr)
            dst_c0, dst_c1 = max(0, -dc), cols - max(0, dc)
            src_r0, src_r1 = dst_r0 + dr, dst_r1 + dr
            src_c0, src_c1 = dst_c0 + dc, dst_c1 + dc
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = state[src_r0:src_r1, src_c0:src_c1]
            parts.append(shifted)
        return np.concatenate(parts, axis=-1)

    def _step(self, state: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
        perception = self._perceive(state)  # (rows, cols, 5C)
        flat = perception.reshape(-1, self.input_dim)
        hidden = np.tanh(flat @ w1 + b1)
        delta = hidden @ w2 + b2
        delta = delta.reshape(state.shape)
        return state + delta  # residual, synchronous (whole grid computed from old state)

    def develop(self, genome: np.ndarray, return_trajectory: bool = True):
        w1, b1, w2, b2 = self._unpack(genome)
        state = self._initial_state.copy()

        def readout(s):
            return np.argmax(s[:, :, :N_MATERIALS], axis=-1).astype(int)

        trajectory = [readout(state)] if return_trajectory else None

        for _ in range(self.dev_steps):
            state = self._step(state, w1, b1, w2, b2)
            if return_trajectory:
                trajectory.append(readout(state))

        if return_trajectory:
            return trajectory
        return [readout(state)]

    def decode(self, genome: np.ndarray) -> np.ndarray:
        trajectory = self.develop(genome, return_trajectory=True)
        return trajectory[-1]

    def develop_trajectory(self, genome: np.ndarray):
        """The full t=0..dev_steps sequence of argmax bodies -- see
        develop(return_trajectory=True), which decode() already uses to
        obtain its own final result."""
        return self.develop(genome, return_trajectory=True)

    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        genome = np.asarray(genome, dtype=float).copy()
        n = genome.shape[0]
        n_mutate = int(round(self.mutation_fraction * n))
        if n_mutate > 0:
            indices = rng.choice(n, size=n_mutate, replace=False)
            genome[indices] += rng.normal(scale=self.mutation_sigma, size=n_mutate)
        return genome

    def describe(self) -> dict:
        return {
            "name": self.name,
            "factors": dataclasses.asdict(self.factors),
            "grid_shape": list(self.grid_shape),
            "hidden_channels": self.hidden_channels,
            "hidden_width": self.hidden_width,
            "dev_steps": self.dev_steps,
            "neighbourhood": self.neighbourhood,
            "init_condition": self.factors.init_condition,
            "init_scale": self.init_scale,
            "mutation_sigma": self.mutation_sigma,
            "mutation_scale": self.factors.mutation_scale,
            "mutation_fraction": self.mutation_fraction,
            "channels": self.channels,
            "genome_length": self._genome_length,
        }
