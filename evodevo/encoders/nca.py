"""Neural Cellular Automata developmental encoder.

The genome is weights-only: a flat vector of the perception network's
parameters. Development always starts from the same fixed initial cell
state (see _INIT_STATE_SEED below) and runs a fixed number of synchronous,
residual update steps; only the network's weights are evolved.
"""
from __future__ import annotations

import numpy as np

from evodevo.encoders.base import Encoder

N_MATERIALS = 5  # EMPTY, RIGID, SOFT, H_ACT, V_ACT
NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

# Seeds the fixed, non-evolved initial cell state s(0). It is generated once,
# deterministically, from this seed and is identical for every individual
# and every generation -- development only ever differs through the genome
# (network weights), never through the starting condition. A perfectly
# symmetric initial state (e.g. all zeros, or a mirror-symmetric pattern)
# would keep every cell's perception identical to its mirror cell forever
# under a deterministic, synchronous update, so left-right/up-down
# differentiation could never emerge; drawing small non-zero, non-symmetric
# noise avoids that failure mode.
_INIT_STATE_SEED = 20260902
_INIT_STATE_SCALE = 0.1


def _make_initial_state(grid_shape: tuple, channels: int) -> np.ndarray:
    rng = np.random.default_rng(_INIT_STATE_SEED)
    rows, cols = grid_shape
    return rng.normal(scale=_INIT_STATE_SCALE, size=(rows, cols, channels))


class NCAEncoder(Encoder):
    def __init__(
        self,
        grid_shape=(5, 5),
        hidden_channels: int = 3,
        hidden_width: int = 16,
        dev_steps: int = 10,
        neighbourhood: str = "von_neumann",
        init_scale: float = 0.5,
        mutation_sigma: float = 0.1,
        mutation_fraction: float = 1.0,
    ):
        if neighbourhood != "von_neumann":
            raise ValueError(
                f"unsupported neighbourhood '{neighbourhood}'; only 'von_neumann' "
                "(4-neighbour, cardinal) is implemented"
            )

        self.grid_shape = tuple(grid_shape)
        self.hidden_channels = hidden_channels
        self.hidden_width = hidden_width
        self.dev_steps = dev_steps
        self.neighbourhood = neighbourhood
        self.init_scale = init_scale
        self.mutation_sigma = mutation_sigma
        self.mutation_fraction = mutation_fraction

        self.channels = N_MATERIALS + hidden_channels
        self.input_dim = 5 * self.channels  # own state + 4 cardinal neighbours

        self._initial_state = _make_initial_state(self.grid_shape, self.channels)

        w1_size = self.input_dim * hidden_width
        b1_size = hidden_width
        w2_size = hidden_width * self.channels
        b2_size = self.channels
        self._sizes = (w1_size, b1_size, w2_size, b2_size)
        self._genome_length = w1_size + b1_size + w2_size + b2_size

    @property
    def name(self) -> str:
        return "nca"

    def genome_length(self) -> int:
        return self._genome_length

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
            "grid_shape": list(self.grid_shape),
            "hidden_channels": self.hidden_channels,
            "hidden_width": self.hidden_width,
            "dev_steps": self.dev_steps,
            "neighbourhood": self.neighbourhood,
            "init_scale": self.init_scale,
            "mutation_sigma": self.mutation_sigma,
            "mutation_fraction": self.mutation_fraction,
            "channels": self.channels,
            "genome_length": self._genome_length,
        }
