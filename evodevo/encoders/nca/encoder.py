"""Neural Cellular Automata developmental encoder.

The genome is a flat standard-normal z-space vector z ~ N(0, 1)^792 (weights
only), NOT the perception network's weights directly: _unpack applies
per-layer variance scaling (layer 1 fan-in 5C, layer 2 fan-in h) governed by
`init_gain` and `bias_scale`, so a single mutation sigma in z-space has a
consistent relative effect on both layers regardless of their different
fan-ins -- see results/nca/INIT_STUDY.md. Development starts from the
ExperimentFactors-selected initial cell state (see
evodevo.core.initial_conditions) and runs factors.dev_steps synchronous
update steps; only the network's weights are evolved.

Two update rules are available via `update_rule`:
  bounded_leaky (default): s_{t+1} = (1-decay)*s_t + decay*state_bound*tanh(f(p_t)).
      Provably bounded (Proposition 1: |s_t| stays within
      max(|s_0|_inf, state_bound) for all t) -- see results/depth/DEPTH_STUDY.md
      "Diagnosis of non-settling genomes" for why the previous unbounded rule
      caused most genomes to diverge rather than converge.
  unbounded_residual: s_{t+1} = s_t + f(p_t). The original, unbounded rule,
      kept for reproducing pre-fix behaviour and before/after comparisons.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from evodevo.core.factors import ExperimentFactors
from evodevo.core.initial_conditions import get_initial_condition
from evodevo.core.materials import N_MATERIALS
from evodevo.encoders.base import Encoder

NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


VALID_UPDATE_RULES = ("bounded_leaky", "unbounded_residual")
VALID_BOUNDARIES = ("zero_pad", "replicate", "toroidal")
VALID_READOUT_MODES = ("fixed_t", "mode_of_window")

# The mode-of-window readout implementation now lives in
# evodevo/analysis/windowed_readout.py -- a SHARED utility used by both NCA
# and rule_based's encoders (see docs/decisions.md's "unified T_primary /
# windowed readout for both arms" row), so there is exactly one definition
# of "mode of window" in the codebase. `_mode_of_window` is kept as a
# module-level alias here ONLY for backward compatibility with existing
# imports (`from evodevo.encoders.nca.encoder import _mode_of_window`) --
# new code should import `mode_of_window` from `windowed_readout` directly.
from evodevo.analysis.windowed_readout import mode_of_window as _mode_of_window  # noqa: E402


class NCAEncoder(Encoder):
    def __init__(
        self,
        factors: ExperimentFactors,
        grid_shape=(5, 5),
        hidden_width: int = 16,
        init_gain: float = 1.0,
        bias_scale: float = 0.1,
        mutation_sigma: float = 0.1,
        mutation_fraction: float = 1.0,
        update_rule: str = "bounded_leaky",
        state_bound: float = 5.0,
        decay: float = 0.2,
        boundary: str = "zero_pad",
        update_prob: float = 1.0,
        spectral_normalise: bool = False,
        readout_mode: str = "mode_of_window",
        readout_window: int = 50,
        init_scale: float = None,
    ):
        if init_scale is not None:
            warnings.warn(
                "NCAEncoder's `init_scale` is deprecated and maps directly onto "
                "`init_gain` (init_gain=init_scale); the genome is now standard-normal "
                "z-space with layer-wise variance scaling applied during unpacking, so "
                "init_scale no longer scales the raw genome the way it used to. Pass "
                "`init_gain` explicitly instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            init_gain = init_scale

        if factors.neighbourhood != "von_neumann":
            raise ValueError(
                f"unsupported neighbourhood '{factors.neighbourhood}'; only 'von_neumann' "
                "(4-neighbour, cardinal) is implemented"
            )
        if update_rule not in VALID_UPDATE_RULES:
            raise ValueError(f"unsupported update_rule '{update_rule}'; must be one of {VALID_UPDATE_RULES}")
        if boundary not in VALID_BOUNDARIES:
            raise ValueError(f"unsupported boundary '{boundary}'; must be one of {VALID_BOUNDARIES}")
        if readout_mode not in VALID_READOUT_MODES:
            raise ValueError(f"unsupported readout_mode '{readout_mode}'; must be one of {VALID_READOUT_MODES}")
        if readout_window < 1:
            raise ValueError(f"readout_window must be >= 1, got {readout_window}")

        self.grid_shape = tuple(grid_shape)
        self.factors = factors
        self.hidden_channels = factors.hidden_channels
        self.hidden_width = hidden_width
        self.dev_steps = factors.dev_steps
        self.neighbourhood = factors.neighbourhood
        self.init_gain = init_gain
        self.bias_scale = bias_scale
        # factors.mutation_scale multiplies this arm's own perturbation
        # sigma -- it does not change the mutation operator itself (still a
        # per-weight additive Gaussian perturbation).
        self.mutation_sigma = mutation_sigma * factors.mutation_scale
        self.mutation_fraction = mutation_fraction

        self.update_rule = update_rule
        self.state_bound = state_bound
        self.decay = decay
        self.boundary = boundary
        self.update_prob = update_prob
        self.spectral_normalise = spectral_normalise
        self.readout_mode = readout_mode
        self.readout_window = readout_window

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
        # Standard normal z-space, independent of init_gain: the gain (and its
        # per-layer fan-in scaling) lives entirely in _unpack, which is what
        # makes an additive Gaussian mutation in z-space have a consistent
        # relative effect across layers of different fan-in.
        return rng.normal(scale=1.0, size=self._genome_length)

    def _unpack(self, genome: np.ndarray):
        genome = np.asarray(genome, dtype=float)
        w1_size, b1_size, w2_size, b2_size = self._sizes
        i = 0
        z_w1 = genome[i : i + w1_size].reshape(self.input_dim, self.hidden_width)
        i += w1_size
        z_b1 = genome[i : i + b1_size]
        i += b1_size
        z_w2 = genome[i : i + w2_size].reshape(self.hidden_width, self.channels)
        i += w2_size
        z_b2 = genome[i : i + b2_size]

        fan_in1 = self.input_dim  # 5C
        fan_in2 = self.hidden_width  # h
        bias_std = np.sqrt(self.init_gain) * self.bias_scale

        w1 = z_w1 * np.sqrt(self.init_gain / fan_in1)
        b1 = z_b1 * bias_std
        w2 = z_w2 * np.sqrt(self.init_gain / fan_in2)
        b2 = z_b2 * bias_std

        if self.spectral_normalise:
            w1, w2 = self._apply_spectral_normalisation(w1, w2)

        return w1, b1, w2, b2

    def _pack(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Inverse of the (spectral_normalise=False) branch of _unpack: maps
        weight-space (w1, b1, w2, b2) back to a flat z-space genome. Used for
        round-trip testing of the layer-wise variance scaling."""
        fan_in1 = self.input_dim
        fan_in2 = self.hidden_width
        bias_std = np.sqrt(self.init_gain) * self.bias_scale

        z_w1 = w1 / np.sqrt(self.init_gain / fan_in1)
        z_b1 = b1 / bias_std
        z_w2 = w2 / np.sqrt(self.init_gain / fan_in2)
        z_b2 = b2 / bias_std

        return np.concatenate([z_w1.flatten(), z_b1, z_w2.flatten(), z_b2])

    def _apply_spectral_normalisation(self, w1: np.ndarray, w2: np.ndarray):
        """Rescale w1, w2 so state_bound * sqrt(5) * sigma_max(w1) * sigma_max(w2) <= 0.9.

        This is a sufficient (Lipschitz) contraction certificate for the
        bounded_leaky update: tanh is 1-Lipschitz, so the network's output is
        Lipschitz in its input with constant sigma_max(w1) * sigma_max(w2);
        the factor sqrt(5) upper-bounds the perception gather's operator norm
        (5 concatenated blocks, each a coordinate projection).
        """
        target = 0.9
        sigma1 = np.linalg.svd(w1, compute_uv=False)[0] if w1.size else 0.0
        sigma2 = np.linalg.svd(w2, compute_uv=False)[0] if w2.size else 0.0
        current = self.state_bound * np.sqrt(5.0) * sigma1 * sigma2
        if current <= target or current == 0.0:
            return w1, w2
        scale = np.sqrt(target / current)
        return w1 * scale, w2 * scale

    def _perceive(self, state: np.ndarray) -> np.ndarray:
        """state: (rows, cols, C) -> perception: (rows, cols, 5*C).

        Own state is always the first block. The four neighbour blocks are
        gathered according to self.boundary:
          zero_pad:  out-of-grid neighbours read as 0 (original behaviour).
          replicate: out-of-grid indices clamp to the nearest edge cell.
          toroidal:  out-of-grid indices wrap around the grid.
        """
        rows, cols, channels = state.shape
        parts = [state]
        for dr, dc in NEIGHBOUR_OFFSETS:
            if self.boundary == "zero_pad":
                shifted = np.zeros_like(state)
                dst_r0, dst_r1 = max(0, -dr), rows - max(0, dr)
                dst_c0, dst_c1 = max(0, -dc), cols - max(0, dc)
                src_r0, src_r1 = dst_r0 + dr, dst_r1 + dr
                src_c0, src_c1 = dst_c0 + dc, dst_c1 + dc
                shifted[dst_r0:dst_r1, dst_c0:dst_c1] = state[src_r0:src_r1, src_c0:src_c1]
            else:
                row_idx = np.arange(rows) + dr
                col_idx = np.arange(cols) + dc
                if self.boundary == "replicate":
                    row_idx = np.clip(row_idx, 0, rows - 1)
                    col_idx = np.clip(col_idx, 0, cols - 1)
                elif self.boundary == "toroidal":
                    row_idx = row_idx % rows
                    col_idx = col_idx % cols
                shifted = state[np.ix_(row_idx, col_idx)]
            parts.append(shifted)
        return np.concatenate(parts, axis=-1)

    def _step(self, state: np.ndarray, w1, b1, w2, b2, rng: np.random.Generator = None) -> np.ndarray:
        perception = self._perceive(state)  # (rows, cols, 5C)
        flat = perception.reshape(-1, self.input_dim)
        hidden = np.tanh(flat @ w1 + b1)
        raw = hidden @ w2 + b2  # (rows*cols, C)
        raw = raw.reshape(state.shape)

        if self.update_rule == "unbounded_residual":
            new_state = state + raw
        elif self.update_rule == "bounded_leaky":
            new_state = (1.0 - self.decay) * state + self.decay * self.state_bound * np.tanh(raw)
        else:  # pragma: no cover -- validated in __init__
            raise ValueError(f"unsupported update_rule '{self.update_rule}'")

        if self.update_prob < 1.0:
            if rng is None:
                raise ValueError("update_prob < 1.0 requires an explicit rng")
            mask = rng.random(state.shape[:2]) < self.update_prob
            new_state = np.where(mask[:, :, None], new_state, state)

        return new_state

    def develop(self, genome: np.ndarray, return_trajectory: bool = True, rng: np.random.Generator = None):
        if self.update_prob < 1.0 and rng is None:
            raise ValueError("update_prob < 1.0 makes develop()/decode() RNG-dependent; pass an explicit rng")

        w1, b1, w2, b2 = self._unpack(genome)
        state = self._initial_state.copy()

        def readout(s):
            return np.argmax(s[:, :, :N_MATERIALS], axis=-1).astype(int)

        trajectory = [readout(state)] if return_trajectory else None

        for _ in range(self.dev_steps):
            state = self._step(state, w1, b1, w2, b2, rng=rng)
            if return_trajectory:
                trajectory.append(readout(state))

        if return_trajectory:
            return trajectory
        return [readout(state)]

    def decode(self, genome: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Decode a genome into its final body grid.

        Two readout policies (`self.readout_mode`), both selectable:
          "fixed_t" (default): body = argmax(s_T) at the single final step T
              = self.dev_steps. Simple and correct for genomes whose argmax
              readout has genuinely reached a fixed point by T, but for a
              genome that is still changing at T (see
              results/nca/HORIZON_EXTENSION.md, results/nca/
              DYNAMICS_CLASSIFICATION.md), the returned body is an arbitrary
              snapshot that depends on exactly which phase of an ongoing
              (possibly cyclic, possibly slowly-drifting) process T happens
              to land on -- see results/nca/READOUT_POLICY_SELECTION.md's
              T-shift sensitivity measurement for how large this effect is
              in practice.
          "mode_of_window": body = the most frequent WHOLE-GRID argmax
              configuration among the last `self.readout_window` (=W) steps
              (t = T-W+1 .. T; fewer if T < W). Ties between equally-frequent
              configurations are broken deterministically by the
              lexicographically smallest flattened (row-major) material-ID
              sequence (i.e. "lowest material ID first" at the first cell
              where the tied configurations differ) -- deterministic, no RNG
              involved. For a genome whose readout has already reached a
              literal fixed point, every snapshot in the window is identical
              and mode-of-window returns exactly the same body as fixed_t
              (verified in tests/encoders/test_nca_readout_policy.py and
              results/nca/READOUT_POLICY_SELECTION.md's equivalence check) --
              it is a strict generalisation, not a different answer for
              genomes that were already fine.

              CAVEAT (do not use at T << W): when `dev_steps` is much
              smaller than W, the window is dominated by the EARLY,
              near-initial-condition steps rather than steady-state
              behaviour -- at shallow T this can make mode_of_window return
              a near-empty/degenerate body even for genomes whose eventual
              steady state is not degenerate at all (observed directly:
              re-running the depth-sweep diagnostics' shallow-T probes
              under mode_of_window collapsed valid_rate from ~0.63 to
              ~0.01 at T=10, purely a window/T mismatch artifact, not a
              property of the readout at its intended scale). Only use
              mode_of_window where T is comparable to or larger than W --
              this holds by construction at PRIMARY's own resolved
              (T_primary, W) = (40, 50) (see below), but callers building
              their own encoder must not default W=50 with a much smaller
              custom `dev_steps` without checking this. `resolved_config.py`
              ::PRIMARY carries the resolved (readout_mode="mode_of_window",
              readout_window=50) explicitly for whichever callers construct
              "the primary NCA arm" -- NCAEncoder's own class default
              remains "fixed_t"/10 precisely so that depth-sweep-style
              diagnostics probing many different T values (shallow ones
              included) keep their existing, correct behaviour rather than
              silently picking up a readout mismatched to their own T.

        When self.update_prob == 1.0 (the default) decode is a pure,
        deterministic function of the genome. When self.update_prob < 1.0,
        cell updates are masked by independent Bernoulli draws each step
        (canonical stochastic-NCA update), so decode is NOT a pure function
        of the genome alone: it requires an explicit rng/seed and repeated
        calls with different rng state will generally return different
        bodies.
        """
        trajectory = self.develop(genome, return_trajectory=True, rng=rng)
        if self.readout_mode == "fixed_t":
            return trajectory[-1]
        elif self.readout_mode == "mode_of_window":
            window = trajectory[-self.readout_window :]
            return _mode_of_window(window)
        raise ValueError(f"unsupported readout_mode '{self.readout_mode}'")  # pragma: no cover -- validated in __init__

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
            "init_gain": self.init_gain,
            "bias_scale": self.bias_scale,
            # Deprecated constructor alias for init_gain; not a live parameter
            # of this instance (never stored), included only so describe()
            # covers every constructor parameter per the Encoder contract.
            "init_scale": None,
            "mutation_sigma": self.mutation_sigma,
            "mutation_scale": self.factors.mutation_scale,
            "mutation_fraction": self.mutation_fraction,
            "channels": self.channels,
            "genome_length": self._genome_length,
            "update_rule": self.update_rule,
            "state_bound": self.state_bound,
            "decay": self.decay,
            "boundary": self.boundary,
            "update_prob": self.update_prob,
            "spectral_normalise": self.spectral_normalise,
            "readout_mode": self.readout_mode,
            "readout_window": self.readout_window,
        }
