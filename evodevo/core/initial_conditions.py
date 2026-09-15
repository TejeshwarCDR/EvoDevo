"""Named initial conditions, usable identically by every developmental
encoder.

Each InitialCondition exposes two representations of the SAME starting
condition: a discrete voxel grid (`as_discrete_grid`, for the rule-based
interpreter) and a continuous material-logit + hidden-channel state tensor
(`as_state_tensor`, for the NCA's representation). Round-trip consistency
between the two (argmax(as_state_tensor(...)[..., :N_MATERIALS]) ==
as_discrete_grid()) is what makes "the same named condition" a provable,
falsifiable claim across encoding arms rather than an assertion -- see
tests/core/test_initial_conditions.py.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from evodevo.core.materials import EMPTY, N_MATERIALS, SOFT

# Logit magnitude used to make a material's argmax unambiguous in
# as_state_tensor: "favouring" a material means giving it this logit and
# every other material 0, not an arbitrary probability distribution.
_FAVOUR_LOGIT = 5.0

# Matches the NCA encoder's own historical hidden-channel/init-state scale
# (its previous hardcoded _INIT_STATE_SCALE), so a named condition's hidden
# channels sit on the same footing as its material logits rather than an
# independently chosen constant.
_HIDDEN_SCALE = 0.1

# Seed/scale for the fixed non-symmetric fully-populated state that used to
# be the NCA encoder's only initial condition (hardcoded _INIT_STATE_SEED).
# Ported unchanged here so "full_nonsymmetric" reproduces that arm's exact
# historical behaviour as one of the three now-explicit initial conditions.
_NONSYMMETRIC_SEED = 20260902
_NONSYMMETRIC_SCALE = 0.1


@runtime_checkable
class InitialCondition(Protocol):
    name: str

    def as_discrete_grid(self, grid_shape: tuple) -> np.ndarray:
        """Return a (rows, cols) int array of material IDs."""
        ...

    def as_state_tensor(self, grid_shape: tuple, channels: int) -> np.ndarray:
        """Return a (rows, cols, channels) float array: N_MATERIALS material
        logits followed by (channels - N_MATERIALS) hidden channels."""
        ...


class CenterSeed:
    """One SOFT voxel at the grid centre, everything else EMPTY."""

    name = "center_seed"

    def as_discrete_grid(self, grid_shape: tuple) -> np.ndarray:
        rows, cols = grid_shape
        grid = np.zeros(grid_shape, dtype=int)
        grid[rows // 2, cols // 2] = SOFT
        return grid

    def as_state_tensor(self, grid_shape: tuple, channels: int) -> np.ndarray:
        rows, cols = grid_shape
        state = np.zeros((rows, cols, channels), dtype=float)
        state[:, :, EMPTY] = _FAVOUR_LOGIT
        state[rows // 2, cols // 2, EMPTY] = 0.0
        state[rows // 2, cols // 2, SOFT] = _FAVOUR_LOGIT
        if channels > N_MATERIALS:
            state[:, :, N_MATERIALS:] = _HIDDEN_SCALE
        return state


class FullNonsymmetric:
    """The NCA arm's original fixed asymmetric initial state; decoded to a
    discrete grid via argmax so the rule-based arm can start from the exact
    same condition."""

    name = "full_nonsymmetric"

    def as_state_tensor(self, grid_shape: tuple, channels: int) -> np.ndarray:
        # Material logits are drawn independently of any hidden channels, so
        # the material portion (and therefore as_discrete_grid()) is the
        # same regardless of how many hidden channels the caller asks for --
        # drawing all `channels` at once would interleave the two draws and
        # make the material logits depend on hidden_channels, breaking the
        # round-trip consistency that makes this a well-defined condition.
        rng = np.random.default_rng(_NONSYMMETRIC_SEED)
        rows, cols = grid_shape
        material = rng.normal(scale=_NONSYMMETRIC_SCALE, size=(rows, cols, N_MATERIALS))
        if channels <= N_MATERIALS:
            return material[:, :, :channels]
        hidden = rng.normal(scale=_NONSYMMETRIC_SCALE, size=(rows, cols, channels - N_MATERIALS))
        return np.concatenate([material, hidden], axis=-1)

    def as_discrete_grid(self, grid_shape: tuple) -> np.ndarray:
        state = self.as_state_tensor(grid_shape, N_MATERIALS)
        return np.argmax(state, axis=-1).astype(int)


class FullUniform:
    """Every cell SOFT -- a symmetry-breaking control. A deterministic local
    update rule cannot differentiate cells starting from a perfectly
    uniform, symmetric state, so this condition is EXPECTED to fail to
    differentiate; confirming that failure is itself the validation of the
    locality argument used to justify the other two conditions."""

    name = "full_uniform"

    def as_discrete_grid(self, grid_shape: tuple) -> np.ndarray:
        return np.full(grid_shape, SOFT, dtype=int)

    def as_state_tensor(self, grid_shape: tuple, channels: int) -> np.ndarray:
        rows, cols = grid_shape
        state = np.zeros((rows, cols, channels), dtype=float)
        state[:, :, SOFT] = _FAVOUR_LOGIT
        if channels > N_MATERIALS:
            state[:, :, N_MATERIALS:] = _HIDDEN_SCALE
        return state


_REGISTRY = {
    "center_seed": CenterSeed,
    "full_nonsymmetric": FullNonsymmetric,
    "full_uniform": FullUniform,
}


def get_initial_condition(name: str) -> InitialCondition:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown initial condition '{name}'; available: {sorted(_REGISTRY)}")
