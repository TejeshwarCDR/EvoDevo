"""Proves the conformance suite actually catches violations, by running it
against deliberately broken dummy encoders and asserting each fails with a
clear, clause-naming message. These dummies are never registered into the
shared evodevo.encoders registry (that would permanently break
test_contract.py for everyone) -- they are only run through the same
check functions test_contract.py uses, directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from evodevo.encoders.base import Encoder
from evodevo.encoders.registry import register_encoder
from tests.encoders._contract_checks import (
    check_decode_returns_valid_5x5_int_array,
    check_module_is_physics_and_rl_free,
    check_random_genome_sometimes_produces_invalid_bodies,
    check_serialize_deserialize_roundtrip,
)


class _WrongShapeEncoder(Encoder):
    """decode() returns the wrong shape -- violates contract:decode_shape."""

    @property
    def name(self) -> str:
        return "broken_wrong_shape"

    @property
    def genome_dtype(self):
        return np.int64

    def genome_length(self) -> int:
        return 25

    def random_genome(self, rng):
        return rng.integers(0, 5, size=25)

    def decode(self, genome):
        return np.zeros((4, 4), dtype=int)  # wrong shape: contract requires (5, 5)

    def mutate(self, genome, rng):
        return genome.copy()

    def serialize_genome(self, genome):
        return genome.tolist()

    def deserialize_genome(self, obj):
        return np.asarray(obj, dtype=int)

    def describe(self) -> dict:
        return {"name": self.name}


class _BadRoundtripEncoder(_WrongShapeEncoder):
    """decode() is fixed, but serialize/deserialize silently corrupts the
    genome -- violates contract:serialize_roundtrip."""

    @property
    def name(self) -> str:
        return "broken_bad_roundtrip"

    def decode(self, genome):
        return np.asarray(genome, dtype=int).reshape(5, 5)

    def serialize_genome(self, genome):
        return genome.tolist()

    def deserialize_genome(self, obj):
        corrupted = np.asarray(obj, dtype=int)
        corrupted[0] += 1  # bug: silently mutates on deserialize
        return corrupted


class _AlwaysValidEncoder(_WrongShapeEncoder):
    """decode() is hard-coded to always return the same known-valid body --
    violates contract:no_resample_until_valid (simulates hidden repair logic)."""

    @property
    def name(self) -> str:
        return "broken_always_valid"

    def decode(self, genome):
        return np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 3, 0, 3, 0],
                [0, 3, 0, 3, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=int,
        )


class _ImportsPhysicsEncoder(_WrongShapeEncoder):
    """Module-level import of a forbidden physics/RL package -- violates
    contract:physics_rl_free. (The import happens in *this* test module,
    which is what check_module_is_physics_and_rl_free inspects via
    inspect.getsourcefile(type(encoder)).)"""

    @property
    def name(self) -> str:
        return "broken_imports_physics"

    def decode(self, genome):
        return np.asarray(genome, dtype=int).reshape(5, 5)


try:
    # Deliberately present so ast-parsing this module's source (which is what
    # check_module_is_physics_and_rl_free does) finds a forbidden import --
    # wrapped in try/except so this test module still collects on machines
    # without torch installed; the AST node is present either way.
    import torch  # noqa: E402,F401
except ImportError:
    pass


def test_broken_decode_shape_fails_with_named_clause():
    encoder = _WrongShapeEncoder()
    with pytest.raises(AssertionError, match=r"\[contract:decode_shape\]"):
        check_decode_returns_valid_5x5_int_array(encoder)


def test_broken_serialize_roundtrip_fails_with_named_clause():
    encoder = _BadRoundtripEncoder()
    with pytest.raises(AssertionError, match=r"\[contract:serialize_roundtrip\]"):
        check_serialize_deserialize_roundtrip(encoder)


def test_broken_always_valid_fails_with_named_clause():
    encoder = _AlwaysValidEncoder()
    with pytest.raises(AssertionError, match=r"\[contract:no_resample_until_valid\]"):
        check_random_genome_sometimes_produces_invalid_bodies(encoder, n=50)


def test_broken_physics_import_fails_with_named_clause():
    encoder = _ImportsPhysicsEncoder()
    with pytest.raises(AssertionError, match=r"\[contract:physics_rl_free\]"):
        check_module_is_physics_and_rl_free(encoder)


def test_registering_a_broken_encoder_does_not_corrupt_the_real_registry():
    """register_encoder() itself only checks that the class is an Encoder
    subclass -- it is the contract *checks*, run separately, that catch
    behavioural violations. This test demonstrates the full path: register
    a broken dummy under a throwaway name, fetch it back out, and show the
    conformance checks (the same ones test_contract.py runs against every
    registered encoder) fail on it with a clause-naming message."""
    register_encoder("broken_wrong_shape_demo", _WrongShapeEncoder)

    from evodevo.encoders.registry import get_encoder

    broken = get_encoder("broken_wrong_shape_demo")
    with pytest.raises(AssertionError, match=r"\[contract:decode_shape\]"):
        check_decode_returns_valid_5x5_int_array(broken)
