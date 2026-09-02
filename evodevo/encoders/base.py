"""The frozen Encoder contract. Every representation (direct, NCA, and any
future encoding such as a symbolic-rule arm) must implement exactly this
interface without changing method signatures.

Invalid-genome handling (what happens when decode() produces a body that
fails morphology validation) is deliberately NOT part of this interface.
It is a pipeline concern, enforced once in the shared evaluation/EA code,
and applied identically to every encoding. An Encoder must never loop,
resample, or repair to avoid producing an invalid body.

This module, together with evodevo/encoders/registry.py, is the one place
where the plugin contract is defined. A contributor adding a new encoding
package should never need to edit this file, registry.py, evolution/ea.py,
evaluation/ppo_runner.py, or anything under morphology/ -- see
docs/adding-an-encoder.md. If satisfying your encoder genuinely requires
touching one of those, that is a signal the contract itself is incomplete
and should be extended here (and re-discussed), not worked around per-encoder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Encoder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def genome_length(self) -> int:
        ...

    @property
    @abstractmethod
    def genome_dtype(self) -> Any:
        """The genome's element type, for callers (the EA, the run logger)
        that need to allocate or introspect genomes without special-casing
        each encoding. For a fixed-shape numeric genome (direct, NCA) this
        is a numpy dtype (e.g. np.int64, np.float64). For a variable-length,
        non-numeric genome (e.g. a list of rule objects) this should be the
        Python type `object` -- callers must not assume a numpy array."""
        ...

    @abstractmethod
    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, genome: np.ndarray) -> np.ndarray:
        """Return a (5, 5) int array of material IDs. No validity handling here."""
        ...

    @abstractmethod
    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a mutated copy of genome. Must not loop until a valid body results."""
        ...

    def supports_crossover(self) -> bool:
        """Whether this encoder implements a real crossover(). Defaults to
        False; encoders with no meaningful recombination operator (e.g.
        direct, NCA) should leave this as-is rather than implement a
        crossover() that merely raises."""
        return False

    def crossover(self, parent_a, parent_b, rng: np.random.Generator):
        """Recombine two parent genomes into a child genome. Only called when
        supports_crossover() is True; the default implementation raises so
        that an encoder cannot be silently used for crossover without an
        explicit, deliberate override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support crossover "
            "(supports_crossover() is False)"
        )

    @abstractmethod
    def serialize_genome(self, genome) -> Any:
        """Convert a genome to a JSON-serialisable object (dict/list/str/
        number), so it can be written to a run log or a saved-genome file
        regardless of whether the genome is a numpy array (direct, NCA) or
        a variable-length list of non-numeric objects (e.g. rule-based)."""
        ...

    @abstractmethod
    def deserialize_genome(self, obj: Any):
        """Inverse of serialize_genome: reconstruct a genome from the
        JSON-serialisable object serialize_genome produced. Must round-trip
        exactly: deserialize_genome(serialize_genome(g)) == g."""
        ...

    @abstractmethod
    def describe(self) -> dict:
        """JSON-serialisable hyperparameters, written into every run log."""
        ...

    def develop_trajectory(self, genome) -> Any:
        """For a DEVELOPMENTAL encoder (one whose decode() runs an iterated
        process rather than a single static transformation): return the
        list of intermediate (5, 5) int bodies at each developmental step,
        t=0..T, ending with decode()'s own final result. Defaults to None --
        a non-developmental encoder (e.g. direct, whose decode() is a pure
        reshape with no notion of "steps") has nothing meaningful to return
        here. Callers (evodevo/visualization/development.py) must treat
        None as "this encoder has no trajectory to show", not an error."""
        return None
