"""The frozen Encoder contract. Every representation (direct, NCA, and any
future encoding such as a symbolic-rule arm) must implement exactly this
interface without changing method signatures.

Invalid-genome handling (what happens when decode() produces a body that
fails morphology validation) is deliberately NOT part of this interface.
It is a pipeline concern, enforced once in the shared evaluation/EA code,
and applied identically to every encoding. An Encoder must never loop,
resample, or repair to avoid producing an invalid body.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def genome_length(self) -> int:
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

    @abstractmethod
    def describe(self) -> dict:
        """JSON-serialisable hyperparameters, written into every run log."""
        ...
