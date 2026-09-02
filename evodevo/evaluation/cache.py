"""Body-hash keyed fitness memoization, shared identically by both arms."""
from __future__ import annotations

from typing import Optional


class FitnessCache:
    def __init__(self):
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, body_hash: str):
        if body_hash in self._store:
            self.hits += 1
            return self._store[body_hash]
        self.misses += 1
        return None

    def put(self, body_hash: str, result) -> None:
        self._store[body_hash] = result

    def __len__(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> Optional[float]:
        total = self.hits + self.misses
        if total == 0:
            return None
        return self.hits / total
