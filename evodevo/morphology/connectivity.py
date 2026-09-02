"""Canonical 4-connected component sizing, shared by validate.py and metrics.py."""
from __future__ import annotations

import numpy as np


def largest_component_size(occupied: np.ndarray) -> int:
    """Largest 4-connected component among True cells in a boolean grid."""
    visited = np.zeros_like(occupied, dtype=bool)
    best = 0
    rows, cols = occupied.shape
    for r in range(rows):
        for c in range(cols):
            if not occupied[r, c] or visited[r, c]:
                continue
            size = 0
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                size += 1
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and occupied[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            best = max(best, size)
    return best
