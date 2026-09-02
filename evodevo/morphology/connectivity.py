"""Canonical 4-connected component sizing, shared by validate.py and metrics.py."""
from __future__ import annotations

import numpy as np


def get_full_connectivity(robot: np.ndarray) -> np.ndarray:
    """Native equivalent of ``evogym.get_full_connectivity``.

    Returns a `(2, k)` array specifying `k` pairwise voxel connections
    between orthogonally adjacent non-empty voxels. Voxels are indexed by
    their position in the flattened (row-major) grid, matching evogym's
    exact iteration order and edge set (right-neighbor then down-neighbor,
    for each non-empty cell in row-major flat order).
    """
    out = []
    height, width = robot.shape

    for i in range(robot.size):
        x = i % width
        y = i // width

        if robot[y][x] == 0:
            continue

        nx, ny = x + 1, y
        if 0 <= nx < width and 0 <= ny < height and robot[ny][nx] != 0:
            out.append([x + width * y, nx + width * ny])

        nx, ny = x, y + 1
        if 0 <= nx < width and 0 <= ny < height and robot[ny][nx] != 0:
            out.append([x + width * y, nx + width * ny])

    if len(out) == 0:
        return np.empty((0, 2)).T
    return np.array(out).T


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
