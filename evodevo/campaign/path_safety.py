"""Filename rules shared by the local campaign tree and NTFS/FUSE mirror."""
from __future__ import annotations

from pathlib import Path

NTFS_FORBIDDEN = set('<>:"\\|?*')


def unsafe_ntfs_component(component: str) -> bool:
    return (not component or any(char in NTFS_FORBIDDEN for char in component) or
            component.endswith((".", " ")))


def ntfs_unsafe_paths(root) -> list[str]:
    """Return relative paths whose *components* cannot be mirrored to NTFS."""
    root = Path(root)
    if not root.exists():
        return []
    return [str(path.relative_to(root)) for path in root.rglob("*")
            if any(unsafe_ntfs_component(component) for component in path.relative_to(root).parts)]


def assert_ntfs_safe_tree(root) -> None:
    unsafe = ntfs_unsafe_paths(root)
    if unsafe:
        raise ValueError("NTFS-unsafe campaign path(s): " + ", ".join(unsafe))
