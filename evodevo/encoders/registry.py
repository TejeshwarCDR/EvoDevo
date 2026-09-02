"""The encoder plugin registry.

Registration is explicit (via imports in evodevo/encoders/__init__.py), never
dynamic filesystem scanning -- a registration failure (bad import, missing
class) must raise immediately and loudly at import time, not be silently
skipped because a directory scan couldn't find or load it.
"""
from __future__ import annotations

from typing import Dict, Type

from evodevo.encoders.base import Encoder

_REGISTRY: Dict[str, Type[Encoder]] = {}


def register_encoder(name: str, cls: Type[Encoder]) -> None:
    if not (isinstance(cls, type) and issubclass(cls, Encoder)):
        raise TypeError(f"cannot register '{name}': {cls!r} is not an Encoder subclass")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(
            f"encoder name '{name}' is already registered to {_REGISTRY[name]!r}; "
            f"refusing to overwrite with {cls!r}"
        )
    _REGISTRY[name] = cls


def get_encoder(name: str, **kwargs) -> Encoder:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown encoder '{name}'; available: {sorted(_REGISTRY)}")
    return cls(**kwargs)


def list_encoders() -> list:
    return sorted(_REGISTRY)
