"""All randomness in this project must flow from these two functions."""
from __future__ import annotations

import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
