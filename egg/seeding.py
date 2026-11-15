"""Seeding: make random numbers repeatable.

Computers generate "random" numbers from a starting value called the
seed. The same seed always produces the same sequence, so a run started
with the same seed and the same config gives the same result.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch. Returns a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)
