"""Save and load training state.

A checkpoint holds everything needed to continue training exactly where
it stopped: the genome, optimizer state, generation counter and seed.
Written atomically (temp file then rename) so a crash during saving never
leaves a half-written checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=True refuses arbitrary pickled code; our checkpoints only
    # contain tensors, numbers, strings and dicts.
    return torch.load(Path(path), map_location=map_location, weights_only=True)
