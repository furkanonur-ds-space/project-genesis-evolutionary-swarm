"""Compute device selection (CPU or GPU).

The same code must run on a 4 GB laptop GPU and on large workstation
GPUs. Only the "device" setting in the config changes.
"""

from __future__ import annotations

from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(requested: str = "auto") -> Any:
    """Return a torch.device. "auto" picks CUDA when available, else CPU."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this environment. See README.md for setup."
        ) from exc

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    return device


def describe_device(requested: str = "auto") -> str:
    if not torch_available():
        return "torch not installed (numpy only)"
    import torch

    device = resolve_device(requested)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{device} {props.name} {props.total_memory / 2**30:.1f} GB"
    return str(device)
