"""Typed experiment configuration loaded from TOML files.

A config file is the single source of truth for an experiment. Every run
saves a frozen copy of the config it used, so any result can be traced
back to the exact settings that produced it.

Unknown keys raise an error on purpose: a typo such as "sede = 3" must
fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file or override is invalid."""


@dataclass
class RunConfig:
    name: str = "unnamed"
    seed: int = 0
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", ...
    output_dir: str = "runs"


@dataclass
class ESConfig:
    """Evolution Strategies settings (see egg/es.py)."""

    population: int = 64        # lifetimes per generation; must be even (antithetic pairs)
    sigma: float = 0.05         # std of the Gaussian perturbation
    lr: float = 0.02            # Adam step size on the shared genome
    weight_decay: float = 0.0   # L2 pull toward zero, added to the update
    generations: int = 100
    log_every: int = 1
    checkpoint_every: int = 0   # 0 = only at the end


@dataclass
class ToyConfig:
    """Toy objective used to test the ES trainer before real tasks."""

    function: str = "sphere"    # "sphere" or "rastrigin"
    dim: int = 100
    init_scale: float = 3.0     # initial genome is drawn from N(0, init_scale^2)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    es: ESConfig = field(default_factory=ESConfig)
    toy: ToyConfig = field(default_factory=ToyConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _from_dict(cls: type, data: dict[str, Any], path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a table, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{path or 'config'}: unknown key(s) {unknown}; allowed: {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        key_path = f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value, key_path)
        else:
            kwargs[name] = _check_scalar(hint, value, key_path)
    return cls(**kwargs)


def _check_scalar(hint: Any, value: Any, path: str) -> Any:
    # bool is a subclass of int in Python, so check it explicitly.
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    return value


def _parse_override_value(raw: str) -> Any:
    """Parse a value using TOML rules; fall back to a plain string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply "section.key=value" overrides to a raw config dict (in place)."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {key} is not a table")
        node[keys[-1]] = _parse_override_value(raw.strip())
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config file (or defaults when path is None) and apply overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    apply_overrides(data, overrides or [])
    return _from_dict(Config, data, "")
