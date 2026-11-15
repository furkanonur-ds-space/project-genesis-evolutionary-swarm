"""Genome layout: one flat vector <-> named weight tensors.

ES works on a single flat vector theta. Models need named matrices. A
GenomeSpec records the name and shape of every tensor, in order, and
converts in both directions. Conversion is batched: a (population, size)
matrix of genomes becomes a dict of (population, *shape) tensors, so the
whole population is evaluated in one pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamDef:
    name: str
    shape: tuple[int, ...]
    init: str = "fan_in"   # "fan_in" (normal / sqrt(fan_in)), "zeros"

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class GenomeSpec:
    def __init__(self, params: list[ParamDef]) -> None:
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names in {names}")
        self.params = params
        self.offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for p in params:
            self.offsets[p.name] = (cursor, cursor + p.size)
            cursor += p.size
        self.size = cursor

    def init(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Initial genome. Weight matrices are scaled by 1/sqrt(fan_in)."""
        parts = []
        for p in self.params:
            if p.init == "zeros":
                parts.append(torch.zeros(p.size))
            elif p.init == "fan_in":
                fan_in = p.shape[0] if len(p.shape) > 1 else 1
                parts.append(torch.randn(p.size, generator=generator) / math.sqrt(fan_in))
            else:
                raise ValueError(f"unknown init {p.init!r} for {p.name}")
        return torch.cat(parts)

    def unflatten(self, genomes: torch.Tensor) -> dict[str, torch.Tensor]:
        """(population, size) -> {name: (population, *shape)} views."""
        if genomes.dim() == 1:
            genomes = genomes[None]
        if genomes.shape[-1] != self.size:
            raise ValueError(f"genome size {genomes.shape[-1]} != spec size {self.size}")
        pop = genomes.shape[0]
        return {
            p.name: genomes[:, a:b].reshape(pop, *p.shape)
            for p, (a, b) in zip(self.params, self.offsets.values())
        }

    def summary(self) -> str:
        lines = [f"{p.name:12s} {str(p.shape):14s} {p.size:7d}" for p in self.params]
        lines.append(f"{'total':12s} {'':14s} {self.size:7d}")
        return "\n".join(lines)
