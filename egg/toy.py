"""Toy objectives with a known optimum, used to validate the ES trainer.

Both take a batch of genomes (population, dim) and return fitness
(population,). Higher is better; the best possible value is 0 at x = 0.

- sphere:    a smooth bowl. Any working optimizer must solve it.
- rastrigin: a bowl covered in many local dips (local optima). Tests that
             the noise lets the search look past small traps.
"""

from __future__ import annotations

import math

import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    return -(x * x).sum(dim=-1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    a = 10.0
    return -(a * x.shape[-1] + (x * x - a * torch.cos(2 * math.pi * x)).sum(dim=-1))


TOY_FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin}
