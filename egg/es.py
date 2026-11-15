"""OpenAI-style Evolution Strategies (Salimans et al. 2017).

The shared genome is a single parameter vector theta (the "soul"). One
generation:

  1. ask():  draw half a population of Gaussian noise vectors eps and
             form 2 candidates per vector: theta + sigma*eps and
             theta - sigma*eps (antithetic, "mirrored" sampling).
  2. evaluate every candidate (one lifetime each) -> fitness.
  3. tell(): replace raw fitness by centered ranks in [-0.5, 0.5],
             estimate the gradient
                 g = 1/(n*sigma) * sum_i rank_i * eps_i
             and move theta uphill with Adam.

Every lifetime contributes to the update in proportion to how it ranked,
not only the best one.

Noise is generated from a seed derived from (run seed, generation), so a
whole generation of noise can be rebuilt from two integers. This is what
later lets several machines share work by exchanging only seeds and
fitness values.
"""

from __future__ import annotations

from typing import Any

import torch


def centered_ranks(fitness: torch.Tensor) -> torch.Tensor:
    """Map fitness values to ranks scaled to [-0.5, 0.5].

    Only the order matters, so the update is immune to fitness scale and
    to outliers (one lucky lifetime cannot dominate).
    """
    n = fitness.numel()
    if n < 2:
        return torch.zeros_like(fitness, dtype=torch.float32)
    ranks = torch.empty(n, device=fitness.device, dtype=torch.float32)
    ranks[fitness.argsort()] = torch.arange(n, device=fitness.device, dtype=torch.float32)
    return ranks / (n - 1) - 0.5


class OpenAIES:
    def __init__(
        self,
        theta: torch.Tensor,
        population: int,
        sigma: float,
        lr: float,
        weight_decay: float = 0.0,
        seed: int = 0,
        lr_decay: float = 1.0,
        lr_min: float = 0.0,
        sigma_decay: float = 1.0,
        sigma_min: float = 0.0,
    ) -> None:
        if population < 2 or population % 2 != 0:
            raise ValueError(f"population must be an even number >= 2, got {population}")
        if theta.dim() != 1:
            raise ValueError("theta must be a 1-D parameter vector")

        self.theta = torch.nn.Parameter(theta.detach().clone().float())
        self.population = population
        self.sigma = float(sigma)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.lr_decay, self.lr_min = float(lr_decay), float(lr_min)
        self.sigma_decay, self.sigma_min = float(sigma_decay), float(sigma_min)
        self.generation = 0
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        self._noise: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.theta.device

    @property
    def num_params(self) -> int:
        return self.theta.numel()

    def _generation_noise(self) -> torch.Tensor:
        """Noise for the current generation, rebuilt from (seed, generation)."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed * 1_000_003 + self.generation)
        half = self.population // 2
        return torch.randn(half, self.num_params, generator=gen, device=self.device)

    def ask(self) -> torch.Tensor:
        """Return candidate genomes, shape (population, num_params).

        Row 2k is theta + sigma*eps_k and row 2k+1 is theta - sigma*eps_k.
        """
        self._noise = self._generation_noise()
        with torch.no_grad():
            plus = self.theta + self.sigma * self._noise
            minus = self.theta - self.sigma * self._noise
            return torch.stack([plus, minus], dim=1).reshape(self.population, -1)

    def tell(self, fitness: torch.Tensor) -> dict[str, float]:
        """Update theta from the fitness of the candidates returned by ask()."""
        if self._noise is None:
            raise RuntimeError("call ask() before tell()")
        fitness = torch.as_tensor(fitness, device=self.device, dtype=torch.float32).flatten()
        if fitness.numel() != self.population:
            raise ValueError(f"expected {self.population} fitness values, got {fitness.numel()}")

        ranks = centered_ranks(fitness).reshape(-1, 2)
        # Antithetic pair: (rank of +eps) - (rank of -eps) weights eps.
        weights = ranks[:, 0] - ranks[:, 1]
        grad = (weights @ self._noise) / (self.population * self.sigma)
        if self.weight_decay:
            grad = grad - self.weight_decay * self.theta.detach()

        self.optimizer.zero_grad(set_to_none=False)
        self.theta.grad = -grad  # optimizers minimize; we want to climb
        self.optimizer.step()

        self._noise = None
        self.generation += 1
        # Schedules (Lange 2022, evosax): explore broadly first, then refine.
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        for group in self.optimizer.param_groups:
            group["lr"] = max(group["lr"] * self.lr_decay, self.lr_min)
        return {
            "fitness_mean": float(fitness.mean()),
            "fitness_max": float(fitness.max()),
            "fitness_min": float(fitness.min()),
            "grad_norm": float(grad.norm()),
            "theta_norm": float(self.theta.detach().norm()),
            "sigma": self.sigma,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "theta": self.theta.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "generation": self.generation,
            "seed": self.seed,
            "population": self.population,
            "sigma": self.sigma,
            "weight_decay": self.weight_decay,
            "schedule": [self.lr_decay, self.lr_min, self.sigma_decay, self.sigma_min],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            self.theta.copy_(state["theta"].to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self.generation = int(state["generation"])
        self.seed = int(state["seed"])
        self.population = int(state["population"])
        self.sigma = float(state["sigma"])
        self.weight_decay = float(state["weight_decay"])
        if "schedule" in state:  # absent in checkpoints written before schedules existed
            self.lr_decay, self.lr_min, self.sigma_decay, self.sigma_min = map(float, state["schedule"])
