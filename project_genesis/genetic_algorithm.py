from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from brain import NeuralNetwork


class EvolvableAgent(Protocol):
    brain: NeuralNetwork

    @property
    def fitness(self) -> float:
        ...


@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 50
    elite_fraction: float = 0.2
    mutation_rate: float = 0.05
    mutation_strength: float = 0.15


class GeneticAlgorithm:
    def __init__(self, config: GeneticAlgorithmConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.config = config or GeneticAlgorithmConfig()
        self.rng = rng or np.random.default_rng()
        self.generation = 1
        self.best_fitness_seen = 0.0
        self.last_generation_best = 0.0

    def _select_elites(self, agents: list[EvolvableAgent]) -> list[EvolvableAgent]:
        sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        elite_count = max(1, int(len(sorted_agents) * self.config.elite_fraction))
        return sorted_agents[:elite_count]

    def _sample_parent(self, elites: list[EvolvableAgent]) -> EvolvableAgent:
        fitnesses = np.array([max(0.0, a.fitness) for a in elites], dtype=np.float64)
        total = fitnesses.sum()

        if total <= 0.0:
            return elites[int(self.rng.integers(0, len(elites)))]

        probs = fitnesses / total
        index = int(self.rng.choice(len(elites), p=probs))
        return elites[index]

    def create_next_generation(self, agents: list[EvolvableAgent]) -> list[NeuralNetwork]:
        if not agents:
            raise ValueError("Cannot evolve empty agent population")

        elites = self._select_elites(agents)
        self.last_generation_best = elites[0].fitness
        self.best_fitness_seen = max(self.best_fitness_seen, self.last_generation_best)

        next_brains: list[NeuralNetwork] = []
        elite_count = len(elites)

        for i in range(self.config.population_size):
            if i < elite_count:
                child = elites[i].brain.clone()
            else:
                parent_a = self._sample_parent(elites)
                parent_b = self._sample_parent(elites)
                child = parent_a.brain.crossover(parent_b.brain, rng=self.rng)
                child.mutate(
                    mutation_rate=self.config.mutation_rate,
                    mutation_strength=self.config.mutation_strength,
                    rng=self.rng,
                )
            next_brains.append(child)

        self.generation += 1
        return next_brains
