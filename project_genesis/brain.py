from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkShape:
    input_size: int = 3
    hidden_size: int = 8
    output_size: int = 2


class NeuralNetwork:
    def __init__(self, shape: NetworkShape | None = None, rng: np.random.Generator | None = None) -> None:
        self.shape = shape or NetworkShape()
        self.rng = rng or np.random.default_rng()

        self.w1 = self.rng.normal(0.0, 0.6, size=(self.shape.hidden_size, self.shape.input_size))
        self.b1 = self.rng.normal(0.0, 0.2, size=(self.shape.hidden_size,))
        self.w2 = self.rng.normal(0.0, 0.6, size=(self.shape.output_size, self.shape.hidden_size))
        self.b2 = self.rng.normal(0.0, 0.2, size=(self.shape.output_size,))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        if x.shape != (self.shape.input_size,):
            raise ValueError(f"Expected input shape {(self.shape.input_size,)}, got {x.shape}")

        hidden = np.tanh(self.w1 @ x + self.b1)
        output = np.tanh(self.w2 @ hidden + self.b2)
        return output

    def clone(self) -> "NeuralNetwork":
        cloned = NeuralNetwork(shape=self.shape, rng=self.rng)
        cloned.w1 = self.w1.copy()
        cloned.b1 = self.b1.copy()
        cloned.w2 = self.w2.copy()
        cloned.b2 = self.b2.copy()
        return cloned

    def get_dna(self) -> np.ndarray:
        return np.concatenate(
            [
                self.w1.ravel(),
                self.b1.ravel(),
                self.w2.ravel(),
                self.b2.ravel(),
            ]
        )

    @classmethod
    def from_dna(
        cls,
        dna: np.ndarray,
        shape: NetworkShape | None = None,
        rng: np.random.Generator | None = None,
    ) -> "NeuralNetwork":
        shape = shape or NetworkShape()
        dna = np.asarray(dna, dtype=np.float64)

        expected = (
            shape.hidden_size * shape.input_size
            + shape.hidden_size
            + shape.output_size * shape.hidden_size
            + shape.output_size
        )

        if dna.size != expected:
            raise ValueError(f"DNA length mismatch: expected {expected}, got {dna.size}")

        network = cls(shape=shape, rng=rng)
        cursor = 0

        w1_size = shape.hidden_size * shape.input_size
        network.w1 = dna[cursor : cursor + w1_size].reshape(shape.hidden_size, shape.input_size)
        cursor += w1_size

        b1_size = shape.hidden_size
        network.b1 = dna[cursor : cursor + b1_size]
        cursor += b1_size

        w2_size = shape.output_size * shape.hidden_size
        network.w2 = dna[cursor : cursor + w2_size].reshape(shape.output_size, shape.hidden_size)
        cursor += w2_size

        b2_size = shape.output_size
        network.b2 = dna[cursor : cursor + b2_size]

        return network

    def crossover(self, other: "NeuralNetwork", rng: np.random.Generator | None = None) -> "NeuralNetwork":
        rng = rng or self.rng
        dna_a = self.get_dna()
        dna_b = other.get_dna()

        if dna_a.shape != dna_b.shape:
            raise ValueError("Parent DNA shapes must match for crossover")

        mix_mask = rng.random(dna_a.size) < 0.5
        child_dna = np.where(mix_mask, dna_a, dna_b)
        return NeuralNetwork.from_dna(child_dna, shape=self.shape, rng=rng)

    def mutate(
        self,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or self.rng
        dna = self.get_dna()

        mask = rng.random(dna.size) < mutation_rate
        if np.any(mask):
            dna[mask] += rng.normal(0.0, mutation_strength, size=np.count_nonzero(mask))
            mutated = NeuralNetwork.from_dna(dna, shape=self.shape, rng=rng)
            self.w1 = mutated.w1
            self.b1 = mutated.b1
            self.w2 = mutated.w2
            self.b2 = mutated.b2
