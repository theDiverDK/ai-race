"""Small, dependency-free neural networks and an evolutionary population."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


WEIGHT_MUTATION_STD = 0.15
BIAS_MUTATION_STD = 0.10


@dataclass
class Network:
    """A fully connected network with tanh steering, drive, and brake outputs."""

    sizes: tuple[int, ...]
    weights: list[list[list[float]]]
    biases: list[list[float]]

    @classmethod
    def random(cls, sizes: tuple[int, ...], rng: random.Random) -> "Network":
        weights = []
        biases = []
        for inputs, outputs in zip(sizes, sizes[1:]):
            scale = math.sqrt(2.0 / inputs)
            weights.append(
                [[rng.gauss(0, scale) for _ in range(inputs)] for _ in range(outputs)]
            )
            biases.append([rng.gauss(0, 0.25) for _ in range(outputs)])
        # A slight initial preference for forward drive still permits reverse.
        biases[-1][1] += 0.35
        return cls(sizes, weights, biases)

    def forward(self, inputs: list[float]) -> tuple[float, float, float]:
        if len(inputs) != self.sizes[0]:
            raise ValueError(f"Expected {self.sizes[0]} inputs, got {len(inputs)}")
        values = inputs
        for weights, biases in zip(self.weights, self.biases):
            values = [
                math.tanh(sum(weight * value for weight, value in zip(row, values)) + bias)
                for row, bias in zip(weights, biases)
            ]
        return values[0], values[1], values[2]

    def copy(self) -> "Network":
        return Network(
            self.sizes,
            [[row[:] for row in layer] for layer in self.weights],
            [layer[:] for layer in self.biases],
        )

    def child(self, other: "Network", rng: random.Random) -> "Network":
        if self.sizes != other.sizes:
            raise ValueError("Parents must have the same architecture")
        child = self.copy()
        for layer_index, layer in enumerate(child.weights):
            for row_index, row in enumerate(layer):
                for column_index, value in enumerate(row):
                    if rng.random() < 0.5:
                        value = other.weights[layer_index][row_index][column_index]
                    if rng.random() < 0.12:
                        value += rng.gauss(0, WEIGHT_MUTATION_STD)
                    row[column_index] = value
        for layer_index, layer in enumerate(child.biases):
            for index, value in enumerate(layer):
                if rng.random() < 0.5:
                    value = other.biases[layer_index][index]
                if rng.random() < 0.18:
                    value += rng.gauss(0, BIAS_MUTATION_STD)
                layer[index] = value
        return child


def next_generation(
    ranked: list[tuple[tuple[int, float, float], Network]], rng: random.Random, population: int
) -> list[Network]:
    """Keep champions, breed from strong drivers, and add some newcomers."""
    if not ranked or population < 4:
        raise ValueError("A ranked population of at least four cars is required")
    ranked = sorted(ranked, key=lambda item: item[0], reverse=True)
    pool = [network for _, network in ranked[: max(4, population // 4)]]
    result = [network.copy() for network in pool[:2]]
    newcomers = max(2, population // 10)
    while len(result) < population - newcomers:
        a = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        b = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        result.append(a.child(b, rng))
    while len(result) < population:
        result.append(Network.random(pool[0].sizes, rng))
    return result
