"""Small, dependency-free neural networks and an evolutionary population."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


WEIGHT_MUTATION_STD = 0.15
BIAS_MUTATION_STD = 0.10
MINOR_WEIGHT_STD = 0.03
MINOR_BIAS_STD = 0.02
MINOR_MUTATION_RATE = 0.10
SAVE_VERSION = 2


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

    def minor_mutation(self, rng: random.Random) -> "Network":
        """A near-copy: a few small nudges, never a crossover."""
        clone = self.copy()
        for layer in clone.weights:
            for row in layer:
                for index, value in enumerate(row):
                    if rng.random() < MINOR_MUTATION_RATE:
                        row[index] = value + rng.gauss(0, MINOR_WEIGHT_STD)
        for layer in clone.biases:
            for index, value in enumerate(layer):
                if rng.random() < MINOR_MUTATION_RATE:
                    layer[index] = value + rng.gauss(0, MINOR_BIAS_STD)
        return clone

    def to_dict(self) -> dict:
        return {"sizes": list(self.sizes), "weights": self.weights, "biases": self.biases}

    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        sizes = tuple(int(size) for size in data["sizes"])
        weights = [[[float(v) for v in row] for row in layer] for layer in data["weights"]]
        biases = [[float(v) for v in layer] for layer in data["biases"]]
        if len(sizes) < 2 or sizes[-1] != 3 or len(weights) != len(sizes) - 1 or len(biases) != len(weights):
            raise ValueError("Malformed network")
        for (inputs, outputs), layer, bias in zip(zip(sizes, sizes[1:]), weights, biases):
            if len(layer) != outputs or len(bias) != outputs or any(len(row) != inputs for row in layer):
                raise ValueError("Network shape does not match its layer sizes")
        return cls(sizes, weights, biases)

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


def newcomer_count(population: int) -> int:
    return max(2, population // 10)


def slot_role(index: int, population: int) -> str:
    """How the car in this slot of a new generation was made by next_generation()."""
    if index == 0:
        return "champion"
    if index == 1:
        return "runner_up"
    if index < 4:
        return "mutant"
    if index >= population - newcomer_count(population):
        return "newcomer"
    return "child"


def next_generation(
    ranked: list[tuple[tuple[int, float, float], Network]], rng: random.Random, population: int
) -> list[Network]:
    """Keep the two champions unchanged, add two lightly mutated copies of
    them, breed from strong drivers, and add some newcomers."""
    if not ranked or population < 4:
        raise ValueError("A ranked population of at least four cars is required")
    ranked = sorted(ranked, key=lambda item: item[0], reverse=True)
    pool = [network for _, network in ranked[: max(4, population // 4)]]
    elites = pool[:2]
    result = [network.copy() for network in elites]
    # Two near-copies of the champions explore the neighbourhood of what works.
    result += [elites[i % len(elites)].minor_mutation(rng) for i in range(2)]
    newcomers = newcomer_count(population)
    while len(result) < population - newcomers:
        a = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        b = pool[min(int(rng.random() ** 2 * len(pool)), len(pool) - 1)]
        result.append(a.child(b, rng))
    while len(result) < population:
        result.append(Network.random(pool[0].sizes, rng))
    return result


@dataclass
class SaveData:
    networks: list[Network]
    tracks_completed: int = 0
    best_score: float = 0.0


def save_networks(
    path: Path, networks: list[Network], tracks_completed: int, best_score: float = 0.0
) -> None:
    """Write the champions atomically so a crash never corrupts the file."""
    payload = {
        "version": SAVE_VERSION,
        "tracks_completed": tracks_completed,
        "best_score": best_score,
        "networks": [network.to_dict() for network in networks],
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def load_networks(path: Path) -> SaveData | None:
    """Return the saved champions and records, or None if missing or unusable."""
    try:
        payload = json.loads(Path(path).read_text())
        if payload["version"] != SAVE_VERSION:
            return None
        networks = [Network.from_dict(item) for item in payload["networks"]]
        if not networks or len({network.sizes for network in networks}) != 1:
            return None
        return SaveData(
            networks,
            max(0, int(payload["tracks_completed"])),
            max(0.0, float(payload.get("best_score", 0.0))),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
