"""Small, dependency-free neural networks and an evolutionary population."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


WEIGHT_MUTATION_STD = 0.15
BIAS_MUTATION_STD = 0.10
WEIGHT_MUTATION_RATE = 0.12
BIAS_MUTATION_RATE = 0.18
CROSSOVER_RATE = 0.5  # chance a gene comes from the second parent; 0 = mutated clone
MINOR_WEIGHT_STD = 0.03
MINOR_BIAS_STD = 0.02
MINOR_MUTATION_RATE = 0.10
# Inputs also feed the outputs directly, next to the hidden layers. Simple reflexes
# such as steering toward the open side or braking when fast with a wall ahead then
# need a handful of weights instead of a path through two random tanh layers.
SKIP_CONNECTIONS = True
SAVE_VERSION = 3  # v3: networks have a speed input


@lru_cache(maxsize=64)
def _forward_for(sizes: tuple[int, ...]):
    """Specialize the arithmetic to a topology, reading live parameters each call."""
    lines = ["def forward(brain, inputs):"]
    previous = [f"v0_{i}" for i in range(sizes[0])]
    lines.append("    " + ", ".join(previous) + ", = inputs")
    for layer, (width, count) in enumerate(zip(sizes, sizes[1:])):
        lines.append(f"    weights = brain.weights[{layer}]")
        lines.append(f"    biases = brain.biases[{layer}]")
        current = []
        for i in range(count):
            name = f"v{layer + 1}_{i}"
            current.append(name)
            names = [f"w{j}" for j in range(width)]
            lines.append("    " + ", ".join(names) + f", = weights[{i}]")
            terms = ", ".join(f"w{j} * {previous[j]}" for j in range(width))
            expression = f"sum(({terms},)) + biases[{i}]"
            if layer == len(sizes) - 2:
                lines.append(f"    shortcut = brain.skip[{i}] if brain.skip else ()")
                terms = ", ".join(f"shortcut[{j}] * v0_{j}" for j in range(sizes[0]))
                expression = f"({expression} + sum(({terms},))) if shortcut else ({expression})"
            lines.append(f"    {name} = tanh({expression})")
        previous = current
    lines.append("    return " + ", ".join(previous[:3]))
    namespace = {"tanh": math.tanh}
    exec("\n".join(lines), namespace)
    return namespace["forward"]


@dataclass
class Network:
    """A fully connected network with tanh steering, drive, and brake outputs."""

    sizes: tuple[int, ...]
    weights: list[list[list[float]]]
    biases: list[list[float]]
    skip: list[list[float]] = field(default_factory=list)  # outputs x inputs, added before the output tanh

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
        skip_scale = math.sqrt(2.0 / sizes[0])
        skip = [
            [rng.gauss(0, skip_scale) if SKIP_CONNECTIONS else 0.0 for _ in range(sizes[0])]
            for _ in range(sizes[-1])
        ]
        return cls(sizes, weights, biases, skip)

    def forward(self, inputs: list[float]) -> tuple[float, float, float]:
        if len(inputs) != self.sizes[0]:
            raise ValueError(f"Expected {self.sizes[0]} inputs, got {len(inputs)}")
        return _forward_for(self.sizes)(self, inputs)

    def copy(self) -> "Network":
        return Network(
            self.sizes,
            [[row[:] for row in layer] for layer in self.weights],
            [layer[:] for layer in self.biases],
            [row[:] for row in self.skip],
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
        if SKIP_CONNECTIONS:
            for row in clone.skip:
                for index, value in enumerate(row):
                    if rng.random() < MINOR_MUTATION_RATE:
                        row[index] = value + rng.gauss(0, MINOR_WEIGHT_STD)
        return clone

    def to_dict(self) -> dict:
        return {"sizes": list(self.sizes), "weights": self.weights, "biases": self.biases, "skip": self.skip}

    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        sizes = tuple(int(size) for size in data["sizes"])
        weights = [[[float(v) for v in row] for row in layer] for layer in data["weights"]]
        biases = [[float(v) for v in layer] for layer in data["biases"]]
        # Files written before skip connections existed have none: all zeros keeps them identical.
        skip = [[float(v) for v in row] for row in data.get("skip") or [[0.0] * sizes[0] for _ in range(sizes[-1])]]
        if len(sizes) < 2 or sizes[-1] != 3 or len(weights) != len(sizes) - 1 or len(biases) != len(weights):
            raise ValueError("Malformed network")
        for (inputs, outputs), layer, bias in zip(zip(sizes, sizes[1:]), weights, biases):
            if len(layer) != outputs or len(bias) != outputs or any(len(row) != inputs for row in layer):
                raise ValueError("Network shape does not match its layer sizes")
        if len(skip) != sizes[-1] or any(len(row) != sizes[0] for row in skip):
            raise ValueError("Skip connections do not match the layer sizes")
        return cls(sizes, weights, biases, skip)

    def child(self, other: "Network", rng: random.Random) -> "Network":
        if self.sizes != other.sizes:
            raise ValueError("Parents must have the same architecture")
        child = self.copy()
        for layer_index, layer in enumerate(child.weights):
            for row_index, row in enumerate(layer):
                for column_index, value in enumerate(row):
                    if rng.random() < CROSSOVER_RATE:
                        value = other.weights[layer_index][row_index][column_index]
                    if rng.random() < WEIGHT_MUTATION_RATE:
                        value += rng.gauss(0, WEIGHT_MUTATION_STD)
                    row[column_index] = value
        for layer_index, layer in enumerate(child.biases):
            for index, value in enumerate(layer):
                if rng.random() < CROSSOVER_RATE:
                    value = other.biases[layer_index][index]
                if rng.random() < BIAS_MUTATION_RATE:
                    value += rng.gauss(0, BIAS_MUTATION_STD)
                layer[index] = value
        if SKIP_CONNECTIONS:
            for row_index, row in enumerate(child.skip):
                for column_index, value in enumerate(row):
                    if rng.random() < CROSSOVER_RATE:
                        value = other.skip[row_index][column_index]
                    if rng.random() < WEIGHT_MUTATION_RATE:
                        value += rng.gauss(0, WEIGHT_MUTATION_STD)
                    row[column_index] = value
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
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
