"""Live, separate-window view of a racing car's neural network."""

from __future__ import annotations

import math

import pygame
from pygame._sdl2.video import Renderer, Texture, Window


SIZE = (1180, 800)
BG = (13, 31, 35)
PANEL = (15, 24, 33)
TEXT = (231, 242, 239)
MUTED = (137, 160, 165)
POSITIVE = (97, 231, 188)
NEGATIVE = (255, 178, 101)


def activations(brain, inputs: list[float]) -> list[list[float]]:
    """Return the input and every layer's live outputs."""
    layers = [inputs[:]]
    for weights, biases in zip(brain.weights, brain.biases):
        layers.append([
            math.tanh(sum(weight * value for weight, value in zip(row, layers[-1])) + bias)
            for row, bias in zip(weights, biases)
        ])
    return layers


class NetworkInspector:
    def __init__(self) -> None:
        self.window = Window("Best car network", size=SIZE)
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface(SIZE)
        self.font = pygame.font.SysFont("Avenir Next", 16)
        self.small = pygame.font.SysFont("Avenir Next", 13)
        self.title = pygame.font.SysFont("Avenir Next", 25, bold=True)
        self.selected: tuple[int, int] | None = None
        self.scroll = 0
        self.mouse = (-1, -1)
        self.positions: list[list[tuple[int, int]]] = []

    def close(self) -> None:
        self.renderer = None
        self.window.destroy()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return false when the inspector's close button was pressed."""
        window = getattr(event, "window", None)
        if getattr(window, "id", window) != self.window.id:
            return True
        if event.type == pygame.WINDOWCLOSE:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.mouse = event.pos
                self.selected = self._node_at(event.pos)
                self.scroll = 0
            elif event.button in (4, 5):
                self.scroll = max(0, self.scroll + (-1 if event.button == 4 else 1) * 3)
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll = max(0, self.scroll - event.y * 3)
        return True

    def _node_at(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        for layer, points in enumerate(self.positions):
            for index, (x, y) in enumerate(points):
                if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 14 ** 2:
                    return layer, index
        return None

    def _text(self, value: str, x: int, y: int, color=TEXT, font=None) -> None:
        self.surface.blit((font or self.small).render(value, True, color), (x, y))

    def draw(self, race, paused: bool) -> None:
        car = race.best_car
        brain = car.brain
        inputs = car.sensors or race.track.sense(car.x, car.y, car.angle, brain.sizes[0])
        layers = activations(brain, inputs)
        if self.selected and (
            self.selected[0] >= len(layers)
            or self.selected[1] >= len(layers[self.selected[0]])
        ):
            self.selected = None
        self.surface.fill(BG)
        self._text("BEST CAR NETWORK", 26, 17, POSITIVE, self.title)
        state = "paused" if paused else ("driving" if car.alive else "stopped")
        result = (
            f"fastest lap {car.fastest_lap:.2f} s" if car.fastest_lap is not None
            else f"progress {car.best_progress / race.track.length:.2f} laps"
        )
        self._text(
            f"Generation {race.generation}  |  {result}  |  {state}  |  "
            f"speed {car.speed:.1f}  |  {len(brain.sizes)} layers", 28, 53, MUTED, self.font,
        )
        self._text("Green: positive weight    Orange: negative weight    Brighter: stronger", 28, 77, MUTED)

        left, right = 65, 720
        top, bottom = 139, 742
        self.positions = []
        for layer, count in enumerate(brain.sizes):
            x = round(left + (right - left) * layer / (len(brain.sizes) - 1))
            self.positions.append([
                (x, round(top + (bottom - top) * (i + 0.5) / count))
                for i in range(count)
            ])
            name = "Sensors" if layer == 0 else "Outputs" if layer == len(brain.sizes) - 1 else f"Hidden {layer}"
            self._text(name, x - 27, 112, MUTED)

        selected = self.selected
        for layer, matrix in enumerate(brain.weights):
            for target, row in enumerate(matrix):
                end = self.positions[layer + 1][target]
                for source, weight in enumerate(row):
                    start = self.positions[layer][source]
                    highlighted = selected in ((layer, source), (layer + 1, target))
                    intensity = min(abs(weight) / 1.5, 1.0)
                    base = POSITIVE if weight >= 0 else NEGATIVE
                    floor = 39 if highlighted else 23
                    factor = (0.35 if highlighted else 0.15) + intensity * (0.65 if highlighted else 0.45)
                    color = tuple(round(floor + (component - floor) * factor) for component in base)
                    pygame.draw.line(self.surface, color, start, end, 2 if highlighted else 1)

        for layer, points in enumerate(self.positions):
            for index, (x, y) in enumerate(points):
                value = layers[layer][index]
                color = POSITIVE if value >= 0 else NEGATIVE
                if selected == (layer, index):
                    pygame.draw.circle(self.surface, TEXT, (x, y), 13, 2)
                pygame.draw.circle(self.surface, color, (x, y), 8)
                self._text(f"{value:+.2f}", x + 12, y - 8, TEXT if selected == (layer, index) else MUTED)

        pygame.draw.rect(self.surface, PANEL, (805, 0, 375, SIZE[1]))
        pygame.draw.line(self.surface, (52, 76, 78), (805, 0), (805, SIZE[1]), 2)
        self._draw_details(brain, layers)
        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        self.renderer.blit(texture)
        self.renderer.present()

    def _draw_details(self, brain, layers: list[list[float]]) -> None:
        x = 829
        self._text("NEURON DETAILS", x, 23, POSITIVE, self.title)
        self._text("Click a neuron to inspect its values.", x, 58, MUTED, self.font)
        selected = self.selected
        if selected is None:
            self._text("All connections are drawn on the left.", x, 99, TEXT)
            self._text("Select a neuron for exact weights.", x, 120, TEXT)
            return
        layer, index = selected
        name = "Sensor" if layer == 0 else "Output" if layer == len(layers) - 1 else f"Hidden {layer}"
        if layer == len(layers) - 1:
            name += (" (steer)", " (drive)", " (brake)")[index]
        self._text(f"{name} {index + 1} / {len(layers[layer])}", x, 99, TEXT, self.font)
        self._text(f"Activation: {layers[layer][index]:+.6f}", x, 129)
        rows: list[tuple[str, str]] = []
        if layer:
            bias = brain.biases[layer - 1][index]
            weighted = sum(
                w * value for w, value in zip(brain.weights[layer - 1][index], layers[layer - 1])
            )
            self._text(f"Bias: {bias:+.6f}", x, 151)
            self._text(f"Weighted sum + bias: {weighted + bias:+.6f}", x, 173)
            rows.append(("INCOMING CONNECTIONS", ""))
            for source, weight in enumerate(brain.weights[layer - 1][index]):
                rows.append((f"L{layer - 1}:{source + 1}  →  L{layer}:{index + 1}", f"{weight:+.6f}"))
        else:
            self._text("Input: normalized road-edge distance", x, 151)
        if layer < len(layers) - 1:
            rows.append(("OUTGOING CONNECTIONS", ""))
            for target, matrix_row in enumerate(brain.weights[layer]):
                rows.append((f"L{layer}:{index + 1}  →  L{layer + 1}:{target + 1}", f"{matrix_row[index]:+.6f}"))
        visible = (SIZE[1] - 234) // 22
        self.scroll = min(self.scroll, max(0, len(rows) - visible))
        for row_index, (label, value) in enumerate(rows[self.scroll:self.scroll + visible]):
            y = 218 + row_index * 22
            if value:
                self._text(label, x, y, MUTED)
                self._text(value, 1081, y, POSITIVE if value[0] == "+" else NEGATIVE)
            else:
                self._text(label, x, y, POSITIVE)
        if len(rows) > visible:
            self._text(f"Scroll for more  ({self.scroll + 1}-{min(self.scroll + visible, len(rows))}/{len(rows)})", x, 768, MUTED)
