"""Separate window showing the road through the leading car's windscreen."""

from __future__ import annotations

import math

import numpy as np
import pygame
from pygame._sdl2.video import Renderer, Texture, Window


SIZE = (960, 540)
RENDER = (480, 270)  # drawn at low resolution, then scaled up
HORIZON = 108  # render rows above the road
FOV = math.radians(70)
CAMERA_HEIGHT = 9.0  # world pixels above the road
VIEW_DISTANCE = 700.0
GROUND = (13, 31, 35)
SKY_TOP = (17, 40, 58)
SKY_HORIZON = (70, 104, 117)
TEXT = (231, 242, 239)
MUTED = (137, 160, 165)
HOOD = (97, 231, 188)


class DriverView:
    def __init__(self) -> None:
        self.window = Window("Leading car cockpit", size=SIZE)
        self.renderer = Renderer(self.window)
        self.surface = pygame.Surface(SIZE)
        self.frame = pygame.Surface(RENDER)
        self.font = pygame.font.SysFont("Avenir Next", 18, bold=True)
        self.small = pygame.font.SysFont("Avenir Next", 14)
        self._track = None
        self._texture: np.ndarray | None = None
        # Each road pixel below the horizon looks at a fixed spot relative to
        # the camera: `forward` ahead of it and `right` to its side.
        width, height = RENDER
        focal = width / 2 / math.tan(FOV / 2)
        rows = np.arange(height - HORIZON, dtype=np.float64)[None, :] + 0.5
        columns = np.arange(width, dtype=np.float64)[:, None] - width / 2 + 0.5
        self.forward = np.broadcast_to(CAMERA_HEIGHT * focal / rows, (width, rows.size))
        self.right = columns * self.forward / focal
        self.fog = np.clip(self.forward / VIEW_DISTANCE, 0.0, 1.0)[..., None] ** 1.5
        self.sky = np.array([
            [SKY_TOP[c] + (SKY_HORIZON[c] - SKY_TOP[c]) * y / (HORIZON - 1) for c in range(3)]
            for y in range(HORIZON)
        ], dtype=np.uint8)[None, :, :].repeat(width, axis=0)

    def close(self) -> None:
        self.renderer = None
        self.window.destroy()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return false when the cockpit's close button was pressed."""
        return event.type != pygame.WINDOWCLOSE

    def road_pixels(self, track, x: float, y: float, angle: float) -> np.ndarray:
        """Colours of the road ahead, as a (width, height, 3) array of render pixels."""
        if track is not self._track:
            self._track = track
            self._texture = pygame.surfarray.array3d(track.art)
        texture = self._texture
        cos, sin = math.cos(angle), math.sin(angle)
        world_x = (x + self.forward * cos - self.right * sin).astype(np.int32)
        world_y = (y + self.forward * sin + self.right * cos).astype(np.int32)
        inside = (world_x >= 0) & (world_x < texture.shape[0]) & (world_y >= 0) & (world_y < texture.shape[1])
        colors = np.empty((*world_x.shape, 3), dtype=np.float64)
        colors[:] = GROUND
        colors[inside] = texture[world_x[inside], world_y[inside]]
        return (colors + (np.array(SKY_HORIZON) - colors) * self.fog).astype(np.uint8)

    def draw(self, race, paused: bool) -> None:
        car = race.leader or race.best_car
        pixels = np.concatenate((self.sky, self.road_pixels(race.track, car.x, car.y, car.angle)), axis=1)
        pygame.surfarray.blit_array(self.frame, pixels)
        pygame.transform.scale(self.frame, SIZE, self.surface)

        # The bonnet leans with the steering so turns read at a glance.
        width, height = SIZE
        lean = 40 * car.steering
        pygame.draw.polygon(self.surface, HOOD, [
            (width * 0.18, height), (width * 0.36 + lean, height - 64),
            (width * 0.64 + lean, height - 64), (width * 0.82, height),
        ])
        state = "paused" if paused else ("driving" if car.alive else "stopped")
        self.surface.blit(self.font.render(f"{race.track.label}  •  lap {car.laps_completed + 1}", True, TEXT), (20, 16))
        self.surface.blit(self.small.render(
            f"speed {car.speed:.0f}  |  steer {car.steering:+.2f}  |  drive {car.drive:+.2f}  |  "
            f"brake {car.brake:+.2f}  |  {state}", True, MUTED,
        ), (20, 44))

        texture = Texture.from_surface(self.renderer, self.surface)
        self.renderer.clear()
        self.renderer.blit(texture)
        self.renderer.present()
