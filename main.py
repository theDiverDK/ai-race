"""Neural Circuit: a Pygame race track with evolving neural drivers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame

from neural import Network, next_generation


WORLD_W, HEIGHT, PANEL_W = 900, 800, 360
WIDTH = WORLD_W + PANEL_W
FPS = 60
POPULATION = 50
DEFAULT_MAX_RUNTIME = 25
SENSOR_RANGE = 175

# Later circuits add tighter bends and narrower tarmac. The numbers are the
# strengths of four waves applied to a closed, elliptical centerline.
ROAD_SPECS = [
    ("Open Oval", 104, 0.000, 0.000, 0.000, 0.000, 0.0),
    ("Long Sweep", 100, 0.035, 0.000, 0.000, 0.000, 0.4),
    ("Rolling Ring", 96, 0.055, 0.012, 0.000, 0.000, 0.8),
    ("Triple Bend", 92, 0.075, 0.025, 0.000, 0.000, 1.2),
    ("S Curve", 88, 0.085, 0.040, 0.008, 0.000, 0.5),
    ("Ripple Road", 84, 0.120, 0.080, 0.035, 0.020, 1.0),
    ("Clover Run", 80, 0.140, 0.100, 0.045, 0.025, 1.5),
    ("Switchback", 76, 0.150, 0.120, 0.055, 0.035, 0.3),
    ("Tight Turns", 72, 0.170, 0.130, 0.065, 0.055, 0.8),
    ("Expert Loop", 68, 0.180, 0.150, 0.075, 0.065, 1.3),
]

BG = (13, 31, 35)
PANEL = (15, 24, 33)
CARD = (25, 39, 49)
TEXT = (231, 242, 239)
MUTED = (137, 160, 165)
ACCENT = (97, 231, 188)
ORANGE = (255, 178, 101)


def catmull_rom(points: list[tuple[float, float]], samples: int = 16) -> list[tuple[float, float]]:
    """Make a smooth, closed centerline from a few hand-placed waypoints."""
    result = []
    count = len(points)
    for i in range(count):
        p0, p1, p2, p3 = (points[(i + offset) % count] for offset in (-1, 0, 1, 2))
        for step in range(samples):
            t = step / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((x, y))
    return result


def draw_band(surface: pygame.Surface, points: list[tuple[float, float]], color, width: int) -> None:
    """Draw overlapping round segments so thick curves have no square gaps."""
    radius = width // 2
    for x, y in points:
        pygame.draw.circle(surface, color, (round(x), round(y)), radius)


class Track:
    def __init__(self, level: int = 1) -> None:
        if not 1 <= level <= len(ROAD_SPECS):
            raise ValueError("Road number must be between 1 and 10")
        self.level = level
        self.name, self.road_width, a3, a5, a7, a9, phase = ROAD_SPECS[level - 1]
        if level == 1:
            controls = [
                (215, 155), (430, 112), (665, 155), (775, 265),
                (780, 453), (675, 626), (470, 684), (251, 644),
                (127, 535), (120, 345),
            ]
            self.points = catmull_rom(controls)
        else:
            self.points = []
            radius_x, radius_y = (260, 220) if level >= 6 else (300, 248)
            for i in range(240):
                theta = -2.25 + 2 * math.pi * i / 240
                radius = (
                    1
                    + a3 * math.sin(3 * theta + phase)
                    + a5 * math.sin(5 * theta - 0.7 * phase)
                    + a7 * math.cos(7 * theta + 1.3 * phase)
                    + a9 * math.sin(9 * theta - 0.4 * phase)
                )
                self.points.append(
                    (450 + radius_x * radius * math.cos(theta),
                     400 + radius_y * radius * math.sin(theta))
                )
        self.count = len(self.points)
        self.lengths = []
        self.cumulative = [0.0]
        for i, (x, y) in enumerate(self.points):
            nx, ny = self.points[(i + 1) % self.count]
            length = math.hypot(nx - x, ny - y)
            self.lengths.append(length)
            self.cumulative.append(self.cumulative[-1] + length)
        self.length = self.cumulative[-1]

        road_alpha = pygame.Surface((WORLD_W, HEIGHT), pygame.SRCALPHA)
        draw_band(road_alpha, self.points, (255, 255, 255), self.road_width)
        self.mask = pygame.mask.from_surface(road_alpha)
        self.art = self._make_art()

    def on_road(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        return 0 <= ix < WORLD_W and 0 <= iy < HEIGHT and bool(self.mask.get_at((ix, iy)))

    def sense(self, x: float, y: float, angle: float, count: int) -> list[float]:
        # Every input neuron corresponds to one evenly spaced road-edge ray.
        spread = math.radians(105)
        values = []
        for i in range(count):
            ray_angle = angle + (-spread + 2 * spread * i / (count - 1))
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            distance = SENSOR_RANGE
            for step in range(6, SENSOR_RANGE + 1, 5):
                if not self.on_road(x + dx * step, y + dy * step):
                    distance = step
                    break
            values.append(distance / SENSOR_RANGE)
        return values

    def progress(self, x: float, y: float, near: int) -> tuple[float, int]:
        """Project a car onto nearby centerline segments for smooth lap progress."""
        best_distance = float("inf")
        best_progress = 0.0
        best_index = near
        for offset in range(-8, 9):
            i = (near + offset) % self.count
            ax, ay = self.points[i]
            bx, by = self.points[(i + 1) % self.count]
            vx, vy = bx - ax, by - ay
            segment_squared = vx * vx + vy * vy
            t = max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / segment_squared))
            distance = (x - ax - t * vx) ** 2 + (y - ay - t * vy) ** 2
            if distance < best_distance:
                best_distance = distance
                best_progress = self.cumulative[i] + t * self.lengths[i]
                best_index = i
        return best_progress, best_index

    def _make_art(self) -> pygame.Surface:
        art = pygame.Surface((WORLD_W, HEIGHT))
        art.fill(BG)
        for x in range(0, WORLD_W, 40):
            pygame.draw.line(art, (18, 40, 43), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 40):
            pygame.draw.line(art, (18, 40, 43), (0, y), (WORLD_W, y))
        rng = random.Random(29)
        for _ in range(150):
            x, y = rng.randrange(WORLD_W), rng.randrange(HEIGHT)
            if not self.on_road(x, y):
                radius = rng.randrange(2, 7)
                pygame.draw.circle(art, rng.choice([(23, 54, 49), (27, 60, 52), (18, 47, 43)]), (x, y), radius)

        draw_band(art, self.points, (7, 15, 22), self.road_width + 16)
        draw_band(art, self.points, (173, 191, 188), self.road_width + 8)
        draw_band(art, self.points, (49, 62, 69), self.road_width)
        for i in range(0, self.count, 5):
            p1 = self.points[i]
            p2 = self.points[(i + 2) % self.count]
            pygame.draw.line(art, (111, 126, 125), p1, p2, 2)

        # Start line across the road, with a simple checker pattern.
        x, y = self.points[0]
        nx, ny = self.points[1]
        angle = math.atan2(ny - y, nx - x)
        px, py = -math.sin(angle), math.cos(angle)
        for cell in range(-6, 6):
            start = cell * self.road_width / 12
            end = (cell + 1) * self.road_width / 12
            color = (240, 246, 237) if cell % 2 == 0 else (33, 42, 47)
            pygame.draw.line(
                art, color,
                (x + px * start, y + py * start),
                (x + px * end, y + py * end), 9,
            )
        return art


@dataclass
class Car:
    brain: Network
    x: float
    y: float
    angle: float
    speed: float = 45.0
    alive: bool = True
    time: float = 0.0
    progress: float = 0.0
    best_progress: float = 0.0
    nearest: int = 0
    sensors: list[float] = field(default_factory=list)
    steering: float = 0.0
    drive: float = 0.0

    @property
    def fitness(self) -> float:
        return self.best_progress + min(self.time, 10) * 0.05

    def update(self, track: Track, dt: float) -> None:
        if not self.alive:
            return
        self.time += dt
        self.sensors = track.sense(self.x, self.y, self.angle, self.brain.sizes[0])
        self.steering, self.drive = self.brain.forward(self.sensors)
        acceleration = 200 * max(0.0, self.drive) - 290 * max(0.0, -self.drive)
        self.speed = max(0.0, min(220.0, self.speed + (acceleration - 25 - 0.07 * self.speed) * dt))
        self.angle += self.steering * (1.25 + 0.005 * self.speed) * dt
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        if not track.on_road(self.x, self.y):
            self.alive = False
            return
        position, self.nearest = track.progress(self.x, self.y, self.nearest)
        current = self.progress % track.length
        delta = (position - current + track.length / 2) % track.length - track.length / 2
        if -15 < delta < 15:
            self.progress += delta
        self.best_progress = max(self.best_progress, self.progress)
        if self.time > 4.0 and self.best_progress < 10:
            self.alive = False
        if self.time > 8.0 and self.speed < 3:
            self.alive = False


class Race:
    def __init__(
        self, track: Track, inputs: int, hidden: list[int],
        max_runtime: int = DEFAULT_MAX_RUNTIME, time_limit_enabled: bool = True,
    ) -> None:
        self.track = track
        self.rng = random.Random()
        self.inputs = inputs
        self.hidden = hidden[:]
        self.max_runtime = max_runtime
        self.time_limit_enabled = time_limit_enabled
        self.sizes = (inputs, *hidden, 2)
        self.generation = 1
        self.elapsed = 0.0
        self.best_ever = 0.0
        self.history: list[float] = []
        self.cars: list[Car] = []
        self._spawn([Network.random(self.sizes, self.rng) for _ in range(POPULATION)])

    def _spawn(self, brains: list[Network]) -> None:
        x, y = self.track.points[0]
        nx, ny = self.track.points[1]
        angle = math.atan2(ny - y, nx - x)
        px, py = -math.sin(angle), math.cos(angle)
        start_spread = min(15, self.track.road_width * 0.18)
        self.cars = [
            Car(brain, x + px * self.rng.uniform(-start_spread, start_spread),
                y + py * self.rng.uniform(-start_spread, start_spread), angle)
            for brain in brains
        ]
        self.elapsed = 0.0

    def change_track(self, track: Track) -> None:
        """Re-run this generation on another circuit, keeping its networks."""
        self.track = track
        self.best_ever = 0.0
        self.history.clear()
        self._spawn([car.brain for car in self.cars])

    def update(self, dt: float) -> None:
        self.elapsed += dt
        for car in self.cars:
            car.update(self.track, dt)
        self.best_ever = max(self.best_ever, *(car.best_progress for car in self.cars))
        time_expired = self.time_limit_enabled and self.elapsed >= self.max_runtime
        if time_expired or not any(car.alive for car in self.cars):
            ranked = [(car.fitness, car.brain) for car in self.cars]
            self.history.append(max(score for score, _ in ranked))
            self.history = self.history[-40:]
            self._spawn(next_generation(ranked, self.rng, POPULATION))
            self.generation += 1

    @property
    def leader(self) -> Car | None:
        living = [car for car in self.cars if car.alive]
        return max(living, key=lambda car: car.best_progress) if living else None


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Neural Circuit")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Avenir Next", 17)
        self.small = pygame.font.SysFont("Avenir Next", 14)
        self.tiny = pygame.font.SysFont("Avenir Next", 12)
        self.bold = pygame.font.SysFont("Avenir Next", 20, bold=True)
        self.title = pygame.font.SysFont("Avenir Next", 32, bold=True)
        self.road_level = 1
        self.track = Track(self.road_level)
        self.inputs = 7
        self.hidden = [12, 12]
        self.max_runtime = DEFAULT_MAX_RUNTIME
        self.time_limit_enabled = True
        self.race = Race(self.track, self.inputs, self.hidden, self.max_runtime, self.time_limit_enabled)
        self.paused = False
        self.speed = 1
        self.show_sensors = True
        self.buttons: list[tuple[pygame.Rect, str]] = []

    def label(self, text: str, x: int, y: int, color=TEXT, font=None) -> None:
        self.screen.blit((font or self.font).render(text, True, color), (x, y))

    def button(self, rect: pygame.Rect, title: str, action: str, primary=False) -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = ACCENT if primary else ((45, 65, 73) if hovered else CARD)
        if primary and hovered:
            fill = (129, 247, 205)
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        color = (15, 35, 39) if primary else TEXT
        rendered = self.small.render(title, True, color)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))
        self.buttons.append((rect, action))

    def stepper(self, y: int, caption: str, value: int, key: str) -> None:
        x = WORLD_W + 23
        self.label(caption, x, y, MUTED, self.small)
        self.button(pygame.Rect(WIDTH - 122, y - 1, 26, 27), "−", f"{key}:-")
        value_text = self.font.render(str(value), True, TEXT)
        self.screen.blit(value_text, value_text.get_rect(center=(WIDTH - 77, y + 13)))
        self.button(pygame.Rect(WIDTH - 50, y - 1, 26, 27), "+", f"{key}:+")

    def handle_action(self, action: str) -> None:
        if action == "pause":
            self.paused = not self.paused
        elif action == "speed":
            self.speed = {1: 2, 2: 4, 4: 1}[self.speed]
        elif action == "limit":
            self.time_limit_enabled = not self.time_limit_enabled
            self.race.time_limit_enabled = self.time_limit_enabled
        elif action == "apply":
            self.race = Race(self.track, self.inputs, self.hidden, self.max_runtime, self.time_limit_enabled)
            self.paused = False
        else:
            key, sign = action.split(":")
            change = 1 if sign == "+" else -1
            if key == "road":
                self.road_level = (self.road_level - 1 + change) % len(ROAD_SPECS) + 1
                self.track = Track(self.road_level)
                self.race.change_track(self.track)
            elif key == "runtime":
                self.max_runtime = max(5, min(120, self.max_runtime + 5 * change))
                self.race.max_runtime = self.max_runtime
            elif key == "inputs":
                self.inputs = max(3, min(15, self.inputs + change))
            elif key == "layers":
                new_count = max(1, min(5, len(self.hidden) + change))
                while len(self.hidden) < new_count:
                    self.hidden.append(12)
                self.hidden = self.hidden[:new_count]
            elif key.startswith("hidden"):
                index = int(key[6:])
                self.hidden[index] = max(2, min(24, self.hidden[index] + change))

    def draw_cars(self) -> None:
        leader = self.race.leader
        if leader and self.show_sensors:
            for index, value in enumerate(leader.sensors):
                ray_angle = leader.angle + math.radians(-105 + 210 * index / (len(leader.sensors) - 1))
                length = value * SENSOR_RANGE
                end = (leader.x + math.cos(ray_angle) * length, leader.y + math.sin(ray_angle) * length)
                pygame.draw.line(self.screen, (73, 175, 162), (leader.x, leader.y), end, 1)
                pygame.draw.circle(self.screen, ACCENT, (int(end[0]), int(end[1])), 2)
        for car in self.race.cars:
            if not car.alive:
                continue
            forward = (math.cos(car.angle), math.sin(car.angle))
            side = (-forward[1], forward[0])
            points = [
                (car.x + forward[0] * 11, car.y + forward[1] * 11),
                (car.x - forward[0] * 8 + side[0] * 5, car.y - forward[1] * 8 + side[1] * 5),
                (car.x - forward[0] * 8 - side[0] * 5, car.y - forward[1] * 8 - side[1] * 5),
            ]
            color = ACCENT if car is leader else ORANGE
            pygame.draw.polygon(self.screen, (9, 22, 27), points, 5)
            pygame.draw.polygon(self.screen, color, points)

    def draw_world(self) -> None:
        self.screen.blit(self.track.art, (0, 0))
        self.draw_cars()
        pygame.draw.rect(self.screen, (11, 22, 30), (22, 22, 233, 88), border_radius=12)
        self.label("NEURAL CIRCUIT", 37, 32, ACCENT, self.small)
        self.label(f"Generation {self.race.generation:02d}", 37, 56, TEXT, self.bold)
        alive = sum(car.alive for car in self.race.cars)
        self.label(f"{alive:02d} / {POPULATION} cars on track", 38, 84, MUTED, self.small)

        pygame.draw.rect(self.screen, (11, 22, 30), (22, HEIGHT - 58, 580, 37), border_radius=8)
        self.label("P  pause     R  restart     V  sensors     TAB  speed     ARROWS  road", 36, HEIGHT - 49, MUTED, self.small)
        if self.paused:
            shade = pygame.Surface((WORLD_W, HEIGHT), pygame.SRCALPHA)
            shade.fill((5, 15, 20, 100))
            self.screen.blit(shade, (0, 0))
            text = self.title.render("PAUSED", True, TEXT)
            self.screen.blit(text, text.get_rect(center=(WORLD_W // 2, HEIGHT // 2)))

    def draw_panel(self) -> None:
        x = WORLD_W
        self.buttons.clear()
        pygame.draw.rect(self.screen, PANEL, (x, 0, PANEL_W, HEIGHT))
        pygame.draw.line(self.screen, (52, 76, 78), (x, 0), (x, HEIGHT), 2)
        self.label("CONTROL ROOM", x + 23, 18, ACCENT, self.small)
        self.label("Train the drivers", x + 22, 39, TEXT, self.title)
        self.label("Brains evolve after each race.", x + 24, 80, MUTED, self.small)

        self.button(pygame.Rect(x + 22, 109, 146, 32), "RESUME" if self.paused else "PAUSE", "pause")
        self.button(pygame.Rect(x + 177, 109, 160, 32), f"SPEED  {self.speed}×", "speed")
        pygame.draw.line(self.screen, (46, 65, 72), (x + 22, 151), (WIDTH - 22, 151))

        alive = sum(car.alive for car in self.race.cars)
        self.label("GENERATION", x + 23, 162, MUTED, self.tiny)
        self.label("BEST ON THIS ROAD", x + 184, 162, MUTED, self.tiny)
        self.label(str(self.race.generation), x + 23, 178, TEXT, self.bold)
        self.label(f"{self.race.best_ever / 10:.0f} m", x + 184, 178, TEXT, self.bold)
        self.label("CARS RUNNING", x + 23, 215, MUTED, self.tiny)
        self.label("LAPS", x + 184, 215, MUTED, self.tiny)
        self.label(f"{alive} / {POPULATION}", x + 23, 231, TEXT, self.font)
        self.label(f"{self.race.best_ever / self.track.length:.1f}", x + 184, 231, TEXT, self.font)

        pygame.draw.line(self.screen, (46, 65, 72), (x + 22, 263), (WIDTH - 22, 263))
        self.label("ROAD & RACE", x + 23, 274, ACCENT, self.small)
        self.button(pygame.Rect(x + 22, 301, 38, 32), "‹", "road:-")
        self.button(pygame.Rect(x + 300, 301, 38, 32), "›", "road:+")
        road_number = self.bold.render(f"ROAD {self.road_level:02d} / 10", True, TEXT)
        self.screen.blit(road_number, road_number.get_rect(center=(x + 180, 317)))
        road_name = self.small.render(self.track.name, True, MUTED)
        self.screen.blit(road_name, road_name.get_rect(center=(x + 180, 345)))
        self.stepper(374, "Max runtime (seconds)", self.max_runtime, "runtime")
        self.button(
            pygame.Rect(x + 22, 407, 133, 27),
            "LIMIT ON" if self.time_limit_enabled else "LIMIT OFF", "limit",
            primary=not self.time_limit_enabled,
        )
        heat_text = (
            f"Heat {self.race.elapsed:.1f} / {self.max_runtime} s"
            if self.time_limit_enabled else f"Heat {self.race.elapsed:.1f} s"
        )
        self.label(heat_text, x + 166, 411, MUTED, self.small)

        pygame.draw.line(self.screen, (46, 65, 72), (x + 22, 443), (WIDTH - 22, 443))
        self.label("NETWORK SETUP", x + 23, 451, ACCENT, self.small)
        self.stepper(477, "Input neurons / sensors", self.inputs, "inputs")
        self.stepper(509, "Hidden layers", len(self.hidden), "layers")
        self.label("NEURONS IN EACH HIDDEN LAYER", x + 23, 543, MUTED, self.tiny)
        for i, size in enumerate(self.hidden):
            self.stepper(563 + i * 29, f"Layer {i + 1}", size, f"hidden{i}")

        self.label("OUTPUTS", x + 23, 711, MUTED, self.tiny)
        self.label("steer: left / right    •    drive: speed / brake", x + 23, 727, TEXT, self.small)
        dirty = self.inputs != self.race.inputs or self.hidden != self.race.hidden
        self.button(
            pygame.Rect(x + 22, 755, PANEL_W - 44, 38),
            "APPLY & RESTART" if dirty else "RESTART TRAINING", "apply", primary=True,
        )

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.handle_action("pause")
                    elif event.key == pygame.K_r:
                        self.handle_action("apply")
                    elif event.key == pygame.K_v:
                        self.show_sensors = not self.show_sensors
                    elif event.key == pygame.K_TAB:
                        self.handle_action("speed")
                    elif event.key == pygame.K_LEFT:
                        self.handle_action("road:-")
                    elif event.key == pygame.K_RIGHT:
                        self.handle_action("road:+")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, action in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.handle_action(action)
                            break

            if not self.paused:
                for _ in range(self.speed):
                    self.race.update(1 / FPS)
            self.draw_world()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == "__main__":
    App().run()
