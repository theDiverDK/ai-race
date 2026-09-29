"""Neural Circuit: a Pygame race track with evolving neural drivers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pygame
from pygame._sdl2.video import Window

from inspector import NetworkInspector
from neural import Network, load_networks, next_generation, save_networks


WORLD_W, HEIGHT, PANEL_W = 900, 800, 360
WIDTH = WORLD_W + PANEL_W
FPS = 60
POPULATION = 50
DEFAULT_MAX_RUNTIME = 25
DEFAULT_HIDDEN_WIDTH = 16
SENSOR_RANGE = 175
LAPS_PER_ROAD = 5
FINAL_ROAD_LAPS = 10
FINAL_ROAD_RUNTIME = 60
# A challenger must be this much faster (or finish when the champion has not)
# before it takes the title, so a near-identical clone cannot reset progress.
BEAT_MARGIN = 0.03
SENSOR_NOISE = 0.02
START_ANGLE_JITTER = 0.10
SAVE_PATH = Path(__file__).with_name("best_network.json")

# Each new circuit combines different bends. Selected circuits narrow at the
# listed angular positions; the first circuit keeps its original outline.
ROAD_SPECS = [
    ("Open Oval", 104, (), (), 0.0),
    ("Gentle Chicane", 100, ((3, .075, .2), (5, .045, 1.4)), (), 0.0),
    ("Twin Esses", 96, ((4, .110, 1.1), (7, .055, .3)), (), 0.0),
    ("Clover Bend", 92, ((3, .185, 2.2), (6, .065, .1)), (), 0.0),
    ("Offset Slalom", 88, ((5, .165, .5), (2, .085, 1.8)), (), 0.0),
    ("Pinch Point", 84, ((4, .140, 1.7), (7, .065, 2.6)), (.4, 3.4), .34),
    ("Sawtooth Sweep", 80, ((6, .120, .4), (3, .100, 2.0)), (2.1,), .38),
    ("Serpentine", 76, ((5, .155, 1.9), (8, .070, .8)), (1.0, 4.1), .40),
    ("Narrow Gates", 72, ((7, .140, 1.2), (4, .090, 2.8)), (.6, 2.8, 5.0), .44),
    ("Switchback Circuit", 68, ((9, .075, .7), (5, .105, 2.3)), (1.5, 3.7, 5.6), .46),
    ("Hairpin Ladder", 58, (), (), 0.0),
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


def draw_road_band(
    surface: pygame.Surface, points: list[tuple[float, float]],
    widths: list[int], color, padding: int = 0,
) -> None:
    """Paint a road whose width can change along its centerline."""
    for (x, y), width in zip(points, widths):
        pygame.draw.circle(surface, color, (round(x), round(y)), (width + padding) // 2)


def pinched_width(width: int, theta: float, centers: tuple[float, ...], depth: float) -> int:
    """Cosine-shaped bottlenecks blend smoothly into the normal road width."""
    half_span = .34
    reduction = 0.0
    for center in centers:
        distance = abs((theta - center + math.pi) % (2 * math.pi) - math.pi)
        if distance < half_span:
            reduction = max(reduction, depth * (1 + math.cos(math.pi * distance / half_span)) / 2)
    return round(width * (1 - reduction))


class Track:
    def __init__(self, level: int = 1) -> None:
        if not 1 <= level <= len(ROAD_SPECS):
            raise ValueError(f"Road number must be between 1 and {len(ROAD_SPECS)}")
        self.level = level
        self.name, self.road_width, waves, pinch_centers, pinch_depth = ROAD_SPECS[level - 1]
        if level == 1:
            controls = [
                (215, 155), (430, 112), (665, 155), (775, 265),
                (780, 453), (675, 626), (470, 684), (251, 644),
                (127, 535), (120, 345),
            ]
            self.points = catmull_rom(controls)
            self.widths = [self.road_width] * len(self.points)
        elif level == 11:
            # Six long vertical legs, five alternating U-turns, and a low
            # return straight that closes the circuit without crossing it.
            controls = [
                (130, 610), (130, 400), (130, 200),
                (145, 135), (185, 105), (225, 135), (240, 200),
                (240, 400), (240, 530),
                (255, 595), (295, 625), (335, 595), (350, 530),
                (350, 400), (350, 200),
                (365, 135), (405, 105), (445, 135), (460, 200),
                (460, 400), (460, 530),
                (475, 595), (515, 625), (555, 595), (570, 530),
                (570, 400), (570, 200),
                (585, 135), (625, 105), (665, 135), (680, 200),
                (680, 400), (680, 610), (695, 690), (660, 735),
                (520, 735), (350, 735), (180, 735), (130, 690),
            ]
            self.points = catmull_rom(controls)
            self.widths = [self.road_width] * len(self.points)
        else:
            self.points = []
            self.widths = []
            for i in range(320):
                theta = -2.25 + 2 * math.pi * i / 320
                radius = 1 + sum(
                    amplitude * math.sin(frequency * theta + phase)
                    for frequency, amplitude, phase in waves
                )
                self.points.append(
                    (450 + 260 * radius * math.cos(theta),
                     400 + 215 * radius * math.sin(theta))
                )
                self.widths.append(pinched_width(self.road_width, theta, pinch_centers, pinch_depth))
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
        draw_road_band(road_alpha, self.points, self.widths, (255, 255, 255))
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

        draw_road_band(art, self.points, self.widths, (7, 15, 22), 16)
        draw_road_band(art, self.points, self.widths, (173, 191, 188), 8)
        draw_road_band(art, self.points, self.widths, (49, 62, 69))
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
            start = cell * self.widths[0] / 12
            end = (cell + 1) * self.widths[0] / 12
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
    distance_travelled: float = 0.0
    progress: float = 0.0
    best_progress: float = 0.0
    laps_completed: int = 0
    fastest_lap: float | None = None
    last_lap_crossing_time: float = 0.0
    nearest: int = 0
    sensors: list[float] = field(default_factory=list)
    steering: float = 0.0
    drive: float = 0.0
    brake: float = 0.0
    sensor_noise: float = SENSOR_NOISE

    @property
    def rank_key(self) -> tuple[int, float, float]:
        if self.fastest_lap is not None:
            return 1, -self.fastest_lap, self.best_progress
        return 0, self.best_progress, min(self.time, 10) * 0.05

    def update(self, track: Track, dt: float) -> None:
        if not self.alive:
            return
        self.time += dt
        self.sensors = track.sense(self.x, self.y, self.angle, self.brain.sizes[0])
        if self.sensor_noise:
            # Noisy eyes stop networks memorising exact distances on one road.
            self.sensors = [
                min(1.0, max(0.0, value + random.gauss(0, self.sensor_noise)))
                for value in self.sensors
            ]
        self.steering, self.drive, self.brake = self.brain.forward(self.sensors)
        # Drive is signed: negative accelerates backwards. Brake always acts
        # against the current motion and cannot reverse the car by itself.
        self.speed += 200 * self.drive * dt
        drag = (25 + 0.07 * abs(self.speed)) * dt
        braking = 290 * max(0.0, self.brake) * dt
        if self.speed > 0:
            self.speed = max(0.0, self.speed - drag - braking)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + drag + braking)
        self.speed = max(-110.0, min(220.0, self.speed))
        self.distance_travelled += abs(self.speed) * dt
        turn_direction = -1 if self.speed < 0 else 1
        self.angle += self.steering * (1.25 + 0.005 * abs(self.speed)) * turn_direction * dt
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        if not track.on_road(self.x, self.y):
            self.alive = False
            return
        position, self.nearest = track.progress(self.x, self.y, self.nearest)
        previous_progress = self.progress
        current = self.progress % track.length
        delta = (position - current + track.length / 2) % track.length - track.length / 2
        if -15 < delta < 15:
            self.progress += delta
        while self.progress >= (self.laps_completed + 1) * track.length:
            finish_distance = (self.laps_completed + 1) * track.length
            finish_fraction = (finish_distance - previous_progress) / (self.progress - previous_progress)
            finish_time = self.time - dt + finish_fraction * dt
            lap_time = finish_time - self.last_lap_crossing_time
            self.fastest_lap = lap_time if self.fastest_lap is None else min(self.fastest_lap, lap_time)
            self.last_lap_crossing_time = finish_time
            self.laps_completed += 1
        self.best_progress = max(self.best_progress, self.progress)
        if self.time > 4.0 and self.distance_travelled < 10:
            self.alive = False
        if self.time > 8.0 and abs(self.speed) < 3:
            self.alive = False


class Race:
    """Runs heats and tracks the champion through the gauntlet.

    The champion (slot 0 of every generation) must finish 5 laps on each road
    in turn, 10 on the final road, then faces random roads. A crash or a
    clearly better challenger crowns a new champion and restarts at road 1.
    """

    def __init__(
        self, track: Track, inputs: int, hidden: list[int],
        max_runtime: int = DEFAULT_MAX_RUNTIME, time_limit_enabled: bool = True,
        seed_networks: list[Network] | None = None, saved_tracks: int = 0,
        save_path: Path | None = None,
    ) -> None:
        self.track = track
        self.rng = random.Random()
        self.inputs = inputs
        self.hidden = hidden[:]
        self.max_runtime = max_runtime
        self.time_limit_enabled = time_limit_enabled
        self.sizes = (inputs, *hidden, 3)
        self.generation = 1
        self.elapsed = 0.0
        self.best_ever = 0.0
        self.fastest_ever: float | None = None
        self.history: list[float] = []
        self.cars: list[Car] = []
        self.champion: Network | None = None
        self.tracks_completed = 0
        self.laps_banked = 0  # champion laps finished on this road in earlier heats
        self.random_phase = False
        self.saved_tracks = saved_tracks
        self.save_path = save_path
        self._tracks = {track.level: track}
        if seed_networks:
            ranked = [((0, -index, 0.0), network) for index, network in enumerate(seed_networks)]
            brains = next_generation(ranked, self.rng, POPULATION)
            self.champion = brains[0]
        else:
            brains = [Network.random(self.sizes, self.rng) for _ in range(POPULATION)]
        self._spawn(brains)

    def _spawn(self, brains: list[Network]) -> None:
        x, y = self.track.points[0]
        nx, ny = self.track.points[1]
        base_angle = math.atan2(ny - y, nx - x)
        px, py = -math.sin(base_angle), math.cos(base_angle)
        start_spread = min(15, self.track.road_width * 0.18)
        self.cars = [
            Car(brain, x + px * self.rng.uniform(-start_spread, start_spread),
                y + py * self.rng.uniform(-start_spread, start_spread),
                base_angle + self.rng.uniform(-START_ANGLE_JITTER, START_ANGLE_JITTER))
            for brain in brains
        ]
        self.elapsed = 0.0

    def track_for(self, level: int) -> Track:
        if level not in self._tracks:
            self._tracks[level] = Track(level)
        return self._tracks[level]

    @property
    def laps_required(self) -> int:
        return FINAL_ROAD_LAPS if self.track.level == len(ROAD_SPECS) else LAPS_PER_ROAD

    @property
    def heat_limit(self) -> int:
        if self.track.level == len(ROAD_SPECS):
            return max(FINAL_ROAD_RUNTIME, self.max_runtime)
        return self.max_runtime

    @property
    def champion_car(self) -> Car | None:
        return self.cars[0] if self.champion is not None else None

    @property
    def champion_laps(self) -> int:
        car = self.champion_car
        return self.laps_banked + (car.laps_completed if car else 0)

    def _reset_gauntlet(self) -> None:
        self.tracks_completed = 0
        self.laps_banked = 0
        self.random_phase = False

    def _beats(self, challenger: Car, champion: Car) -> bool:
        if challenger.fastest_lap is not None:
            return champion.fastest_lap is None or challenger.fastest_lap < champion.fastest_lap * (1 - BEAT_MARGIN)
        if champion.fastest_lap is not None:
            return False
        return challenger.best_progress > champion.best_progress * (1 + BEAT_MARGIN)

    def _save_champion(self, networks: list[Network]) -> None:
        """Persist the champion, but only for a new record of completed tracks."""
        if not self.save_path or self.tracks_completed <= self.saved_tracks:
            return
        try:
            save_networks(self.save_path, networks, self.tracks_completed)
        except OSError:
            return
        self.saved_tracks = self.tracks_completed

    def change_track(self, track: Track) -> None:
        """Jump to another road: the heat restarts and the gauntlet begins there."""
        self.track = track
        self._tracks.setdefault(track.level, track)
        self.best_ever = 0.0
        self.fastest_ever = None
        self.history.clear()
        self._reset_gauntlet()
        self._spawn([car.brain for car in self.cars])

    def update(self, dt: float) -> None:
        self.elapsed += dt
        for car in self.cars:
            car.update(self.track, dt)
        self.best_ever = max(self.best_ever, *(car.best_progress for car in self.cars))
        for car in self.cars:
            if car.fastest_lap is not None:
                self.fastest_ever = (
                    car.fastest_lap if self.fastest_ever is None
                    else min(self.fastest_ever, car.fastest_lap)
                )
        goal = self.champion is not None and self.champion_laps >= self.laps_required
        time_expired = self.time_limit_enabled and self.elapsed >= self.heat_limit
        if goal or time_expired or not any(car.alive for car in self.cars):
            self._end_heat(goal)

    def _end_heat(self, goal: bool) -> None:
        ranked = [(car.rank_key, car.brain) for car in self.cars]
        self.history.append(max(car.best_progress for car in self.cars))
        self.history = self.history[-40:]
        brains = next_generation(ranked, self.rng, POPULATION)
        champion_car = self.champion_car
        next_level = 1
        if champion_car is None:
            self.champion = brains[0]
            self._reset_gauntlet()
        else:
            best = max(self.cars, key=lambda car: car.rank_key)
            crashed = not champion_car.alive and not goal
            if crashed or (best is not champion_car and self._beats(best, champion_car)):
                # Crash, or a clearly better car: new champion, back to road 1.
                self.champion = brains[0]
                self._reset_gauntlet()
            else:
                brains[0] = self.champion  # the champion always survives unchanged
                next_level = self.track.level
                if goal:
                    self.tracks_completed += 1
                    self.laps_banked = 0
                    self._save_champion([self.champion, brains[1]])
                    if self.random_phase or self.track.level == len(ROAD_SPECS):
                        self.random_phase = True
                        next_level = self.rng.randint(1, len(ROAD_SPECS))
                    else:
                        next_level = self.track.level + 1
                else:
                    self.laps_banked += champion_car.laps_completed
        if next_level != self.track.level:
            self.track = self.track_for(next_level)
            self.best_ever = 0.0
            self.fastest_ever = None
            self.history.clear()
        self._spawn(brains)
        self.generation += 1

    @property
    def leader(self) -> Car | None:
        living = [car for car in self.cars if car.alive]
        return max(living, key=lambda car: car.rank_key) if living else None

    @property
    def best_car(self) -> Car:
        return max(self.cars, key=lambda car: car.rank_key)


class App:
    def __init__(self, save_path: Path | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Neural Circuit")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.main_window = Window.from_display_module()
        self.main_window_id = self.main_window.id
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Avenir Next", 17)
        self.small = pygame.font.SysFont("Avenir Next", 14)
        self.tiny = pygame.font.SysFont("Avenir Next", 12)
        self.bold = pygame.font.SysFont("Avenir Next", 20, bold=True)
        self.title = pygame.font.SysFont("Avenir Next", 32, bold=True)
        self.save_path = save_path or SAVE_PATH
        self.inputs = 7
        self.hidden = [DEFAULT_HIDDEN_WIDTH, DEFAULT_HIDDEN_WIDTH]
        self.max_runtime = DEFAULT_MAX_RUNTIME
        self.time_limit_enabled = True
        # Resume from the saved champions when possible, otherwise start from scratch.
        saved = load_networks(self.save_path)
        seed, saved_tracks = None, 0
        if saved:
            seed, saved_tracks = saved
            self.inputs, self.hidden = seed[0].sizes[0], list(seed[0].sizes[1:-1])
        self.road_level = 1  # the gauntlet always starts on road 1
        self.track = Track(self.road_level)
        self.race = Race(
            self.track, self.inputs, self.hidden, self.max_runtime, self.time_limit_enabled,
            seed, saved_tracks, self.save_path,
        )
        self.paused = False
        self.speed = 4
        self.show_sensors = True
        self.buttons: list[tuple[pygame.Rect, str]] = []
        self.inspector: NetworkInspector | None = None

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
            self.speed = {1: 2, 2: 4, 4: 8, 8: 1}[self.speed]
        elif action == "limit":
            self.time_limit_enabled = not self.time_limit_enabled
            self.race.time_limit_enabled = self.time_limit_enabled
        elif action == "apply":
            # A fresh start on road 1; the saved file is only replaced by a new record.
            self.road_level = 1
            self.track = Track(self.road_level)
            self.race = Race(
                self.track, self.inputs, self.hidden, self.max_runtime, self.time_limit_enabled,
                saved_tracks=self.race.saved_tracks, save_path=self.save_path,
            )
            self.paused = False
        elif action == "inspect":
            if self.inspector is None:
                self.inspector = NetworkInspector()
            else:
                self.inspector.window.focus()
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
                    self.hidden.append(DEFAULT_HIDDEN_WIDTH)
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
        pygame.draw.rect(self.screen, (11, 22, 30), (22, 22, 233, 140), border_radius=12)
        self.label("NEURAL CIRCUIT", 37, 32, ACCENT, self.small)
        self.label(f"Generation {self.race.generation:02d}", 37, 56, TEXT, self.bold)
        alive = sum(car.alive for car in self.race.cars)
        self.label(f"{alive:02d} / {POPULATION} cars on track", 38, 84, MUTED, self.small)
        self.label(f"Tracks completed: {self.race.tracks_completed}", 37, 108, ACCENT, self.font)
        lap_text = f"Lap {min(self.race.champion_laps, self.race.laps_required)}/{self.race.laps_required}"
        self.label(lap_text + (" · random roads" if self.race.random_phase else " on this road"), 38, 133, MUTED, self.small)

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
        best_label = "FASTEST LAP" if self.race.fastest_ever is not None else "BEST DISTANCE"
        self.label(best_label, x + 184, 162, MUTED, self.tiny)
        self.label(str(self.race.generation), x + 23, 178, TEXT, self.bold)
        best_value = (
            f"{self.race.fastest_ever:.2f} s" if self.race.fastest_ever is not None
            else f"{self.race.best_ever / 10:.0f} m"
        )
        self.label(best_value, x + 184, 178, TEXT, self.bold)
        self.label("CARS RUNNING", x + 23, 215, MUTED, self.tiny)
        self.label("LAPS", x + 184, 215, MUTED, self.tiny)
        self.label(f"{alive} / {POPULATION}", x + 23, 231, TEXT, self.font)
        self.label(f"{self.race.best_ever / self.track.length:.1f}", x + 184, 231, TEXT, self.font)

        pygame.draw.line(self.screen, (46, 65, 72), (x + 22, 263), (WIDTH - 22, 263))
        self.label("ROAD & RACE", x + 23, 274, ACCENT, self.small)
        self.button(pygame.Rect(x + 22, 301, 38, 32), "‹", "road:-")
        self.button(pygame.Rect(x + 300, 301, 38, 32), "›", "road:+")
        road_number = self.bold.render(f"ROAD {self.road_level:02d} / {len(ROAD_SPECS)}", True, TEXT)
        self.screen.blit(road_number, road_number.get_rect(center=(x + 180, 317)))
        road_name = self.small.render(self.track.name, True, MUTED)
        self.screen.blit(road_name, road_name.get_rect(center=(x + 180, 345)))
        auto_text = (
            f"Tracks completed: {self.race.tracks_completed}  •  "
            f"lap {min(self.race.champion_laps, self.race.laps_required)}/{self.race.laps_required}"
            + ("  •  random roads" if self.race.random_phase else "")
        )
        self.label(auto_text, x + 23, 357, MUTED, self.tiny)
        self.stepper(374, "Max runtime (seconds)", self.max_runtime, "runtime")
        self.button(
            pygame.Rect(x + 22, 407, 133, 27),
            "LIMIT ON" if self.time_limit_enabled else "LIMIT OFF", "limit",
            primary=not self.time_limit_enabled,
        )
        heat_text = (
            f"Heat {self.race.elapsed:.1f} / {self.race.heat_limit} s"
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
        self.label("steer: L/R  •  drive: F/R  •  brake: slow/stop", x + 23, 727, TEXT, self.small)
        dirty = self.inputs != self.race.inputs or self.hidden != self.race.hidden
        self.button(pygame.Rect(x + 22, 755, 153, 38), "VIEW NETWORK", "inspect")
        self.button(
            pygame.Rect(x + 184, 755, 154, 38),
            "APPLY & RESTART" if dirty else "RESTART TRAINING", "apply", primary=True,
        )

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                window = getattr(event, "window", None)
                window_id = getattr(window, "id", window)
                if event.type == pygame.QUIT:
                    running = False
                elif self.inspector and window_id == self.inspector.window.id:
                    if not self.inspector.handle_event(event):
                        self.inspector.close()
                        self.inspector = None
                elif event.type == pygame.WINDOWCLOSE and window_id == self.main_window_id:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if window_id not in (None, self.main_window_id):
                        continue
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
                    if window_id not in (None, self.main_window_id):
                        continue
                    for rect, action in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.handle_action(action)
                            break

            if not self.paused:
                for _ in range(self.speed):
                    self.race.update(1 / FPS)
                if self.track is not self.race.track:
                    self.track = self.race.track
                    self.road_level = self.track.level
            self.draw_world()
            self.draw_panel()
            pygame.display.flip()
            if self.inspector:
                self.inspector.draw(self.race, self.paused)
            self.clock.tick(FPS)
        if self.inspector:
            self.inspector.close()
        pygame.quit()


if __name__ == "__main__":
    App().run()
