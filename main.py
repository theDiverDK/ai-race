"""Neural Circuit: a Pygame race track with evolving neural drivers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pygame
from pygame._sdl2.video import Window

from inspector import NetworkInspector
from neural import Network, load_networks, next_generation, save_networks, slot_role


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
# A challenger takes the title only when its fitness is more than (1 + this) times the
# champion's. Fitness is noisy and the population keeps improving, so a small margin
# (3%) made nearly every heat a takeover and the run never got past road 1.
BEAT_MARGIN = 1.0
# Score: each finished lap earns lap points plus a pace bonus, and the lap in
# progress earns a fraction. Later roads are worth more. A champion's run score
# adds this up across roads, so a higher score means a better model.
SCORE_LAP_POINTS = 100
SCORE_PACE_POINTS = 50
SCORE_ROAD_BONUS = 0.10
TOP_SPEED = 220.0
HOW_MADE = {
    "champion": "the unchanged champion",
    "runner_up": "the runner-up, an unchanged copy of the 2nd best",
    "mutant": "a lightly mutated copy of a top car",
    "child": "a bred child of two top cars (crossover + mutation)",
    "newcomer": "a brand-new random network",
}
# Selection fitness. Every brain also drives probe roads (other orientations of
# the same road, plus a nearby road) in the same heat, and is judged on its
# distance there too, so it cannot win by learning one road or one direction.
FITNESS_MEAN_WEIGHT = 0.5  # the rest goes to the worst road, so weak spots hurt
REFERENCE_SPEED = 0.5 * 220.0  # px/s used to scale distance to about 0..1
ORIENTATIONS = [(False, False), (True, False), (False, True), (True, True)]
ORIENTATION_NAMES = {
    (False, False): "", (True, False): "mirrored",
    (False, True): "reversed", (True, True): "mirrored + reversed",
}
# The brain also feels its own speed (signed, scaled by top speed): without it
# a network cannot tell how hard to brake for a bend it is approaching.
EXTRA_INPUTS = 1
# Every brain also drives PROBE_COUNT probe roads per heat. They are drawn from the
# hard roads (PROBE_MIN_LEVEL and up) in any orientation, whatever road the gauntlet
# is on. Training only on easy roads first teaches "always full speed"; measured over
# an hour of training, hard probes from the start lift the average share of a lap a
# champion drives on 22 test roads from about 14.5 to about 19.
PROBE_COUNT = 2
PROBE_MIN_LEVEL = 5
# A car that does not gain STAGNATION_DISTANCE px of forward progress for
# STAGNATION_SECONDS is out. Without this, a network can drive in circles (or back
# and forth) on the road forever: it never crashes, so a champion doing it is never
# replaced and the run, and its score, freeze on that road.
STAGNATION_SECONDS = 4.0
STAGNATION_DISTANCE = 30.0
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
    def __init__(self, level: int = 1, mirror: bool = False, reverse: bool = False) -> None:
        if not 1 <= level <= len(ROAD_SPECS):
            raise ValueError(f"Road number must be between 1 and {len(ROAD_SPECS)}")
        self.level = level
        self.mirror = mirror
        self.reverse = reverse
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
        # Mirroring swaps every left turn for a right turn; reversing drives the
        # same road the other way round. Together they give four roads from one.
        if mirror:
            self.points = [(WORLD_W - x, y) for x, y in self.points]
        if reverse:
            self.points = self.points[::-1]
            self.widths = self.widths[::-1]
        self.count = len(self.points)
        self.lengths = []
        self.cumulative = [0.0]
        for i, (x, y) in enumerate(self.points):
            nx, ny = self.points[(i + 1) % self.count]
            length = math.hypot(nx - x, ny - y)
            self.lengths.append(length)
            self.cumulative.append(self.cumulative[-1] + length)
        self.length = self.cumulative[-1]

        segments = []
        for i, (ax, ay) in enumerate(self.points):
            bx, by = self.points[(i + 1) % self.count]
            vx, vy = bx - ax, by - ay
            segments.append((i, ax, ay, vx, vy, vx * vx + vy * vy,
                             self.cumulative[i], self.lengths[i]))
        self._progress_windows = [
            tuple(segments[(near + offset) % self.count] for offset in range(-8, 9))
            for near in range(self.count)
        ]

        road_alpha = pygame.Surface((WORLD_W, HEIGHT), pygame.SRCALPHA)
        draw_road_band(road_alpha, self.points, self.widths, (255, 255, 255))
        self.mask = pygame.mask.from_surface(road_alpha)
        # Road pixels are either transparent or fully opaque. A byte lookup avoids
        # allocating a coordinate tuple and crossing into Mask.get_at for each ray sample.
        self._road_pixels = pygame.image.tobytes(road_alpha, "RGBA")[3::4]
        self._art: pygame.Surface | None = None
        suffix = ORIENTATION_NAMES[(mirror, reverse)]
        self.label = f"Road {level}" + (f" {suffix}" if suffix else "")
        if suffix:
            self.name = f"{self.name} ({suffix})"

    @property
    def art(self) -> pygame.Surface:
        """Drawn on first use: probe roads are only driven, never shown."""
        if self._art is None:
            self._art = self._make_art()
        return self._art

    def on_road(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        return 0 <= ix < WORLD_W and 0 <= iy < HEIGHT and bool(self.mask.get_at((ix, iy)))

    def sense(self, x: float, y: float, angle: float, count: int) -> list[float]:
        # Every input neuron corresponds to one evenly spaced road-edge ray.
        spread = math.radians(105)
        values = []
        pixels = self._road_pixels
        width, height = WORLD_W, HEIGHT
        steps = range(6, SENSOR_RANGE + 1, 5)
        for i in range(count):
            ray_angle = angle + (-spread + 2 * spread * i / (count - 1))
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            distance = SENSOR_RANGE
            for step in steps:
                ix, iy = int(x + dx * step), int(y + dy * step)
                if not (0 <= ix < width and 0 <= iy < height and pixels[iy * width + ix]):
                    distance = step
                    break
            values.append(distance / SENSOR_RANGE)
        return values

    def progress(self, x: float, y: float, near: int) -> tuple[float, int]:
        """Project a car onto nearby centerline segments for smooth lap progress."""
        best_distance = float("inf")
        best_progress = 0.0
        best_index = near
        for i, ax, ay, vx, vy, segment_squared, start, length in self._progress_windows[near]:
            t = ((x - ax) * vx + (y - ay) * vy) / segment_squared
            if t <= 0.0:
                t = 0.0
            elif t >= 1.0:
                t = 1.0
            distance = (x - ax - t * vx) ** 2 + (y - ay - t * vy) ** 2
            if distance < best_distance:
                best_distance = distance
                best_progress = start + t * length
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
    lap_times: list[float] = field(default_factory=list)
    last_lap_crossing_time: float = 0.0
    nearest: int = 0
    sensors: list[float] = field(default_factory=list)
    inputs: list[float] = field(default_factory=list)  # what the brain actually sees
    steering: float = 0.0
    drive: float = 0.0
    brake: float = 0.0
    sensor_noise: float = SENSOR_NOISE
    progress_mark: float = 0.0  # best_progress when it last moved forward enough
    mark_time: float = 0.0

    @property
    def rank_key(self) -> tuple[int, float, float]:
        if self.fastest_lap is not None:
            return 1, -self.fastest_lap, self.best_progress
        return 0, self.best_progress, min(self.time, 10) * 0.05

    def lap_score(self, track: Track) -> float:
        """Points for finished laps only: lap points plus a pace bonus per lap."""
        ideal = track.length / TOP_SPEED
        laps = sum(SCORE_LAP_POINTS + SCORE_PACE_POINTS * min(1.0, ideal / t) for t in self.lap_times)
        return laps * (1 + SCORE_ROAD_BONUS * (track.level - 1))

    def score_on(self, track: Track) -> float:
        """Finished laps plus the fraction of the current lap driven so far."""
        partial = max(0.0, self.best_progress / track.length - self.laps_completed)
        weight = 1 + SCORE_ROAD_BONUS * (track.level - 1)
        return self.lap_score(track) + SCORE_LAP_POINTS * min(partial, 1.0) * weight

    def update(self, track: Track, dt: float) -> None:
        if not self.alive:
            return
        self.time += dt
        self.sensors = track.sense(self.x, self.y, self.angle, self.brain.sizes[0] - EXTRA_INPUTS)
        if self.sensor_noise:
            # Noisy eyes stop networks memorising exact distances on one road.
            self.sensors = [
                min(1.0, max(0.0, value + random.gauss(0, self.sensor_noise)))
                for value in self.sensors
            ]
        self.inputs = self.sensors + [self.speed / TOP_SPEED] * EXTRA_INPUTS
        self.steering, self.drive, self.brake = self.brain.forward(self.inputs)
        # Drive is signed: negative accelerates backwards. Brake always acts
        # against the current motion and cannot reverse the car by itself.
        self.speed += 200 * self.drive * dt
        drag = (25 + 0.07 * abs(self.speed)) * dt
        braking = 290 * max(0.0, self.brake) * dt
        if self.speed > 0:
            self.speed = max(0.0, self.speed - drag - braking)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + drag + braking)
        self.speed = max(-110.0, min(TOP_SPEED, self.speed))
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
            self.lap_times.append(lap_time)
            self.last_lap_crossing_time = finish_time
            self.laps_completed += 1
        self.best_progress = max(self.best_progress, self.progress)
        if self.best_progress > self.progress_mark + STAGNATION_DISTANCE:
            self.progress_mark = self.best_progress
            self.mark_time = self.time
        elif self.time - self.mark_time > STAGNATION_SECONDS:
            self.alive = False
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
        seed_networks: list[Network] | None = None, record_tracks: int = 0,
        save_path: Path | None = None, best_score: float = 0.0,
        record_brain: Network | None = None,
    ) -> None:
        self.track = track
        self.rng = random.Random()
        self.inputs = inputs
        self.hidden = hidden[:]
        self.max_runtime = max_runtime
        self.time_limit_enabled = time_limit_enabled
        self.sizes = (inputs + EXTRA_INPUTS, *hidden, 3)
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
        self.save_path = save_path
        self.run_banked = 0.0  # points from finished laps in the champion's earlier heats
        self.current_score = 0.0  # the champion's run score
        # The record is tied to the model that set it, so higher means a better saved model.
        self.best_score_ever = best_score
        self.saved_best_score = best_score
        self.record_tracks = record_tracks
        self.record_brain = record_brain or (seed_networks[0] if seed_networks else None)
        # Slots only mean something once a generation was bred from ranked parents.
        self.roles_known = bool(seed_networks)
        self.last_result: tuple[str, str] | None = None  # who won the previous generation
        self._tracks = {(track.level, track.mirror, track.reverse): track}
        self.probes: list[tuple[Track, list[Car]]] = []
        if seed_networks:
            ranked = [((0, -index, 0.0), network) for index, network in enumerate(seed_networks)]
            brains = next_generation(ranked, self.rng, POPULATION)
            self.champion = brains[0]
        else:
            brains = [Network.random(self.sizes, self.rng) for _ in range(POPULATION)]
        self._spawn(brains)

    def _make_cars(self, track: Track, brains: list[Network]) -> list[Car]:
        x, y = track.points[0]
        nx, ny = track.points[1]
        base_angle = math.atan2(ny - y, nx - x)
        px, py = -math.sin(base_angle), math.cos(base_angle)
        start_spread = min(15, track.road_width * 0.18)
        return [
            Car(brain, x + px * self.rng.uniform(-start_spread, start_spread),
                y + py * self.rng.uniform(-start_spread, start_spread),
                base_angle + self.rng.uniform(-START_ANGLE_JITTER, START_ANGLE_JITTER))
            for brain in brains
        ]

    def _spawn(self, brains: list[Network]) -> None:
        self.cars = self._make_cars(self.track, brains)
        self.probes = [(track, self._make_cars(track, brains)) for track in self._choose_probes()]
        self.elapsed = 0.0

    def _choose_probes(self) -> list[Track]:
        """Hard roads, in random orientations, that every car also drives this heat."""
        current = (self.track.level, self.track.mirror, self.track.reverse)
        low = min(PROBE_MIN_LEVEL, len(ROAD_SPECS))
        picks: list[tuple[int, bool, bool]] = []
        while len(picks) < PROBE_COUNT:
            other = (self.rng.randint(low, len(ROAD_SPECS)), *self.rng.choice(ORIENTATIONS))
            if other != current and other not in picks:
                picks.append(other)
        return [self.track_for(*pick) for pick in picks]

    def track_for(self, level: int, mirror: bool = False, reverse: bool = False) -> Track:
        key = (level, mirror, reverse)
        if key not in self._tracks:
            self._tracks[key] = Track(level, mirror, reverse)
        return self._tracks[key]

    def fitness(self, index: int) -> float:
        """How well brain `index` drove the shown road and every probe road.

        Distance is scaled to the same units on every road. The score blends the
        average with the worst road, so a brain must cope with all of them.
        """
        reference = REFERENCE_SPEED * self.heat_limit
        values = [self.cars[index].best_progress / reference]
        values += [cars[index].best_progress / reference for _, cars in self.probes]
        mean = sum(values) / len(values)
        return FITNESS_MEAN_WEIGHT * mean + (1 - FITNESS_MEAN_WEIGHT) * min(values)

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
        self.run_banked = 0.0
        self.random_phase = False

    @property
    def run_score(self) -> float:
        """The champion's points since it was crowned.

        Once the champion has crashed its run is over, so the score follows the
        best car still racing, starting from zero. The same goes for the first
        generation, before any champion exists.
        """
        car = self.champion_car
        if car is not None and car.alive:
            return self.run_banked + car.score_on(self.track)
        return self.scoring_car.score_on(self.track)

    @property
    def scoring_car(self) -> Car:
        """The car whose model the current run score belongs to."""
        car = self.champion_car
        if car is not None and car.alive:
            return car
        return self.leader or self.best_car

    def save_now(self) -> None:
        """Write the record-setting model and score if the record is not saved yet."""
        if not self.save_path or self.record_brain is None or self.best_score_ever <= self.saved_best_score:
            return
        try:
            save_networks(self.save_path, [self.record_brain], self.record_tracks, self.best_score_ever)
        except OSError:
            return
        self.saved_best_score = self.best_score_ever

    def change_track(self, track: Track) -> None:
        """Jump to another road: the heat restarts and the gauntlet begins there."""
        self.track = track
        self._tracks.setdefault((track.level, track.mirror, track.reverse), track)
        self.best_ever = 0.0
        self.fastest_ever = None
        self.history.clear()
        self._reset_gauntlet()
        self.last_result = None
        self._spawn([car.brain for car in self.cars])

    def update(self, dt: float) -> None:
        self.elapsed += dt
        for car in self.cars:
            car.update(self.track, dt)
        for track, cars in self.probes:
            for car in cars:
                car.update(track, dt)
        self.best_ever = max(self.best_ever, *(car.best_progress for car in self.cars))
        for car in self.cars:
            if car.fastest_lap is not None:
                self.fastest_ever = (
                    car.fastest_lap if self.fastest_ever is None
                    else min(self.fastest_ever, car.fastest_lap)
                )
        self.current_score = self.run_score
        if self.current_score > self.best_score_ever:
            self.best_score_ever = self.current_score
            self.record_tracks = self.tracks_completed
            self.record_brain = self.scoring_car.brain
        goal = self.champion is not None and self.champion_laps >= self.laps_required
        time_expired = self.time_limit_enabled and self.elapsed >= self.heat_limit
        anyone_alive = any(car.alive for car in self.cars) or any(
            car.alive for _, cars in self.probes for car in cars
        )
        if goal or time_expired or not anyone_alive:
            self._end_heat(goal)

    def _end_heat(self, goal: bool) -> None:
        fitness = [self.fitness(index) for index in range(len(self.cars))]
        ranked = [((value,), car.brain) for value, car in zip(fitness, self.cars)]
        self.history.append(max(car.best_progress for car in self.cars))
        self.history = self.history[-40:]
        brains = next_generation(ranked, self.rng, POPULATION)
        self.save_now()
        champion_car = self.champion_car
        winner_index = max(range(len(self.cars)), key=fitness.__getitem__)
        if self.roles_known:
            winner_text = HOW_MADE[slot_role(winner_index, POPULATION)]
        else:
            winner_text = "a random starting network"
        headline = f"Generation {self.generation} was won by {winner_text}."
        next_track = self.track_for(1)
        if champion_car is None:
            self.champion = brains[0]
            self._reset_gauntlet()
            outcome = "It is the first champion; the run starts on road 1."
        else:
            crashed = not champion_car.alive and not goal
            # A challenger must be clearly fitter over all roads to take the title.
            beaten = fitness[winner_index] > fitness[0] * (1 + BEAT_MARGIN) and winner_index != 0
            if crashed or beaten:
                # Crash, or a clearly better car: new champion, back to road 1.
                self.champion = brains[0]
                self._reset_gauntlet()
                outcome = (
                    "The champion crashed, so the best car of the heat is the new champion; back to road 1."
                    if crashed else
                    "It beat the champion by enough to take over; back to road 1."
                )
            else:
                outcome = "The champion keeps its title."
                brains[0] = self.champion  # the champion always survives unchanged
                next_track = self.track  # laps are still owed on this road
                self.run_banked += champion_car.lap_score(self.track)
                if goal:
                    self.tracks_completed += 1
                    self.laps_banked = 0
                    outcome = f"The champion finished road {self.track.level} and has now completed {self.tracks_completed}."
                    if self.random_phase or self.track.level == len(ROAD_SPECS):
                        self.random_phase = True
                        # After the last road, roads and orientations are random.
                        next_track = self.track_for(
                            self.rng.randint(1, len(ROAD_SPECS)), *self.rng.choice(ORIENTATIONS))
                    else:
                        next_track = self.track_for(self.track.level + 1)
                else:
                    self.laps_banked += champion_car.laps_completed
        if next_track is not self.track:
            self.track = next_track
            self.best_ever = 0.0
            self.fastest_ever = None
            self.history.clear()
        self._spawn(brains)
        self.generation += 1
        self.roles_known = True
        self.last_result = (headline, outcome)
        self.current_score = self.run_score

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
        seed, record_tracks, best_score = None, 0, 0.0
        if saved:
            seed, record_tracks, best_score = saved.networks, saved.tracks_completed, saved.best_score
            self.inputs, self.hidden = seed[0].sizes[0] - EXTRA_INPUTS, list(seed[0].sizes[1:-1])
        self.road_level = 1  # the gauntlet always starts on road 1
        self.track = Track(self.road_level)
        self.race = Race(
            self.track, self.inputs, self.hidden, self.max_runtime, self.time_limit_enabled,
            seed, record_tracks, self.save_path, best_score,
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
                record_tracks=self.race.record_tracks, save_path=self.save_path,
                best_score=self.race.best_score_ever, record_brain=self.race.record_brain,
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
        pygame.draw.rect(self.screen, (11, 22, 30), (22, 22, 233, 196), border_radius=12)
        self.label("NEURAL CIRCUIT", 37, 32, ACCENT, self.small)
        self.label(f"Generation {self.race.generation:02d}", 37, 56, TEXT, self.bold)
        alive = sum(car.alive for car in self.race.cars)
        self.label(f"{alive:02d} / {POPULATION} cars on track", 38, 84, MUTED, self.small)
        self.label(f"Tracks completed: {self.race.tracks_completed}", 37, 108, ACCENT, self.font)
        lap_text = f"Lap {min(self.race.champion_laps, self.race.laps_required)}/{self.race.laps_required}"
        self.label(lap_text + (" · random roads" if self.race.random_phase else " on this road"), 38, 133, MUTED, self.small)
        self.label(f"Run score: {self.race.current_score:,.0f}", 37, 160, ORANGE, self.font)
        self.label(f"Best score ever: {self.race.best_score_ever:,.0f}", 38, 186, MUTED, self.small)
        probes = ", ".join(track.label for track, _ in self.race.probes)
        lines = [*(self.race.last_result or ()), f"Every car is also tested on: {probes}"]
        top = HEIGHT - 66 - 21 * len(lines)
        pygame.draw.rect(self.screen, (11, 22, 30), (22, top - 6, 640, 21 * len(lines) + 10), border_radius=8)
        for row, line in enumerate(lines):
            self.label(line, 36, top + 21 * row, TEXT if row == 0 and self.race.last_result else MUTED, self.small)

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
        self.race.save_now()
        pygame.quit()


if __name__ == "__main__":
    App().run()
