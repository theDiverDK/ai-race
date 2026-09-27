import os
import math
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from main import App, Car, POPULATION, ROAD_SPECS, Race, Track


class RaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_ten_distinct_drivable_roads_get_narrower(self):
        self.assertEqual(len(ROAD_SPECS), 10)
        tracks = [Track(level) for level in range(1, 11)]
        self.assertEqual(len({tuple(track.points[:2]) for track in tracks}), 10)
        self.assertEqual([track.road_width for track in tracks], sorted(
            (track.road_width for track in tracks), reverse=True
        ))
        for track in tracks:
            self.assertTrue(all(track.on_road(x, y) for x, y in track.points))
            self.assertTrue(all(0 < x < 900 and 0 < y < 800 for x, y in track.points))
        turns = []
        for track in tracks:
            points = track.points
            headings = [
                math.atan2(
                    points[(i + 1) % len(points)][1] - points[i][1],
                    points[(i + 1) % len(points)][0] - points[i][0],
                )
                for i in range(len(points))
            ]
            turns.append(sum(
                abs((headings[(i + 1) % len(points)] - headings[i] + math.pi)
                    % (2 * math.pi) - math.pi)
                for i in range(len(points))
            ))
        self.assertGreater(turns[5], turns[4] * 1.4)
        self.assertTrue(all(a < b for a, b in zip(turns[5:], turns[6:])))

    def test_runtime_and_live_road_change(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        self.assertEqual(len(race.cars), POPULATION)
        race.update(5)
        self.assertEqual(race.generation, 2)
        brains = [car.brain for car in race.cars]
        race.change_track(Track(8))
        self.assertEqual(race.generation, 2)
        self.assertEqual(race.track.level, 8)
        self.assertEqual(race.elapsed, 0)
        self.assertEqual(race.best_ever, 0)
        self.assertTrue(all(car.brain is brain for car, brain in zip(race.cars, brains)))

    def test_ui_controls_update_active_race(self):
        app = App()
        brains = [car.brain for car in app.race.cars]
        app.handle_action("runtime:+")
        app.handle_action("limit")
        app.handle_action("road:+")
        self.assertEqual(app.race.max_runtime, 30)
        self.assertFalse(app.race.time_limit_enabled)
        self.assertEqual(app.road_level, 2)
        self.assertEqual(app.race.track.level, 2)
        self.assertTrue(all(car.brain is brain for car, brain in zip(app.race.cars, brains)))
        app.handle_action("apply")
        self.assertFalse(app.race.time_limit_enabled)

    def test_no_time_limit_waits_for_last_car(self):
        race = Race(Track(1), 7, [8], max_runtime=5, time_limit_enabled=False)
        for car in race.cars[1:]:
            car.alive = False
        with patch.object(Car, "update", return_value=None):
            race.update(6)
            self.assertEqual(race.generation, 1)
            self.assertEqual(race.elapsed, 6)
            race.cars[0].alive = False
            race.update(0)
            self.assertEqual(race.generation, 2)


if __name__ == "__main__":
    unittest.main()
