import os
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import main
from main import App, Car, POPULATION, ROAD_SPECS, Race, Track, WORLD_W
from neural import Network, load_networks, save_networks


class RaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        # Never touch the real best_network.json from tests.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.save_path = Path(directory.name) / "best_network.json"
        patcher = patch.object(main, "SAVE_PATH", self.save_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_distinct_drivable_roads_with_both_turns_and_pinches(self):
        self.assertEqual(len(ROAD_SPECS), 11)
        tracks = [Track(level) for level in range(1, 12)]
        self.assertEqual(tracks[0].points[0], (215, 155))
        self.assertEqual(set(tracks[0].widths), {104})
        self.assertEqual(len({tuple(track.points[:2]) for track in tracks}), 11)
        self.assertEqual([track.road_width for track in tracks], sorted(
            (track.road_width for track in tracks), reverse=True
        ))
        for track in tracks:
            self.assertTrue(all(track.on_road(x, y) for x, y in track.points))
            self.assertTrue(all(0 < x < 900 and 0 < y < 800 for x, y in track.points))
        for track in tracks[1:]:
            points = track.points
            crosses = [
                (points[i][0] - points[i - 1][0])
                * (points[(i + 1) % len(points)][1] - points[i][1])
                - (points[i][1] - points[i - 1][1])
                * (points[(i + 1) % len(points)][0] - points[i][0])
                for i in range(len(points))
            ]
            self.assertTrue(any(turn > 0 for turn in crosses), track.name)
            self.assertTrue(any(turn < 0 for turn in crosses), track.name)
        for track in tracks[5:10]:
            self.assertLess(min(track.widths), track.road_width * .7)
            index = track.widths.index(min(track.widths))
            x, y = track.points[index]
            nx, ny = track.points[(index + 1) % track.count]
            heading = math.atan2(ny - y, nx - x)
            side_x, side_y = -math.sin(heading), math.cos(heading)
            inside = track.widths[index] / 2 - 5
            outside = track.widths[index] / 2 + 10
            self.assertTrue(track.on_road(x + side_x * inside, y + side_y * inside))
            self.assertFalse(track.on_road(x + side_x * outside, y + side_y * outside))

    def test_hairpin_ladder_alternates_up_and_down_without_merging_lanes(self):
        track = Track(11)
        self.assertEqual(track.name, "Hairpin Ladder")
        for control_index, x, direction in (
            (1, 130, -1), (7, 240, 1), (13, 350, -1),
            (19, 460, 1), (25, 570, -1), (31, 680, 1),
        ):
            index = control_index * 16
            self.assertEqual(track.points[index], (x, 400))
            self.assertEqual(
                math.copysign(1, track.points[index + 1][1] - track.points[index - 1][1]),
                direction,
            )
        for left in (130, 240, 350, 460, 570):
            self.assertFalse(track.on_road(left + 55, 400))
        self.assertTrue(track.on_road(350, 735))

    def test_middle_roads_have_bends_tighter_than_full_speed_turning(self):
        full_speed_turn_radius = 220 / (1.25 + 0.005 * 220)
        for level in range(3, 10):
            track = Track(level)
            radii = []
            for index in range(track.count):
                before = track.points[index - 1]
                center = track.points[index]
                after = track.points[(index + 1) % track.count]
                incoming = (center[0] - before[0], center[1] - before[1])
                outgoing = (after[0] - center[0], after[1] - center[1])
                turn = abs((
                    math.atan2(outgoing[1], outgoing[0])
                    - math.atan2(incoming[1], incoming[0]) + math.pi
                ) % (2 * math.pi) - math.pi)
                if turn > 1e-5:
                    segment_length = (math.hypot(*incoming) + math.hypot(*outgoing)) / 2
                    radii.append(segment_length / turn)
            self.assertLess(min(radii) + track.road_width / 2, full_speed_turn_radius, track.name)

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

    def test_reverse_drive_and_separate_brake(self):
        track = Track(1)
        x, y = track.points[0]
        next_x, next_y = track.points[1]
        angle = math.atan2(next_y - y, next_x - x)

        def car_with_controls(outputs, speed):
            brain = Mock(sizes=(7, 3))
            brain.forward.return_value = outputs
            return Car(brain, x, y, angle, speed=speed)

        reversing = car_with_controls((0.5, -1.0, -1.0), 0)
        reversing.update(track, 0.1)
        self.assertLess(reversing.speed, 0)
        self.assertLess(reversing.angle, angle)
        self.assertTrue(reversing.alive)

        forward_coast = car_with_controls((0, 0, -1), 80)
        forward_brake = car_with_controls((0, 0, 1), 80)
        reverse_coast = car_with_controls((0, 0, -1), -80)
        reverse_brake = car_with_controls((0, 0, 1), -80)
        for car in (forward_coast, forward_brake, reverse_coast, reverse_brake):
            car.update(track, 0.1)
        self.assertGreater(forward_brake.speed, 0)
        self.assertLess(forward_brake.speed, forward_coast.speed)
        self.assertLess(reverse_brake.speed, 0)
        self.assertLess(abs(reverse_brake.speed), abs(reverse_coast.speed))

        stopped = car_with_controls((0, 0, 1), 1)
        stopped.update(track, 0.1)
        self.assertEqual(stopped.speed, 0)

        open_track = Mock(length=1000)
        open_track.sense.return_value = [1.0] * 7
        open_track.on_road.return_value = True
        open_track.progress.return_value = (0.0, 0)
        long_reverse = car_with_controls((0, -1, -1), 0)
        for _ in range(50):
            long_reverse.update(open_track, 0.1)
        self.assertEqual(long_reverse.speed, -110)
        self.assertGreater(long_reverse.distance_travelled, 10)
        self.assertTrue(long_reverse.alive)

    def test_completed_laps_record_the_fastest_time(self):
        track = Mock(length=100)
        track.sense.return_value = [1.0] * 7
        track.on_road.return_value = True
        track.progress.side_effect = [((step % 10) * 10.0, 0) for step in range(1, 21)]
        brain = Mock(sizes=(7, 3))
        brain.forward.return_value = (0.0, 1.0, -1.0)
        car = Car(brain, 0, 0, 0)
        for _ in range(10):
            car.update(track, 1.0)
        self.assertEqual(car.laps_completed, 1)
        self.assertAlmostEqual(car.fastest_lap, 10.0)
        for _ in range(10):
            car.update(track, 0.8)
        self.assertEqual(car.laps_completed, 2)
        self.assertAlmostEqual(car.fastest_lap, 8.0)

    def test_fastest_finisher_is_selected_before_distance(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        farther = race.cars[0]
        faster = race.cars[1]
        slower = race.cars[2]
        farther.best_progress = race.track.length * 0.9
        faster.best_progress = race.track.length
        faster.fastest_lap = 9.0
        slower.best_progress = race.track.length * 2
        slower.fastest_lap = 12.0
        self.assertIs(race.best_car, faster)
        self.assertIs(race.leader, faster)
        with patch.object(Car, "update", return_value=None):
            race.update(5)
        self.assertEqual(race.fastest_ever, 9.0)
        self.assertEqual(race.cars[0].brain.weights, faster.brain.weights)

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

    def test_network_button_accepts_window_object_events(self):
        app = App()
        app.draw_panel()
        click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"window": app.main_window, "pos": (WORLD_W + 50, 770), "button": 1},
        )
        with patch("pygame.event.get", side_effect=[[click], [pygame.event.Event(pygame.QUIT)]]):
            with patch.object(app, "handle_action", wraps=app.handle_action) as handle_action:
                app.run()
        handle_action.assert_any_call("inspect")

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

    def one_lap_runner(self, race):
        def one_car_completes_lap(car, track, dt):
            if car is race.cars[0]:
                car.best_progress = track.length
                car.fastest_lap = 5.0
        return patch.object(Car, "update", one_car_completes_lap)

    def test_next_generation_keeps_champions_and_adds_minor_mutants(self):
        from neural import next_generation
        rng = random.Random(3)
        ranked = [((0, float(i), 0.0), Network.random((5, 6, 3), rng)) for i in range(12)]
        offspring = next_generation(ranked, rng, 12)
        first, second = ranked[-1][1], ranked[-2][1]
        self.assertEqual(offspring[0].weights, first.weights)
        self.assertEqual(offspring[1].weights, second.weights)
        for mutant, parent in ((offspring[2], first), (offspring[3], second)):
            self.assertNotEqual(mutant.weights, parent.weights)
            diffs = [abs(a - b) for la, lb in zip(mutant.weights, parent.weights)
                     for ra, rb in zip(la, lb) for a, b in zip(ra, rb)]
            self.assertLess(max(diffs), 0.3)

    def gauntlet(self, race, laps_per_heat=None, crash=False):
        """Patch cars so slot 0 (the champion) finishes laps and others go nowhere."""
        def fake_update(car, track, dt):
            if car is race.cars[0]:
                if crash:
                    car.alive = False
                else:
                    car.laps_completed = laps_per_heat
                    car.best_progress = track.length * laps_per_heat
                    car.fastest_lap = 5.0
        return patch.object(Car, "update", fake_update)

    def finish_heat(self, race):
        race.elapsed = race.heat_limit
        race.update(0)

    def crowned_race(self, level=1, **kwargs):
        race = Race(Track(level), 7, [8], max_runtime=5, **kwargs)
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(race)  # generation 1 crowns the first champion
        return race

    def test_first_generation_crowns_a_champion_on_road_one(self):
        race = Race(Track(4), 7, [8], max_runtime=5)
        self.assertIsNone(race.champion)
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(race)
        self.assertIs(race.champion, race.cars[0].brain)
        self.assertEqual((race.track.level, race.tracks_completed), (1, 0))

    def test_five_laps_complete_a_road_and_move_to_the_next(self):
        race = self.crowned_race()
        champion = race.champion
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertEqual(race.tracks_completed, 1)
        self.assertEqual(race.track.level, 2)
        self.assertEqual(race.laps_banked, 0)
        self.assertIs(race.cars[0].brain, champion)

    def test_laps_accumulate_over_several_heats(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=2):
            self.finish_heat(race)
            self.assertEqual((race.laps_banked, race.track.level, race.tracks_completed), (2, 1, 0))
            self.finish_heat(race)
            self.assertEqual(race.laps_banked, 4)
            self.assertEqual(race.champion_laps, 4)
        with self.gauntlet(race, laps_per_heat=1):
            race.update(0.01)
        self.assertEqual((race.tracks_completed, race.track.level), (1, 2))

    def test_crash_restarts_from_road_one_with_zero_count(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertEqual(race.track.level, 2)
        with self.gauntlet(race, crash=True):
            self.finish_heat(race)
        self.assertEqual((race.track.level, race.tracks_completed, race.laps_banked), (1, 0, 0))

    def test_clearly_better_challenger_replaces_champion_and_resets(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        champion = race.champion

        def challenger_wins(car, track, dt):
            if car is race.cars[0]:
                car.laps_completed, car.best_progress, car.fastest_lap = 1, track.length, 5.0
            elif car is race.cars[7]:
                car.best_progress, car.fastest_lap = track.length, 4.0
        with patch.object(Car, "update", challenger_wins):
            self.finish_heat(race)
        self.assertIsNot(race.champion, champion)
        self.assertEqual((race.track.level, race.tracks_completed), (1, 0))

    def test_marginally_faster_clone_does_not_dethrone_champion(self):
        race = self.crowned_race()
        champion = race.champion

        def clone_slightly_faster(car, track, dt):
            car.best_progress = track.length
            if car is race.cars[0]:
                car.laps_completed, car.fastest_lap = 1, 5.00
            elif car is race.cars[3]:
                car.fastest_lap = 4.95
        with patch.object(Car, "update", clone_slightly_faster):
            self.finish_heat(race)
        self.assertIs(race.champion, champion)
        self.assertIs(race.cars[0].brain, champion)
        self.assertEqual(race.laps_banked, 1)

    def test_final_road_needs_ten_laps_sixty_seconds_then_random_roads(self):
        race = self.crowned_race()
        race.change_track(Track(11))
        self.assertEqual((race.laps_required, race.heat_limit), (10, 60))
        with self.gauntlet(race, laps_per_heat=9):
            race.update(0.01)
        self.assertEqual(race.tracks_completed, 0)
        with self.gauntlet(race, laps_per_heat=10):
            race.update(0.01)
        self.assertEqual(race.tracks_completed, 1)
        self.assertTrue(race.random_phase)
        seen = set()
        with self.gauntlet(race, laps_per_heat=10):
            for _ in range(40):
                race.update(0.01)
                seen.add(race.track.level)
        self.assertGreater(len(seen), 3)
        self.assertGreater(race.tracks_completed, 5)  # keeps counting on random roads

    def test_manual_road_change_restarts_gauntlet_there(self):
        race = self.crowned_race()
        race.laps_banked, race.tracks_completed = 3, 2
        race.change_track(Track(3))
        self.assertEqual((race.track.level, race.tracks_completed, race.laps_banked), (3, 0, 0))

    def test_save_only_for_a_new_record_and_load_round_trip(self):
        race = self.crowned_race(save_path=self.save_path)
        self.assertFalse(self.save_path.exists())
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        loaded, tracks = load_networks(self.save_path)
        self.assertEqual((len(loaded), tracks), (2, 1))
        self.assertEqual(loaded[0].weights, race.champion.weights)
        # A crashed, restarted run below the record leaves the file alone.
        with self.gauntlet(race, crash=True):
            self.finish_heat(race)
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertEqual(load_networks(self.save_path)[1], 1)

    def test_bad_or_missing_save_files_are_ignored(self):
        self.assertIsNone(load_networks(self.save_path))
        self.save_path.write_text("not json")
        self.assertIsNone(load_networks(self.save_path))
        self.save_path.write_text('{"version": 2, "tracks_completed": 1, "networks": [{"sizes": [5, 3], "weights": [[[1]]], "biases": [[0, 0, 0]]}]}')
        self.assertIsNone(load_networks(self.save_path))

    def test_app_starts_on_road_one_and_loads_saved_network(self):
        brains = [Network.random((5, 6, 4, 3), random.Random(i)) for i in range(2)]
        save_networks(self.save_path, brains, 7)
        app = App()
        self.assertEqual((app.inputs, app.hidden, app.road_level), (5, [6, 4], 1))
        self.assertEqual(app.race.saved_tracks, 7)
        self.assertEqual(app.race.cars[0].brain.weights, brains[0].weights)
        self.assertEqual(app.race.tracks_completed, 0)
        self.assertEqual(len(app.race.cars), POPULATION)

    def test_app_starts_from_scratch_without_a_file(self):
        app = App()
        self.assertIsNone(app.race.champion)
        self.assertEqual(app.race.track.level, 1)

    def test_app_follows_the_race_to_the_next_road_and_draws_counter(self):
        app = App()
        app.speed = 1
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(app.race)
        app.draw_world()  # the counter renders
        app.draw_panel()
        with self.gauntlet(app.race, laps_per_heat=5):
            with patch("pygame.event.get", side_effect=[[], [pygame.event.Event(pygame.QUIT)]]):
                app.run()
        # The loop runs two frames, each finishing one road.
        self.assertEqual((app.road_level, app.race.tracks_completed), (3, 2))
        self.assertIs(app.track, app.race.track)


if __name__ == "__main__":
    unittest.main()
