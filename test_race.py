import os
import json
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
from neural import Network, load_networks, save_networks, slot_role


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
        for _ in range(35):  # under the 4 s no-progress limit: reversing is not a stall
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

    def test_ui_controls_update_active_race(self):
        app = App()
        self.assertFalse(app.race.time_limit_enabled)
        brains = [car.brain for car in app.race.cars]
        app.handle_action("runtime:+")
        app.handle_action("limit")
        app.handle_action("road:+")
        self.assertEqual(app.race.max_runtime, 30)
        self.assertTrue(app.race.time_limit_enabled)
        self.assertEqual(app.road_level, 2)
        self.assertEqual(app.race.track.level, 2)
        self.assertTrue(all(car.brain is brain for car, brain in zip(app.race.cars, brains)))
        app.handle_action("apply")
        self.assertTrue(app.race.time_limit_enabled)

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

    def test_evolution_runs_past_25_seconds_until_a_car_finishes_five_laps(self):
        app = App()
        race = app.race
        self.assertFalse(race.time_limit_enabled)
        with patch.object(Car, "update", return_value=None):
            race.update(30)
            self.assertEqual(race.generation, 1)
            race.cars[7].laps_completed = 4
            race.cars[7].best_progress = race.track.length * 4
            race.cars[7].progress = race.cars[7].best_progress
            race.update(0)
            self.assertEqual(race.generation, 1)
            race.cars[7].laps_completed = 5
            race.cars[7].best_progress = race.track.length * 5
            race.cars[7].progress = race.cars[7].best_progress
            race.update(0)
        self.assertEqual(race.generation, 2)
        self.assertEqual(race.track.level, 2)
        self.assertEqual(race.transition_history[-1]["reason"], "five_laps_completed")
        self.assertEqual(race.transition_history[-1]["leader_before"]["index"], 7)

    def test_banked_champion_laps_do_not_end_an_unlimited_generation_early(self):
        race = Race(Track(3), 7, [8], time_limit_enabled=False)
        race.champion = race.cars[0].brain
        race.laps_banked = 4
        with patch.object(Car, "update", return_value=None):
            race.cars[0].laps_completed = 1
            race.cars[0].best_progress = race.track.length
            race.update(30)
            self.assertEqual(race.generation, 1)
            race.cars[0].laps_completed = 5
            race.cars[0].best_progress = race.track.length * 5
            race.update(0)
        self.assertEqual(race.generation, 2)
        self.assertEqual(race.track.level, 4)

    def test_leader_is_car_furthest_through_race(self):
        race = Race(Track(1), 7, [8], time_limit_enabled=False)
        race.cars[1].laps_completed = 1
        race.cars[1].best_progress = race.track.length
        race.cars[1].progress = race.track.length
        race.cars[1].fastest_lap = 3.0
        race.cars[2].laps_completed = 4
        race.cars[2].best_progress = race.track.length * 4
        race.cars[2].progress = race.track.length * 4
        race.cars[2].fastest_lap = 8.0
        self.assertIs(race.leader, race.cars[2])
        race.cars[2].progress = race.track.length / 2
        self.assertIs(race.leader, race.cars[1])
        self.assertIs(race.scoring_car, race.cars[2])

    def test_road_eight_records_why_a_live_leader_started_a_new_generation(self):
        race = Race(Track(8), 7, [8], time_limit_enabled=False)
        race.champion = race.cars[0].brain
        with patch.object(Car, "update", return_value=None):
            race.update(30)
            self.assertEqual(race.generation, 1)
            race.cars[0].laps_completed = 5
            race.cars[0].best_progress = race.track.length * 5
            race.cars[0].progress = race.cars[0].best_progress
            race.update(0)
        transition = race.transition_history[-1]
        self.assertEqual(race.generation, 2)
        self.assertEqual(race.track.level, 9)
        self.assertEqual(transition["reason"], "five_laps_completed")
        self.assertFalse(transition["time_limit_enabled"])
        self.assertTrue(transition["leader_before"]["alive"])
        self.assertEqual(transition["leader_before"]["laps_completed"], 5)
        self.assertEqual((transition["road_before"], transition["road_after"]), (8, 9))

    def test_debug_button_saves_generation_reason_and_car_state(self):
        app = App()
        with patch.object(Car, "update", return_value=None):
            app.race.cars[4].laps_completed = 5
            app.race.cars[4].best_progress = app.race.track.length * 5
            app.race.cars[4].progress = app.race.cars[4].best_progress
            app.race.update(0)
        app.draw_panel()
        self.assertTrue(any(action == "debug" for _, action in app.buttons))
        app.handle_action("debug")
        payload = json.loads((self.save_path.parent / "debug_snapshot.json").read_text())
        transition = payload["evolution_recent_transitions"][-1]
        self.assertEqual(transition["reason"], "five_laps_completed")
        self.assertEqual(transition["leader_before"]["index"], 4)
        self.assertEqual((transition["road_before"], transition["road_after"]), (1, 2))
        self.assertEqual(payload["race"]["generation"], 2)
        self.assertEqual(len(payload["cars"]), POPULATION)

    def test_debug_button_is_clickable(self):
        app = App()
        app.draw_panel()
        button = next(rect for rect, action in app.buttons if action == "debug")
        click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"window": app.main_window, "pos": button.center, "button": 1},
        )
        with patch.object(app.race, "update", return_value=None):
            with patch("pygame.event.get", side_effect=[[click], [pygame.event.Event(pygame.QUIT)]]):
                app.run()
        self.assertTrue((self.save_path.parent / "debug_snapshot.json").exists())

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
                    car.lap_times = [track.length / 220 * 2] * laps_per_heat  # half of top pace
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

    def test_first_champion_keeps_completed_lap_score_on_next_road(self):
        race = Race(Track(1), 7, [8], time_limit_enabled=False)
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0)
        self.assertEqual(race.track.level, 2)
        self.assertGreater(race.run_banked, 0)
        self.assertAlmostEqual(race.current_score, race.run_banked)

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
            self.finish_heat(race)
        self.assertEqual((race.tracks_completed, race.track.level), (1, 2))

    def test_crash_restarts_from_road_one_with_zero_count(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertEqual(race.track.level, 2)
        with self.gauntlet(race, crash=True):
            self.finish_heat(race)
        self.assertEqual((race.track.level, race.tracks_completed, race.laps_banked), (1, 0, 0))

    def test_manual_road_selection_survives_failed_generations(self):
        race = Race(Track(1), 7, [8], time_limit_enabled=False)
        race.change_track(Track(8))
        with patch.object(Car, "update", return_value=None):
            for _ in range(2):
                for car in race.cars:
                    car.alive = False
                race.update(0)
                self.assertEqual(race.track.level, 8)
                self.assertEqual(race.restart_level, 8)
        self.assertEqual(race.generation, 3)

    def test_clearly_better_challenger_replaces_champion_and_resets(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        champion = race.champion

        def challenger_wins(car, track, dt):
            if car is race.cars[0]:
                car.laps_completed, car.best_progress, car.fastest_lap = 1, track.length, 5.0
            elif car is race.cars[7]:
                car.best_progress, car.fastest_lap = track.length * 3, 4.0
        with patch.object(Car, "update", challenger_wins):
            self.finish_heat(race)
        self.assertIsNot(race.champion, champion)
        self.assertEqual((race.track.level, race.tracks_completed), (1, 0))

    def test_marginally_faster_clone_does_not_dethrone_champion(self):
        race = self.crowned_race()
        champion = race.champion

        def clone_slightly_faster(car, track, dt):
            if car is race.cars[0]:
                car.laps_completed, car.fastest_lap = 1, 5.00
                car.best_progress = track.length
            elif car is race.cars[3]:
                car.best_progress = track.length * 1.02  # under the 3% margin
        with patch.object(Car, "update", clone_slightly_faster):
            self.finish_heat(race)
        self.assertIs(race.champion, champion)
        self.assertIs(race.cars[0].brain, champion)
        self.assertEqual(race.laps_banked, 1)

    def test_final_road_needs_five_laps_then_random_roads(self):
        race = self.crowned_race()
        race.change_track(Track(11))
        self.assertEqual((race.laps_required, race.heat_limit), (5, 60))
        with self.gauntlet(race, laps_per_heat=4):
            race.update(0.01)
        self.assertEqual(race.tracks_completed, 0)
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertEqual(race.tracks_completed, 1)
        self.assertTrue(race.random_phase)
        seen = set()
        with self.gauntlet(race, laps_per_heat=5):
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

    def test_score_counts_laps_pace_partial_lap_and_road_weight(self):
        track1, track5 = Track(1), Track(5)
        car = Car(Network.random((7, 8, 3), random.Random(1)), 0, 0, 0)
        self.assertEqual(car.score_on(track1), 0)
        car.best_progress = track1.length / 2
        self.assertAlmostEqual(car.score_on(track1), 50)
        car.best_progress = track1.length * 1.5
        car.laps_completed, car.lap_times = 1, [track1.length / 220 * 2]  # half of top pace
        self.assertAlmostEqual(car.lap_score(track1), 100 + 25)
        self.assertAlmostEqual(car.score_on(track1), 125 + 50)
        car.lap_times = [track1.length / 220]  # top pace
        self.assertAlmostEqual(car.lap_score(track1), 150)
        car.best_progress, car.laps_completed = track5.length, 1
        car.lap_times = [track5.length / 220]
        self.assertAlmostEqual(car.lap_score(track5), 150 * 1.4)

    def test_run_score_accumulates_across_roads_and_resets_on_crash(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)  # road 1 finished: 5 * (100 + 25)
        self.assertAlmostEqual(race.run_banked, 625)
        self.assertEqual(race.track.level, 2)
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)  # road 2 is worth 10% more
        self.assertAlmostEqual(race.run_banked, 625 + 625 * 1.1)
        self.assertAlmostEqual(race.best_score_ever, 625 + 625 * 1.1)
        with self.gauntlet(race, crash=True):
            self.finish_heat(race)
        self.assertEqual(race.run_banked, 0)
        self.assertAlmostEqual(race.best_score_ever, 625 + 625 * 1.1)  # the record survives

    def test_live_run_score_includes_the_lap_in_progress(self):
        race = self.crowned_race()

        def half_lap(car, track, dt):
            if car is race.cars[0]:
                car.best_progress = track.length / 2
        with patch.object(Car, "update", half_lap):
            race.update(0.01)
        self.assertAlmostEqual(race.current_score, 50)
        self.assertAlmostEqual(race.best_score_ever, 50)

    def test_score_before_any_champion_uses_the_best_car(self):
        race = Race(Track(1), 7, [8], max_runtime=5)

        def score_by_slot(car, track, dt):
            if car in race.cars:
                car.best_progress = track.length * (race.cars.index(car) / 100)
                car.progress = car.best_progress
        with patch.object(Car, "update", score_by_slot):
            race.update(1)
        self.assertAlmostEqual(race.current_score, 49)

    def test_record_saves_the_model_that_set_it_and_is_loaded_back(self):
        race = self.crowned_race(save_path=self.save_path)
        champion = race.champion
        self.assertFalse(self.save_path.exists())
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
            self.finish_heat(race)
        saved = load_networks(self.save_path)
        self.assertAlmostEqual(saved.best_score, race.best_score_ever)
        self.assertEqual(saved.networks[0].weights, champion.weights)
        # A worse later run leaves the saved model and score alone.
        with self.gauntlet(race, crash=True):
            self.finish_heat(race)
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(race)
        self.assertAlmostEqual(load_networks(self.save_path).best_score, saved.best_score)
        app = App()
        self.assertAlmostEqual(app.race.best_score_ever, saved.best_score)
        app.draw_world()
        # Restarting keeps the record and never overwrites the saved model with a weaker one.
        app.handle_action("apply")
        self.assertAlmostEqual(app.race.best_score_ever, saved.best_score)
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(app.race)
        app.race.save_now()
        self.assertEqual(load_networks(self.save_path).networks[0].weights, champion.weights)

    def test_champion_crash_keeps_the_race_going_and_score_follows_the_best_car(self):
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)  # road 1 done: banked points, now on road 2
        generation = race.generation
        self.assertGreater(race.run_banked, 0)

        def champion_crashes_others_run(car, track, dt):
            if car is race.cars[0]:
                car.alive = False
            elif car is race.cars[5]:
                car.best_progress = track.length * 0.3
                car.progress = car.best_progress
        with patch.object(Car, "update", champion_crashes_others_run):
            race.update(0.01)
            self.assertEqual(race.generation, generation)  # the heat carries on
            # 30% of a lap on road 2; the old run's banked points no longer count.
            self.assertAlmostEqual(race.current_score, 30 * 1.1)
            self.assertIs(race.scoring_car, race.cars[5])
            self.finish_heat(race)
        self.assertEqual(race.generation, generation + 1)
        self.assertIn("crashed", race.last_result[1])
        self.assertEqual((race.track.level, race.run_banked), (1, 0))

    def test_score_never_freezes_while_the_champion_races(self):
        race = self.crowned_race()
        scores = []

        def creeping(car, track, dt):
            if car is race.cars[0] and car.alive:
                car.best_progress += 1
        with patch.object(Car, "update", creeping):
            for _ in range(30):
                race.update(0.01)
                scores.append(race.current_score)
        self.assertEqual(scores, sorted(set(scores)))

    def test_slot_roles_match_how_next_generation_builds_cars(self):
        roles = [slot_role(i, 50) for i in range(50)]
        self.assertEqual(roles[:4], ["champion", "runner_up", "mutant", "mutant"])
        self.assertEqual(roles[4:45], ["child"] * 41)
        self.assertEqual(roles[45:], ["newcomer"] * 5)

    def test_result_line_says_who_won_and_how_it_was_made(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        with patch.object(Car, "update", return_value=None):
            self.finish_heat(race)
        self.assertIn("random starting network", race.last_result[0])
        self.assertIn("first champion", race.last_result[1])

        def winner_is(slot, lap=5.0):
            def update(car, track, dt):
                if car is race.cars[slot]:
                    car.laps_completed, car.best_progress, car.fastest_lap = 1, track.length, lap
            return patch.object(Car, "update", update)
        for slot, text in ((0, "unchanged champion"), (1, "runner-up"), (2, "lightly mutated"),
                           (20, "bred child"), (48, "brand-new random")):
            race = self.crowned_race()
            with winner_is(slot):
                self.finish_heat(race)
            self.assertIn(text, race.last_result[0], slot)
        race = self.crowned_race()
        with winner_is(20):  # beats a champion that has no lap
            self.finish_heat(race)
        self.assertIn("take over", race.last_result[1])
        race = self.crowned_race()
        with self.gauntlet(race, laps_per_heat=5):
            race.update(0.01)
        self.assertIn("finished road 1", race.last_result[1])
        app = App()
        app.race.last_result = ("Generation 3 was won by x.", "y")
        app.draw_world()

    def test_roads_can_be_mirrored_and_reversed(self):
        for level in range(1, len(ROAD_SPECS) + 1):
            base = Track(level)
            for mirror, reverse in ((True, False), (False, True), (True, True)):
                variant = Track(level, mirror, reverse)
                self.assertTrue(variant.on_road(*variant.points[0]), (level, mirror, reverse))
                self.assertNotEqual(variant.points[:3], base.points[:3])
                self.assertAlmostEqual(variant.length, base.length, places=3)
                self.assertTrue(all(0 <= x <= WORLD_W for x, _ in variant.points))
        base, mirrored = Track(1), Track(1, mirror=True)
        self.assertEqual(mirrored.points[0], (WORLD_W - base.points[0][0], base.points[0][1]))
        self.assertEqual(Track(1, reverse=True).points, base.points[::-1])
        self.assertIsNone(Track(1, mirror=True)._art)  # not drawn until shown
        self.assertIsNotNone(Track(1, mirror=True).art)

    def test_only_visible_cars_drive_the_displayed_road(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        with patch.object(Car, "update", autospec=True) as update:
            race.update(1 / 60)
        self.assertEqual(update.call_count, POPULATION)
        self.assertTrue(all(call.args[1] is race.track for call in update.call_args_list))

    def test_champion_needs_to_be_more_than_twice_as_fit_to_be_replaced(self):
        race = self.crowned_race()
        champion = race.champion

        def challenger(factor):
            def update(car, track, dt):
                if car is race.cars[0]:
                    car.best_progress = track.length
                elif car is race.cars[7]:
                    car.best_progress = track.length * factor
            return patch.object(Car, "update", update)
        with challenger(1.9):
            self.finish_heat(race)
        self.assertIs(race.champion, champion)
        race = self.crowned_race()
        champion = race.champion
        with challenger(2.5):
            self.finish_heat(race)
        self.assertIsNot(race.champion, champion)

    def test_speed_is_the_last_input_of_every_brain(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        self.assertEqual(race.sizes[0], 8)
        car = race.cars[0]
        car.sensor_noise = 0.0
        car.speed = 110.0
        car.update(race.track, 1 / 60)
        self.assertEqual(len(car.sensors), 7)
        self.assertEqual(len(car.inputs), 8)
        self.assertAlmostEqual(car.inputs[-1], 110.0 / 220)  # the speed the brain felt this step

    def test_fitness_uses_only_distance_on_the_displayed_road(self):
        race = Race(Track(1), 7, [8], max_runtime=5)
        reference = 110.0 * race.heat_limit
        race.cars[0].best_progress = reference * 3
        race.cars[1].best_progress = reference
        self.assertGreater(race.fitness(0), race.fitness(1))
        self.assertAlmostEqual(race.fitness(1), 1.0)
        self.assertAlmostEqual(race.fitness(0), 3.0)

    def test_next_generation_breeds_from_the_best_shown_road_driver(self):
        race = self.crowned_race()
        reference = 110.0 * race.heat_limit
        champion = race.champion

        def winner_is_car_9(car, track, dt):
            if car is race.cars[0]:
                car.best_progress = reference
            elif car is race.cars[9]:
                car.best_progress = reference * 3
        with patch.object(Car, "update", winner_is_car_9):
            self.finish_heat(race)
        self.assertIs(race.champion, race.cars[0].brain)
        self.assertIsNot(race.champion, champion)

    def test_random_phase_shows_random_orientations(self):
        race = self.crowned_race()
        race.change_track(Track(11))
        seen = set()
        with self.gauntlet(race, laps_per_heat=10):
            for _ in range(60):
                race.update(0.01)
                seen.add((race.track.mirror, race.track.reverse))
        self.assertGreater(len(seen), 2)

    def test_a_car_that_stops_making_progress_is_out_even_if_it_keeps_driving(self):
        class Cruiser:
            sizes = (8, 3)

            def forward(self, inputs):
                return 0.0, 0.3, 0.0

        track = Track(1)
        car = Car(Cruiser(), *track.points[0], 0.0, sensor_noise=0.0)
        with patch.object(track, "on_road", return_value=True):
            with patch.object(track, "progress", return_value=(100.0, 0)):  # stuck at one spot
                for _ in range(int(3.9 * 60)):
                    car.update(track, 1 / 60)
                self.assertTrue(car.alive)
                for _ in range(int(0.3 * 60)):
                    car.update(track, 1 / 60)
        self.assertFalse(car.alive)
        self.assertGreater(abs(car.speed), 20)  # it was still driving, just going nowhere

    def test_a_car_that_keeps_advancing_is_not_timed_out(self):
        class Cruiser:
            sizes = (8, 3)

            def forward(self, inputs):
                return 0.0, 0.3, 0.0

        track = Track(1)
        car = Car(Cruiser(), *track.points[0], 0.0, sensor_noise=0.0)
        distance = {"value": 0.0}

        def advancing(x, y, near):
            distance["value"] += 1.0  # 60 px per simulated second
            return distance["value"], 0
        with patch.object(track, "on_road", return_value=True):
            with patch.object(track, "progress", side_effect=advancing):
                for _ in range(10 * 60):
                    car.update(track, 1 / 60)
        self.assertTrue(car.alive)

    def test_a_circling_champion_cannot_freeze_the_run(self):
        race = self.crowned_race()

        def circler(car, track, dt):
            if car is race.cars[0]:
                # Alive and "driving" but never gaining ground, like a car going in circles.
                car.time += dt
                if car.time - car.mark_time > 4.0:
                    car.alive = False
        with patch.object(Car, "update", circler):
            for _ in range(int(6 / 0.05)):
                race.update(0.05)
        self.assertIn("crashed", race.last_result[1])

    def test_bad_or_missing_save_files_are_ignored(self):
        self.assertIsNone(load_networks(self.save_path))
        self.save_path.write_text("not json")
        self.assertIsNone(load_networks(self.save_path))
        self.save_path.write_text('{"version": 2, "tracks_completed": 1, "networks": []}')
        self.assertIsNone(load_networks(self.save_path))  # older format without the speed input
        self.save_path.write_text('{"version": 3, "tracks_completed": 1, "networks": [{"sizes": [5, 3], "weights": [[[1]]], "biases": [[0, 0, 0]]}]}')
        self.assertIsNone(load_networks(self.save_path))

    def test_app_starts_on_road_one_and_loads_saved_network(self):
        brains = [Network.random((6, 6, 4, 3), random.Random(i)) for i in range(2)]  # 5 sensors + speed
        save_networks(self.save_path, brains, 7)
        app = App()
        self.assertEqual((app.inputs, app.hidden, app.road_level), (5, [6, 4], 1))
        self.assertEqual(app.race.sizes, (6, 6, 4, 3))
        self.assertEqual(app.race.record_tracks, 7)
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
        app.race.time_limit_enabled = True
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
