import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import torch

import main
from main import App, Car, PPORace, Track
from neural import Network
from ppo import PPOAgent


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch.object(main, "SAVE_PATH", self.directory / "best_network.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exported_actor_matches_deterministic_policy(self):
        agent = PPOAgent((8, 12, 8, 3))
        inputs = np.array([[0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.1, -0.2]], dtype=np.float32)
        with torch.no_grad():
            means, _ = agent.model(torch.as_tensor(inputs))
            expected = torch.tanh(means)[0].tolist()
        actual = agent.export_network().forward(inputs[0].tolist())
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_training_rollout_updates_policy_and_checkpoint(self):
        checkpoint = self.directory / "ppo_checkpoint.pt"
        race = PPORace(Track(1), 7, [8], 5, True, checkpoint, resume=False)
        self.assertIs(race.cars[0], race.worker_cars[0])
        before = race.agent.model.actor_head.weight.detach().clone()
        with patch.object(main, "PPO_ROLLOUT_STEPS", 8):
            for _ in range(30):
                race.update(1 / 60)
                if race.agent.updates:
                    break
        self.assertEqual(race.agent.updates, 1)
        self.assertFalse(torch.equal(before, race.agent.model.actor_head.weight))
        self.assertFalse(race.agent.optimizing)
        race.save_now()
        loaded = PPOAgent.load(checkpoint, race.sizes)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.updates, 1)
        np.testing.assert_allclose(
            loaded.export_network().weights[-1], race.agent.export_network().weights[-1]
        )
        resumed = PPORace(Track(1), 7, [8], 5, True, checkpoint, resume=True)
        self.assertEqual(resumed.agent.updates, 1)

    def test_visible_worker_keeps_driving_during_optimization(self):
        race = PPORace(Track(1), 7, [8], 5, True, self.directory / "ppo_checkpoint.pt", resume=False)
        with patch.object(main, "PPO_ROLLOUT_STEPS", 2):
            race.update(1 / 60)
            race.update(1 / 60)
        self.assertTrue(race.agent.optimizing)
        car = race.cars[0]
        old_time = car.time
        race.update(1 / 60)
        self.assertTrue(race.cars[0] is not car or race.cars[0].time > old_time)
        self.assertIs(race.cars[0], race.worker_cars[0])

    def test_algorithm_switch_preserves_both_in_memory_trainers(self):
        app = App()
        evolution = app.race
        app.handle_action("algorithm")
        self.assertEqual(app.algorithm, "ppo")
        ppo = app.race
        ppo.update(1 / 60)
        app.draw_world()
        app.draw_panel()
        app.handle_action("algorithm")
        self.assertIs(app.race, evolution)
        app.handle_action("algorithm")
        self.assertIs(app.race, ppo)
        app.handle_action("road:+")
        self.assertEqual(app.track.level, 2)
        self.assertIs(app.race.cars[0], app.race.worker_cars[0])

    def test_algorithm_button_switches_mode_in_event_loop(self):
        app = App()
        app.draw_panel()
        button = next(rect for rect, action in app.buttons if action == "algorithm")
        click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"window": app.main_window, "pos": button.center, "button": 1},
        )
        with patch("pygame.event.get", side_effect=[[click], [pygame.event.Event(pygame.QUIT)]]):
            app.run()
        self.assertEqual(app.algorithm, "ppo")

    def test_network_controls_apply_to_ppo_policy(self):
        app = App()
        app.handle_action("algorithm")
        app.handle_action("inputs:+")
        app.handle_action("layers:+")
        app.handle_action("hidden2:+")
        app.handle_action("apply")
        self.assertEqual(app.race.sizes, (9, 16, 16, 17, 3))
        self.assertEqual(app.race.agent.sizes, app.race.sizes)

    def test_shared_physics_matches_evolutionary_update(self):
        track = Track(1)
        brain = Network.random((8, 6, 3), main.random.Random(7))
        x, y = track.points[0]
        angle = main.math.atan2(track.points[1][1] - y, track.points[1][0] - x)
        first = Car(brain, x, y, angle, sensor_noise=0)
        second = Car(brain, x, y, angle, sensor_noise=0)
        first.update(track, 1 / 60)
        inputs = second.observe(track)
        second.move(track, 1 / 60, brain.forward(inputs))
        for field in ("x", "y", "angle", "speed", "progress", "best_progress", "time"):
            self.assertEqual(getattr(first, field), getattr(second, field))


if __name__ == "__main__":
    unittest.main()
