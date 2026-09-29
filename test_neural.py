import random
import unittest

from neural import Network, next_generation


class NeuralTests(unittest.TestCase):
    def test_architecture_and_outputs(self):
        brain = Network.random((7, 9, 4, 3), random.Random(1))
        steering, drive, brake = brain.forward([0.5] * 7)
        self.assertTrue(-1 <= steering <= 1)
        self.assertTrue(-1 <= drive <= 1)
        self.assertTrue(-1 <= brake <= 1)
        with self.assertRaises(ValueError):
            brain.forward([0.5] * 6)

    def test_breeding_keeps_population_and_architecture(self):
        rng = random.Random(2)
        ranked = [((0, float(i), 0.0), Network.random((5, 6, 3), rng)) for i in range(12)]
        offspring = next_generation(ranked, rng, 12)
        self.assertEqual(len(offspring), 12)
        self.assertTrue(all(brain.sizes == (5, 6, 3) for brain in offspring))
        self.assertIsNot(offspring[0], ranked[-1][1])
        self.assertEqual(offspring[0].weights, ranked[-1][1].weights)


if __name__ == "__main__":
    unittest.main()
