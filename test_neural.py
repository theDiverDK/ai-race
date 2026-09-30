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

    def test_direct_input_connections_reach_the_outputs(self):
        brain = Network.random((5, 4, 3), random.Random(3))
        self.assertEqual((len(brain.skip), len(brain.skip[0])), (3, 5))
        inputs = [0.2, -0.4, 0.9, 0.1, 0.5]
        before = brain.forward(inputs)
        brain.skip[1][2] += 1.0  # only the shortcut into output 1 changes
        after = brain.forward(inputs)
        self.assertEqual(after[0], before[0])
        self.assertNotEqual(after[1], before[1])
        self.assertEqual(after[2], before[2])

    def test_skip_weights_are_copied_bred_mutated_and_saved(self):
        rng = random.Random(4)
        a, b = Network.random((5, 4, 3), rng), Network.random((5, 4, 3), rng)
        clone = a.copy()
        self.assertEqual(clone.skip, a.skip)
        clone.skip[0][0] += 1
        self.assertNotEqual(clone.skip, a.skip)  # a deep copy
        self.assertNotEqual(a.minor_mutation(rng).skip, a.skip)
        self.assertNotEqual(a.child(b, rng).skip, a.skip)
        again = Network.from_dict(a.to_dict())
        self.assertEqual((again.weights, again.biases, again.skip), (a.weights, a.biases, a.skip))

    def test_networks_saved_without_shortcuts_still_load_unchanged(self):
        brain = Network.random((5, 4, 3), random.Random(5))
        data = brain.to_dict()
        del data["skip"]
        old = Network.from_dict(data)
        self.assertTrue(all(value == 0 for row in old.skip for value in row))
        brain.skip = [[0.0] * 5 for _ in range(3)]
        self.assertEqual(old.forward([0.3] * 5), brain.forward([0.3] * 5))
        data["skip"] = [[0.0] * 4 for _ in range(3)]  # wrong shape
        with self.assertRaises(ValueError):
            Network.from_dict(data)

    def test_inspector_shows_what_the_network_computes(self):
        import inspector
        brain = Network.random((8, 6, 3), random.Random(6))
        inputs = [0.5] * 7 + [0.3]
        self.assertEqual([round(v, 9) for v in inspector.activations(brain, inputs)[-1]],
                         [round(v, 9) for v in brain.forward(inputs)])


if __name__ == "__main__":
    unittest.main()
