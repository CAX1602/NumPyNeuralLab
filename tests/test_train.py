"""训练循环的单元测试：损失下降、准确率、可复现、历史记录行为。"""

import unittest

import numpy as np

from data import make_xor, split_data
from network import initialize_parameters, forward, binary_cross_entropy
from train import train, accuracy

X_CLEAN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_CLEAN = np.array([[0], [1], [1], [0]], dtype=float)


class TestTrain(unittest.TestCase):
    def test_loss_decreases(self):
        p0 = initialize_parameters(4, seed=42)
        before = binary_cross_entropy(Y_CLEAN, forward(X_CLEAN, p0))
        p, _ = train(X_CLEAN, Y_CLEAN, hidden_size=4, learning_rate=1.0,
                     epochs=500, seed=42, print_every=0)
        after = binary_cross_entropy(Y_CLEAN, forward(X_CLEAN, p))
        self.assertLess(after, before)

    def test_reaches_perfect_on_clean_xor(self):
        p, _ = train(X_CLEAN, Y_CLEAN, hidden_size=4, learning_rate=1.0,
                     epochs=1000, seed=42, print_every=0)
        self.assertEqual(accuracy(X_CLEAN, Y_CLEAN, p), 1.0)

    def test_training_deterministic(self):
        X, y = make_xor(200, 0.1, seed=42)
        p1, h1 = train(X, y, hidden_size=4, learning_rate=1.0, epochs=200,
                       seed=42, print_every=0)
        p2, h2 = train(X, y, hidden_size=4, learning_rate=1.0, epochs=200,
                       seed=42, print_every=0)
        for name in p1:
            self.assertTrue(np.array_equal(p1[name], p2[name]))
        self.assertEqual(h1["train"], h2["train"])

    def test_history_length_with_validation(self):
        X, y = make_xor(200, 0.1, seed=42)
        Xtr, ytr, Xva, yva, _, _ = split_data(X, y, seed=42)
        _, h = train(Xtr, ytr, epochs=50, seed=42, print_every=0,
                     X_validation=Xva, y_validation=yva)
        self.assertEqual(len(h["train"]), 50)
        self.assertEqual(len(h["validation"]), 50)

    def test_history_empty_without_validation(self):
        _, h = train(X_CLEAN, Y_CLEAN, epochs=50, seed=42, print_every=0)
        self.assertEqual(len(h["train"]), 0)
        self.assertEqual(len(h["validation"]), 0)

    def test_accuracy_in_range(self):
        p = initialize_parameters(4, seed=0)
        acc = accuracy(X_CLEAN, Y_CLEAN, p)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()