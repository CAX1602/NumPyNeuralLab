"""网络本体的单元测试：形状、数值范围、损失性质、梯度检查、确定性。

运行（项目根目录）：
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from network import (
    initialize_parameters, forward, backward, sigmoid, binary_cross_entropy,
)
from gradients import numeric_gradient

X_CLEAN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_CLEAN = np.array([[0], [1], [1], [0]], dtype=float)


class TestShapes(unittest.TestCase):
    def test_init_shapes(self):
        for h in (1, 2, 4, 8):
            p = initialize_parameters(hidden_size=h, seed=0)
            self.assertEqual(p["W1"].shape, (2, h))
            self.assertEqual(p["b1"].shape, (1, h))
            self.assertEqual(p["W2"].shape, (h, 1))
            self.assertEqual(p["b2"].shape, (1, 1))

    def test_forward_shape(self):
        p = initialize_parameters(4, seed=0)
        for m in (1, 4, 80):
            X = np.random.default_rng(1).normal(size=(m, 2))
            self.assertEqual(forward(X, p).shape, (m, 1))

    def test_backward_shapes_match_params(self):
        for h in (1, 2, 4, 8):
            p = initialize_parameters(h, seed=0)
            g = backward(X_CLEAN, Y_CLEAN, p)
            self.assertEqual(set(g), set(p))
            for name in p:
                self.assertEqual(g[name].shape, p[name].shape)


class TestNumerics(unittest.TestCase):
    def test_sigmoid_range_and_clipping(self):
        z = np.array([-1e6, -500, 0, 500, 1e6])
        out = sigmoid(z)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0) & (out <= 1)))
        self.assertAlmostEqual(float(out[2]), 0.5, places=12)

    def test_forward_finite_on_boundary_inputs(self):
        p = initialize_parameters(4, seed=0)
        cases = [
            np.zeros((5, 2)),
            np.ones((5, 2)),
            np.full((5, 2), 1e3),
            np.full((5, 2), -1e3),
            np.full((5, 2), 1e6),
        ]
        for X in cases:
            out = forward(X, p)
            self.assertTrue(np.all(np.isfinite(out)), msg=f"X={X[0]}")
            self.assertTrue(np.all((out >= 0) & (out <= 1)))

    def test_bce_half_is_ln2(self):
        y = Y_CLEAN
        loss = binary_cross_entropy(y, np.full_like(y, 0.5))
        self.assertAlmostEqual(loss, np.log(2), places=10)

    def test_bce_perfect_near_zero(self):
        loss = binary_cross_entropy(Y_CLEAN, Y_CLEAN.copy())
        self.assertLess(loss, 1e-6)

    def test_bce_non_negative(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            y = rng.integers(0, 2, size=(10, 1)).astype(float)
            p = rng.uniform(0, 1, size=(10, 1))
            self.assertGreaterEqual(binary_cross_entropy(y, p), 0.0)


class TestGradientCheck(unittest.TestCase):
    def test_analytic_matches_numeric(self):
        for h in (1, 2, 4, 8):
            p = initialize_parameters(h, seed=42)
            analytic = backward(X_CLEAN, Y_CLEAN, p)
            numeric = numeric_gradient(X_CLEAN, Y_CLEAN, p)
            worst = 0.0
            for name in p:
                diff = np.linalg.norm(analytic[name] - numeric[name])
                total = (np.linalg.norm(analytic[name])
                         + np.linalg.norm(numeric[name]))
                worst = max(worst, diff / total)
            self.assertLess(worst, 1e-7, msg=f"hidden={h} worst={worst:.2e}")


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_init(self):
        a = initialize_parameters(4, seed=7)
        b = initialize_parameters(4, seed=7)
        for name in a:
            self.assertTrue(np.array_equal(a[name], b[name]))

    def test_different_seed_different_init(self):
        a = initialize_parameters(4, seed=7)
        b = initialize_parameters(4, seed=8)
        self.assertFalse(np.array_equal(a["W1"], b["W1"]))


if __name__ == "__main__":
    unittest.main()