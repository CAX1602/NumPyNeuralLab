"""落点线性基线的单元测试：最小二乘性质、像素换算、确定性、obs 接口。

全部用合成数据，不读 shots.npz。
"""

import unittest

import numpy as np

from landing import flatten_obs, rmse
from landing_baseline import fit, predict, predict_landing_px
from physics import COURT_W


def synth_linear(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 8))
    w = np.array([[0.5], [-0.3], [0.2], [0.0],
                  [0.1], [0.0], [-0.2], [0.4]])
    y = 0.2 + X @ w
    return X, y, w


class TestLinearBaseline(unittest.TestCase):
    def test_fit_and_predict_shapes(self):
        X, y, _ = synth_linear()
        p = fit(X, y)
        self.assertEqual(p["W"].shape, (8, 1))
        self.assertEqual(p["b"].shape, (1, 1))
        self.assertEqual(predict(X, p).shape, (200, 1))

    def test_recovers_exact_linear(self):
        X, y, w = synth_linear(400, seed=1)
        p = fit(X, y)
        self.assertLess(rmse(y, predict(X, p)), 1e-8)
        np.testing.assert_allclose(p["W"], w, atol=1e-8)
        self.assertAlmostEqual(float(p["b"][0, 0]), 0.2, places=8)

    def test_residual_orthogonal_to_features(self):
        # 正规方程：无论数据是否非线性，残差都与 [X, 1] 张成的空间正交
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, size=(300, 8))
        y = 0.1 * X[:, 0:1] * X[:, 2:3] + 0.05 * np.sin(6 * X[:, 1:2])
        p = fit(X, y)
        residual = y - predict(X, p)
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        np.testing.assert_allclose(A.T @ residual, 0.0, atol=1e-8)

    def test_pixel_scaling(self):
        X, y, _ = synth_linear(50, seed=3)
        p = fit(X, y)
        np.testing.assert_allclose(
            predict_landing_px(X, p), predict(X, p) * COURT_W)

    def test_deterministic(self):
        X, y, _ = synth_linear(100, seed=4)
        p1, p2 = fit(X, y), fit(X, y)
        self.assertTrue(np.array_equal(p1["W"], p2["W"]))
        self.assertTrue(np.array_equal(p1["b"], p2["b"]))

    def test_accepts_flattened_obs(self):
        rng = np.random.default_rng(5)
        obs = rng.uniform(0, COURT_W, size=(60, 4, 2))
        X = flatten_obs(obs)
        y = 0.3 + 0.4 * X[:, 0:1]
        p = fit(X, y)
        self.assertLess(rmse(y, predict(X, p)), 1e-8)


if __name__ == "__main__":
    unittest.main()
