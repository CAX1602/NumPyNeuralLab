"""落点回归网络的单元测试：形状、线性输出、损失性质、梯度检查、合成数据收敛、确定性。

只用合成数据，不读 shots.npz——物理未定稿，真实训练留待 11-27。
运行（项目根目录）：.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from landing import (
    COURT_W, initialize_parameters, forward, backward, mse_loss, compute_loss,
    rmse, train, flatten_obs, predict_landing_px,
)


def synth(n=200, seed=0):
    """答案已知的合成函数：线性项 + 一点非线性，模拟「观测 → 落点」。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 8))
    y = (0.2 + 0.5 * X[:, 0:1] - 0.3 * X[:, 1:2]
         + 0.1 * X[:, 0:1] * X[:, 2:3])
    return X, y


def numeric_gradient(X, y, parameters, epsilon=1e-4):
    """中心差分数值梯度，与 gradients.py 同法，损失换成 0.5·MSE。"""
    gradients = {}
    for name, value in parameters.items():
        g = np.zeros_like(value)
        for row in range(value.shape[0]):
            for col in range(value.shape[1]):
                plus = {k: v.copy() for k, v in parameters.items()}
                minus = {k: v.copy() for k, v in parameters.items()}
                plus[name][row, col] += epsilon
                minus[name][row, col] -= epsilon
                g[row, col] = ((compute_loss(X, y, plus)
                                - compute_loss(X, y, minus)) / (2 * epsilon))
        gradients[name] = g
    return gradients


class TestShapes(unittest.TestCase):
    def test_init_shapes(self):
        for h in (4, 8, 16, 32, 64):
            p = initialize_parameters(8, h, seed=0)
            self.assertEqual(p["W1"].shape, (8, h))
            self.assertEqual(p["b1"].shape, (1, h))
            self.assertEqual(p["W2"].shape, (h, 1))
            self.assertEqual(p["b2"].shape, (1, 1))

    def test_default_is_8_to_16_to_1(self):
        p = initialize_parameters()
        self.assertEqual(sum(v.size for v in p.values()), 161)

    def test_forward_shape(self):
        p = initialize_parameters(8, 16, seed=0)
        for m in (1, 5, 100):
            X = np.random.default_rng(1).uniform(size=(m, 8))
            self.assertEqual(forward(X, p).shape, (m, 1))

    def test_backward_shapes_match_params(self):
        X, y = synth(20)
        for h in (4, 16):
            p = initialize_parameters(8, h, seed=0)
            g = backward(X, y, p)
            self.assertEqual(set(g), set(p))
            for name in p:
                self.assertEqual(g[name].shape, p[name].shape)


class TestLinearOutputAndLoss(unittest.TestCase):
    def test_output_not_squashed_by_sigmoid(self):
        p = initialize_parameters(8, 4, seed=0)
        for name in ("W1", "b1", "W2"):
            p[name][:] = 0.0
        X = np.zeros((3, 8))
        for value in (3.7, -2.0, 0.5):
            p["b2"][:] = value
            out = forward(X, p)
            self.assertTrue(np.allclose(out, value),
                            msg=f"b2={value} 被压成了 {out[0]}")

    def test_mse_hand_computed(self):
        y = np.array([[1.0], [0.0]])
        pred = np.full_like(y, 0.5)          # 两个误差都是 ±0.5
        self.assertAlmostEqual(mse_loss(y, pred), 0.125, places=12)

    def test_mse_perfect_is_zero(self):
        _, y = synth(10)
        self.assertEqual(mse_loss(y, y.copy()), 0.0)

    def test_mse_non_negative(self):
        X, y = synth(30)
        p = initialize_parameters(8, 4, seed=0)
        self.assertGreaterEqual(compute_loss(X, y, p), 0.0)

    def test_rmse(self):
        _, y = synth(10)
        self.assertEqual(rmse(y, y.copy()), 0.0)
        pred = y + 0.1
        self.assertAlmostEqual(rmse(y, pred), 0.1, places=12)


class TestGradientCheck(unittest.TestCase):
    def test_analytic_matches_numeric(self):
        X, y = synth(40, seed=1)
        for h in (4, 16):
            p = initialize_parameters(8, h, seed=42)
            analytic = backward(X, y, p)
            numeric = numeric_gradient(X, y, p)
            worst = 0.0
            for name in p:
                diff = np.linalg.norm(analytic[name] - numeric[name])
                total = (np.linalg.norm(analytic[name])
                         + np.linalg.norm(numeric[name]))
                worst = max(worst, diff / total)
            self.assertLess(worst, 1e-7, msg=f"hidden={h} worst={worst:.2e}")


class TestTrainSynthetic(unittest.TestCase):
    def test_loss_decreases(self):
        X, y = synth(100)
        p0 = initialize_parameters(8, 16, seed=42)
        before = compute_loss(X, y, p0)
        p, _ = train(X, y, hidden_size=16, learning_rate=0.1, epochs=300,
                     seed=42, print_every=0)
        self.assertLess(compute_loss(X, y, p), before)

    def test_fits_synthetic_function(self):
        X, y = synth(400, seed=0)
        X_val, y_val = synth(80, seed=99)
        p, _ = train(X, y, hidden_size=16, learning_rate=0.3, epochs=3000,
                     seed=42, print_every=0)
        self.assertLess(rmse(y, forward(X, p)), 0.02)
        self.assertLess(rmse(y_val, forward(X_val, p)), 0.03)

    def test_training_deterministic(self):
        X, y = synth(100)
        p1, h1 = train(X, y, hidden_size=8, epochs=100, seed=42, print_every=0)
        p2, h2 = train(X, y, hidden_size=8, epochs=100, seed=42, print_every=0)
        for name in p1:
            self.assertTrue(np.array_equal(p1[name], p2[name]))
        self.assertEqual(h1["train"], h2["train"])

    def test_history_length_with_validation(self):
        X, y = synth(60)
        X_val, y_val = synth(20, seed=7)
        _, h = train(X, y, epochs=50, seed=42, print_every=0,
                     X_validation=X_val, y_validation=y_val)
        self.assertEqual(len(h["train"]), 50)
        self.assertEqual(len(h["validation"]), 50)

    def test_history_empty_without_validation(self):
        X, y = synth(60)
        _, h = train(X, y, epochs=50, seed=42, print_every=0)
        self.assertEqual(len(h["train"]), 0)
        self.assertEqual(len(h["validation"]), 0)


class TestDataInterface(unittest.TestCase):
    def test_flatten_obs_layout(self):
        obs = np.arange(24, dtype=float).reshape(3, 4, 2)
        X = flatten_obs(obs)
        self.assertEqual(X.shape, (3, 8))
        for i in range(3):
            self.assertTrue(np.allclose(X[i], obs[i].reshape(-1) / COURT_W))

    def test_predict_landing_px_scale(self):
        p = initialize_parameters(8, 4, seed=0)
        for name in ("W1", "b1", "W2"):
            p[name][:] = 0.0
        p["b2"][:] = 0.5
        out = predict_landing_px(np.zeros((2, 8)), p)
        self.assertTrue(np.allclose(out, 0.5 * COURT_W))


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_init(self):
        a = initialize_parameters(8, 16, seed=7)
        b = initialize_parameters(8, 16, seed=7)
        for name in a:
            self.assertTrue(np.array_equal(a[name], b[name]))

    def test_different_seed_different_init(self):
        a = initialize_parameters(8, 16, seed=7)
        b = initialize_parameters(8, 16, seed=8)
        self.assertFalse(np.array_equal(a["W1"], b["W1"]))


class TestTrainedModelInterface(unittest.TestCase):
    def test_predict_with_model_matches_manual(self):
        from landing import predict_with_model
        rng = np.random.default_rng(11)
        X = rng.uniform(0, 1, size=(20, 8))
        p = initialize_parameters(8, 16, seed=3)
        norm = {"x_mean": rng.uniform(0, 1, 8),
                "x_std": rng.uniform(0.05, 0.3, 8),
                "y_mean": 0.5, "y_std": 0.25, "court_w": 1200.0}
        manual = (forward((X - norm["x_mean"]) / norm["x_std"], p)
                  * norm["y_std"] + norm["y_mean"]) * norm["court_w"]
        np.testing.assert_allclose(predict_with_model(X, p, norm), manual)

    def test_load_model_roundtrip(self):
        import tempfile
        from pathlib import Path
        from landing import load_model, predict_with_model
        p = initialize_parameters(8, 16, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            np.savez(path, **p, x_mean=np.zeros(8), x_std=np.ones(8),
                     y_mean=0.5, y_std=0.25, court_w=1200.0)
            p2, norm = load_model(path)
        for name in p:
            self.assertTrue(np.array_equal(p[name], p2[name]))
        self.assertEqual(norm["court_w"], 1200.0)
        X = np.zeros((1, 8))
        np.testing.assert_allclose(
            predict_with_model(X, p2, norm),
            (forward(X, p) * 0.25 + 0.5) * 1200.0)


if __name__ == "__main__":
    unittest.main()
