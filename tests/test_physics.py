"""物理模拟器的单元测试：击球判定单调性、杠杆范围、阻力效应、批量积分形状。"""

import unittest

import numpy as np

import physics
from physics import (
    resolve_hit, lever, v_base, simulate_batch, simulate_path,
    SWING_THETA, THETA_MIN, THETA_MAX, NET_X,
)


class TestResolveHit(unittest.TestCase):
    def test_launch_angle_matches_game_formula(self):
        # 与 game.py try_contact 同一公式：angle = θ − 0.50 − 0.45·low²，最低 −0.61
        theta = np.linspace(THETA_MIN, THETA_MAX, 11)
        v0, angle = resolve_hit(theta, 0.5, 1.0)
        low = np.clip((1.00 - theta) / 0.50, 0.0, 1.0)
        expected = np.maximum(theta - 0.50 - 0.45 * low * low, -0.61)
        self.assertTrue(np.allclose(angle, expected))
        self.assertTrue(np.all(angle <= theta))   # 只会比拍面角更下压

    def test_launch_angle_hand_value(self):
        # 手算：θ=0.9 → low=0.2 → 0.9 − 0.5 − 0.45·0.04 = 0.382 rad ≈ 21.9°
        _, angle = resolve_hit(0.9, 1.0, 1.0)
        self.assertAlmostEqual(float(angle), 0.382, places=6)

    def test_launch_angle_monotonic_and_floor(self):
        theta = np.linspace(THETA_MIN, THETA_MAX, 101)
        _, angle = resolve_hit(theta, 0.5, 1.0)
        self.assertTrue(np.all(np.diff(angle) >= 0))
        self.assertTrue(np.all(angle >= -0.61))

    def test_speed_decreases_with_theta(self):
        theta = np.linspace(THETA_MIN, THETA_MAX, 11)
        v0, _ = resolve_hit(theta, 0.5, 1.0)
        self.assertTrue(np.all(np.diff(v0) < 0))

    def test_lever_bounds_and_monotonic(self):
        s = np.linspace(0, 1, 11)
        lv = lever(s)
        self.assertAlmostEqual(float(lv[0]), 0.6)
        self.assertAlmostEqual(float(lv[-1]), 1.0)
        self.assertTrue(np.all(np.diff(lv) > 0))

    def test_power_scales_linearly(self):
        v1, _ = resolve_hit(0.3, 0.7, 1.0)
        v2, _ = resolve_hit(0.3, 0.7, 1.14)
        self.assertAlmostEqual(float(v2 / v1), 1.14, places=6)

    def test_swing_ranges_within_global(self):
        for lo, hi in SWING_THETA.values():
            self.assertGreaterEqual(lo, THETA_MIN)
            self.assertLessEqual(hi, THETA_MAX)


class TestDrag(unittest.TestCase):
    def test_drag_shortens_range(self):
        kw = dict(x0=200, y0=160, v0=800, angle=1.0, direction=1)
        _, xd, _ = simulate_path(drag_k=physics.DRAG_K, **kw)
        _, xv, _ = simulate_path(drag_k=0.0, **kw)
        self.assertLess(xd[-1], xv[-1])

    def test_drag_trajectory_finite(self):
        t, x, y = simulate_path(200, 160, 800, 1.0, 1)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLessEqual(y[-1], 0.0)


class TestSimulateBatch(unittest.TestCase):
    def setUp(self):
        # 只取保证过网的参数区间（11-24 按 5 档球速 + 1200 场宽重定标）：
        # 高弧（出射角 0.8–1.1）、1500–2000 px/s，最小射程也足够从 x0=320 过网；
        # 低平/软球或深区击球本来就可能落本方半场，那是合法物理，不在此用例范围。
        n = 32
        rng = np.random.default_rng(0)
        self.x0 = rng.uniform(320, NET_X - 60, n)
        self.y0 = rng.uniform(120, 260, n)
        self.v0 = rng.uniform(1500, 2000, n)
        self.angle = rng.uniform(0.8, 1.1, n)
        self.dir = np.ones(n)

    def test_shapes_and_first_frame(self):
        landing, ftime, obs = simulate_batch(
            self.x0, self.y0, self.v0, self.angle, self.dir)
        self.assertEqual(landing.shape, (32,))
        self.assertEqual(ftime.shape, (32,))
        self.assertEqual(obs.shape, (32, 4, 2))
        self.assertTrue(np.allclose(obs[:, 0, 0], self.x0))
        self.assertTrue(np.allclose(obs[:, 0, 1], self.y0))

    def test_landing_beyond_net_for_upward_shots(self):
        landing, _, _ = simulate_batch(
            self.x0, self.y0, self.v0, self.angle, self.dir)
        self.assertTrue(np.all(np.isfinite(landing)))
        self.assertTrue(np.all(landing > NET_X))

    def test_deterministic(self):
        a, _, _ = simulate_batch(self.x0, self.y0, self.v0, self.angle, self.dir)
        b, _, _ = simulate_batch(self.x0, self.y0, self.v0, self.angle, self.dir)
        self.assertTrue(np.array_equal(a, b))

    def test_direction_mirrors(self):
        lp, _, _ = simulate_batch(self.x0, self.y0, self.v0, self.angle,
                                  np.ones(32))
        lm, _, _ = simulate_batch(physics.COURT_W - self.x0, self.y0, self.v0,
                                  self.angle, -np.ones(32))
        self.assertTrue(np.allclose(lp + lm, physics.COURT_W, atol=1e-6))

    def test_short_flight_obs_rest_at_landing(self):
        # 快球在 0.15 s 观察窗内落地：落地后的帧必须是落点静止位置，
        # 不能是 (0,0)（否则归一化后变成「球在左上角」的坏输入）
        landing, ftime, obs = simulate_batch(
            np.array([500.0]), np.array([60.0]), np.array([3000.0]),
            np.array([-0.6]), np.array([1.0]))
        self.assertLess(float(ftime[0]), 0.15)
        self.assertAlmostEqual(float(obs[0, 0, 0]), 500.0)      # 第 0 帧 = 击球点
        for frame in (1, 2, 3):
            self.assertAlmostEqual(float(obs[0, frame, 0]), float(landing[0]))
            self.assertAlmostEqual(float(obs[0, frame, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()