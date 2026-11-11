"""物理模拟器的单元测试：击球判定单调性、杠杆范围、阻力效应、批量积分形状。"""

import unittest

import numpy as np

import physics
from physics import (
    resolve_hit, lever, v_base, simulate_batch, simulate_path,
    SWING_THETA, THETA_MIN, THETA_MAX, NET_X,
)


class TestResolveHit(unittest.TestCase):
    def test_angle_equals_theta(self):
        theta = np.linspace(THETA_MIN, THETA_MAX, 11)
        v0, angle = resolve_hit(theta, 0.5, 1.0)
        self.assertTrue(np.allclose(angle, theta))

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
        # 只取保证过网的参数区间。二次阻力下全场最大射程约 345 px，
        # 本参数盒（高弧、700–820 px/s）最小射程约 174 px（角点 1.1/700/120），
        # 所以击球点必须靠近网：x0 >= 320 时落点 >= 494 > 450。
        # 低平/软球或深区击球本来就可能落本方半场，那是合法物理，不在此用例范围。
        n = 32
        rng = np.random.default_rng(0)
        self.x0 = rng.uniform(320, NET_X - 60, n)
        self.y0 = rng.uniform(120, 260, n)
        self.v0 = rng.uniform(700, 820, n)
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


if __name__ == "__main__":
    unittest.main()