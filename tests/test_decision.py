"""决策网络的单元测试：可行规则、教练价值、策略路由、多输出梯度。全部小样本，不读数据文件。"""

import unittest

import numpy as np

from decision import (
    ACTIONS, FAIL, choose_action, decide, feasible_mask, load_model,
    normalize_state, oracle_values, value_of_outcome,
)
from landing import backward, initialize_parameters
from physics import COURT_W, NET_X


class TestFeasibleMask(unittest.TestCase):
    def test_low_ball_only_lift(self):
        mask = feasible_mask(np.array([[400.0, 60.0, 0.8, 900.0]]))
        self.assertEqual(mask[0].tolist(), [True, False, False])

    def test_high_ball_no_time_no_jump(self):
        mask = feasible_mask(np.array([[400.0, 200.0, 0.10, 900.0]]))
        self.assertEqual(mask[0].tolist(), [False, True, False])

    def test_high_ball_with_time_clear_or_jump(self):
        mask = feasible_mask(np.array([[400.0, 200.0, 0.8, 900.0]]))
        self.assertEqual(mask[0].tolist(), [False, True, True])


class TestOracleValue(unittest.TestCase):
    def test_net_and_out_are_failures(self):
        self.assertEqual(float(value_of_outcome(NET_X - 50, 0.8, 900.0)), FAIL)
        self.assertEqual(float(value_of_outcome(COURT_W + 50, 0.8, 900.0)), FAIL)
        self.assertEqual(float(value_of_outcome(np.nan, 0.8, 900.0)), FAIL)

    def test_farther_from_opponent_scores_higher(self):
        near = float(value_of_outcome(NET_X + 300, 0.8, NET_X + 310))
        far = float(value_of_outcome(NET_X + 300, 0.8, COURT_W - 60))
        self.assertGreater(far, near)

    def test_recovery_bonus_rewards_flight_time(self):
        # 对手难度相同（|L−x_h|/T 相等）时，飞得久给自己回位时间 → 价值更高
        landing = 900.0
        short = float(value_of_outcome(landing, 0.5, landing - 0.5 * 390))
        long = float(value_of_outcome(landing, 1.0, landing - 1.0 * 390))
        self.assertGreater(long, short)

    def test_oracle_marks_infeasible_actions(self):
        # 低球取 x0=300：x0=400、离地 60 的挑球到网时低于网顶，撞网判 FAIL
        state = np.array([[300.0, 60.0, 0.8, 900.0],     # 低球
                          [400.0, 200.0, 0.10, 900.0]])  # 高球但来不及跳
        values = oracle_values(state)
        self.assertEqual(values[0, 1], FAIL)
        self.assertEqual(values[0, 2], FAIL)
        self.assertEqual(values[1, 2], FAIL)
        self.assertGreater(values[0, 0], FAIL)   # 低球挑球是真实仿真分

    def test_oracle_jump_into_net_is_failure(self):
        # 后场（x0=100）跳杀：接触点 330 px、出球约 −18°，到网时只剩约 100 px 高，
        # 游戏里会被网挡下；教练必须判 FAIL，而不是按「落点在对方半场」给分
        values = oracle_values(np.array([[100.0, 200.0, 0.8, 900.0],
                                         [500.0, 200.0, 0.8, 900.0]]))
        self.assertEqual(values[0, 2], FAIL)
        self.assertGreater(values[0, 1], FAIL)      # 同一点打高远照样过网
        self.assertGreater(values[1, 2], FAIL)      # 前场跳杀不受影响


class TestHelpers(unittest.TestCase):
    def test_normalize_state_scales_columns(self):
        X = normalize_state(np.array([[600.0, 240.0, 0.75, 1200.0]]))
        np.testing.assert_allclose(X[0], [0.5, 0.2, 0.5, 1.0])

    def test_choose_action_argmax(self):
        values = np.array([[0.1, 0.9, -1.0], [-1.0, 0.2, 0.8]])
        self.assertEqual(choose_action(values).tolist(), [1, 2])

    def test_decide_routes_by_rules_and_probability(self):
        # 全零参数 → 跳杀概率 0.5（不跳）；b2 调大 → 概率 > 0.5（跳）
        p = initialize_parameters(4, 4, seed=0, n_output=1)
        for k in p:
            p[k] = np.zeros_like(p[k])
        norm = {"x_mean": np.zeros(4), "x_std": np.ones(4)}
        state = np.array([[400.0, 60.0, 0.8, 900.0],      # 低球 → 挑
                          [400.0, 200.0, 0.8, 900.0],     # 高球 → 高远（0.5 不跳）
                          [400.0, 200.0, 0.10, 900.0]])   # 来不及跳 → 高远
        self.assertEqual(decide(state, p, norm).tolist(), [0, 1, 1])
        p["b2"] = np.ones((1, 1)) * 2.0                   # 概率 ≈ 0.88 → 跳
        self.assertEqual(decide(state, p, norm).tolist(), [0, 2, 1])

    def test_load_model_roundtrip(self):
        import tempfile
        from pathlib import Path
        p = initialize_parameters(4, 4, seed=1, n_output=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            np.savez(path, **p, x_mean=np.zeros(4), x_std=np.ones(4))
            p2, norm = load_model(path)
        for name in p:
            self.assertTrue(np.array_equal(p[name], p2[name]))
        self.assertEqual(norm["x_std"].shape, (4,))


class TestMultiOutputGradients(unittest.TestCase):
    def test_backward_matches_numeric_for_three_outputs(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0, 1, size=(12, 4))
        y = rng.uniform(-1, 1, size=(12, 3))
        p = initialize_parameters(4, 4, seed=42, n_output=3)
        analytic = backward(X, y, p)

        def loss(params):
            from landing import compute_loss
            return compute_loss(X, y, params)

        eps = 1e-5
        worst = 0.0
        for name in p:
            numeric = np.zeros_like(p[name])
            it = np.nditer(p[name], flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                plus = {k: v.copy() for k, v in p.items()}
                minus = {k: v.copy() for k, v in p.items()}
                plus[name][idx] += eps
                minus[name][idx] -= eps
                numeric[idx] = (loss(plus) - loss(minus)) / (2 * eps)
            diff = np.linalg.norm(analytic[name] - numeric)
            total = np.linalg.norm(analytic[name]) + np.linalg.norm(numeric)
            worst = max(worst, diff / total)
        self.assertLess(worst, 1e-6)


if __name__ == "__main__":
    unittest.main()
