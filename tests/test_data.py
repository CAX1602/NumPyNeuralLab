"""数据生成与划分的单元测试：形状、标签、可复现、划分比例与互不重叠。"""

import unittest

import numpy as np

from data import make_xor, split_data


class TestMakeXor(unittest.TestCase):
    def test_shapes_and_labels(self):
        X, y = make_xor(n_samples=400, noise=0.1, seed=42)
        self.assertEqual(X.shape, (400, 2))
        self.assertEqual(y.shape, (400, 1))
        self.assertTrue(set(np.unique(y)).issubset({0.0, 1.0}))

    def test_reproducible(self):
        X1, y1 = make_xor(400, 0.1, seed=42)
        X2, y2 = make_xor(400, 0.1, seed=42)
        self.assertTrue(np.array_equal(X1, X2))
        self.assertTrue(np.array_equal(y1, y2))

    def test_seed_changes_data(self):
        X1, _ = make_xor(400, 0.1, seed=42)
        X2, _ = make_xor(400, 0.1, seed=43)
        self.assertFalse(np.array_equal(X1, X2))

    def test_noise_spreads_points(self):
        X0, _ = make_xor(400, noise=0.0, seed=42)
        X1, _ = make_xor(400, noise=0.5, seed=42)
        # 无噪声时点只落在四个角上；有噪声时应明显散开
        self.assertLessEqual(np.unique(X0, axis=0).shape[0], 4)
        self.assertGreater(np.unique(X1, axis=0).shape[0], 4)


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_xor(400, 0.1, seed=42)

    def test_ratios(self):
        Xtr, ytr, Xva, yva, Xte, yte = split_data(self.X, self.y, seed=42)
        self.assertEqual(len(Xtr), 240)
        self.assertEqual(len(Xva), 80)
        self.assertEqual(len(Xte), 80)
        self.assertEqual(len(ytr), 240)
        self.assertEqual(len(yva), 80)
        self.assertEqual(len(yte), 80)

    def test_disjoint(self):
        Xtr, _, Xva, _, Xte, _ = split_data(self.X, self.y, seed=42)
        a = {tuple(r) for r in Xtr}
        b = {tuple(r) for r in Xva}
        c = {tuple(r) for r in Xte}
        self.assertFalse(a & b)
        self.assertFalse(a & c)
        self.assertFalse(b & c)
        self.assertEqual(len(a | b | c), 400)

    def test_reproducible(self):
        s1 = split_data(self.X, self.y, seed=42)
        s2 = split_data(self.X, self.y, seed=42)
        for a, b in zip(s1, s2):
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()