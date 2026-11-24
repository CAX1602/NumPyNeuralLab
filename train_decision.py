"""训练决策网络（11-24 终版）：二分类「跳杀 vs 高远」。

运行：
    .venv\\Scripts\\python.exe train_decision.py

数据：decision_shots.npz（make_decision.py 生成，模拟器教练价值分）。
只取「可决策」局面（高球且来得及起跳，clear 与 jump 都可行），
标签 = 教练价值更高的那个动作；低球由游戏规则自动挑球，不需要网络。
划分：70% 训练 / 15% 验证 / 15% 测试（seed=42）。
网络：4 → 16(tanh) → 1(sigmoid)，BCE——与 XOR 分类网络（network.py）同公式，
初始化用 landing.initialize_parameters（已泛化输入维度）。
输出：decision_model.npz（参数 + 训练集标准化统计量 + 测试指标）。

为什么从价值回归改成分类（11-24 实测）：两个动作的价值差平均只有 0.025，
MSE 的大部分容量拿去拟合「-1 失败区」和简单区域，argmax 一致率仅 58.9%；
直接学分界面，300 轮 BCE 就降到 0.10。「任务选对损失函数」的又一例证。
"""

import time
from pathlib import Path

import numpy as np

from decision import feasible_mask
from landing import initialize_parameters
from network import forward, backward, binary_cross_entropy

HIDDEN = 16
LR = 1.0
EPOCHS = 2000
SEED = 42


def main():
    d = np.load(Path(__file__).parent / "decision_shots.npz")
    X_all, values, state = (d["X"].astype(float), d["values"].astype(float),
                            d["state"].astype(float))
    mask = feasible_mask(state)
    both = mask[:, 1] & mask[:, 2]
    X = X_all[both]
    v = values[both]                       # 保留价值分用于策略价值评估
    y = (v[:, 2:3] > v[:, 1:2]).astype(float)
    print(f"可决策局面 {X.shape[0]} 个；跳杀占优比例 {y.mean():.1%}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(X.shape[0])
    n = X.shape[0]
    tr, va, te = idx[:int(0.7 * n)], idx[int(0.7 * n):int(0.85 * n)], idx[int(0.85 * n):]

    x_mean, x_std = X[tr].mean(axis=0), X[tr].std(axis=0)
    Xs = (X - x_mean) / x_std
    Xtr, Xva, Xte = Xs[tr], Xs[va], Xs[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    params = initialize_parameters(4, HIDDEN, SEED, 1)
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        gradients = backward(Xtr, ytr, params)
        for name in params:
            params[name] -= LR * gradients[name]
        if epoch % 500 == 0:
            print(f"第 {epoch:5d} 轮   训练 BCE "
                  f"{binary_cross_entropy(ytr, forward(Xtr, params)):.6f}   "
                  f"验证 BCE {binary_cross_entropy(yva, forward(Xva, params)):.6f}")
    print(f"训练耗时 {time.perf_counter() - t0:.1f} s")

    proba_te = forward(Xte, params)
    acc = float(((proba_te > 0.5).astype(float) == yte).mean())
    print(f"\ntest 策略一致率（= 分类准确率）：{acc:.1%}")

    choose_jump = (proba_te > 0.5).ravel()
    oracle_best = (v[te][:, 2] > v[te][:, 1])
    best_value = np.where(oracle_best, v[te][:, 2], v[te][:, 1]).mean()
    net_value = np.where(choose_jump, v[te][:, 2], v[te][:, 1]).mean()
    print(f"教练平均分：最优 {best_value:.3f} | 网络策略 {net_value:.3f} | "
          f"永远高远 {v[te][:, 1].mean():.3f} | 永远跳杀 {v[te][:, 2].mean():.3f}")

    out = Path(__file__).parent / "decision_model.npz"
    np.savez(out, **params, x_mean=x_mean, x_std=x_std,
             hidden_size=HIDDEN, learning_rate=LR, epochs=EPOCHS, seed=SEED,
             test_accuracy=acc)
    print(f"模型已保存 {out.name}")


if __name__ == "__main__":
    main()
