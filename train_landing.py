"""物理定稿（11-24）后的真实数据训练：落点网络 vs 线性基线。

运行：
    .venv\\Scripts\\python.exe train_landing.py

数据：shots.npz（已对齐游戏：1200 场宽、非线性出球角、5 档球速 1900/4100）。
划分：16000 训练 / 2000 验证 / 2000 测试（seed=42，打乱后切，互不重叠）。
输出：landing_model.npz（网络参数 + 训练集标准化统计量 + 元数据），供游戏 predict() 加载。
对照：同一份训练集上拟合最小二乘线性基线，报同一把尺（RMSE，像素）。

关键做法（11-24 实测）：输入/目标都用**训练集**统计量标准化（z-score）。
原始 ÷COURT_W 归一化下各特征 std 差异 4 倍以上（0.051~0.224），
梯度下降在病态 Hessian 上爬得极慢：3000 轮 test RMSE 120 px；
同样的轮数与学习率，标准化后 17 px，6000 轮约 15 px。
标准化统计量只从训练集估计（无泄漏），并随模型保存供推理复用。
"""

import time
from pathlib import Path

import numpy as np

from landing import load_dataset, train, forward, rmse
from landing_baseline import fit as linear_fit, predict as linear_predict
from physics import COURT_W

HIDDEN = 16
LR = 0.5
EPOCHS = 6000
SEED = 42


def split(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    tr, va, te = idx[:16000], idx[16000:18000], idx[18000:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def main():
    X, y = load_dataset()
    print(f"数据 X{X.shape} y{y.shape}；归一化常数 COURT_W={COURT_W:.0f}")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)

    # 标准化统计量只从训练集估计
    x_mean, x_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    Xtr_s, Xva_s, Xte_s = (Xtr - x_mean) / x_std, (Xva - x_mean) / x_std, (Xte - x_mean) / x_std
    ytr_s = (ytr - y_mean) / y_std

    t0 = time.perf_counter()
    params, history = train(Xtr_s, ytr_s, hidden_size=HIDDEN, learning_rate=LR,
                            epochs=EPOCHS, seed=SEED, print_every=1000,
                            X_validation=Xva_s, y_validation=(yva - y_mean) / y_std)
    print(f"网络训练耗时 {time.perf_counter() - t0:.1f} s")

    linear = linear_fit(Xtr, ytr)

    def net_px(Xx_s):
        return (forward(Xx_s, params) * y_std + y_mean) * COURT_W

    print(f"\n{'集合':<6} {'网络 RMSE':>12} {'线性基线 RMSE':>14}")
    results = {}
    for name, (Xx_s, Xx, yy) in (("train", (Xtr_s, Xtr, ytr)),
                                 ("val", (Xva_s, Xva, yva)),
                                 ("test", (Xte_s, Xte, yte))):
        net = rmse(yy * COURT_W, net_px(Xx_s))
        base = rmse(yy, linear_predict(Xx, linear)) * COURT_W
        results[name] = (net, base)
        print(f"{name:<6} {net:>10.2f} px {base:>12.2f} px")

    out = Path(__file__).parent / "landing_model.npz"
    np.savez(out, **params, x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std, court_w=COURT_W,
             hidden_size=HIDDEN, learning_rate=LR, epochs=EPOCHS, seed=SEED,
             test_rmse_net=results["test"][0],
             test_rmse_linear=results["test"][1])
    print(f"\n模型已保存 {out.name}"
          f"（test RMSE：网络 {results['test'][0]:.2f} px，"
          f"基线 {results['test'][1]:.2f} px）")


if __name__ == "__main__":
    main()
