"""真实数据训练（11-24 晚终版）：双输出落点网络 vs 线性基线。

运行：
    .venv\\Scripts\\python.exe train_landing.py

数据：shots.npz（对齐游戏：1200 场宽、非线性出球角、10 档球速 3800/8200、
DRAG_K 0.00375）。y = [落点 x, 截击点 x]（截击点 = 下行穿过 250 高度的 x，
平球取落点）——AI 跑位用截击点；只预测落点会让 AI 站得太深、挥拍晚了
变成「往后砸」，11-24 接入实测 12-0 的教训。
划分：16000 训练 / 2000 验证 / 2000 测试（seed=42）。
输出：landing_model.npz（参数 + 训练集标准化统计量 + 元数据）。

关键做法：输入/目标都用**训练集**统计量 z-score 标准化（统计量随模型保存，
推理复用，无泄漏）。不归一化特征尺度差 4 倍以上时，同样轮数误差约 8 倍。
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
TARGETS = ("落点", "截击点")


def split(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    tr, va, te = idx[:16000], idx[16000:18000], idx[18000:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def main():
    X, y = load_dataset(with_ic=True)
    print(f"数据 X{X.shape} y{y.shape}（{TARGETS}）；归一化常数 COURT_W={COURT_W:.0f}")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)

    x_mean, x_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    y_mean, y_std = ytr.mean(axis=0), ytr.std(axis=0)
    Xtr_s, Xva_s, Xte_s = (Xtr - x_mean) / x_std, (Xva - x_mean) / x_std, (Xte - x_mean) / x_std
    ytr_s = (ytr - y_mean) / y_std

    t0 = time.perf_counter()
    params, history = train(Xtr_s, ytr_s, hidden_size=HIDDEN, learning_rate=LR,
                            epochs=EPOCHS, seed=SEED, print_every=1000, n_output=2,
                            X_validation=Xva_s, y_validation=(yva - y_mean) / y_std)
    print(f"网络训练耗时 {time.perf_counter() - t0:.1f} s")

    linear = linear_fit(Xtr, ytr)
    results = {}
    print(f"\n{'集合':<6} {'目标':<6} {'网络 RMSE':>12} {'线性基线 RMSE':>14}")
    for name, (Xx_s, Xx, yy) in (("train", (Xtr_s, Xtr, ytr)),
                                 ("val", (Xva_s, Xva, yva)),
                                 ("test", (Xte_s, Xte, yte))):
        net = (forward(Xx_s, params) * y_std + y_mean) * COURT_W
        base = linear_predict(Xx, linear)            # 保持归一化单位，rmse 后再乘场宽
        for j, target in enumerate(TARGETS):
            n_rmse = rmse(yy[:, j] * COURT_W, net[:, j])
            b_rmse = rmse(yy[:, j], base[:, j]) * COURT_W
            results[(name, target)] = (n_rmse, b_rmse)
            print(f"{name:<6} {target:<6} {n_rmse:>10.2f} px {b_rmse:>12.2f} px")

    out = Path(__file__).parent / "landing_model.npz"
    np.savez(out, **params, x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std, court_w=COURT_W,
             hidden_size=HIDDEN, learning_rate=LR, epochs=EPOCHS, seed=SEED,
             test_rmse_net=results[("test", "落点")][0],
             test_rmse_linear=results[("test", "落点")][1],
             test_rmse_ic=results[("test", "截击点")][0])
    print(f"\n模型已保存 {out.name}（test：落点 {results[('test', '落点')][0]:.2f} px / "
          f"截击点 {results[('test', '截击点')][0]:.2f} px）")


if __name__ == "__main__":
    main()
