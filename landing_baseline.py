"""落点预测的线性基线：最小二乘闭式解，不迭代、不训练。

用途：报告与 12-05 对照。和 landing.py 的网络吃同一份 X(n,8) / y(n,1)
（坐标已 ÷COURT_W 归一化），回答「网络相比一条直线多学会了什么」。
真实数据同样等物理定稿后再用；现在只用合成数据验证正确性。
"""

import numpy as np

from physics import COURT_W


def fit(X, y):
    """最小二乘：对 [X, 1] 解一次线性方程，返回 {"W": (d,1), "b": (1,1)}。"""
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"W": coef[:-1], "b": coef[-1:]}


def predict(X, params):
    """线性预测，输出与 y 同单位（归一化落点）。"""
    return X @ params["W"] + params["b"]


def predict_landing_px(X, params):
    """归一化输入 → 落点像素坐标，与 landing.predict_landing_px 同接口。"""
    return predict(X, params) * COURT_W


if __name__ == "__main__":
    from landing import rmse

    def synth(n, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, size=(n, 8))
        y = (0.2 + 0.5 * X[:, 0:1] - 0.3 * X[:, 1:2]
             + 0.1 * X[:, 0:1] * X[:, 2:3])
        return X, y

    X, y = synth(400, seed=0)
    X_val, y_val = synth(80, seed=99)
    params = fit(X, y)

    print(f"线性基线 train RMSE {rmse(y, predict(X, params)) * COURT_W:.2f} px")
    print(f"线性基线 val   RMSE {rmse(y_val, predict(X_val, params)) * COURT_W:.2f} px")
    print("注意：这份合成目标太接近线性（0.1*x0*x2 交互项的大部分能被最佳线性"
          "逼近吸收），闭式基线比 3000 轮梯度下降的网络（约 13/15 px）还准。"
          "合成数据只用于验证正确性；网络 vs 线性的真正对比要等真实弹道数据。")
