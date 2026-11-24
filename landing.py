"""落点回归网络：8 → 16(tanh) → 1(线性)，0.5·MSE 损失。

与 XOR 分类网络（network.py）同一套结构，只有两处不同：
1. 输出层不做 sigmoid，直接输出线性值——落点 x 是连续量，不是概率。
2. 损失从交叉熵换成 0.5·MSE。0.5 这个系数让输出层的 dZ2 = (A2 - y) / m
   与 sigmoid+BCE 逐字同形，隐藏层反向公式完全复用——「损失函数决定任务」的直接证据。

输入 = 击球后 0.15 s 的 4 帧观测（obs(n,4,2) 展平成 8 维），输出 = 落点 x。
坐标统一除以场宽 COURT_W 做归一化（固定常数，不是从数据统计出来的，无泄漏）。

**训练已完成（11-24）**：train_landing.py 用训练集 z-score 标准化训出
landing_model.npz（test RMSE 14.58 px），推理走 load_model + predict_with_model。
本模块的 initialize/train 已泛化出 n_output 参数，决策网络（decision.py）复用。
"""

import numpy as np

from physics import COURT_W


def initialize_parameters(n_input=8, hidden_size=16, seed=42, n_output=1):
    """创建旋钮。8→16→1 时共 8*16+16+16+1 = 161 个参数。

    n_output 默认 1（落点回归）；决策网络传 3（每个动作一个价值分）。
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.5, size=(n_input, hidden_size)),
        "b1": np.zeros((1, hidden_size)),
        "W2": rng.normal(0, 0.5, size=(hidden_size, n_output)),
        "b2": np.zeros((1, n_output)),
    }


def forward(X, parameters):
    """前向传播：tanh 隐藏层 + 线性输出（不压 sigmoid）。"""
    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = np.tanh(Z1)
    return A1 @ parameters["W2"] + parameters["b2"]


def mse_loss(y, predictions):
    """0.5 · 均方误差。取 0.5 是为了反向传播与分类网络同形。"""
    return 0.5 * float(np.mean((predictions - y) ** 2))


def compute_loss(X, y, parameters):
    return mse_loss(y, forward(X, parameters))


def rmse(y, predictions):
    """均方根误差，和 y 同单位（用于报告「平均差多少」）。"""
    return float(np.sqrt(np.mean((predictions - y) ** 2)))


def backward(X, y, parameters):
    """反向传播：除输出层少了 sigmoid 导数外，与 network.backward 完全一致。"""
    m = X.shape[0]

    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = np.tanh(Z1)
    A2 = A1 @ parameters["W2"] + parameters["b2"]

    # 0.5·MSE + 线性输出：dZ2 与 sigmoid+BCE 同一形式。
    # 损失对全部输出元素取平均 → 除以元素总数（单输出时 A2.size = m，与原来一致）
    dZ2 = (A2 - y) / A2.size
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ parameters["W2"].T
    dZ1 = dA1 * (1 - A1 ** 2)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(X, y, hidden_size=16, learning_rate=0.1, epochs=3000,
          seed=42, print_every=500, X_validation=None, y_validation=None,
          n_output=1):
    """全批量梯度下降，接口与 train.py 对齐。返回 (参数, 历史)。

    学习率默认 0.1：回归的梯度尺度与分类不同，0.5 会在几百轮内发散（11-19 实测），
    0.1–0.3 在合成数据上稳定；与「lr=10 在 4 点集发散、400 点集收敛」同一课——
    稳定学习率不能跨设置照搬。真实训练的学习率与标准化见 train_landing.py。
    """
    parameters = initialize_parameters(X.shape[1], hidden_size, seed, n_output)
    history = {"train": [], "validation": []}

    for epoch in range(epochs):
        loss = compute_loss(X, y, parameters)

        validation_loss = None
        if X_validation is not None:
            validation_loss = compute_loss(X_validation, y_validation, parameters)
            history["train"].append(loss)
            history["validation"].append(validation_loss)

        gradients = backward(X, y, parameters)
        for name in parameters:
            parameters[name] -= learning_rate * gradients[name]

        if print_every and epoch % print_every == 0:
            message = f"第 {epoch:5d} 轮   训练损失 {loss:.6f}"
            if validation_loss is not None:
                message += f"   验证损失 {validation_loss:.6f}"
            print(message)

    return parameters, history


def flatten_obs(obs):
    """obs(n,4,2) → X(n,8)：按帧展平 [x0,y0,x1,y1,...]，坐标除以场宽。"""
    return (obs.reshape(obs.shape[0], -1) / COURT_W).astype(float)


def load_dataset(path="shots.npz", with_ic=False):
    """读 shots.npz → (X(n,8), y(n,1))，全部归一化到场宽单位。只读，不训练。

    with_ic=True 时 y 为 (n,2)：[落点 x, 截击点 x]（下行穿过 250 高度的 x），
    供双输出模型训练——AI 跑位用截击点，不是落点。
    """
    d = np.load(path)
    X = flatten_obs(d["obs"])
    if with_ic:
        y = (np.stack([d["landing_x"], d["ic_x"]], axis=1) / COURT_W).astype(float)
    else:
        y = (d["landing_x"].reshape(-1, 1) / COURT_W).astype(float)
    return X, y


def predict_landing_px(X, parameters):
    """归一化输入 → 落点像素坐标，供 11-28 游戏内 predict() 调用。

    注意：只适用于未做 z-score 标准化的参数。train_landing.py 训出的
    landing_model.npz 带训练集标准化统计量，请用 load_model + predict_with_model。
    """
    return forward(X, parameters) * COURT_W


def load_model(path="landing_model.npz"):
    """读 train_landing.py 保存的模型，返回 (parameters, norm)。

    norm 里有训练集的标准化统计量（x_mean/x_std/y_mean/y_std）与 court_w，
    推理时必须用同一套统计量，否则输入分布对不上。
    """
    d = np.load(path)
    parameters = {k: d[k] for k in ("W1", "b1", "W2", "b2")}
    norm = {"x_mean": d["x_mean"], "x_std": d["x_std"],
            "y_mean": d["y_mean"], "y_std": d["y_std"],   # 标量或逐输出向量（双输出）
            "court_w": float(d["court_w"])}
    return parameters, norm


def predict_with_model(X, parameters, norm):
    """游戏推理入口：归一化输入 X(n,8)（÷COURT_W）→ 落点像素坐标。"""
    z = forward((X - norm["x_mean"]) / norm["x_std"], parameters)
    return (z * norm["y_std"] + norm["y_mean"]) * norm["court_w"]


# 合成数据自检：答案已知的函数，不碰 shots.npz
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(400, 8))
    y = (0.2 + 0.5 * X[:, 0:1] - 0.3 * X[:, 1:2]
         + 0.1 * X[:, 0:1] * X[:, 2:3])       # 线性项 + 一点非线性
    X_val = rng.uniform(0, 1, size=(80, 8))
    y_val = (0.2 + 0.5 * X_val[:, 0:1] - 0.3 * X_val[:, 1:2]
             + 0.1 * X_val[:, 0:1] * X_val[:, 2:3])

    parameters, _ = train(X, y, hidden_size=16, learning_rate=0.3,
                          epochs=3000, seed=42, print_every=500,
                          X_validation=X_val, y_validation=y_val)

    print(f"\n训练 RMSE {rmse(y, forward(X, parameters)) * COURT_W:.2f} px")
    print(f"验证 RMSE {rmse(y_val, forward(X_val, parameters)) * COURT_W:.2f} px")
