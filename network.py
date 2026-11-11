import numpy as np


def initialize_parameters(hidden_size=4, seed=42):
    """创建网络里的参数，也就是后来需要调整的“旋钮”。"""
    rng = np.random.default_rng(seed)

    parameters = {
        "W1": rng.normal(0, 0.5, size=(2, hidden_size)),
        "b1": np.zeros((1, hidden_size)),
        "W2": rng.normal(0, 0.5, size=(hidden_size, 1)),
        "b2": np.zeros((1, 1))
    }

    return parameters


def sigmoid(Z):
    """把任意数字压缩到 0 和 1 之间。"""
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def forward(X, parameters):
    """前向传播：从输入一路计算到预测概率。"""
    # 第一层：两个输入 → 隐藏层
    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = np.tanh(Z1)

    # 第二层：隐藏层 → 一个输出
    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = sigmoid(Z2)

    return A2


def backward(X, y, parameters):
    """反向传播：一次前向 + 一次从后往前，就算出全部 17 个旋钮的导数。"""
    m = X.shape[0]

    # 先把前向传播走一遍，中间结果要留着，反方向要用
    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = sigmoid(Z2)

    # 输出层：预测减答案，再按样本数平均
    dZ2 = (A2 - y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # 隐藏层：把输出层的误差按权重分摊回 4 个神经元，再乘 tanh 自己的导数
    dA1 = dZ2 @ parameters["W2"].T
    dZ1 = dA1 * (1 - A1 ** 2)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }


def binary_cross_entropy(y, probabilities):
    """计算预测结果与正确答案之间的损失。"""
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    loss = -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )

    return loss

# 下面的代码只在直接运行 network.py 时执行
if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    parameters = initialize_parameters(hidden_size=4)
    probabilities = forward(X, parameters)
    loss = binary_cross_entropy(y, probabilities)

    print("网络对四道题的猜测：")
    print(probabilities)

    print("\n正确答案：")
    print(y.ravel())

    print("\n当前损失：", loss)
    print("预测结果的形状：", probabilities.shape)
