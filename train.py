import numpy as np

from network import (
    initialize_parameters,
    forward,
    backward,
    binary_cross_entropy
)


def accuracy(X, y, parameters):
    """把概率按 0.5 切成 0/1，再和答案比较，返回答对的比例。"""
    predictions = (forward(X, parameters) > 0.5).astype(float)
    return float((predictions == y).mean())


def train(X, y, hidden_size=4, learning_rate=1.0, epochs=3000,
          seed=42, print_every=500,
          X_validation=None, y_validation=None):
    """梯度下降。返回 (参数, 历史记录)，历史记录里存每轮的损失。"""
    parameters = initialize_parameters(hidden_size, seed)
    history = {"train": [], "validation": []}

    for epoch in range(epochs):
        probabilities = forward(X, parameters)
        loss = binary_cross_entropy(y, probabilities)

        validation_loss = None
        if X_validation is not None:
            validation_loss = binary_cross_entropy(
                y_validation, forward(X_validation, parameters)
            )
            history["train"].append(loss)
            history["validation"].append(validation_loss)

        gradients = backward(X, y, parameters)

        # 这一步就是全部的训练：往梯度的反方向挪
        for name in parameters:
            parameters[name] -= learning_rate * gradients[name]

        if print_every and epoch % print_every == 0:
            message = f"第 {epoch:5d} 轮   训练损失 {loss:.6f}"
            if validation_loss is not None:
                message += f"   验证损失 {validation_loss:.6f}"
            print(message)

    return parameters, history


# 四道最干净的 XOR 题，直接运行 train.py 时的自检
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

    learning_rate = 1.0

    parameters, _ = train(X, y, learning_rate=learning_rate)

    probabilities = forward(X, parameters)
    loss = binary_cross_entropy(y, probabilities)

    print(f"\n学习率 {learning_rate}，训练结束，最终损失 {loss:.8f}")
    print("\n预测概率：", np.round(probabilities.ravel(), 6))
    print("正确答案：", y.ravel())
    print("准确率：  ", accuracy(X, y, parameters))
