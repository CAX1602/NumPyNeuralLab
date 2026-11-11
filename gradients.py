import numpy as np

from network import forward, binary_cross_entropy, initialize_parameters, backward

# 四道最干净的 XOR 题，和 network.py 里演示用的是同一组
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


def compute_loss(X, y, parameters):
    """给一套旋钮，算出网络现在错得多离谱（一个数字）。"""
    probabilities = forward(X, parameters)
    return binary_cross_entropy(y, probabilities)


def numeric_gradient_single(X, y, parameters, key, row, col, epsilon=1e-4):
    """只算一个旋钮的导数。

    做法：把这个旋钮拧大一点点，算一次损失；拧小一点点，再算一次损失。
    损失的变化量 ÷ 旋钮的变化量，就是这个旋钮的导数。
    """
    # 复制两份参数，避免把原始参数改坏
    plus = {name: value.copy() for name, value in parameters.items()}
    minus = {name: value.copy() for name, value in parameters.items()}

    # 只动指定的那一个旋钮
    plus[key][row, col] += epsilon
    minus[key][row, col] -= epsilon

    loss_plus = compute_loss(X, y, plus)
    loss_minus = compute_loss(X, y, minus)

    return (loss_plus - loss_minus) / (2 * epsilon)


def numeric_gradient(X, y, parameters, epsilon=1e-4):
    """一次算出全部旋钮的导数，返回的字典形状和 parameters 完全一样。"""
    gradients = {}

    for name, value in parameters.items():
        one_gradient = np.zeros_like(value)

        for row in range(value.shape[0]):
            for col in range(value.shape[1]):
                one_gradient[row, col] = numeric_gradient_single(
                    X, y, parameters, name, row, col, epsilon
                )

        gradients[name] = one_gradient

    return gradients


if __name__ == "__main__":
    parameters = initialize_parameters(hidden_size=4)

    print("当前损失：", compute_loss(X, y, parameters))

    gradients = numeric_gradient(X, y, parameters)

    knob_count = sum(value.size for value in parameters.values())
    print(f"\n旋钮总数：{knob_count} 个，损失函数被调用 {knob_count * 2} 次")

    print("\n全部导数（数值法）：")
    for name in ["W1", "b1", "W2", "b2"]:
        print(f"\n{name}：")
        print(gradients[name])

    analytic = backward(X, y, parameters)

    print("\n全部导数（反向传播，解析法）：")
    for name in ["W1", "b1", "W2", "b2"]:
        print(f"\n{name}：")
        print(analytic[name])

    print("\n两种算法的梯度检查（标准公式：范数差 ÷ 范数和）：")
    worst = 0.0
    for name in ["W1", "b1", "W2", "b2"]:
        analytic_matrix = analytic[name]
        numeric_matrix = gradients[name]

        difference = np.linalg.norm(analytic_matrix - numeric_matrix)
        total = (np.linalg.norm(analytic_matrix)
                 + np.linalg.norm(numeric_matrix))
        check = difference / total
        worst = max(worst, check)

        print(f"  {name}: 梯度检查 {check:.2e}，最大逐元素绝对差 "
              f"{np.abs(analytic_matrix - numeric_matrix).max():.2e}")

    print(f"\n最严的一格：{worst:.2e}")
    print("小于 1e-7 说明反向传播写对了。" if worst < 1e-7 else "偏大了，反向传播可能有错。")

    print("\n导数形状：", {name: value.shape for name, value in gradients.items()})
