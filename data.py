import numpy as np


def make_xor(n_samples=400, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)

    points = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    answers = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    # 随机从四类点中抽取 n_samples 次
    indices = rng.integers(0, len(points), size=n_samples)

    X = points[indices]
    y = answers[indices]

    # 给每个点加入少量随机偏移
    X = X + rng.normal(0, noise, X.shape)

    return X, y


def split_data(X, y, seed=42):
    rng = np.random.default_rng(seed)

    # 随机打乱题目的编号
    indices = rng.permutation(len(X))

    # 60% 训练，20% 验证，20% 测试
    train_end = int(len(X) * 0.6)
    validation_end = int(len(X) * 0.8)

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_validation = X[validation_indices]
    y_validation = y[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test
    )


# 下面的代码只在直接运行 data.py 时执行，被 import 时不会画图
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = make_xor()

    print("前 5 道题的输入：")
    print(X[:5])

    print("\n前 5 道题的正确答案：")
    print(y[:5].ravel())

    print("\nX 的形状：", X.shape)
    print("y 的形状：", y.shape)

    # 按正确答案把点分成两组
    points_0 = X[y.ravel() == 0]
    points_1 = X[y.ravel() == 1]

    plt.figure(figsize=(6, 6))

    plt.scatter(
        points_0[:, 0],
        points_0[:, 1],
        color="tab:blue",
        alpha=0.7,
        label="Answer 0"
    )

    plt.scatter(
        points_1[:, 0],
        points_1[:, 1],
        color="tab:orange",
        alpha=0.7,
        label="Answer 1"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("XOR Dataset")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

    (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test
    ) = split_data(X, y)

    print("\n训练集：", X_train.shape, y_train.shape)
    print("验证集：", X_validation.shape, y_validation.shape)
    print("测试集：", X_test.shape, y_test.shape)
