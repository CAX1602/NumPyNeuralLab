from pathlib import Path

import matplotlib.pyplot as plt

from data import make_xor, split_data
from network import forward, binary_cross_entropy
from train import train, accuracy

# 400 道带噪声的 XOR 题
X, y = make_xor(n_samples=400, noise=0.1, seed=42)

(
    X_train, y_train,
    X_validation, y_validation,
    X_test, y_test
) = split_data(X, y, seed=42)

print("训练集：", X_train.shape, "验证集：", X_validation.shape,
      "测试集：", X_test.shape)

parameters, history = train(
    X_train, y_train,
    hidden_size=4,
    learning_rate=1.0,
    epochs=2000,
    seed=42,
    print_every=400,
    X_validation=X_validation,
    y_validation=y_validation
)

print("\n最终结果：")
print("  训练损失：", f"{binary_cross_entropy(y_train, forward(X_train, parameters)):.6f}")
print("  训练准确率：", accuracy(X_train, y_train, parameters))
print("  验证准确率：", accuracy(X_validation, y_validation, parameters))
print("  测试准确率：", accuracy(X_test, y_test, parameters))

# 画损失曲线
plt.figure(figsize=(7, 4.5))
plt.plot(history["train"], label="train")
plt.plot(history["validation"], label="validation", linestyle="--")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("binary cross-entropy loss (log scale)")
plt.title("Training on noisy XOR (400 samples)")
plt.legend()
plt.grid(alpha=0.25, which="both")
plt.tight_layout()

docs = Path(__file__).parent / "docs"
docs.mkdir(exist_ok=True)
output = docs / "loss_curve.png"
plt.savefig(output, dpi=150)
print("\n损失曲线已保存到：", output)

plt.show()
