import numpy as np

# 故意用一眼能认出来的数字，方便看 broadcasting 到底做了什么
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

W1 = np.array([
    [1, 2, 3, 4],
    [10, 20, 30, 40]
], dtype=float)

b1 = np.array([
    [0.5, -0.5, 1.0, 100.0]
], dtype=float)

print("X 的形状：", X.shape)
print("W1 的形状：", W1.shape)
print("X @ W1 的形状：", (X @ W1).shape)
print("b1 的形状：", b1.shape)

print("\nX @ W1（还没加偏置）：")
print(X @ W1)

print("\nb1：")
print(b1)

print("\nX @ W1 + b1 的结果：")
Z1 = X @ W1 + b1
print(Z1)
print("形状：", Z1.shape)

print("\n——— 把 broadcasting 手写回真正的线性代数 ———")
ones = np.ones((4, 1))
print("np.ones((4,1)) @ b1 的形状：", (ones @ b1).shape)
print(ones @ b1)
print("和 X @ W1 + ones @ b1 完全相等吗？",
      np.array_equal(Z1, X @ W1 + ones @ b1))

print("\n——— 偏置的含义：输入全 0 时这一层的输出 ———")
print("样本 1 的输入是 (0, 0)，它的 X @ W1 =", (X @ W1)[0])
print("加上 b1 之后变成", Z1[0], "—— 正好就是 b1 本身")

print("\n——— 三种写法对比 ———")
print("b1 形状 (1, 4)：", (X @ W1 + np.array([[0.5, -0.5, 1.0, 100.0]])).shape)
print("b1 形状 (4,)  ：", (X @ W1 + np.array([0.5, -0.5, 1.0, 100.0])).shape)
print("b1 形状 (4, 1)：", (X @ W1 + np.ones((4, 1))).shape,
      "（这是每道题加同一个数，不是每层加偏置，含义完全不同）")

print("\n——— 形状对不上会直接报错 ———")
try:
    np.ones((4, 2)) + np.ones((1, 4))
except ValueError as error:
    print("ValueError:", error)
