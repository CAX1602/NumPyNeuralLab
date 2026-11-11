"""11-14 核心实验：一次跑完五组对照，全部存图 + 一张证据表。

运行：
    .venv\\Scripts\\python.exe experiments.py

产出（都在 docs/ 下）：
    decision_boundary.png   网络 vs 线性基线的决策边界
    noise_experiment.png    噪声 0.1 / 0.3 / 0.5 的准确率退化
    learning_rate.png       学习率 0.01 / 0.1 / 1 / 10 的损失曲线
    hidden_nodes.png        隐藏节点 1 / 2 / 4 / 8 的训练/测试准确率
    experiment_results.csv  证据表（可直接贴进报告）

图里用英文标签，避免 matplotlib 默认字体缺中文；结论写在进展记录里。
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt
import numpy as np

from data import make_xor, split_data
from network import forward, binary_cross_entropy, sigmoid
from train import train, accuracy

DOCS = Path(__file__).parent / "docs"
DOCS.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]          # 多 seed 取平均，减少偶然性
EPOCHS = 2000
ROWS = []                  # 证据表的每一行


def train_once(noise=0.1, hidden=4, lr=1.0, epochs=EPOCHS, seed=42, n=400):
    """生成数据 → 划分 → 训练，返回准确率、损失和用到的数据。"""
    X, y = make_xor(n_samples=n, noise=noise, seed=seed)
    Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, seed=seed)
    params, hist = train(
        Xtr, ytr, hidden_size=hidden, learning_rate=lr, epochs=epochs,
        seed=seed, print_every=0, X_validation=Xva, y_validation=yva,
    )
    return {
        "params": params, "hist": hist,
        "train_acc": accuracy(Xtr, ytr, params),
        "val_acc": accuracy(Xva, yva, params),
        "test_acc": accuracy(Xte, yte, params),
        "train_loss": binary_cross_entropy(ytr, forward(Xtr, params)),
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": yte,
    }


def train_linear(X, y, lr=1.0, epochs=EPOCHS, seed=42):
    """线性基线：没有隐藏层的逻辑回归，同一个 BCE + 梯度下降。"""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.5, size=(2, 1))
    b = np.zeros((1, 1))
    m = X.shape[0]
    hist = []
    for _ in range(epochs):
        A = sigmoid(X @ W + b)
        hist.append(binary_cross_entropy(y, A))
        dZ = (A - y) / m
        W -= lr * (X.T @ dZ)
        b -= lr * dZ.sum(axis=0, keepdims=True)
    return {"W": W, "b": b}, hist


def linear_forward(X, p):
    return sigmoid(X @ p["W"] + p["b"])


def linear_acc(X, y, p):
    return float(((linear_forward(X, p) > 0.5) == y).mean())


def add_row(experiment, setting, train_acc, val_acc, test_acc, train_loss, note=""):
    ROWS.append({
        "experiment": experiment, "setting": setting,
        "train_acc": f"{train_acc:.4f}", "val_acc": f"{val_acc:.4f}",
        "test_acc": f"{test_acc:.4f}", "train_loss": f"{train_loss:.6f}",
        "note": note,
    })


def scatter_points(ax, X, y):
    c0 = X[y.ravel() == 0]
    c1 = X[y.ravel() == 1]
    ax.scatter(c0[:, 0], c0[:, 1], c="tab:blue", s=14, alpha=0.7,
               edgecolors="white", linewidths=0.3, label="y=0")
    ax.scatter(c1[:, 0], c1[:, 1], c="tab:orange", s=14, alpha=0.7,
               edgecolors="white", linewidths=0.3, label="y=1")


# ---------------------------------------------------------------------------
# 1. 决策边界 + 线性基线
# ---------------------------------------------------------------------------
def experiment_boundary():
    res = train_once(noise=0.1, hidden=4, lr=1.0, seed=42)
    lp, _ = train_linear(res["Xtr"], res["ytr"], lr=1.0, seed=42)
    lin_test = linear_acc(res["Xte"], res["yte"], lp)
    lin_train = linear_acc(res["Xtr"], res["ytr"], lp)
    lin_val = linear_acc(res["Xva"], res["yva"], lp)

    grid = np.linspace(-0.5, 1.5, 60)
    xx, yy = np.meshgrid(grid, grid)
    G = np.c_[xx.ravel(), yy.ravel()]
    Pn = forward(G, res["params"]).reshape(xx.shape)
    Pl = linear_forward(G, lp).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = [
        (axes[0], Pn, "Network 2-4-1 (tanh)", res["test_acc"]),
        (axes[1], Pl, "Linear baseline (logistic)", lin_test),
    ]
    cf = None
    for ax, P, title, acc in panels:
        cf = ax.contourf(xx, yy, P, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
        scatter_points(ax, res["Xtr"], res["ytr"])
        ax.set_title(f"{title}\ntest accuracy {acc:.0%}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(cf, ax=axes, label="P(y=1)", shrink=0.85)
    fig.text(0.5, 0.01, "Noisy XOR · 400 samples · noise=0.1",
             ha="center", fontsize=9)
    fig.savefig(DOCS / "decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    add_row("boundary", "network hidden=4 lr=1", res["train_acc"],
            res["val_acc"], res["test_acc"], res["train_loss"])
    add_row("boundary", "linear logistic lr=1", lin_train, lin_val,
            lin_test, binary_cross_entropy(res["ytr"],
                                           linear_forward(res["Xtr"], lp)),
            "no hidden layer")
    print(f"[boundary] network test {res['test_acc']:.0%} | "
          f"linear test {lin_test:.0%}")


# ---------------------------------------------------------------------------
# 2. 噪声实验
# ---------------------------------------------------------------------------
def experiment_noise():
    noises = [0.1, 0.3, 0.5]
    splits = ["train_acc", "val_acc", "test_acc"]
    means = {s: [] for s in splits}
    stds = {s: [] for s in splits}
    for noise in noises:
        runs = [train_once(noise=noise, seed=s) for s in SEEDS]
        for s in splits:
            vals = [r[s] for r in runs]
            means[s].append(float(np.mean(vals)))
            stds[s].append(float(np.std(vals)))
        add_row("noise", f"noise={noise}", means["train_acc"][-1],
                means["val_acc"][-1], means["test_acc"][-1],
                float(np.mean([r["train_loss"] for r in runs])),
                f"mean of {len(SEEDS)} seeds")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"train_acc": "train", "val_acc": "validation", "test_acc": "test"}
    for s in splits:
        ax.errorbar(noises, means[s], yerr=stds[s], marker="o", capsize=3,
                    label=labels[s])
    ax.set_xlabel("label noise (std of Gaussian jitter)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(noises)
    ax.set_title("Accuracy vs noise (400 samples, hidden=4, 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOCS / "noise_experiment.png", dpi=150)
    plt.close(fig)
    print("[noise] test acc:",
          {n: f"{m:.0%}" for n, m in zip(noises, means["test_acc"])})


# ---------------------------------------------------------------------------
# 3. 学习率对比
# ---------------------------------------------------------------------------
def experiment_learning_rate():
    lrs = [0.01, 0.1, 1.0, 10.0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in lrs:
        res = train_once(lr=lr, seed=42)
        loss = np.array(res["hist"]["train"], dtype=float)
        ax.plot(loss, label=f"lr={lr}")
        final = res["train_loss"]
        note = "" if np.isfinite(final) else "diverged (NaN)"
        add_row("learning_rate", f"lr={lr}", res["train_acc"],
                res["val_acc"], res["test_acc"], final, note)
        print(f"[lr={lr}] final loss {final:.4f} | "
              f"test {res['test_acc']:.0%} {note}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("Learning rate comparison (hidden=4, noise=0.1)")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(DOCS / "learning_rate.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 隐藏节点对比
# ---------------------------------------------------------------------------
def experiment_hidden_nodes():
    sizes = [1, 2, 4, 8]
    train_m, train_s, test_m, test_s = [], [], [], []
    for h in sizes:
        runs = [train_once(hidden=h, seed=s) for s in SEEDS]
        tr = [r["train_acc"] for r in runs]
        te = [r["test_acc"] for r in runs]
        train_m.append(float(np.mean(tr))); train_s.append(float(np.std(tr)))
        test_m.append(float(np.mean(te))); test_s.append(float(np.std(te)))
        add_row("hidden_nodes", f"hidden={h}", train_m[-1],
                float(np.mean([r["val_acc"] for r in runs])), test_m[-1],
                float(np.mean([r["train_loss"] for r in runs])),
                f"mean of {len(SEEDS)} seeds")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sizes, train_m, yerr=train_s, marker="o", capsize=3, label="train")
    ax.errorbar(sizes, test_m, yerr=test_s, marker="s", capsize=3, label="test")
    ax.set_xlabel("hidden units")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(sizes)
    ax.set_title("Hidden units: train vs test (400 samples, noise=0.1, 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOCS / "hidden_nodes.png", dpi=150)
    plt.close(fig)
    print("[hidden] test acc:",
          {h: f"{m:.0%}" for h, m in zip(sizes, test_m)})


def write_table():
    path = DOCS / "experiment_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"\n证据表已保存到：{path}")


def main():
    experiment_boundary()
    experiment_noise()
    experiment_learning_rate()
    experiment_hidden_nodes()
    write_table()
    print("全部完成，图在 docs/ 下。")


if __name__ == "__main__":
    main()