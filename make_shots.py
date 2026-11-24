"""生成落点网络的训练数据 shots.npz（11-20；11-24 对齐交付版游戏）。

对齐项：场宽 1200、非线性出球角（physics.launch_angle，与 game.py 同公式）、
球速 5 档 V_MIN/V_MAX = 1900/4100。

运行：
    .venv\\Scripts\\python.exe make_shots.py

覆盖：12 个角色的 power、两种挥拍可达的全部拍角、接触点 s∈[0,1]、左右两个方向。
每条样本存：前 0.15 s 的 4 帧观测 obs(n,4,2)、落点 landing_x、以及生成参数（meta）。
另画一张「阻力 vs 真空」对照图 docs/drag_trajectories.png，证明阻力可见。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import physics
from characters import CHARACTERS
from physics import (
    COURT_W, NET_X, NET_HEIGHT, SWING_THETA, resolve_hit,
    simulate_batch, simulate_path,
)

DOCS = Path(__file__).parent / "docs"
DOCS.mkdir(exist_ok=True)
N = 20000
SEED = 42


def sample_shots(n, seed):
    rng = np.random.default_rng(seed)
    powers = np.array([c["power"] for c in CHARACTERS])
    power = rng.choice(powers, size=n)
    direction = rng.choice([-1.0, 1.0], size=n)
    swing = rng.choice(["clear", "lift"], size=n)
    theta = np.empty(n)
    for name, (lo, hi) in SWING_THETA.items():
        mask = swing == name
        theta[mask] = rng.uniform(lo, hi, size=mask.sum())
    s = rng.uniform(0.0, 1.0, size=n)
    # 击球点在本方半场；高度覆盖地面到起跳可达
    x0 = np.where(direction > 0,
                  rng.uniform(60, NET_X - 60, size=n),
                  rng.uniform(NET_X + 60, COURT_W - 60, size=n))
    y0 = rng.uniform(40, 260, size=n)
    v0, angle = resolve_hit(theta, s, power)
    return dict(theta=theta, s=s, power=power, direction=direction,
                swing=swing, x0=x0, y0=y0, v0=v0, angle=angle)


def main():
    # 只保留「飞过网」的球（真实回合里对手的回球必然过网）；
    # 拒绝采样：不断生成候选，直到攒够 N 条过网球。
    chunks = []
    kept = 0
    candidates = 0
    offset = 0
    while kept < N:
        meta = sample_shots(N, SEED + offset)
        landing, ftime, obs = simulate_batch(
            meta["x0"], meta["y0"], meta["v0"], meta["angle"], meta["direction"])
        d = meta["direction"]
        cross = np.isfinite(landing) & np.where(
            d > 0, landing > NET_X, landing < NET_X)
        chunks.append({k: v[cross] for k, v in meta.items()} | {
            "landing_x": landing[cross], "flight_time": ftime[cross],
            "obs": obs[cross]})
        kept += int(cross.sum())
        candidates += N
        offset += 1
    data = {k: np.concatenate([c[k] for c in chunks])[:N]
            for k in chunks[0]}
    print(f"候选 {candidates} 条，保留过网 {kept} 条，取前 {N} 条"
          f"（过网率 {kept / candidates:.1%}）")

    data["obs"] = data["obs"].astype(np.float32)
    for k in ("theta", "s", "power", "direction", "x0", "y0", "v0", "angle",
              "landing_x", "flight_time"):
        data[k] = data[k].astype(np.float32)

    out = Path(__file__).parent / "shots.npz"
    np.savez_compressed(out, **data)
    print(f"已保存 {out}（{out.stat().st_size / 1e6:.1f} MB）")

    # 覆盖与分布统计
    d = data["direction"]
    in_court = np.where(d > 0,
                        (data["landing_x"] >= NET_X) & (data["landing_x"] <= COURT_W),
                        (data["landing_x"] >= 0) & (data["landing_x"] <= NET_X))
    print(f"拍角范围 [{data['theta'].min():.2f}, {data['theta'].max():.2f}] rad")
    print(f"s 范围 [{data['s'].min():.2f}, {data['s'].max():.2f}]")
    print(f"power 取值 {len(np.unique(data['power']))} 种；"
          f"方向 +1:{int((d > 0).sum())} / -1:{int((d < 0).sum())}")
    print(f"落点在对方场区内比例 {in_court.mean():.1%}")
    print(f"落点分位数 5/50/95: "
          f"{np.percentile(data['landing_x'], [5, 50, 95]).round(0)}")
    print(f"飞行时间分位数 5/50/95: "
          f"{np.percentile(data['flight_time'], [5, 50, 95]).round(2)} s")
    print(f"飞行时间短于 0.15 s（观察窗需截断）比例 "
          f"{(data['flight_time'] < 0.15).mean():.1%}")

    plot_drag_comparison()


def plot_drag_comparison():
    """同一击球，有阻力 vs 真空，证明轨迹明显偏离抛体。

    11-24 示例（5 档球速、1200 场宽）：阻力下落点 828（对方界内），
    真空下同一条轨迹会飞到 1660 —— 直接飞出底线，阻力「救命」。
    """
    kw = dict(x0=300, y0=220, v0=1500, angle=0.5, direction=1)
    _, xd, yd = simulate_path(drag_k=physics.DRAG_K, **kw)
    _, xv, yv = simulate_path(drag_k=0.0, **kw)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xv, yv, "--", color="tab:gray", label="vacuum (no drag)")
    ax.plot(xd, yd, "-", color="tab:red", label=f"drag k={physics.DRAG_K}")
    ax.axvline(NET_X, color="tab:blue", lw=1)
    ax.plot([NET_X, NET_X], [0, NET_HEIGHT], color="tab:blue", lw=3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Air drag visibly shortens and steepens the flight")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOCS / "drag_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"阻力对照图已保存 docs/drag_trajectories.png"
          f"（真空落点 {xv[-1]:.0f} vs 阻力落点 {xd[-1]:.0f}）")


if __name__ == "__main__":
    main()