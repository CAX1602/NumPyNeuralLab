"""生成决策网络的教练数据 decision_shots.npz（11-24）。

运行：
    .venv\\Scripts\\python.exe make_decision.py

在左半场随机采 3 万个局面 [x0, y0, t_in, x_h]，用 decision.oracle_values
（物理模拟器）给三个动作打分。只采过网后的 AI 视角，右半场镜像复用。
"""

from pathlib import Path

import numpy as np

from decision import (
    ACTIONS, JUMP_T, WAIST_Y, feasible_mask, normalize_state, oracle_values,
)
from physics import COURT_W, NET_X

N = 40000             # 11-24 晚：挑球线上调到 200 后可决策局面占比下降，加大采样
SEED = 42


def sample_states(n, seed=SEED):
    rng = np.random.default_rng(seed)
    state = np.empty((n, 4))
    state[:, 0] = rng.uniform(60, NET_X - 40, n)        # 接触点 x（本方半场）
    state[:, 1] = rng.uniform(40, 260, n)               # 接触点高度 y
    state[:, 2] = rng.uniform(0.10, 1.20, n)            # 剩余时间 t_in
    state[:, 3] = rng.uniform(NET_X + 40, COURT_W - 60, n)  # 人类站位 x_h
    return state


def main():
    state = sample_states(N)
    values = oracle_values(state)
    X = normalize_state(state)

    out = Path(__file__).parent / "decision_shots.npz"
    np.savez_compressed(out, X=X.astype(np.float32),
                        values=values.astype(np.float32),
                        state=state.astype(np.float32))
    print(f"已保存 {out.name}：X{X.shape} values{values.shape}")

    mask = feasible_mask(state)
    best = np.argmax(values, axis=1)
    print(f"低球（只能挑，y0 < {WAIST_Y:.0f}）比例 {(state[:, 1] < WAIST_Y).mean():.1%}")
    print(f"高球且来得及跳（真正需要决策）比例 "
          f"{((state[:, 1] >= WAIST_Y) & (state[:, 2] >= JUMP_T)).mean():.1%}")
    for j, name in enumerate(ACTIONS):
        print(f"教练最优动作 = {name:<6} 比例 {(best == j).mean():.1%}；"
              f"可行比例 {mask[:, j].mean():.1%}")
    both = mask[:, 1] & mask[:, 2]
    diff = values[both, 2] - values[both, 1]
    print(f"可决策局面中：跳杀比高远平均多 {diff.mean():.3f} 分；"
          f"跳杀胜出比例 {(diff > 0).mean():.1%}")
    fail = (values == -1.0)
    print(f"三动作结果为 -1（下网/出界/不可行）的比例 {fail.mean():.1%}")


if __name__ == "__main__":
    main()
