"""12-05 三 AI 对照实验（11-24 提前，v2 配对设计）：网络 vs 线性基线 vs 站中间不动。

运行：
    .venv\\Scripts\\python.exe compare_ai.py            # 纯预测（关末端精修，主实验）
    .venv\\Scripts\\python.exe compare_ai.py --refine   # 游戏默认的末端精修也打开（生态效度）

设计（v1 的教训：对手全是深区高远 + 末端精修兜底，三组几乎打平）：
- **同一串 80 个球喂给三个 AI**（完全配对）：随机采样左半场击球，
  只保留落点覆盖整个右半场（620~1160）的来球，平球/高球/杀球混合。
- 被测 AI 在右侧；没有左侧玩家，球由脚本直接发射，R 回击或球落地即下一球。
- 主实验**关闭末端模拟器精修**（game.AIController.refine=False），
  跑位完全由预测器决定，才能把预测质量差异暴露出来。
- 主指标：**接球成功率** = R 成功回击数 / 80。附：回球过网率、平均回合时长。
- 模式：net（双输出网络）/ linear（同特征最小二乘，同一 16000 训练样本）/
  static（站右半场中间 900 不动，只挥拍）。
"""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import game
from characters import CHARACTERS
from landing import flatten_obs, load_dataset
from landing_baseline import fit as linear_fit, predict as linear_predict
from physics import COURT_W, NET_X, simulate_batch

N_SHOTS = 80
SEED = 20261205


def make_shots(n=N_SHOTS, seed=SEED):
    """生成配对的来球序列：落点均匀覆盖右半场（含前后场角落）。"""
    rng = np.random.default_rng(seed)
    shots = []
    while len(shots) < n:
        m = n * 4
        x0 = rng.uniform(150, NET_X - 40, m)
        y0 = rng.uniform(120, 420, m)
        v0 = rng.uniform(1200, 4600, m)
        angle = rng.uniform(-0.55, 0.95, m)
        landing, _, _, ic = simulate_batch(x0, y0, v0, angle, np.ones(m),
                                           with_intercept=True)
        ok = np.isfinite(landing) & (landing > NET_X + 20) & (landing < COURT_W - 40)
        for x, y, v, a in zip(x0[ok], y0[ok], v0[ok], angle[ok]):
            shots.append((float(x), float(y), float(v), float(a)))
            if len(shots) >= n:
                break
    return shots[:n]


class LinearAI(game.AIController):
    linear_params = None

    def predict_pair(self, ball):
        frames = ball.obs_frames()
        if frames is not None and self.linear_params is not None:
            self.net_used += 1
            X = flatten_obs(frames[None, ...])
            out = linear_predict(X, self.linear_params)[0] * COURT_W
            return float(out[0]), float(out[1])
        return super().predict_pair(ball)


class StaticAI(game.AIController):
    def update(self, dt, ball, serve_pending):
        p = self.p
        if ball.live and ball.last_hitter != p.side:
            near = abs(ball.x - p.x) < self.swing_dist
            height_ok = 50 < ball.y < 250 + p.h
            t_arrive = abs(ball.x - p.x) / (abs(ball.vx) + 1e-6)
            if (near or t_arrive < self.lead_time) and height_ok and ball.vy < 150:
                defend = ball.vy < -400.0 and ball.y < 240.0
                p.start_swing(ball, force="lift" if defend else None)
        return 0.0, False


def run(mode, shots, refine):
    g = game.Game(CHARACTERS[0], CHARACTERS[6], smoke=False, race=False)
    g.serve_timer = 1e9                    # 不让游戏自己发球
    if mode == "linear":
        g.ai = LinearAI(g.ai_p, opponent=g.human)
    elif mode == "static":
        g.ai = StaticAI(g.ai_p, opponent=g.human)
    g.ai.refine = refine

    returned = 0
    over_net = 0
    for x0, y0, v0, angle in shots:
        # 每球重置双方站位与状态，保证配对公平
        g.ai_p.x, g.ai_p.h = g.ai_p.recover_x, 0.0
        g.ai.state = "RECOVER"
        g.human.x = 260.0
        g.ball.launch(x0, y0, v0, angle, 1, "L")
        g.hits = 0
        for _ in range(60 * 6):            # 单球最多 6 s
            g.update(1 / 60, {})
            if not g.ball.live:
                break
        if g.hits > 0:
            returned += 1
            over_net += int(g.ball.last_hitter == "R")
    return returned, over_net


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true",
                    help="打开游戏默认的末端模拟器精修（主实验默认关闭）")
    args = ap.parse_args()

    from train_landing import split
    X, y = load_dataset(with_ic=True)
    (Xtr, ytr), _, _ = split(X, y)
    LinearAI.linear_params = linear_fit(Xtr, ytr)

    shots = make_shots()
    print(f"配对来球 {len(shots)} 个（同一串喂给三种 AI）；"
          f"末端精修 {'开' if args.refine else '关（纯预测）'}")
    print(f"\n{'模式':<8} {'接球成功率':>10} {'回球过网率':>10}")
    for mode in ("net", "linear", "static"):
        returned, over_net = run(mode, shots, args.refine)
        print(f"{mode:<8} {returned / len(shots):>9.1%} {over_net / len(shots):>9.1%}"
              f"  ({returned}/{len(shots)})")


if __name__ == "__main__":
    main()
