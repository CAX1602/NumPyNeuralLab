"""决策网络（双网络之二）：局面 → 三种回球动作的价值分，选 argmax。

落点网络（landing.py）负责「球会落哪」，本模块负责「这球怎么回」。
动作全部对应 game.py 现有机制，**不需要新挥拍**：
- lift  挑球：球低于腰部时游戏自动选的动作（θ=0.90, s=0.70）
- clear 高远：高球的基础回球（θ=1.15, s=0.80）
- jump  跳高远（= 杀球效果，用户 11-24 确认）：同一个 clear 挥拍，
        起跳把接触点抬高 130 px，拍面压平 θ=0.55、s=1.00 → 出球角约 −18°、更快

教练 = 物理模拟器：对每种动作模拟落点 L 与飞行时间 T，按
    价值 = min(|L − x_h| / T / (2·OPP_SPEED), 1.5) + RECOVER_W · T
打分——对手（站位 x_h，速度 OPP_SPEED）需要跑多快才够得着，越大越难接；
飞行时间也给自己回位留时间（高远因此不总是输给跳杀）。
下网（按游戏规则从网下穿过即被挡）/ 出界 / 不可行动作 = −1。可行规则与游戏一致：
- 球低于腰部（y0 < 90）：只能挑球；高球：只能高远或跳高远
- 剩余时间 t_in < 0.30 s：来不及起跳，jump 不可行

**终版形态（11-24）**：价值只用来生成标签与评估；真正的决策网络是
**二分类器**「跳杀 vs 高远」（4→16→1 sigmoid + BCE，复用 network.py），
游戏入口 decide()：低球自动挑，高球按概率选。价值回归实验一致率仅 58.9%，
改直接学分界面后见 train_decision.py 的结果。

状态只取 AI 在左半场（向右打）的情形；右半场用镜像 x → COURT_W − x 复用。
"""

import numpy as np

from physics import COURT_W, NET_HEIGHT, NET_X, resolve_hit, simulate_batch
from landing import forward

ACTIONS = ("lift", "clear", "jump")
ACTION_PARAMS = {
    "lift": (0.90, 0.70, 0.0),
    "clear": (1.15, 0.80, 0.0),
    "jump": (0.55, 1.00, 130.0),
}
WAIST_Y = 200.0       # 挑球判定线：离脚 200 px ≈ 胸口（11-24 从腰 90 上调，与 game.WAIST_H 一致）
JUMP_T = 0.30         # 起跳需要的提前时间（s）
JUMP_H = 130.0        # 起跳抬高的接触点（px），与角色 jump 100–155 同量级
OPP_SPEED = 390.0     # 对手移动速度（px/s），取 12 角色速度档的中位
RECOVER_W = 0.25      # 自己回位时间的价值（分/秒）：高远飞得久 = 自己能回位，
                      # 否则跳杀在 87% 局面都占优（11-24 实测），0.25 时约六成，更像真实取舍
FAIL = -1.0           # 下网 / 出界 / 不可行 的统一价值
T_SCALE = 1.5         # t_in 的归一化常数（固定，不是统计量）


def normalize_state(state):
    """原始状态 [x0, y0, t_in, x_h]（物理单位）→ 归一化输入 X(n,4)。"""
    state = np.asarray(state, dtype=float)
    X = state.copy()
    X[:, [0, 1, 3]] /= COURT_W
    X[:, 2] /= T_SCALE
    return X


def feasible_mask(state):
    """每个状态下三个动作是否可行（游戏规则），返回 (n,3) bool。"""
    state = np.asarray(state, dtype=float)
    y0, t_in = state[:, 1], state[:, 2]
    low = y0 < WAIST_Y
    mask = np.zeros((state.shape[0], 3), dtype=bool)
    mask[:, 0] = low                              # 低球只能挑
    mask[:, 1] = ~low                             # 高球才能高远
    mask[:, 2] = (~low) & (t_in >= JUMP_T)        # 高球 + 来得及才跳
    return mask


def value_of_outcome(landing, flight_time, x_h):
    """把一次回球的结果翻译成价值分（向量化）。

    价值 = 对手难度分 + RECOVER_W × 飞行时间（自己回位的收益）；失败 = FAIL。
    """
    landing = np.asarray(landing, dtype=float)
    flight_time = np.asarray(flight_time, dtype=float)
    valid = np.isfinite(landing) & (landing > NET_X) & (landing <= COURT_W)
    required = np.abs(landing - x_h) / np.maximum(flight_time, 0.05)
    value = np.minimum(required / (2 * OPP_SPEED), 1.5) + RECOVER_W * flight_time
    return np.where(valid, value, FAIL)


def oracle_values(state, power=1.0):
    """模拟器教练：对 state(n,4) 逐动作仿真打分，返回 (n,3)。

    不可行动作保持 FAIL；power 取 AI 角色的 power（默认 1.0）。
    """
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    values = np.full((n, 3), FAIL)
    mask = feasible_mask(state)
    x0, y0, x_h = state[:, 0], state[:, 1], state[:, 3]
    for j, name in enumerate(ACTIONS):
        idx = mask[:, j]
        if not idx.any():
            continue
        theta, s, jump_h = ACTION_PARAMS[name]
        v0, angle = resolve_hit(theta, s, power)
        landing, ftime, _ = simulate_batch(
            x0[idx], y0[idx] + jump_h,
            np.full(int(idx.sum()), float(v0)),
            np.full(int(idx.sum()), float(angle)),
            np.ones(int(idx.sum())), net_h=NET_HEIGHT)   # 撞网 = 落点 nan = FAIL
        values[idx, j] = value_of_outcome(landing, ftime, x_h[idx])
    return values


def choose_action(values):
    """价值分 → 动作序号（不可行 = -1 天然垫底，无需再掩码）。"""
    return np.argmax(values, axis=1)


def predict_values(X, parameters, norm):
    """归一化输入 X(n,4) → 三动作价值分（value 回归实验用，已被分类器取代）。"""
    z = forward((X - norm["x_mean"]) / norm["x_std"], parameters)
    return z * norm["y_std"] + norm["y_mean"]


def predict_jump_proba(X, parameters, norm):
    """终版决策网络：归一化输入 X(n,4) → 跳杀优于高远的概率（sigmoid）。"""
    from network import forward as classify_forward
    return classify_forward((X - norm["x_mean"]) / norm["x_std"], parameters)


def decide(state, parameters, norm, jump_threshold=0.5):
    """游戏策略入口：原始状态 state(n,4) → 动作序号（0 挑 / 1 高远 / 2 跳杀）。

    低球按游戏规则直接挑；高球且来得及起跳时按网络概率选跳杀/高远。
    jump_threshold 调低（如 0.4）= AI 更敢杀，调高 = 更保守。
    """
    state = np.asarray(state, dtype=float)
    proba = predict_jump_proba(normalize_state(state), parameters, norm).ravel()
    mask = feasible_mask(state)
    actions = np.zeros(state.shape[0], dtype=int)      # 默认挑球（低球）
    actions[mask[:, 1]] = 1                            # 高球先按高远
    actions[mask[:, 2] & (proba > jump_threshold)] = 2  # 来得及且网络说跳 → 跳杀
    return actions


def load_model(path="decision_model.npz"):
    """读 train_decision.py 保存的模型，返回 (parameters, norm)。"""
    d = np.load(path)
    parameters = {k: d[k] for k in ("W1", "b1", "W2", "b2")}
    norm = {"x_mean": d["x_mean"], "x_std": d["x_std"]}
    return parameters, norm
