"""物理模拟器：带空气阻力的羽毛球飞行 + 击球判定（11-20 数据生成用）。

坐标约定：**y 向上**，表示离地高度（像素）；x 从左到右，范围 [0, COURT_W]。
游戏绘制时再换算成屏幕坐标（屏幕 y 向下）。

只依赖 NumPy，不依赖 pygame，方便批量生成训练数据与单元测试。

击球模型（与进展记录一致，数值 11-21 试玩再调）：
    resolve_hit(theta, s, power) -> (v0, launch_angle)
    - launch_angle = launch_angle(theta)：拍角减去非线性下压（11-24 起与 game.py 同公式）
    - v_base 随 theta 单调递减（越向下基础速度越大）
    - v0 = v_base(theta) * lever(s) * power，lever(s) = 0.6 + 0.4*s
阻力：二次阻力 a_drag = -k * |v| * v（无风）。k 固定公开，轨迹明显偏离真空抛体。
"""

import numpy as np

# --- 场地 / 物理常量（像素单位， provisional，11-21 定标） ---
GRAVITY = 1800.0        # px/s^2，与 characters.py 一致
DRAG_K = 0.00375        # 二次阻力系数（1/px）。11-24 用户：「把空气阻力调高，先调到原来3倍」→ 试过 3 倍后改「1.5 倍」：0.0025 × 1.5
COURT_W = 1200.0        # 场地宽（11-23 游戏加宽 900→1200，11-24 数据生成器对齐）
NET_X = COURT_W / 2     # 网的位置
NET_HEIGHT = 150.0      # 网高（离地），仅游戏规则用，落点模拟不撞网
DT = 1.0 / 240.0        # 积分步长
MAX_T = 3.0             # 单球最长飞行时间

# 两种固定挥拍可达的拍角范围（弧度；0=水平，正=拍面朝上，负=朝下）
SWING_THETA = {
    "clear": (-0.55, 1.20),   # 向上高远：大幅上扫
    "lift": (-0.15, 1.05),    # 向上挑球：低位快挑
}
THETA_MIN = min(lo for lo, _ in SWING_THETA.values())
THETA_MAX = max(hi for _, hi in SWING_THETA.values())

# 基础速度范围：拍角最向下时最快，最向上时最慢
# 11-17 用户三次要求提速（「至少能飞过半场」→「球速还要再加」→「加到 6 倍」），
# 但 6 倍时发现：球到得太快，拍子只能在挥拍弧最开头（拍杆 60°）碰到球，
# 结果变成「很快的高抛球」（滞空 1.51 s），看起来反而慢 —— 那是击球点太高，不是速度不够。
# 现在把挥拍手位改成「架拍 → 瞬间落到胸前」，击球点降到 40° 左右（真实头顶挥拍的角度），
# 同一套速度下滞空只有 ~1.0 s，落点仍在中后场。
# 11-19 用户仍觉得慢（「球速太慢」）→ 再上调到 4 倍（原 380/820 的 4 倍）。
# 11-24 用户拍板「球速定在 5 档，不动了」→ 默认值与游戏 5 档（380×5 / 820×5）对齐，
# 这样 make_shots.py 采出来的数据分布才和游戏里实际打的一致。
# 11-24 晚：验收后锁定 **10 档**（3800/8200），游戏界面不再暴露档位；数据生成器同步。
V_MIN = 3800.0
V_MAX = 8200.0

# 出球角非线性下压（11-24 与 game.py try_contact 完全同公式；两边必须同步改）：
#   angle = theta - LAUNCH_TILT - LAUNCH_SHARP * low^2，low = clip((KNEE-theta)/0.50, 0, 1)
# 效果：头顶（theta≈0.9）出球约 22°；拍面越平，追加的下压越狠；最低夹到 -0.61 rad ≈ -35°。
LAUNCH_TILT = 0.50
LAUNCH_KNEE = 1.00
LAUNCH_SHARP = 0.45
LAUNCH_MIN = -0.61

# 观察窗：前 0.15 s，取 4 帧（0 / 0.05 / 0.10 / 0.15 s）× 2 坐标 = 8 输入
OBS_STEPS = (0, 12, 24, 36)   # DT=1/240 时对应 0/0.05/0.10/0.15 s
IC_H = 250.0            # 截击高度：与 game.IC_HEIGHT 一致；训练数据的第二预测目标


def lever(s):
    """接触点杠杆：s=0 拍柄（弱），s=1 拍头（强）。"""
    return 0.6 + 0.4 * np.asarray(s, dtype=float)


def v_base(theta):
    """基础速度随拍角单调递减：越向下越快。"""
    theta = np.asarray(theta, dtype=float)
    frac = (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)
    return V_MAX + (V_MIN - V_MAX) * frac


def launch_angle(theta):
    """出射角：拍面角 − 常数下压 − 低拍角平方追加下压，最低 LAUNCH_MIN。"""
    theta = np.asarray(theta, dtype=float)
    low = np.clip((LAUNCH_KNEE - theta) / 0.50, 0.0, 1.0)
    return np.maximum(theta - LAUNCH_TILT - LAUNCH_SHARP * low * low, LAUNCH_MIN)


def resolve_hit(theta, s, power):
    """击球判定：返回 (初速度 v0, 出射角 angle)。可单测。"""
    theta = np.asarray(theta, dtype=float)
    v0 = v_base(theta) * lever(s) * np.asarray(power, dtype=float)
    return v0, launch_angle(theta)


def simulate_batch(x0, y0, v0, angle, direction,
                   drag_k=DRAG_K, dt=DT, max_t=MAX_T, with_intercept=False):
    """批量积分一批球，返回 (landing_x, flight_time, obs)（+ ic_x 如果 with_intercept）。

    obs 形状 (n, 4, 2)：前 0.15 s 的 4 帧 (x, y)；球在观察窗内落地时，
    落地后的帧记录为落点静止位置 (landing_x, 0)——球落地就不动了。
    landing_x 为首次 y<=0 的插值落点；未落地为 nan。direction=+1 向 +x 打，-1 向 -x 打。
    ic_x = 下行穿过截击高度 IC_H=250 时的 x（插值）；整段轨迹没高过 250 的平球
    取落点（AI 直接按落点站）。AI 的跑位目标是截击点而不是落点。
    """
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    angle = np.asarray(angle, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n = x0.shape[0]

    pos = np.stack([x0, y0], axis=1)
    vel = np.stack([v0 * np.cos(angle) * direction,
                    v0 * np.sin(angle)], axis=1)
    active = np.ones(n, dtype=bool)
    landed = np.zeros(n, dtype=bool)
    rest = np.zeros_like(pos)           # 落地后的静止位置 (landing_x, 0)
    landing = np.full(n, np.nan)
    ftime = np.full(n, np.nan)
    ic_x = np.full(n, np.nan)           # 下行穿过 250 的 x（首次）
    obs = np.zeros((n, len(OBS_STEPS), 2))
    obs[:, 0, :] = pos

    steps = int(max_t / dt) + 1
    for step in range(steps):
        if not active.any() and step > max(OBS_STEPS):
            break
        # 先记录当前帧（t = step*dt），再积分，保证第 0 帧就是击球点；
        # 已落地的球记录静止位置，而不是把后续帧留成 (0,0)
        if step in OBS_STEPS:
            idx = OBS_STEPS.index(step)
            obs[:, idx, :] = np.where(landed[:, None], rest, pos)
        prev = pos.copy()
        speed = np.linalg.norm(vel, axis=1)
        acc = -drag_k * speed[:, None] * vel
        acc[:, 1] -= GRAVITY
        vel += acc * dt
        pos += vel * dt

        cross = active & (pos[:, 1] <= 0.0) & (vel[:, 1] < 0.0)
        if cross.any():
            py = prev[cross, 1]
            cy = pos[cross, 1]
            frac = py / (py - cy)
            landing[cross] = prev[cross, 0] + frac * (pos[cross, 0] - prev[cross, 0])
            ftime[cross] = step * dt
            rest[cross] = np.stack(
                [landing[cross], np.zeros(int(cross.sum()))], axis=1)
            active[cross] = False
            landed[cross] = True

        if with_intercept:
            ic_hit = (np.isnan(ic_x) & (prev[:, 1] > IC_H)
                      & (pos[:, 1] <= IC_H) & (vel[:, 1] < 0.0))
            if ic_hit.any():
                py = prev[ic_hit, 1]
                frac = (py - IC_H) / (py - pos[ic_hit, 1])
                ic_x[ic_hit] = (prev[ic_hit, 0]
                                + frac * (pos[ic_hit, 0] - prev[ic_hit, 0]))

    if with_intercept:
        # 平球（整段没高过 250）：按落点站
        ic_x = np.where(np.isnan(ic_x), landing, ic_x)
        return landing, ftime, obs, ic_x
    return landing, ftime, obs


def simulate_path(x0, y0, v0, angle, direction,
                  drag_k=DRAG_K, dt=DT, max_t=MAX_T):
    """单球完整轨迹 (t, x, y)，用于画图 / 游戏回放；落地即停。"""
    x, y = float(x0), float(y0)
    vx = float(v0) * np.cos(angle) * direction
    vy = float(v0) * np.sin(angle)
    ts, xs, ys = [0.0], [x], [y]
    t = 0.0
    while t < max_t:
        speed = np.hypot(vx, vy)
        ax = -drag_k * speed * vx
        ay = -GRAVITY - drag_k * speed * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        ts.append(t); xs.append(x); ys.append(y)
        if y <= 0.0 and vy < 0.0:
            break
    return np.array(ts), np.array(xs), np.array(ys)
