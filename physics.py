"""物理模拟器：带空气阻力的羽毛球飞行 + 击球判定（11-20 数据生成用）。

坐标约定：**y 向上**，表示离地高度（像素）；x 从左到右，范围 [0, COURT_W]。
游戏绘制时再换算成屏幕坐标（屏幕 y 向下）。

只依赖 NumPy，不依赖 pygame，方便批量生成训练数据与单元测试。

击球模型（与进展记录一致，数值 11-21 试玩再调）：
    resolve_hit(theta, s, power) -> (v0, launch_angle)
    - launch_angle = theta（拍角直接决定出射角；越向下越尖）
    - v_base 随 theta 单调递减（越向下基础速度越大）
    - v0 = v_base(theta) * lever(s) * power，lever(s) = 0.6 + 0.4*s
阻力：二次阻力 a_drag = -k * |v| * v（无风）。k 固定公开，轨迹明显偏离真空抛体。
"""

import numpy as np

# --- 场地 / 物理常量（像素单位， provisional，11-21 定标） ---
GRAVITY = 1800.0        # px/s^2，与 characters.py 一致
DRAG_K = 0.0025         # 二次阻力系数（1/px），固定公开
COURT_W = 900.0         # 场地宽
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
V_MIN = 1520.0
V_MAX = 3280.0

# 观察窗：前 0.15 s，取 4 帧（0 / 0.05 / 0.10 / 0.15 s）× 2 坐标 = 8 输入
OBS_STEPS = (0, 12, 24, 36)   # DT=1/240 时对应 0/0.05/0.10/0.15 s


def lever(s):
    """接触点杠杆：s=0 拍柄（弱），s=1 拍头（强）。"""
    return 0.6 + 0.4 * np.asarray(s, dtype=float)


def v_base(theta):
    """基础速度随拍角单调递减：越向下越快。"""
    theta = np.asarray(theta, dtype=float)
    frac = (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)
    return V_MAX + (V_MIN - V_MAX) * frac


def resolve_hit(theta, s, power):
    """击球判定：返回 (初速度 v0, 出射角 angle)。可单测。"""
    theta = np.asarray(theta, dtype=float)
    v0 = v_base(theta) * lever(s) * np.asarray(power, dtype=float)
    return v0, theta


def simulate_batch(x0, y0, v0, angle, direction,
                   drag_k=DRAG_K, dt=DT, max_t=MAX_T):
    """批量积分一批球，返回 (landing_x, flight_time, obs)。

    obs 形状 (n, 4, 2)：前 0.15 s 的 4 帧 (x, y)；落地后的帧保留最后一帧
    （即按实际可用帧截断）。landing_x 为首次 y<=0 的插值落点；未落地为 nan。
    direction=+1 向 +x 打，-1 向 -x 打。
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
    landing = np.full(n, np.nan)
    ftime = np.full(n, np.nan)
    obs = np.zeros((n, len(OBS_STEPS), 2))
    obs[:, 0, :] = pos

    steps = int(max_t / dt) + 1
    for step in range(steps):
        if not active.any():
            break
        # 先记录当前帧（t = step*dt），再积分，保证第 0 帧就是击球点
        if step in OBS_STEPS:
            idx = OBS_STEPS.index(step)
            obs[active, idx, :] = pos[active]
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
            active[cross] = False

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
