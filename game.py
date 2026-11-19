"""羽毛球游戏骨架（11-21）：能连续对打的窗口 + 选人界面。

运行：
    .venv\\Scripts\\python.exe game.py            # 选人 → 人机对打
    .venv\\Scripts\\python.exe game.py --smoke    # 双 AI 自动对打 30 s，打印统计后退出

操作：A/D 移动，W 跳，空格挥拍（单键，按球相对腰部高度自动高远/挑球），
P 暂停，Esc 退出，Tab 显示落点预测标记。先到 7 分。

物理与击球模型完全复用 physics.py（二次阻力 k=0.0025、resolve_hit）。
AI 的落点目前用模拟器直接算（占位），11-28 换成网络预测。
"""

import argparse
import math

import pygame

import physics
from characters import CHARACTERS, draw_character, jump_speed, normalize, pros_cons
from game_style_preview import draw_shuttle, draw_trail

W, H = 900, 500
FLOOR_Y = 460                      # 屏幕 y：地板线
SCALE = 1.0
NET_X = physics.NET_X              # 450
NET_H = physics.NET_HEIGHT         # 150
WIN_SCORE = 7

WALL = (168, 178, 174)
WOOD = (198, 156, 106)
INK = (28, 28, 32)

# 挥拍弧线：θ 从大到小扫过（越晚接触越平越快），时长固定。
# 11-17 用户两次要求「挥拍要快」：0.50/0.32 → 0.36/0.24（快 28%）。
# 实测（3.2 倍球速，4 组 AI 对打 ×60 s）：0.36 s → 17.9 次/回合；
# 再短到 0.32 s 会掉到 3.3（拍子扫过球路径的时间窗太窄），0.36 是拐点。
SWING_ARC = {"clear": (1.20, -0.55, 0.36), "lift": (1.05, -0.15, 0.24)}
SERVE_V0 = 2000.0                             # 发球初速（落点约过网 259 px，飞行 1.18 s）
SERVE_ANGLE = 0.90
RACKET_LEN = 70 * SCALE                       # 拍长 = 物理接触段（手→拍头）；改它会改手感，别乱动
DRAW_RACKET_LEN = 104 * SCALE                 # 画出来的拍子比接触段长一点：球看起来打在拍面上（不超出去）
CONTACT_R = 34                                # 判定带半宽（球离扫过的拍杆多近算碰到）
WAIST_H = 90 * SCALE                          # 腰部阈值，随跳跃上移
# 判定带（用户：判定范围扩大成一个带子，球拍从后往前的整个过程都要算进来）：
# 挥拍时拍杆扫过的是一个面，不是一个时刻的一条线 —— 所以每个子步把
# 「本次挥拍从 q=0 到当前 q」之间采样 CONTACT_SAMPLES 个拍杆位置，球碰到其中任何一段都算击中。
# 比「只在当前瞬间那条线段上判定」宽得多，也不会因为按早了就空挥（球在带子里就会被扫到）。
CONTACT_SAMPLES = 12

# 准备姿势（用户给的参考图；拍子举过头顶、拍头在身后 = 正确的架拍）：
#   hi = 架拍待机：手举到头侧、拍面朝后上方，等头顶球（高远/杀球那种高球）
#   lo = 低姿势：拍子垂在腰前、另一只手在腰侧托着球，等挑球和发球
HAND_READY = {"hi": (-38.0, -166.0), "lo": (44.0, -100.0)}
THETA_READY = {"hi": 2.35, "lo": -0.55}
# 非持拍手：架拍时抬到下巴高度（参考图里那只手是抬起来的），低姿势挂在腰侧
FREE_READY = {"hi": (20.0, -144.0), "lo": (-8.0, -96.0)}

# 挥拍手位关键帧：(进度 q, 手 x, 手 y)，相对脚底、面朝方向为正、y 向上为负。
# 拍面角度仍严格走 SWING_ARC（出球只看 θ），手位只负责「人怎么挥」：
#   高远球：架拍（手在头侧、拍头朝后上方）→ 手往前上方送 → 收在身前
#   挑球：  从腰侧低姿势 → 手向后沉一点 → 向前上方兜起
# 两条路径的 q=0 与对应准备姿势的手位完全一致，起手不跳帧。
HAND_IDLE = HAND_READY["hi"]
HAND_PATH = {
    # 高远球：拍子从架拍（手在头后）快速落到胸前再扫出去 —— 手落得越快，
    # 球在越低的位置被接住，出射角就越平（真实头顶挥拍的出球是 35~45°，不是 60°）；
    # 手停在头高会变成「每一拍都是高抛球」，见 11-17 击球层讨论。
    "clear": [(0.00, -38, -166), (0.10, 30, -140), (0.30, 44, -118),
              (0.60, 48, -106), (1.00, 52, -96)],
    "lift": [(0.00, 44, -100), (0.14, 30, -90), (0.36, 22, -84),
             (0.58, 26, -88), (0.80, 38, -100), (1.00, 42, -112)],
}
SWING_SMOOTH = 0.06                           # 起手 0.06 s 内把拍面从准备角度接到弧线


def _path_point(keys, q):
    """按关键帧取手位，段内 smoothstep 平滑（起收都柔）。"""
    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
    for i in range(len(keys) - 1):
        q0, x0, y0 = keys[i]
        q1, x1, y1 = keys[i + 1]
        if q <= q1:
            t = 0.0 if q1 <= q0 else (q - q0) / (q1 - q0)
            t = t * t * (3.0 - 2.0 * t)
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return (float(keys[-1][1]), float(keys[-1][2]))


def sy(world_y):
    """世界高度 → 屏幕 y。"""
    return FLOOR_Y - world_y


class Player:
    def __init__(self, char, side):
        self.char = char
        self.side = side                 # 'L' 或 'R'
        self.facing = 1 if side == "L" else -1
        self.x = 250.0 if side == "L" else 650.0
        self.h = 0.0                     # 跳起高度（世界坐标）
        self.vh = 0.0
        self.swing_t = -1.0              # <0 表示没在挥拍
        self.swing_type = "clear"
        self.contacted = False
        self.recover_x = 260.0 if side == "L" else 640.0

    @property
    def feet_y(self):
        return FLOOR_Y - self.h

    @property
    def waist_y(self):
        return self.feet_y - WAIST_H

    def swing_q(self):
        """挥拍进度 0..1（没在挥拍时返回 0）。"""
        if self.swing_t < 0:
            return 0.0
        return min(self.swing_t / SWING_ARC[self.swing_type][2], 1.0)

    def hand_offset(self):
        """手相对脚底的位置：待机固定举拍，挥拍时走关键帧路径。"""
        if self.swing_t < 0:
            return HAND_IDLE
        return _path_point(HAND_PATH[self.swing_type], self.swing_q())

    def hand(self):
        dx, dy = self.hand_offset()
        return (self.x + self.facing * dx, self.feet_y + dy)

    def theta_now(self):
        hi, lo, dur = SWING_ARC[self.swing_type]
        t = min(self.swing_t / dur, 1.0)
        return hi + (lo - hi) * t

    def draw_theta(self):
        """画拍子用的角度：起手一小段从准备姿势的角度平滑接到挥拍弧上。

        击球判定仍然只看 theta_now()；这里只负责视觉：架拍时拍头朝后上方
        （2.35 rad），起手瞬间必须转身向前扫，所以头 SWING_SMOOTH 秒里把拍面
        从准备角旋到弧线上，看起来就是「架拍 → 转身挥出去」。
        """
        th = self.theta_now()
        if self.swing_t >= 0:
            k = min(self.swing_t / SWING_SMOOTH, 1.0)
            start = THETA_READY["hi" if self.swing_type == "clear" else "lo"]
            th = start + (th - start) * k
        return th

    def racket_dir(self):
        th = self.theta_now()
        return (self.facing * math.cos(th), -math.sin(th))   # 屏幕坐标

    def tip(self):
        hx, hy = self.hand()
        dx, dy = self.racket_dir()
        return (hx + dx * RACKET_LEN, hy + dy * RACKET_LEN)

    def start_swing(self, ball=None, force=None):
        if self.swing_t >= 0:
            return False
        # 球低于腰部 → 挑球，否则高远（阈值随跳跃上移）；force 给发球这类动作指定弧线
        if force is not None:
            self.swing_type = force
        else:
            self.swing_type = "lift" if ball is not None and ball.y_screen > self.waist_y else "clear"
        self.swing_t = 0.0
        self.contacted = False
        return True

    def update(self, dt, move, jump):
        lo, hi = (30, NET_X - 12) if self.side == "L" else (NET_X + 12, W - 30)
        self.x = min(hi, max(lo, self.x + move * self.char["speed"] * dt))
        if jump and self.h <= 0.0:
            self.vh = jump_speed(self.char)
        if self.h > 0.0 or self.vh > 0.0:
            self.vh -= physics.GRAVITY * dt
            self.h += self.vh * dt
            if self.h <= 0.0:
                self.h, self.vh = 0.0, 0.0
        if self.swing_t >= 0.0:
            self.swing_t += dt
            if self.swing_t > SWING_ARC[self.swing_type][2]:
                self.swing_t = -1.0


class Ball:
    def __init__(self):
        self.x, self.y = 250.0, 100.0    # 世界坐标：x 同屏幕，y 离地高度
        self.vx, self.vy = 0.0, 0.0
        self.live = False                # 是否在被击出后的飞行中
        self.last_hitter = "L"
        self.trail = []

    @property
    def y_screen(self):
        return sy(self.y)

    def launch(self, x, y, v0, angle, facing, hitter):
        self.x, self.y = x, y
        self.vx = facing * v0 * math.cos(angle)
        self.vy = v0 * math.sin(angle)
        self.live = True
        self.last_hitter = hitter
        self.trail = []

    def step(self, dt):
        if not self.live:
            return
        speed = math.hypot(self.vx, self.vy)
        ax = -physics.DRAG_K * speed * self.vx
        ay = -physics.DRAG_K * speed * self.vy - physics.GRAVITY
        self.vx += ax * dt
        self.vy += ay * dt
        px = self.x
        self.x += self.vx * dt
        self.y += self.vy * dt
        # 网：从两侧穿越网面时若低于网顶，挡回击球方一侧
        if (px - NET_X) * (self.x - NET_X) < 0 and self.y < NET_H:
            self.x = NET_X - 2 if self.last_hitter == "L" else NET_X + 2
            self.vx = 0.0


class AIController:
    """RECOVER → WAIT → CHASE → SWING。落点暂用模拟器占位（11-28 换网络）。"""

    def __init__(self, player):
        self.p = player
        self.state = "RECOVER"
        self.target = player.recover_x
        self.swing_dist = 130     # 球距多远时起挥（弧线缩短后重标：120→2.2、130→17.9、150→1.0）
        self.stand_off = 25       # 站在落点靠后多少

    def predict(self, ball):
        land, _, _ = physics.simulate_batch(
            [ball.x], [ball.y], [math.hypot(ball.vx, ball.vy)],
            [math.atan2(ball.vy, abs(ball.vx))], [1 if ball.vx > 0 else -1])
        return float(land[0])

    def update(self, dt, ball, serve_pending):
        p = self.p
        move = 0.0
        jump = False
        incoming = ball.live and ball.last_hitter != p.side
        crossed = incoming and ((ball.x > NET_X) if p.side == "R" else (ball.x < NET_X))
        fast = incoming and math.hypot(ball.vx, ball.vy) > 650
        # 球已过网（或快球）就立即追，不等回位——接球第一；
        # 球还在对方半场时才有「先回发球线」的节奏。
        if (crossed or fast) and self.state in ("RECOVER", "WAIT"):
            self.target = self.predict(ball)
            self.state = "CHASE"
        if serve_pending or not ball.live:
            self.state = "RECOVER"
        elif self.state == "RECOVER":
            if abs(p.x - p.recover_x) < 8:
                self.state = "WAIT"
        elif self.state == "WAIT":
            pass
        elif self.state == "CHASE":
            # 站在落点稍靠后一点，让球在身前接触
            want = self.target - p.facing * self.stand_off
            if abs(p.x - want) > 6:
                move = 1.0 if want > p.x else -1.0
            near = abs(ball.x - p.x) < self.swing_dist
            height_ok = 50 < ball.y < 250 + p.h
            if near and height_ok and ball.vy < 150:
                p.start_swing(ball)
                self.state = "SWING"
        elif self.state == "SWING":
            if p.swing_t < 0:
                self.state = "RECOVER"
        if self.state in ("RECOVER",) and abs(p.x - p.recover_x) > 8:
            move = 1.0 if p.recover_x > p.x else -1.0
        return move, jump


class Game:
    def __init__(self, p_char, a_char, smoke=False, race=False):
        self.human = Player(p_char, "L")
        self.ai_p = Player(a_char, "R")
        self.ai = AIController(self.ai_p)
        self.ball = Ball()
        self.score = [0, 0]
        self.server = "L"
        self.serve_timer = 1.0
        self.paused = False
        self.show_pred = False
        self.over = False
        self.smoke = smoke
        self.race = race                  # True = 7 分制；默认练习模式（不限分，一直打）
        self.hits = 0
        self.rallies = 0
        self.pred_x = None

    def serve(self):
        p = self.human if self.server == "L" else self.ai_p
        self.ball.launch(p.x + p.facing * 30, 110, SERVE_V0, SERVE_ANGLE, p.facing, self.server)
        p.start_swing(force="lift")     # 只做发球动作：球已经从腰侧发出去了
        self.rallies += 1

    def point(self, winner):
        self.score[0 if winner == "L" else 1] += 1
        self.server = winner
        self.ball.live = False
        self.serve_timer = 1.2
        self.ai.state = "RECOVER"
        if self.race and max(self.score) >= WIN_SCORE:
            self.over = True

    def racket_at(self, p, q):
        """挥拍进度 q（0..1）时的手位与拍杆角，和 hand()/theta_now() 同一套几何。"""
        hi, lo, _ = SWING_ARC[p.swing_type]
        th = hi + (lo - hi) * q
        hx0, hy0 = _path_point(HAND_PATH[p.swing_type], q)
        return (p.x + p.facing * hx0, p.feet_y + hy0, th)

    def try_contact(self, p):
        if p.swing_t < 0 or p.contacted or not self.ball.live:
            return
        b = self.ball
        on_side = (b.x < NET_X) if p.side == "L" else (b.x > NET_X)
        if not on_side or b.last_hitter == p.side:   # 不能连击自己刚打出的球
            return
        # 判定带：把本次挥拍从 q=0 到当前 q 扫过的整段拍杆都算进来。
        # 从「当前」往「起点」倒着找，取第一个够得着的采样点 —— 优先用当前这一瞬的拍面，
        # 只有球稍稍落在拍子后面/前面时才回退到扫过的路径上（否则全用手后方的早期位置，
        # 出球角会被顶到弧线开头，球又变回高抛）。
        q_now = p.swing_q()
        hit = None                                   # (theta, s)
        for i in range(CONTACT_SAMPLES - 1, -1, -1):
            q = q_now * i / (CONTACT_SAMPLES - 1)
            hx, hy, th = self.racket_at(p, q)
            dx, dy = p.facing * math.cos(th) * RACKET_LEN, -math.sin(th) * RACKET_LEN
            s = ((b.x - hx) * dx + (b.y_screen - hy) * dy) / (RACKET_LEN * RACKET_LEN)
            if not -0.1 <= s <= 1.1:
                continue
            cx, cy = hx + s * dx, hy + s * dy
            if math.hypot(b.x - cx, b.y_screen - cy) < CONTACT_R:
                hit = (th, s)
                break
        if hit is not None:
            th, s = hit
            s = min(1.0, max(0.0, s))
            v0, angle = physics.resolve_hit(th, s, p.char["power"])
            b.launch(b.x, b.y, float(v0), float(angle), p.facing, p.side)
            p.contacted = True
            self.hits += 1

    def update(self, dt, keys):
        if self.paused or self.over:
            return
        if not self.ball.live:
            self.serve_timer -= dt
            if self.serve_timer <= 0:
                self.serve()
        # 人类输入
        move = (keys.get("d", False) - keys.get("a", False)) if not self.smoke else 0.0
        jump = keys.get("w", False) if not self.smoke else False
        if self.smoke:
            ai_l = getattr(self, "ai_left", None)
            if ai_l is None:
                self.ai_left = AIController(self.human)
            move, jump = self.ai_left.update(dt, self.ball, not self.ball.live)
        self.human.update(dt, move, jump)
        amove, ajump = self.ai.update(dt, self.ball, not self.ball.live)
        self.ai_p.update(dt, amove, ajump)
        if keys.get("space", False):
            self.human.start_swing(self.ball if self.ball.live else None)
        # 球按 240 Hz 子步积分，与 physics.py 完全一致
        sub = 4
        for _ in range(sub):
            self.ball.step(dt / sub)
            self.try_contact(self.human)
            self.try_contact(self.ai_p)
            if self.ball.live and self.ball.y <= 0 and self.ball.vy < 0:
                self.ball.y = 0
                self.ball.live = False
                x = self.ball.x
                if x < 0 or x > W:                      # 出界：击球方失分
                    self.point("R" if self.ball.last_hitter == "L" else "L")
                else:
                    self.point("R" if x < NET_X else "L")
                break
        if self.ball.live:
            self.ball.trail.append((self.ball.x, self.ball.y_screen))
            if len(self.ball.trail) > 16:
                self.ball.trail.pop(0)
        if self.show_pred and self.ball.live:
            self.pred_x = self.ai.predict(self.ball)

    # ---------- 绘制 ----------
    def draw_court(self, s):
        s.fill(WALL)
        pygame.draw.rect(s, WOOD, (0, FLOOR_Y, W, H - FLOOR_Y))
        pygame.draw.rect(s, (168, 128, 82), (0, FLOOR_Y, W, 10))
        pygame.draw.line(s, (236, 238, 230), (0, FLOOR_Y), (W, FLOOR_Y), 2)
        for x in range(60, W, 60):                      # 地板虚线
            pygame.draw.line(s, (236, 238, 230), (x, FLOOR_Y + 22), (x + 26, FLOOR_Y + 22), 2)
        # 细网 + 蓝柱
        top = int(sy(NET_H))
        pygame.draw.rect(s, (230, 235, 231), (NET_X - 8, top, 16, FLOOR_Y - top))
        for y in range(top, FLOOR_Y, 8):
            pygame.draw.line(s, (152, 160, 156), (NET_X - 8, y), (NET_X + 8, y), 1)
        pygame.draw.rect(s, (250, 250, 245), (NET_X - 10, top - 5, 20, 6))
        pygame.draw.line(s, (72, 82, 94), (NET_X, top), (NET_X, FLOOR_Y), 3)

    def player_tip_at(self, p, q):
        """挥拍进度 q 处的拍头位置（给残影用，与 hand()/racket_dir() 同一套几何）。"""
        hx, hy = _path_point(HAND_PATH[p.swing_type], q)
        hi, lo, _ = SWING_ARC[p.swing_type]
        th = hi + (lo - hi) * q
        return (p.x + p.facing * (hx + math.cos(th) * RACKET_LEN),
                p.feet_y + hy - math.sin(th) * RACKET_LEN)

    def ready_key(self, p):
        """现在该摆哪套准备姿势：要发球、或来球已经低于腰 → 低姿势（等挑球）。"""
        if not self.ball.live:
            return "lo" if p.side == self.server else "hi"
        if self.ball.last_hitter != p.side and self.ball.y_screen > p.waist_y:
            return "lo"
        return "hi"

    def draw_player(self, s, p):
        swinging = p.swing_t >= 0
        if swinging:
            hand, th = p.hand_offset(), p.draw_theta()
            free_hand = None
        else:
            key = self.ready_key(p)
            hand, th = HAND_READY[key], THETA_READY[key]
            free_hand = FREE_READY[key]
        draw_character(s, p.char, int(p.x), int(p.feet_y),
                       "right" if p.facing > 0 else "left", SCALE,
                       racket_theta=th, hand=hand, free_hand=free_hand,
                       racket_len=DRAW_RACKET_LEN)
        if not self.ball.live and p.side == self.server and not self.over:
            # 发球前另一只手在腰侧托着球（低姿势那张参考图）
            draw_shuttle(s, (int(p.x - p.facing * 18), int(p.feet_y - 104)), math.pi / 2)
        if swinging:                                    # 白色残影：拍头最近扫过的轨迹
            q1 = p.swing_q()
            q0 = max(0.0, q1 - 0.30)
            pts = [tuple(map(int, self.player_tip_at(p, q0 + (q1 - q0) * k / 9.0)))
                   for k in range(10)]
            for k in range(len(pts) - 1):
                f = k / (len(pts) - 2)
                col = (int(WALL[0] + (255 - WALL[0]) * f),
                       int(WALL[1] + (255 - WALL[1]) * f),
                       int(WALL[2] + (255 - WALL[2]) * f))
                pygame.draw.line(s, col, pts[k], pts[k + 1], 2)

    def draw(self, s, fonts):
        self.draw_court(s)
        if self.show_pred and self.pred_x is not None and self.ball.live:
            pygame.draw.line(s, (242, 202, 72), (int(self.pred_x), FLOOR_Y - 14),
                             (int(self.pred_x), FLOOR_Y), 3)
            pygame.draw.circle(s, (242, 202, 72), (int(self.pred_x), FLOOR_Y - 16), 5, 2)
        self.draw_player(s, self.human)
        self.draw_player(s, self.ai_p)
        if self.ball.live or not self.over:
            trail = [(x, y, 2 + 5 * i / 15, i / 15)
                     for i, (x, y) in enumerate(self.ball.trail)]
            draw_trail(s, trail)
            ang = math.atan2(-self.ball.vy, self.ball.vx)
            draw_shuttle(s, (self.ball.x, self.ball.y_screen), ang)
        # 记分牌
        rect = pygame.Rect(W // 2 - 80, 16, 160, 56)
        pygame.draw.rect(s, (10, 10, 12), rect)
        pygame.draw.rect(s, (58, 58, 62), rect, 2)
        t = fonts["score"].render(f"{self.score[0]} - {self.score[1]}", True, (246, 44, 34))
        s.blit(t, t.get_rect(center=rect.center))
        hint = fonts["small"].render("Arrows/AD move  Up/W jump  Space swing  P pause  Tab predict  Esc quit",
                                     True, (246, 246, 240))
        s.blit(hint, (12, H - 24))
        if not self.race:
            t = fonts["small"].render("PRACTICE - no target score", True, (250, 226, 150))
            s.blit(t, (W - t.get_width() - 12, H - 24))
        if self.paused:
            t = fonts["big"].render("PAUSED", True, (248, 248, 244))
            s.blit(t, t.get_rect(center=(W // 2, H // 2)))
        if self.over:
            msg = "YOU WIN!" if self.score[0] > self.score[1] else "AI WINS"
            t = fonts["big"].render(msg, True, (242, 202, 72))
            s.blit(t, t.get_rect(center=(W // 2, H // 2 - 20)))
            t2 = fonts["small"].render("Enter: back to select   Esc: quit", True, (246, 246, 240))
            s.blit(t2, t2.get_rect(center=(W // 2, H // 2 + 20)))


def select_screen(screen, fonts):
    """3×4 网格 + 左侧大图；方向键选择，回车开始。返回玩家角色。"""
    sel = 0
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return None
                if e.key == pygame.K_RETURN:
                    return CHARACTERS[sel]
                if e.key == pygame.K_RIGHT:
                    sel = (sel + 1) % 12
                if e.key == pygame.K_LEFT:
                    sel = (sel - 1) % 12
                if e.key == pygame.K_DOWN:
                    sel = (sel + 3) % 12
                if e.key == pygame.K_UP:
                    sel = (sel - 3) % 12
        screen.fill((18, 20, 24))
        screen.blit(fonts["big"].render("CHOOSE YOUR CHARACTER", True, (236, 238, 232)), (24, 14))
        gx, gy, cw, ch, gap = 330, 62, 182, 100, 6
        for i, c in enumerate(CHARACTERS):
            col, row = i % 3, i // 3
            r = pygame.Rect(gx + col * (cw + gap), gy + row * (ch + gap), cw, ch)
            pygame.draw.rect(screen, (46, 48, 54), r)
            pygame.draw.rect(screen, (242, 202, 72) if i == sel else (92, 94, 100),
                             r, 3 if i == sel else 1)
            draw_character(screen, c, r.centerx, r.bottom - 22, "right", 0.40)
            lab = fonts["small"].render(c["name"], True, (236, 238, 232))
            screen.blit(lab, lab.get_rect(center=(r.centerx, r.bottom - 10)))
        c = CHARACTERS[sel]
        draw_character(screen, c, 150, 330, "right", 1.15)
        n = fonts["mid"].render(c["name"], True, (236, 238, 232))
        screen.blit(n, n.get_rect(center=(150, 352)))
        good, bad = pros_cons(c)
        t = fonts["small"].render(f"{good} / {bad}", True, (156, 162, 158))
        screen.blit(t, t.get_rect(center=(150, 376)))
        nm = normalize(c)
        for j, (key, val) in enumerate([("SPEED", str(c["speed"])),
                                        ("POWER", f"{c['power']:.2f}"),
                                        ("JUMP", str(c["jump"]))]):
            y = 398 + j * 24
            screen.blit(fonts["small"].render(key, True, (156, 162, 158)), (40, y))
            filled = 1 + round(nm[key.lower()] * 7)
            for i in range(8):
                col = tuple(min(255, v + 80) for v in c["accent"]) if i < filled else (54, 56, 62)
                pygame.draw.rect(screen, col, (100 + i * 15, y + 3, 12, 10))
            screen.blit(fonts["small"].render(val, True, (176, 180, 176)), (228, y))
        screen.blit(fonts["small"].render("ARROWS select   ENTER start   Esc quit",
                                          True, (156, 162, 158)), (330, H - 22))
        pygame.display.flip()
        clock.tick(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="双 AI 自动对打 30 s 后退出")
    ap.add_argument("--race", action="store_true", help="7 分制（默认练习模式，不限分一直打）")
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Neural Badminton")
    fonts = {"score": pygame.font.SysFont("consolas", 40, bold=True),
             "big": pygame.font.SysFont("segoeui", 34, bold=True),
             "mid": pygame.font.SysFont("segoeui", 22, bold=True),
             "small": pygame.font.SysFont("segoeui", 14)}
    clock = pygame.time.Clock()

    if args.smoke:
        game = Game(CHARACTERS[8], CHARACTERS[1], smoke=True)   # Bun vs Rex
        frames, t0 = 0, pygame.time.get_ticks()
        fps_min = 999
        while pygame.time.get_ticks() - t0 < 30000 and not game.over:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    break
            game.update(1 / 60, {})
            game.draw(screen, fonts)
            pygame.display.flip()
            clock.tick(60)
            frames += 1
            if frames > 60:
                fps_min = min(fps_min, clock.get_fps())
        secs = (pygame.time.get_ticks() - t0) / 1000
        print(f"smoke | frames {frames} | {secs:.1f}s | avg_fps {frames / secs:.1f} "
              f"| min_fps {fps_min:.1f} | score {game.score[0]}-{game.score[1]} "
              f"| rallies {game.rallies} | hits {game.hits}")
        pygame.quit()
        return

    while True:
        p_char = select_screen(screen, fonts)
        if p_char is None:
            break
        idx = CHARACTERS.index(p_char)
        game = Game(p_char, CHARACTERS[(idx + 6) % 12], race=args.race)
        keys = {}
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if e.key == pygame.K_p:
                        game.paused = not game.paused
                    if e.key == pygame.K_TAB:
                        game.show_pred = not game.show_pred
                    if e.key == pygame.K_RETURN and game.over:
                        break
                    keys[{pygame.K_a: "a", pygame.K_d: "d", pygame.K_w: "w",
                          pygame.K_LEFT: "a", pygame.K_RIGHT: "d", pygame.K_UP: "w",
                          pygame.K_SPACE: "space"}.get(e.key, "")] = True
                if e.type == pygame.KEYUP:
                    keys[{pygame.K_a: "a", pygame.K_d: "d", pygame.K_w: "w",
                          pygame.K_LEFT: "a", pygame.K_RIGHT: "d", pygame.K_UP: "w",
                          pygame.K_SPACE: "space"}.get(e.key, "")] = False
            else:
                try:
                    game.update(1 / 60, keys)
                    game.draw(screen, fonts)
                    pygame.display.flip()
                except Exception:                     # 有 bug 时别让窗口卡死：写日志 + 暂停
                    import traceback
                    tb = traceback.format_exc()
                    print(tb)
                    try:
                        with open("crash.log", "w", encoding="utf-8") as fh:
                            fh.write(tb)
                    except OSError:
                        pass
                    game.paused = True
                clock.tick(60)
                continue
            break
    pygame.quit()


if __name__ == "__main__":
    main()