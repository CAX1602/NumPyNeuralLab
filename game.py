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

W, H = 1200, 800                   # 3:2，和用户屏幕（1440×960 逻辑分辨率）同比例；网在正中 600
FLOOR_Y = H - 100                  # 屏幕 y：地板线（下面留 100 px 地板，底部 38 px 是键位条）
SCALE = 1.0
ZOOM = 0.90                        # 整个世界在窗口里的缩放：1200×800 → 1080×720，装在 1440×960 的桌面里刚好
                                   # 世界坐标仍是 W×H（物理与判定都不变），只把画好的整帧缩放贴到窗口。
ZOOM_MIN, ZOOM_MAX = 0.60, 1.20


def window_size(zoom=None):
    z = ZOOM if zoom is None else zoom
    return max(320, int(W * z)), max(200, int(H * z))


def present(screen, world):
    """把世界画面按 ZOOM 缩放贴到窗口上（ZOOM=1 时就是 1:1）。"""
    size = screen.get_size()
    if size == world.get_size():
        screen.blit(world, (0, 0))
    else:
        pygame.transform.smoothscale(world, size, screen)
    pygame.display.flip()


def draw_hints(s, font, items, x0, y, gap=20, cap_h=20):
    """画一排「键帽 + 说明」，返回结束时的 x（用户：键位要更显眼）。"""
    x = x0
    for key, label in items:
        kw = max(cap_h, font.size(key)[0] + 12)
        r = pygame.Rect(x, y, kw, cap_h)
        pygame.draw.rect(s, (60, 64, 72), r, border_radius=4)
        pygame.draw.rect(s, (118, 124, 132), r, 1, border_radius=4)
        t = font.render(key, True, (244, 244, 238))
        s.blit(t, t.get_rect(center=r.center))
        x += kw + 6
        t2 = font.render(label, True, (180, 186, 182))
        s.blit(t2, (x, r.centery - t2.get_height() // 2))
        x += t2.get_width() + gap
    return x
NET_X = W // 2                     # 600（球场加宽后网在正中；不再取 physics.NET_X）
NET_H = physics.NET_HEIGHT         # 150
WIN_SCORE = 7
WIN_GAMES = 2                      # 经典模式：三局两胜（先拿下 2 局）
AI_ATTACK_Y = 225.0                 # AI 起跳杀球的球高度阈值（世界坐标，球还这么高才值得跳）
AI_ATTACK_DX = 220.0                # 球横向离 AI 多近时可以起跳
AI_ATTACK_ZONE = 400.0              # 落点在自己半场靠网的前 400 px 内（「杀球机会」）；调大喊它更爱杀
IC_HEIGHT = 250.0                  # AI 的截击高度（世界坐标）：球降到这个高度时拍子够得着
SERVICE_LINE = 150.0               # 发球线：距网 150 px。发球必须站在这条线**后面**（用户：要有发球线）
SPEED_DEFAULT = 10.0               # 球速固定 10 档（V_MIN/V_MAX 3800/8200）：用户验收后锁定，界面不再暴露

WALL = (168, 178, 174)
WOOD = (198, 156, 106)
INK = (28, 28, 32)

# 挥拍弧线：θ 从大到小扫过（越晚接触越平越快），时长固定。
# 11-23 用户：「挥拍速度太慢了」→ 0.36/0.42 再压到 0.30/0.32。
SWING_ARC = {"clear": (1.02, -0.55, 0.30), "lift": (1.05, -0.15, 0.32)}
LAUNCH_TILT = 0.50                            # 出球角 = 拍面角 − 这个值（rad）。调大 = 整体更下压。
                                              # 11-23 用户：「出球角度函数常量再加」→ 改成非线性（见下）
LAUNCH_KNEE = 1.00                            # 拍面角低于这个值，才开始「越平越尖」
LAUNCH_SHARP = 0.45                           # 低拍角处额外追加的下压量（rad）；越大越尖
                                              # 出球角 = θ − LAUNCH_TILT − LAUNCH_SHARP * t²，t 是 θ 低于 knee 的程度
                                              # 效果：头顶（θ≈0.9）仍是 23°；球低一点（θ≈0.6）就压到 5°；再低就直接往下打
SERVE_V0 = 2000.0                             # 发球初速（落点约过网 259 px，飞行 1.18 s）
SERVE_ANGLE = 0.90
RACKET_LEN = 70 * SCALE                       # 拍长 = 物理接触段（手→拍头）；改它会改手感，别乱动
DRAW_RACKET_LEN = 104 * SCALE                 # 画出来的拍子比接触段长一点：球看起来打在拍面上（不超出去）
CONTACT_R = 34                                # 判定带半宽（球离扫过的拍杆多近算碰到）
WAIST_H = 200 * SCALE                         # 挑球判定线高度（离脚 200 px ≈ 胸口；原来 90 = 腰）。随跳跃一起上移
# 判定带（用户：判定范围扩大成一个带子，球拍从后往前的整个过程都要算进来）：
# 挥拍时拍杆扫过的是一个面，不是一个时刻的一条线 —— 所以每个子步把
# 「本次挥拍从 q=0 到当前 q」之间采样 CONTACT_SAMPLES 个拍杆位置，球碰到其中任何一段都算击中。
# 比「只在当前瞬间那条线段上判定」宽得多，也不会因为按早了就空挥（球在带子里就会被扫到）。
CONTACT_SAMPLES = 12
RECOVER_DUR = 0.15                            # 收拍后回到准备姿势的过渡时长（用户：挥完别一直晃 → 0.38 缩到 0.15）
                                              # （用户：发完球别往下收，自然往上继续，变成架拍准备打头顶球）

# 准备姿势（用户给的参考图；拍子举过头顶、拍头在身后 = 正确的架拍）：
#   hi = 架拍待机：手举到头侧、拍面朝后上方，等头顶球（高远/杀球那种高球）
#   lo = 低姿势：拍子垂在腰前、另一只手在腰侧托着球，等挑球和发球
HAND_READY = {"hi": (-38.0, -166.0), "lo": (-36.0, -90.0)}
THETA_READY = {"hi": 2.35, "lo": -2.15}
# 非持拍手：架拍时抬到下巴高度（参考图里那只手是抬起来的），低姿势垂在体侧托球
FREE_READY = {"hi": (20.0, -144.0), "lo": (24.0, -72.0)}

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
    "clear": [(0.00, -38, -166), (0.14, 24, -142), (0.34, 42, -118),
              (0.66, 50, -104), (1.00, 22, -88)],
    # 挑球：从腰侧低姿势向前上方兜起（发球也走这条弧线）
    "lift": [(0.00, -36, -90), (0.14, 10, -86), (0.36, 22, -84),
             (0.58, 26, -88), (0.80, 38, -100), (1.00, 42, -112)],
}
SWING_DRAW_LINEAR = True                      # 画出来的拍面角整段匀速转（架拍角 → 弧线终点），
                                              # 不再「前 60 ms 甩完 40%」；判定弧线仍是 SWING_ARC


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
        self.x = W * 0.275 if side == "L" else W * 0.725
        self.h = 0.0                     # 跳起高度（世界坐标）
        self.vh = 0.0
        self.swing_t = -1.0              # <0 表示没在挥拍
        self.swing_type = "clear"
        self.contacted = False
        self.recover_t = 1e9             # 收拍后的过渡计时（从收拍姿势回到准备姿势）
        self.recover_hand = None
        self.recover_key = None          # 过渡期间锁定的准备姿势（防止半路改姿势造成「晃」）
        self.recover_theta = 0.0
        # 11-19 球场加宽到 1200 后重标：0.68W 处回合最长（实测 18 次/回合，0.715W 只有 4.7）
        self.recover_x = W * 0.32 if side == "L" else W * 0.68

    @property
    def feet_y(self):
        return FLOOR_Y - self.h

    @property
    def waist_y(self):
        return self.feet_y - WAIST_H

    def swing_q(self):
        """整段挥拍的时间进度 0..1（沉拍 + 挥击都算）。"""
        if self.swing_t < 0:
            return 0.0
        return min(self.swing_t / SWING_ARC[self.swing_type][2], 1.0)

    def hand_offset(self):
        """手相对脚底的位置：待机固定举拍，挥拍时走关键帧路径（按整段时间）。"""
        if self.swing_t < 0:
            return HAND_IDLE
        return _path_point(HAND_PATH[self.swing_type], self.swing_q())

    def hand(self):
        dx, dy = self.hand_offset()
        return (self.x + self.facing * dx, self.feet_y + dy)

    def theta_now(self):
        hi, lo, _dur = SWING_ARC[self.swing_type]
        return hi + (lo - hi) * self.swing_q()

    def draw_theta(self):
        """画拍子用的角度：先把拍面从准备姿势带到弧线起点，再整段匀速扫过弧线。

        为什么分两段：挑球的准备姿势拍子是朝下的（−0.55），而弧线起点在上方（1.05）——
        以前直接从准备角插到「弧线终点」，画面上几乎不动（只转 0.4 rad），
        看着像「没挥拍」。现在前 22%（挑球 32%）把拍面兜到弧线起点，后面匀速扫完。
        击球判定仍走 theta_now()（物理弧线，没有变）。
        """
        start = THETA_READY["hi" if self.swing_type == "clear" else "lo"]
        if self.swing_t < 0:
            return start
        hi, lo, _dur = SWING_ARC[self.swing_type]
        q = self.swing_q()
        if self.swing_type == "clear":
            # 用户认可的画法：整段匀速从架拍角转到弧线终点（不搞「先快后慢」）
            return start + (lo - start) * q
        prep = 0.55
        if q < prep:
            k = q / prep
            k = k * k * (3.0 - 2.0 * k)
            return start + (hi - start) * k
        k = (q - prep) / (1.0 - prep)
        return hi + (lo - hi) * k

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
                # 收完拍别「定格再瞬移」：记下收拍姿势，后面用 RECOVER_DUR 平滑过渡回准备姿势
                self.recover_t = 0.0
                self.recover_key = None
                self.recover_hand = _path_point(HAND_PATH[self.swing_type], 1.0)
                self.recover_theta = SWING_ARC[self.swing_type][1]
        elif self.recover_hand is not None:
            self.recover_t += dt


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
        self.lead_time = 0.13     # 「多久之后球到我这」的起挥阈值（秒）：改按时间起挥，球速变了也不会晚
        self.stand_off = 25       # 站在落点靠后多少

    def predict(self, ball):
        land, _, _ = physics.simulate_batch(
            [ball.x], [ball.y], [math.hypot(ball.vx, ball.vy)],
            [math.atan2(ball.vy, abs(ball.vx))], [1 if ball.vx > 0 else -1])
        return float(land[0])

    def intercept_x(self, ball, height=IC_HEIGHT):
        """球飞到「球拍够得着的高度」时的 x。

        平快球如果按落点站，球到身前时已经快贴地了（拍子扫不到）；站到这里才能在高处截住。
        返回 None 表示球已经低于这个高度（够不着了）。

        11-19 修的坑：平快球到网这边时往往已经低于 250，旧实现直接返回 None、
        AI 就退回落点站位 → 站过头、球从身后溜走（实测 2.0 次/回合）。
        现在按 250 → 190 → 140 → 100 逐档往下找，总能给一个「还够得着」的接球点。
        """
        v0 = math.hypot(ball.vx, ball.vy)
        if v0 < 1.0:
            return None
        ang = math.atan2(ball.vy, abs(ball.vx))
        d = 1 if ball.vx > 0 else -1
        _, xs, ys = physics.simulate_path(ball.x, ball.y, v0, ang, d)
        for h in (height, 190.0, 140.0, 100.0):
            for x, y in zip(xs, ys):
                if y <= h:
                    return float(x)
        return None

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
            # 每帧重算目标：平快球按落点站会够不着，优先站到「球降到拍子高度」的那个 x；
            # 球已经低于截击高度（快贴地了）就退回按落点站。
            ic = self.intercept_x(ball)
            self.target = ic if ic is not None else self.predict(ball)
            # 站在目标稍靠后一点，让球在身前接触
            want = self.target - p.facing * self.stand_off
            if abs(p.x - want) > 6:
                move = 1.0 if want > p.x else -1.0
            # 攻击球：球还在高处往下掉、已经快到我面前 → 起跳，让手高过球，
            # 拍子从上往下压（这就是模型里能「杀」的唯一方式：接触时拍面朝下）
            land_x = self.predict(ball)
            front = (land_x < NET_X + AI_ATTACK_ZONE) if p.side == "R" else (land_x > NET_X - AI_ATTACK_ZONE)
            if (AI_ATTACK_Y < ball.y and ball.vy < 0 and abs(ball.x - p.x) < AI_ATTACK_DX
                    and front and p.h <= 0.0 and p.swing_t < 0):
                jump = True
            near = abs(ball.x - p.x) < self.swing_dist
            height_ok = 50 < ball.y < 250 + p.h
            # 按「还要多久到」起挥，而不是「还有多远」：平快球用距离判会永远慢半拍
            t_arrive = abs(ball.x - p.x) / (abs(ball.vx) + 1e-6)
            if (near or t_arrive < self.lead_time) and height_ok and ball.vy < 150:
                p.start_swing(ball)
                self.state = "SWING"
        elif self.state == "SWING":
            if p.swing_t < 0:
                self.state = "RECOVER"
        if self.state in ("RECOVER",) and abs(p.x - p.recover_x) > 8:
            move = 1.0 if p.recover_x > p.x else -1.0
        return move, jump


class Game:
    def __init__(self, p_char, a_char, smoke=False, race=True, classic=False, speed=None):
        self.human = Player(p_char, "L")
        self.ai_p = Player(a_char, "R")
        self.ai = AIController(self.ai_p)
        self.ball = Ball()
        self.score = [0, 0]
        self.games = [0, 0]              # 经典模式：已拿下的局数
        self.game_no = 1
        self.classic = classic           # True = 三局两胜；race=False 时是无限练习
        self.banner = ""                 # 中央提示（"GAME 2" / "OUT!"）
        self.banner_sub = ""             # 提示的第二行
        self.banner_rgb = (242, 202, 72)
        self.banner_t = 0.0
        self.server = "L"
        self.serve_timer = 1.0
        self.paused = False
        self.debug = False              # 默认关（正式画面干净）；F1 打开看输入/判定状态
        self.last_key = ""
        self.keys_now = {}              # 本帧轮询到的按键状态（HUD 显示用）
        self.space_prev = False         # 空格上升沿触发（按住只挥一次）
        self.set_speed(SPEED_DEFAULT if speed is None else speed)
        self.over = False
        self.smoke = smoke
        self.race = race                  # True = 7 分制；默认练习模式（不限分，一直打）
        self.hits = 0
        self.rallies = 0
        self.serve_msg = ""               # 发球提示（站错位置/等按键时显示）

    def serve(self):
        p = self.human if self.server == "L" else self.ai_p
        if p is self.human and NET_X - p.x < SERVICE_LINE:   # 站进发球线以内，发球无效
            self.serve_msg = "step behind the service line, then press SPACE"
            return
        # 力度 = 高远球同角度的力度（v_base 是一条随球速档位缩放的曲线），发球不再是半个力度
        v0 = max(SERVE_V0, physics.v_base(SERVE_ANGLE))
        self.ball.launch(p.x + p.facing * 30, 110, v0, SERVE_ANGLE, p.facing, self.server)
        p.start_swing(force="lift")     # 只做发球动作：球已经从腰侧发出去了
        self.serve_msg = ""
        self.rallies += 1

    def point(self, winner, out_by=None):
        if out_by is not None:                           # 谁把球打出了界，当场在屏幕中间说一声
            self.banner = "OUT!"
            self.banner_sub = "YOU hit it out" if out_by == "L" else "AI hit it out"
            self.banner_rgb = (255, 132, 100)
            self.banner_t = 1.6
        self.score[0 if winner == "L" else 1] += 1
        self.server = winner
        self.ball.live = False
        self.serve_timer = 1.2
        self.ai.state = "RECOVER"
        if self.race and max(self.score) >= WIN_SCORE:
            if not self.classic:                   # 单局：到这局结束
                self.over = True
                return
            side = 0 if winner == "L" else 1       # 经典：一局结束，记一分大比分
            self.games[side] += 1
            if max(self.games) >= WIN_GAMES:       # 三局两胜拿到第 2 局 → 整场结束
                self.over = True
            else:                                  # 进入下一局
                self.score = [0, 0]
                self.game_no += 1
                self.banner = f"GAME {self.game_no}"
                self.banner_sub = f"{self.games[0]}-{self.games[1]}"
                self.banner_rgb = (242, 202, 72)
                self.banner_t = 2.0
                self.serve_timer = 2.0

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
            # 出球角的下压（常数 + 低拍角平方追加）已经由 physics.launch_angle() 算好了，
            # 这里**绝对不能再减一次** —— 11-24 两边同时减导致所有球变成 -7° 直钻网子。
            b.launch(b.x, b.y, float(v0), float(angle), p.facing, p.side)
            p.contacted = True
            self.hits += 1

    def update(self, dt, keys):
        if self.paused or self.over:
            return
        if self.banner_t > 0:
            self.banner_t = max(0.0, self.banner_t - dt)
        if not self.ball.live:
            if self.server == "R" or self.smoke:        # AI 自动发球；人必须自己按空格发
                self.serve_timer -= dt
                if self.serve_timer <= 0:
                    self.serve()
            else:
                self.serve_timer = 0.0
                if NET_X - self.human.x < SERVICE_LINE:  # 站在发球线以内不许发
                    self.serve_msg = "step behind the service line, then press SPACE"
                else:
                    self.serve_msg = "press SPACE to serve"
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
        space = bool(keys.get("space", False))
        if space and not self.space_prev:               # 按下那一下挥一次（按住不连挥）
            if not self.ball.live and self.server == "L" and not self.smoke:
                self.serve()                           # 发球：手动按空格
            else:
                self.human.start_swing(self.ball if self.ball.live else None)
        self.space_prev = space
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
                    hitter = self.ball.last_hitter
                    self.point("R" if hitter == "L" else "L", out_by=hitter)
                else:
                    self.point("R" if x < NET_X else "L")
                break
        if self.ball.live:
            self.ball.trail.append((self.ball.x, self.ball.y_screen))
            if len(self.ball.trail) > 16:
                self.ball.trail.pop(0)

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
        # 发球线（距网 SERVICE_LINE px，两条短线 + 小标签）
        for x in (int(NET_X - SERVICE_LINE), int(NET_X + SERVICE_LINE)):
            pygame.draw.line(s, (250, 214, 120), (x, FLOOR_Y - 26), (x, FLOOR_Y), 3)
            pygame.draw.line(s, (250, 214, 120), (x - 10, FLOOR_Y - 26), (x + 10, FLOOR_Y - 26), 2)

    def player_tip_at(self, p, q):
        """挥拍进度 q 处的拍头位置（给残影用，与 hand()/racket_dir() 同一套几何）。"""
        hx, hy = _path_point(HAND_PATH[p.swing_type], q)
        hi, lo, _ = SWING_ARC[p.swing_type]
        th = hi + (lo - hi) * q
        return (p.x + p.facing * (hx + math.cos(th) * RACKET_LEN),
                p.feet_y + hy - math.sin(th) * RACKET_LEN)

    def set_speed(self, k):
        """球速倍率：直接改 physics 的速度表（resolve_hit 仍按同一张表算，物理自洽）。"""
        physics.V_MIN = 380.0 * k
        physics.V_MAX = 820.0 * k
        self.speed_scale = k

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
            # 收拍后的目标姿势在这一段过渡里锁死（不再每帧重算），否则球一过腰就会
            # 在半路改成低姿势、看着像手臂在「晃」
            if p.recover_key is None or p.recover_t >= RECOVER_DUR:
                p.recover_key = self.ready_key(p)
            key = p.recover_key
            hand, th = HAND_READY[key], THETA_READY[key]
            free_hand = FREE_READY[key]
            # 收拍过渡：从刚收拍的姿势快速走回准备姿势（时间短，避免挥完还在动）
            if p.recover_hand is not None and p.recover_t < RECOVER_DUR:
                k = p.recover_t / RECOVER_DUR
                k = k * k * (3.0 - 2.0 * k)
                hand = (p.recover_hand[0] + (hand[0] - p.recover_hand[0]) * k,
                        p.recover_hand[1] + (hand[1] - p.recover_hand[1]) * k)
                th = p.recover_theta + (th - p.recover_theta) * k
        # 拍子不许穿地板：角度朝下时按手上的高度把拍长收短（像临场收腕），最低留 60 px
        hand_y = p.feet_y + hand[1]
        sin_th = math.sin(th)
        rl = DRAW_RACKET_LEN
        if sin_th < -1e-3:
            # 画出来的拍框最远端 = 1.043 × racket_len（柄 26 + 中轴 24 + 半径 23，按 70 归一）
            room = (FLOOR_Y - 8 - hand_y) / ((-sin_th) * (73.0 / 70.0) * SCALE)
            rl = max(60.0, min(rl, room))
        draw_character(s, p.char, int(p.x), int(p.feet_y),
                       "right" if p.facing > 0 else "left", SCALE,
                       racket_theta=th, hand=hand, free_hand=free_hand,
                       racket_len=rl)
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

    def mode_label(self):
        """当前模式的大字（画在球场背景上当水印）。"""
        if not self.race:
            return "ENDLESS"
        if self.classic:
            return "CLASSIC"
        return "FIRST TO 7"

    def draw(self, s, fonts):
        self.draw_court(s)
        # 背景水印：当前模式（对战界面一眼能看出这局是什么模式）
        mark = fonts["title"].render(self.mode_label(), True, (222, 230, 226))
        if mark.get_width() > 760:
            k = 760.0 / mark.get_width()
            mark = pygame.transform.smoothscale(mark, (760, int(mark.get_height() * k)))
        mark.set_alpha(110)
        s.blit(mark, mark.get_rect(center=(W // 2, H // 2 - 40)))
        self.draw_player(s, self.human)
        self.draw_player(s, self.ai_p)
        # 只有球在飞、或已经落地（y<=0）时才画；发球前那个停在半空的初始球不画（用户：「幽灵球」）
        if self.ball.live or self.ball.y <= 0.01:
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
        pygame.draw.rect(s, (22, 22, 26), (0, H - 38, W, 38))       # 底部键位条（加高，键帽样式）
        pygame.draw.line(s, (58, 60, 66), (0, H - 38), (W, H - 38), 1)
        hf = fonts.get("hint") or fonts["small"]
        draw_hints(s, hf, [("←/→  A/D", "move"), ("↑  W", "jump"), ("SPACE", "swing / serve"),
                           ("P", "pause"), ("F1", "debug"), ("ESC", "quit")],
                   14, H - 38 + 9)
        if not self.race:
            txt, col = "ENDLESS - no target score", (250, 226, 150)
        elif self.classic:
            txt = (f"MATCH   [{self.games[0]}-{self.games[1]}]" if self.over else
                   f"GAME {self.game_no}   FIRST TO {WIN_SCORE}   [{self.games[0]}-{self.games[1]}]")
            col = (232, 236, 230)
        else:
            txt, col = f"FIRST TO {WIN_SCORE}", (232, 236, 230)
        t = hf.render(txt, True, col)
        s.blit(t, (W - t.get_width() - 14, H - 38 + 10))
        if self.serve_msg:                              # 发球提示（该按空格 / 站太靠前）
            a = 0.6 + 0.4 * abs(math.sin(pygame.time.get_ticks() / 320.0))
            t = fonts["mid"].render(self.serve_msg, True,
                                    (int(250 * a), int(226 * a), int(150 * a)))
            s.blit(t, t.get_rect(center=(W // 2, H - 68)))
        if self.paused:
            t = fonts["big"].render("PAUSED", True, (248, 248, 244))
            s.blit(t, t.get_rect(center=(W // 2, H // 2)))
        if self.over:
            if self.classic:
                msg = "YOU WIN THE MATCH!" if self.games[0] > self.games[1] else "AI WINS THE MATCH"
                sub = f"games {self.games[0]}-{self.games[1]}"
            else:
                msg = "YOU WIN!" if self.score[0] > self.score[1] else "AI WINS"
                sub = f"{self.score[0]}-{self.score[1]}"
            t = fonts["big"].render(msg, True, (242, 202, 72))
            s.blit(t, t.get_rect(center=(W // 2, H // 2 - 26)))
            t0 = fonts["small"].render(sub, True, (238, 238, 232))
            s.blit(t0, t0.get_rect(center=(W // 2, H // 2 + 4)))
            t2 = fonts["small"].render("Enter: back to select   Esc: quit", True, (246, 246, 240))
            s.blit(t2, t2.get_rect(center=(W // 2, H // 2 + 28)))
        if self.banner_t > 0 and self.banner:            # 中央提示画在最上层：GAME n / OUT!
            a = min(1.0, self.banner_t / 0.4)
            base = self.banner_rgb
            cy = H // 2 - (112 if self.over else 60)    # 结束屏也在中间，让开一点
            t = fonts["big"].render(self.banner, True,
                                    (int(base[0] * a + 40 * (1 - a)),
                                     int(base[1] * a + 40 * (1 - a)),
                                     int(base[2] * a + 40 * (1 - a))))
            s.blit(t, t.get_rect(center=(W // 2, cy)))
            if self.banner_sub:
                t1 = fonts["mid"].render(self.banner_sub, True,
                                         (int(246 * a + 40 * (1 - a)),
                                          int(246 * a + 40 * (1 - a)),
                                          int(240 * a + 40 * (1 - a))))
                s.blit(t1, t1.get_rect(center=(W // 2, cy + 34)))
        if self.debug:                                  # 默认关：排查输入/判定用（F1 打开）
            hp = self.human
            lines = [
                f"keys: {self.last_key!r}  space={self.keys_now.get('space')}  swing {hp.swing_t:+.2f} ({hp.swing_type})  hits {self.hits}",
                f"ball live={self.ball.live} x={self.ball.x:6.1f} y={self.ball.y:5.1f} vx={self.ball.vx:6.0f}",
                f"player x={hp.x:6.1f} paused={self.paused} over={self.over} race={self.race} speed x{self.speed_scale:.1f} zoom {ZOOM:.2f}",
            ]
            for i, ln in enumerate(lines):
                t = fonts["small"].render(ln, True, (255, 250, 200))
                s.blit(t, (12, 84 + 18 * i))


def select_screen(win, world, fonts):
    """设置界面：3×4 网格 + 左侧大图 + 顶部四个设置框（模式 / 球速 / 窗口）。
    方向键选人、M 切模式、S 改球速、- / = 改窗口大小、回车开始。返回 (角色, 模式, 球速)。"""
    global ZOOM
    screen = world                  # 选择界面也按世界坐标画，最后整体缩放
    sel = 0
    modes = [("endless", "ENDLESS", "no target score"),
             ("classic", "CLASSIC", "best of 3 · first to 7")]
    mi = 0
    spd = SPEED_DEFAULT
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return None
                if e.key == pygame.K_RETURN:
                    return CHARACTERS[sel], modes[mi][0], spd
                if e.key in (pygame.K_m, pygame.K_TAB):
                    mi = (mi + 1) % len(modes)
                if e.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    step = 0.05 if e.key in (pygame.K_EQUALS, pygame.K_KP_PLUS) else -0.05
                    ZOOM = max(ZOOM_MIN, min(ZOOM_MAX, ZOOM + step))
                    win = pygame.display.set_mode(window_size())
                if e.key == pygame.K_RIGHT:
                    sel = (sel + 1) % 12
                if e.key == pygame.K_LEFT:
                    sel = (sel - 1) % 12
                if e.key == pygame.K_DOWN:
                    sel = (sel + 3) % 12
                if e.key == pygame.K_UP:
                    sel = (sel - 3) % 12
        screen.fill((18, 20, 24))
        screen.blit(fonts["mid"].render("CHOOSE YOUR CHARACTER", True, (236, 238, 232)), (24, 16))
        # 顶部四个设置框：模式 ×2、球速档位、窗口大小
        boxes = [(f"{label}", sub, k == mi) for k, (_mid, label, sub) in enumerate(modes)]
        boxes.append((f"WINDOW {int(round(ZOOM * 100))}%", "press - / =", True))
        for k, (label, sub, on) in enumerate(boxes):
            r = pygame.Rect(330 + k * 200, 54, 190, 44)
            pygame.draw.rect(screen, (58, 62, 54) if on else (38, 40, 46), r)
            pygame.draw.rect(screen, (242, 202, 72) if on else (92, 94, 100), r, 3 if on else 1)
            lb = fonts["mid"].render(label, True, (244, 226, 150) if on else (168, 172, 168))
            screen.blit(lb, lb.get_rect(center=(r.centerx, r.centery - 7)))
            sb = fonts["small"].render(sub, True, (168, 172, 168) if on else (120, 124, 122))
            screen.blit(sb, sb.get_rect(center=(r.centerx, r.centery + 12)))
        gx, gy, cw, ch, gap = 330, 120, 200, 112, 8
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
        draw_character(screen, c, 150, 600, "right", 1.15)
        n = fonts["mid"].render(c["name"], True, (236, 238, 232))
        screen.blit(n, n.get_rect(center=(150, 636)))
        good, bad = pros_cons(c)
        t = fonts["small"].render(f"{good} / {bad}", True, (156, 162, 158))
        screen.blit(t, t.get_rect(center=(150, 660)))
        nm = normalize(c)
        for j, (key, val) in enumerate([("SPEED", str(c["speed"])),
                                        ("POWER", f"{c['power']:.2f}"),
                                        ("JUMP", str(c["jump"]))]):
            y = 682 + j * 22
            screen.blit(fonts["small"].render(key, True, (156, 162, 158)), (40, y))
            filled = 1 + round(nm[key.lower()] * 7)
            for i in range(8):
                col = tuple(min(255, v + 80) for v in c["accent"]) if i < filled else (54, 56, 62)
                pygame.draw.rect(screen, col, (100 + i * 15, y + 3, 12, 10))
            screen.blit(fonts["small"].render(val, True, (176, 180, 176)), (228, y))
        pygame.draw.rect(screen, (22, 22, 26), (0, H - 38, W, 38))
        pygame.draw.line(screen, (58, 60, 66), (0, H - 38), (W, H - 38), 1)
        draw_hints(screen, fonts.get("hint") or fonts["small"],
                   [("←/→", "select"), ("M", "mode"),
                    ("- / =", "window"), ("ENTER", "start"), ("ESC", "quit")],
                   gx, H - 38 + 9)
        present(win, world)
        clock.tick(60)


def start_screen(win, world, fonts):
    """开始界面：只有标题和「按任意键进入」，按任意键/点一下进设置（选人+模式）界面。"""
    clock = pygame.time.Clock()
    t = 0.0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                return True
            if e.type == pygame.MOUSEBUTTONDOWN:
                return True
        t += 1.0 / 60.0
        world.fill((16, 18, 22))
        pygame.draw.rect(world, (44, 46, 52), (-20, FLOOR_Y, W + 40, H - FLOOR_Y + 20))
        pygame.draw.line(world, (86, 90, 96), (-20, FLOOR_Y), (W + 20, FLOOR_Y), 2)
        for i, (ch, x, face) in enumerate([(CHARACTERS[8], W * 0.26, "right"),
                                           (CHARACTERS[1], W * 0.74, "left")]):
            draw_character(world, ch, x, FLOOR_Y + 2, face, 1.0,
                           racket_theta=THETA_READY["hi"], hand=HAND_READY["hi"],
                           free_hand=FREE_READY["hi"], racket_len=DRAW_RACKET_LEN)
        title = fonts["title"].render("NEURAL BADMINTON", True, (242, 202, 72))
        world.blit(title, title.get_rect(center=(W // 2, 150)))
        sub = fonts["small"].render("physics simulated in NumPy   ·   neural landing predictor", True, (150, 156, 152))
        world.blit(sub, sub.get_rect(center=(W // 2, 202)))
        a = 0.55 + 0.45 * abs(math.sin(t * 2.0))
        pr = fonts["mid"].render("press any key or click to start", True,
                                 (int(236 * a), int(238 * a), int(232 * a)))
        world.blit(pr, pr.get_rect(center=(W // 2, H - 78)))
        present(win, world)
        clock.tick(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="双 AI 自动对打 30 s 后退出")
    ap.add_argument("--practice", action="store_true", help="练习模式（不限分一直打；默认是 7 分制）")
    ap.add_argument("--race", action="store_true", help=argparse.SUPPRESS)   # 旧参数，保留兼容
    args = ap.parse_args()

    pygame.init()
    world = pygame.Surface((W, H))
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption("Neural Badminton")
    fonts = {"title": pygame.font.SysFont("segoeui", 56, bold=True),
             "hint": pygame.font.SysFont("segoeui", 16, bold=True),
             "score": pygame.font.SysFont("consolas", 40, bold=True),
             "big": pygame.font.SysFont("segoeui", 34, bold=True),
             "mid": pygame.font.SysFont("segoeui", 22, bold=True),
             "small": pygame.font.SysFont("segoeui", 14)}
    clock = pygame.time.Clock()

    if args.smoke:
        game = Game(CHARACTERS[8], CHARACTERS[1], smoke=True, race=False)   # Bun vs Rex（冒烟跑满 30 s，不限分）
        frames, t0 = 0, pygame.time.get_ticks()
        fps_min = 999
        while pygame.time.get_ticks() - t0 < 30000 and not game.over:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    break
            game.update(1 / 60, {})
            game.draw(world, fonts)
            present(screen, world)
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

    if not start_screen(screen, world, fonts):     # 开始界面 → 再进设置（选人+模式）
        pygame.quit()
        return

    while True:
        picked = select_screen(screen, world, fonts)
        if picked is None:
            break
        p_char, mode, spd = picked
        if args.practice:                       # 命令行强制无限练习
            mode = "endless"
        idx = CHARACTERS.index(p_char)
        game = Game(p_char, CHARACTERS[(idx + 6) % 12],
                    race=(mode != "endless"), classic=(mode == "classic"), speed=spd)
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
                    if e.key == pygame.K_F1:
                        game.debug = not game.debug
                    game.last_key = pygame.key.name(e.key)
                    if e.key == pygame.K_RETURN and game.over:
                        break
                if e.type == pygame.KEYUP:
                    pass                            # 按住状态改用 get_pressed() 轮询，见下
            else:
                # 按键状态用轮询（不依赖 KEYDOWN/KEYUP 事件，少受输入法/焦点影响）
                pk = pygame.key.get_pressed()
                keys = {
                    "a": bool(pk[pygame.K_a] or pk[pygame.K_LEFT]),
                    "d": bool(pk[pygame.K_d] or pk[pygame.K_RIGHT]),
                    "w": bool(pk[pygame.K_w] or pk[pygame.K_UP]),
                    "space": bool(pk[pygame.K_SPACE]),
                }
                game.keys_now = keys
                try:
                    game.update(1 / 60, keys)
                    game.draw(world, fonts)
                    present(screen, world)
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

