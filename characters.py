"""角色表：一份数据同时决定外观和数值。

12 个角色，每个 = 几个外观开关 + 三个数字：
    speed  左右移动速度（像素/秒）
    power  击球力量（出球速度的倍率）
    jump   跳跃高度（像素）

选人界面、物理和绘制都读这一份表，不写第二份。
"""

import math

GRAVITY = 1800.0
INK = (28, 28, 32)
WHITE = (246, 246, 242)
RACKET = (228, 196, 62)

# 顺序有讲究：第 i 个和第 i+6 个是一对，三项数值互补。
CHARACTERS = [
    {"id": "tophat", "name": "Top Hat", "hat": "top", "hair": None, "extra": None,
     "accent": (52, 52, 60), "speed": 385, "power": 1.02, "jump": 125},
    {"id": "rex", "name": "Rex", "hat": None, "hair": "cap", "extra": None,
     "accent": (178, 44, 40), "speed": 430, "power": 0.92, "jump": 120},
    {"id": "beanie", "name": "Beanie", "hat": "beanie", "hair": None, "extra": None,
     "accent": (58, 118, 190), "speed": 360, "power": 0.96, "jump": 150},
    {"id": "specs", "name": "Specs", "hat": None, "hair": "cap", "extra": "glasses",
     "accent": (150, 150, 150), "speed": 375, "power": 1.13, "jump": 112},
    {"id": "pony", "name": "Ponytail", "hat": None, "hair": "pony", "extra": None,
     "accent": (172, 62, 52), "speed": 425, "power": 0.90, "jump": 115},
    {"id": "helmet", "name": "Helmet", "hat": "helmet", "hair": None, "extra": None,
     "accent": (70, 160, 160), "speed": 355, "power": 0.98, "jump": 155},
    {"id": "phones", "name": "Phones", "hat": None, "hair": None, "extra": "phones",
     "accent": (224, 126, 56), "speed": 405, "power": 0.96, "jump": 115},
    {"id": "cap", "name": "Cap", "hat": "cap", "hair": None, "extra": None,
     "accent": (82, 160, 92), "speed": 410, "power": 1.00, "jump": 130},
    {"id": "bun", "name": "Bun", "hat": None, "hair": "bun", "extra": None,
     "accent": (126, 126, 126), "speed": 390, "power": 1.04, "jump": 120},
    {"id": "antenna", "name": "Antenna", "hat": None, "hair": None, "extra": "antenna",
     "accent": (120, 190, 190), "speed": 365, "power": 1.14, "jump": 100},
    {"id": "scarf", "name": "Scarf", "hat": None, "hair": None, "extra": "scarf",
     "accent": (200, 60, 60), "speed": 400, "power": 0.98, "jump": 128},
    {"id": "bowtie", "name": "Bowtie", "hat": None, "hair": None, "extra": "bowtie",
     "accent": (150, 90, 190), "speed": 395, "power": 1.06, "jump": 118},
]

CHAR_BY_ID = {c["id"]: c for c in CHARACTERS}


def jump_speed(char):
    """从跳跃高度反推起跳速度：v = sqrt(2gh)。"""
    return math.sqrt(2 * GRAVITY * char["jump"])


def _arc(cx, cy, r, a0, a1, n=14):
    pts = []
    for i in range(n):
        a = a0 + (a1 - a0) * i / (n - 1)
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    return pts


def _ellipse(cx, cy, a, b, angle, n=26):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        px, py = a * math.cos(t), b * math.sin(t)
        pts.append((
            int(cx + px * math.cos(angle) - py * math.sin(angle)),
            int(cy + px * math.sin(angle) + py * math.cos(angle)),
        ))
    return pts


def _elbow(shoulder, hand, upper=27.0, fore=25.0):
    """两段手臂 IK：已知肩和手，求肘位置。取肘部朝下的一解，动作像真人。

    坐标与 draw_character 的 P() 同一套：相对脚底、面朝方向为正。
    """
    sx, sy = shoulder
    dx, dy = hand[0] - sx, hand[1] - sy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (sx + upper * 0.5, sy + upper * 0.5)
    ux, uy = dx / dist, dy / dist
    d = min(dist, upper + fore - 0.5)
    cos_a = (upper * upper + d * d - fore * fore) / (2 * upper * d)
    cos_a = max(-1.0, min(1.0, cos_a))
    base = math.atan2(uy, ux)
    ang = math.acos(cos_a)
    e1 = (sx + upper * math.cos(base + ang), sy + upper * math.sin(base + ang))
    e2 = (sx + upper * math.cos(base - ang), sy + upper * math.sin(base - ang))
    return e1 if e1[1] > e2[1] else e2


def draw_character(surface, char, x, feet_y, facing="right", scale=1.0,
                   racket_theta=None, hand=None, free_hand=None, racket_len=70.0):
    import pygame  # 只有绘制需要 pygame；数据表本身可脱离 pygame 导入

    s = scale
    d = 1 if facing == "right" else -1

    def P(dx, dy):
        return (int(x + d * dx * s), int(feet_y + dy * s))

    def W(w):
        return max(1, int(round(w * s)))

    lw = W(6)
    head = P(0, -163)
    radius = int(21 * s)
    accent = char["accent"]

    shadow = pygame.Surface((int(78 * s), int(20 * s)), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
    surface.blit(shadow, (int(x - 39 * s), int(feet_y - 10 * s)))

    pygame.draw.line(surface, INK, P(0, -80), P(-30, 0), lw)
    pygame.draw.line(surface, INK, P(0, -80), P(26, -4), lw)
    pygame.draw.line(surface, INK, P(0, -80), P(0, -138), W(7))

    shoulder = P(0, -128)
    swing_hand = hand if hand is not None else (30.0, -110.0)
    rl = max(40.0, float(racket_len))              # 拍长（手→拍头中心约 0.95 rl）
    rf = rl / 70.0                                 # 相对旧拍长的缩放

    def _draw_swing_arm():
        """挥拍姿势：肩→肘（IK）→手，拍面朝向 = racket_theta。"""
        th = racket_theta
        hx, hy = swing_hand
        elbow = _elbow((0.0, -128.0), (hx, hy))
        hx_s, hy_s = P(hx, hy)
        pygame.draw.line(surface, INK, shoulder, P(*elbow), lw)
        pygame.draw.line(surface, INK, P(*elbow), (hx_s, hy_s), lw)
        dx, dy = d * math.cos(th), -math.sin(th)
        handle = (hx_s + dx * 26 * rf * s, hy_s + dy * 26 * rf * s)
        pygame.draw.line(surface, INK, (hx_s, hy_s),
                         (int(handle[0]), int(handle[1])), W(5 * rf))
        frame = _ellipse(handle[0] + dx * 24 * rf * s, handle[1] + dy * 24 * rf * s,
                         int(23 * rf * s), int(13 * rf * s), -th * d)
        pygame.draw.polygon(surface, RACKET, frame, W(4 * rf))

    # 引拍时手在身体后侧 → 先画手臂和拍，让头挡住，才像「从背后挥出来」
    behind = racket_theta is not None and hand is not None and hand[0] < 8.0
    if behind:
        _draw_swing_arm()

    pygame.draw.circle(surface, WHITE, head, radius)
    pygame.draw.circle(surface, INK, head, radius, W(4))

    hat = char["hat"]
    if hat == "top":
        hat_y = head[1] - radius - int(26 * s)
        pygame.draw.rect(surface, INK, (head[0] - int(13 * s), hat_y, int(26 * s), int(28 * s)))
        pygame.draw.ellipse(surface, INK, (head[0] - int(25 * s), hat_y + int(22 * s),
                                           int(50 * s), int(10 * s)))
    elif hat == "beanie":
        pts = _arc(head[0], head[1], radius + int(3 * s), math.pi, 2 * math.pi)
        pts += [(head[0] + radius + int(3 * s), head[1] - int(7 * s)),
                (head[0] - radius - int(3 * s), head[1] - int(7 * s))]
        pygame.draw.polygon(surface, accent, pts)
        pygame.draw.circle(surface, accent, (head[0], head[1] - radius - int(8 * s)), W(6))
        pygame.draw.line(surface, WHITE, (head[0] - radius, head[1] - int(7 * s)),
                         (head[0] + radius, head[1] - int(7 * s)), W(2))
    elif hat == "cap":
        pts = _arc(head[0], head[1], radius + int(2 * s), math.pi, 2 * math.pi)
        pts += [(head[0] + radius + int(2 * s), head[1] - int(7 * s)),
                (head[0] - radius - int(2 * s), head[1] - int(7 * s))]
        pygame.draw.polygon(surface, accent, pts)
        brim = [(head[0] + d * int(4 * s), head[1] - int(9 * s)),
                (head[0] + d * int(36 * s), head[1] - int(7 * s)),
                (head[0] + d * int(36 * s), head[1] - int(1 * s)),
                (head[0] + d * int(4 * s), head[1] - int(3 * s))]
        pygame.draw.polygon(surface, accent, brim)
    elif hat == "helmet":
        pts = _arc(head[0], head[1], radius + int(4 * s), math.pi * 1.02, math.pi * 1.98)
        pts += [(head[0] + radius + int(4 * s), head[1] - int(2 * s)),
                (head[0] - radius - int(4 * s), head[1] - int(2 * s))]
        pygame.draw.polygon(surface, accent, pts)
        pygame.draw.line(surface, WHITE, (head[0], head[1] - radius - int(3 * s)),
                         (head[0], head[1] - int(8 * s)), W(4))
        pygame.draw.line(surface, accent, (head[0] + d * int(8 * s), head[1] - int(5 * s)),
                         (head[0] + d * int(30 * s), head[1] - int(2 * s)), W(5))

    hair = char["hair"]
    if hair in ("cap", "pony", "bun"):
        pts = _arc(head[0], head[1], radius + 1, math.pi, 2 * math.pi)
        pts += [(head[0] + radius + 1, head[1] - int(5 * s)),
                (head[0] - radius - 1, head[1] - int(5 * s))]
        pygame.draw.polygon(surface, accent, pts)
        if hair == "pony":
            pygame.draw.lines(surface, accent, False, [
                (head[0] - d * int(18 * s), head[1] - int(14 * s)),
                (head[0] - d * int(34 * s), head[1] + int(4 * s)),
                (head[0] - d * int(32 * s), head[1] + int(24 * s)),
            ], W(7))
        if hair == "bun":
            pygame.draw.circle(surface, accent,
                               (head[0] - d * int(7 * s), head[1] - radius - int(5 * s)), W(8))

    extra = char["extra"]
    if extra == "glasses":
        pygame.draw.circle(surface, INK, (head[0] - int(9 * s), head[1] - int(1 * s)),
                           W(6), W(2))
        pygame.draw.circle(surface, INK, (head[0] + int(9 * s), head[1] - int(1 * s)),
                           W(6), W(2))
        pygame.draw.line(surface, INK, (head[0] - int(3 * s), head[1] - int(1 * s)),
                         (head[0] + int(3 * s), head[1] - int(1 * s)), W(2))
    elif extra == "phones":
        pygame.draw.lines(surface, accent, False,
                          _arc(head[0], head[1], radius + int(6 * s), math.pi * 1.12,
                               math.pi * 1.88, 10), W(5))
        for a in (math.pi * 1.12, math.pi * 1.88):
            pygame.draw.circle(surface, accent,
                               (int(head[0] + (radius + int(6 * s)) * math.cos(a)),
                                int(head[1] + (radius + int(6 * s)) * math.sin(a))), W(6))
    elif extra == "antenna":
        pygame.draw.line(surface, INK, (head[0], head[1] - radius),
                         (head[0], head[1] - radius - int(15 * s)), W(3))
        pygame.draw.circle(surface, accent,
                           (head[0], head[1] - radius - int(15 * s)), W(4))
    elif extra == "scarf":
        pygame.draw.line(surface, accent, (head[0] - int(14 * s), head[1] + int(24 * s)),
                         (head[0] + int(14 * s), head[1] + int(24 * s)), W(8))
        pygame.draw.line(surface, accent, (head[0] + int(7 * s), head[1] + int(27 * s)),
                         (head[0] + int(12 * s), head[1] + int(44 * s)), W(5))
    elif extra == "bowtie":
        pygame.draw.polygon(surface, accent, [
            (head[0] - int(14 * s), head[1] + int(19 * s)),
            (head[0] - int(2 * s), head[1] + int(24 * s)),
            (head[0] - int(14 * s), head[1] + int(29 * s)),
        ])
        pygame.draw.polygon(surface, accent, [
            (head[0] + int(14 * s), head[1] + int(19 * s)),
            (head[0] + int(2 * s), head[1] + int(24 * s)),
            (head[0] + int(14 * s), head[1] + int(29 * s)),
        ])
        pygame.draw.circle(surface, accent, (head[0], head[1] + int(24 * s)), W(3))

    if racket_theta is None:
        # 待机姿势（旧静态姿势，正常游戏里由 game.py 传准备姿势角度，不走这里）
        elbow = P(34, -152)
        hand_p = P(50, -186)
        pygame.draw.line(surface, INK, shoulder, elbow, lw)
        pygame.draw.line(surface, INK, elbow, hand_p, lw)
        handle = P(76, -196)
        pygame.draw.line(surface, INK, hand_p, handle, W(5))
        frame = _ellipse(handle[0] + d * int(24 * s), handle[1] - int(10 * s),
                         int(23 * s), int(13 * s), math.radians(-32 * d))
        pygame.draw.polygon(surface, RACKET, frame, W(4))
    elif not behind:
        # 挥拍姿势（手在身前时画在头之后），手位与 game.py 物理接触几何同一套
        _draw_swing_arm()

    # 非持拍手：姿势不同位置不同（架拍时抬到下巴高度，低姿势时挂在腰侧）
    fh = free_hand if free_hand is not None else (-8.0, -96.0)
    f_elbow = _elbow((0.0, -128.0), fh)
    pygame.draw.line(surface, INK, shoulder, P(*f_elbow), lw)
    pygame.draw.line(surface, INK, P(*f_elbow), P(*fh), lw)


def normalize(char):
    return {
        "speed": min(1.0, max(0.0, (char["speed"] - 350) / 85)),
        "power": min(1.0, max(0.0, (char["power"] - 0.89) / 0.26)),
        "jump": min(1.0, max(0.0, (char["jump"] - 95) / 65)),
    }


def pros_cons(char):
    n = normalize(char)
    labels = {
        "speed": ("Fast feet", "Slow feet"),
        "power": ("Heavy smash", "Soft shots"),
        "jump": ("Big jump", "Low jump"),
    }
    best = max(n, key=n.get)
    worst = min(n, key=n.get)
    return labels[best][0], labels[worst][1]
