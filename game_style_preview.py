"""比赛画面的风格样张：用 pygame 画出目标视觉效果（静态，不跑物理）。

参考 4399《火柴人打羽毛球》：固定侧视、火柴人、木地板、网、记分牌、球尾火焰。
运行：python game_style_preview.py
输出：docs/game_style_preview.png
"""

import math
import random
from pathlib import Path

import pygame

from characters import CHAR_BY_ID, draw_character

WIDTH, HEIGHT = 1100, 700
FLOOR_TOP = 545
INK = (28, 28, 32)
WALL = (168, 178, 174)
WOOD = (198, 156, 106)
COURT_LINE = (236, 238, 230)
RACKET = (228, 196, 62)


def lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_background(surface):
    surface.fill(WALL)

    rng = random.Random(7)
    for _ in range(2600):
        x = rng.randrange(WIDTH)
        y = rng.randrange(FLOOR_TOP)
        shade = rng.choice([(150, 160, 156), (183, 191, 187), (158, 168, 164)])
        pygame.draw.circle(surface, shade, (x, y), rng.choice([1, 1, 2]))

    # 两侧稍暗，做出房间的纵深
    pygame.draw.rect(surface, (158, 168, 164), (0, 0, 44, FLOOR_TOP))
    pygame.draw.rect(surface, (158, 168, 164), (WIDTH - 44, 0, 44, FLOOR_TOP))

    pygame.draw.polygon(
        surface, WOOD,
        [(70, FLOOR_TOP), (1030, FLOOR_TOP), (WIDTH, HEIGHT), (0, HEIGHT)],
    )
    for i in range(17):
        t = i / 16
        pygame.draw.line(surface, (178, 138, 90),
                         (70 + t * 960, FLOOR_TOP), (t * WIDTH, HEIGHT), 2)

    pygame.draw.rect(surface, (168, 128, 82), (70, FLOOR_TOP, 960, 14))
    pygame.draw.line(surface, (120, 92, 58), (70, FLOOR_TOP), (1030, FLOOR_TOP), 3)
    pygame.draw.line(surface, COURT_LINE, (70, FLOOR_TOP), (0, HEIGHT), 4)
    pygame.draw.line(surface, COURT_LINE, (1030, FLOOR_TOP), (WIDTH, HEIGHT), 4)
    pygame.draw.line(surface, COURT_LINE, (34, 632), (1066, 632), 4)


def draw_net(surface):
    cx = WIDTH // 2
    top, bottom = 452, FLOOR_TOP + 10

    pygame.draw.line(surface, (70, 76, 82), (cx - 20, top - 8), (cx - 20, bottom), 4)

    rect = pygame.Rect(cx - 13, top, 26, bottom - top)
    pygame.draw.rect(surface, (230, 235, 231), rect)
    for y in range(top, bottom, 9):
        pygame.draw.line(surface, (152, 160, 156), (rect.left, y), (rect.right, y), 1)
    for x in range(rect.left, rect.right + 1, 6):
        pygame.draw.line(surface, (152, 160, 156), (x, top), (x, bottom), 1)
    pygame.draw.rect(surface, (250, 250, 245), (cx - 15, top - 6, 30, 8))
    pygame.draw.rect(surface, (110, 118, 114), (cx - 14, top, 28, bottom - top), 2)
    pygame.draw.rect(surface, (72, 82, 94),
                     (cx - 18, bottom - 12, 36, 14), border_radius=3)


def draw_trail(surface, trail):
    for x, y, r, t in trail:
        glow = lerp(WALL, (255, 150, 40), t ** 1.6)
        pygame.draw.circle(surface, lerp(WALL, glow, 0.45), (int(x), int(y)), int(r + 4))
        pygame.draw.circle(surface, (255, 168, 52), (int(x), int(y)), int(r))


def draw_shuttle(surface, pos, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = -dy, dx
    cork = pos
    tip = (cork[0] - dx * 26, cork[1] - dy * 26)

    feather = [
        (tip[0] + px * 11, tip[1] + py * 11),
        (tip[0] - px * 11, tip[1] - py * 11),
        (cork[0] - px * 5, cork[1] - py * 5),
        (cork[0] + px * 5, cork[1] + py * 5),
    ]
    pygame.draw.polygon(surface, (245, 244, 238), feather)
    pygame.draw.polygon(surface, (170, 176, 172), feather, 2)
    pygame.draw.line(surface, (190, 195, 190),
                     (tip[0] + px * 5, tip[1] + py * 5), (cork[0], cork[1]), 1)
    pygame.draw.line(surface, (190, 195, 190),
                     (tip[0] - px * 5, tip[1] - py * 5), (cork[0], cork[1]), 1)
    pygame.draw.circle(surface, (222, 182, 120), (int(cork[0]), int(cork[1])), 6)
    pygame.draw.circle(surface, (150, 115, 65), (int(cork[0]), int(cork[1])), 6, 1)


def draw_scoreboard(surface, score, font):
    rect = pygame.Rect(WIDTH // 2 - 110, 34, 220, 78)
    pygame.draw.rect(surface, (10, 10, 12), rect)
    pygame.draw.rect(surface, (58, 58, 62), rect, 3)

    text = font.render(f"{score[0]} - {score[1]}", True, (246, 44, 34))
    surface.blit(text, text.get_rect(center=rect.center))


def draw_bubble(surface, x, y, text, font, pointer_to=None):
    label = font.render(text, True, (46, 46, 50))
    w, h = label.get_size()
    rect = pygame.Rect(x, y, w + 34, h + 22)
    pygame.draw.rect(surface, (250, 250, 247), rect, border_radius=12)
    pygame.draw.rect(surface, (116, 124, 120), rect, 2, border_radius=12)
    surface.blit(label, (x + 17, y + 11))

    if pointer_to:
        start = (rect.right - 18, rect.bottom - 6)
        pygame.draw.line(surface, (40, 40, 44), start, pointer_to, 2)
        pygame.draw.circle(surface, (40, 40, 44), pointer_to, 3)


def draw_labels(surface, font):
    shadow = font.render("NEURAL BADMINTON", True, (90, 98, 94))
    surface.blit(shadow, (24, 22))
    surface.blit(font.render("NEURAL BADMINTON", True, (248, 248, 244)), (23, 21))

    bar = pygame.Surface((WIDTH, 38), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 85))
    surface.blit(bar, (0, HEIGHT - 38))
    hint = font.render("<- ->  move      Space  swing      R  restart", True, (246, 246, 240))
    surface.blit(hint, (24, HEIGHT - 32))


def render_frame(score=(1, 1), shuttle=(648, 302), trail_angle=-0.72,
                 taunt="You call that a smash?"):
    surface = pygame.Surface((WIDTH, HEIGHT))
    draw_background(surface)
    draw_net(surface)

    draw_character(surface, CHAR_BY_ID["tophat"], 210, 648, "right", 1.0)
    draw_character(surface, CHAR_BY_ID["rex"], 890, 648, "left", 1.0)

    trail = []
    for i in range(16):
        t = i / 15
        x = 520 + t * 128 + math.sin(t * 3.4) * 14
        y = 452 - t * 150
        trail.append((x + 8, y + 8, 2 + 6 * t, t))
    draw_trail(surface, trail)
    draw_shuttle(surface, shuttle, trail_angle)

    draw_scoreboard(surface, score, pygame.font.SysFont("consolas", 50, bold=True))
    draw_bubble(surface, 690, 300, taunt, pygame.font.SysFont("segoeui", 20),
                pointer_to=(892, 470))
    draw_labels(surface, pygame.font.SysFont("segoeui", 19, bold=True))

    return surface


def main():
    pygame.init()
    frame = render_frame()
    out = Path(__file__).parent / "docs" / "game_style_preview.png"
    out.parent.mkdir(exist_ok=True)
    pygame.image.save(frame, str(out))
    print("saved:", out)


if __name__ == "__main__":
    main()
