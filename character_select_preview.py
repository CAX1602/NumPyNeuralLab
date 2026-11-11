"""选人界面的风格样张：左侧大图 + 右侧 3×4 网格（照参考图 1 的布局）。

运行：python character_select_preview.py
输出：docs/character_select.png
"""

from pathlib import Path

import pygame

from characters import CHARACTERS, draw_character, normalize, pros_cons

WIDTH, HEIGHT = 1100, 700
BG = (18, 20, 24)
CELL = (46, 48, 54)
CELL_EDGE = (92, 94, 100)
SELECTED = (242, 202, 72)
TEXT = (236, 238, 232)
MUTED = (156, 162, 158)


def draw_bar(surface, label, value_text, frac, y, font, accent):
    surface.blit(font.render(label, True, MUTED), (34, y))
    x0, blocks = 112, 8
    filled = 1 + round(frac * (blocks - 1))
    lit = tuple(min(255, c + 80) for c in accent)
    for i in range(blocks):
        rect = pygame.Rect(x0 + i * 15, y + 3, 12, 10)
        pygame.draw.rect(surface, lit if i < filled else (54, 56, 62), rect)
    surface.blit(font.render(value_text, True, (176, 180, 176)), (x0 + blocks * 15 + 8, y))


def main():
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    screen.fill(BG)

    # 左侧聚光灯
    pygame.draw.polygon(screen, (44, 42, 33),
                        [(118, 0), (250, 0), (400, HEIGHT), (0, HEIGHT)])
    pygame.draw.ellipse(screen, (62, 58, 40), (18, 518, 306, 64))

    title_font = pygame.font.SysFont("segoeui", 38, bold=True)
    screen.blit(title_font.render("CHOOSE YOUR CHARACTER", True, TEXT), (36, 24))

    small = pygame.font.SysFont("segoeui", 14)
    name_font = pygame.font.SysFont("segoeui", 17, bold=True)
    big_name = pygame.font.SysFont("segoeui", 28, bold=True)
    start_font = pygame.font.SysFont("segoeui", 26, bold=True)

    selected = 0
    char = CHARACTERS[selected]

    # 右侧网格
    gx, gy, cw, ch, gap = 350, 100, 238, 134, 8
    for i, item in enumerate(CHARACTERS):
        col, row = i % 3, i // 3
        rect = pygame.Rect(gx + col * (cw + gap), gy + row * (ch + gap), cw, ch)
        pygame.draw.rect(screen, CELL, rect)
        pygame.draw.rect(screen, SELECTED if i == selected else CELL_EDGE, rect,
                         3 if i == selected else 1)

        draw_character(screen, item, rect.centerx, rect.bottom - 26, "right", 0.42)
        label = name_font.render(item["name"], True, TEXT if i == selected else (206, 208, 204))
        screen.blit(label, label.get_rect(center=(rect.centerx, rect.bottom - 14)))

    # 左侧大图
    draw_character(screen, char, 168, 540, "right", 1.3)
    name = big_name.render(char["name"], True, TEXT)
    screen.blit(name, name.get_rect(center=(168, 566)))

    good, bad = pros_cons(char)
    line = small.render(f"{good}  ·  {bad}", True, MUTED)
    screen.blit(line, line.get_rect(center=(168, 592)))

    n = normalize(char)
    draw_bar(screen, "SPEED", str(char["speed"]), n["speed"], 614, small, char["accent"])
    draw_bar(screen, "POWER", f"{char['power']:.2f}", n["power"], 636, small, char["accent"])
    draw_bar(screen, "JUMP", str(char["jump"]), n["jump"], 658, small, char["accent"])

    hint = small.render("ARROWS  select      ENTER  start", True, MUTED)
    screen.blit(hint, (gx, HEIGHT - 24))
    start = start_font.render("Start >", True, SELECTED)
    screen.blit(start, start.get_rect(right=WIDTH - 28, centery=HEIGHT - 26))

    out = Path(__file__).parent / "docs" / "character_select.png"
    out.parent.mkdir(exist_ok=True)
    pygame.image.save(screen, str(out))
    print("saved:", out)


if __name__ == "__main__":
    main()
