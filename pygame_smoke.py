"""pygame-ce 冒烟测试：确认窗口、键盘、计时三件事都正常。

运行前（只需一次）：
    .venv\\Scripts\\python.exe -m pip install pygame-ce

运行：
    .venv\\Scripts\\python.exe pygame_smoke.py

看到窗口后检查：
    1. 按住 ←/→ 或 A/D，小球应跟着左右移动；
    2. 空格或 ↑ 让小球跳起来；
    3. 左上角应显示 60 左右的 FPS；
    4. 按 Esc 退出。

退出后终端会打印一行统计，把那一行连同报错（如果有）一起贴回来。
"""

import pygame

WIDTH, HEIGHT = 900, 500
GROUND = HEIGHT - 60
BALL_RADIUS = 18
SPEED = 420        # 像素/秒
GRAVITY = 1400     # 像素/秒²
JUMP_SPEED = -650


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("pygame-ce smoke test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    x = WIDTH / 2
    y = GROUND - BALL_RADIUS
    vy = 0.0
    on_ground = True

    frames = 0
    elapsed = 0.0
    moved_px = 0.0
    jumps = 0
    fps_samples = []

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        frames += 1
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP) and on_ground:
                    vy = JUMP_SPEED
                    on_ground = False
                    jumps += 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            x -= SPEED * dt
            moved_px += SPEED * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            x += SPEED * dt
            moved_px += SPEED * dt

        # 别让小球跑出画面
        x = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, x))

        vy += GRAVITY * dt
        y += vy * dt
        if y >= GROUND - BALL_RADIUS:
            y = GROUND - BALL_RADIUS
            vy = 0.0
            on_ground = True

        screen.fill((18, 24, 38))
        pygame.draw.line(screen, (90, 110, 140), (0, GROUND), (WIDTH, GROUND), 2)
        pygame.draw.circle(screen, (255, 190, 80), (int(x), int(y)), BALL_RADIUS)

        hint = font.render(
            f"FPS {clock.get_fps():5.1f}    <-/-> move   Space jump   Esc quit",
            True,
            (200, 210, 225),
        )
        screen.blit(hint, (16, 14))

        pygame.display.flip()

        # 跳过启动第一秒，避免把初始化耗时算进帧率
        if elapsed > 1.0:
            fps_samples.append(clock.get_fps())

    pygame.quit()

    if fps_samples:
        avg_fps = sum(fps_samples) / len(fps_samples)
        min_fps = min(fps_samples)
        print(
            f"pygame-ce {pygame.version.ver} | window {WIDTH}x{HEIGHT} | "
            f"frames {frames} | {elapsed:.1f}s | "
            f"avg_fps {avg_fps:.1f} | min_fps {min_fps:.1f} | "
            f"moved {moved_px:.0f}px | jumps {jumps}"
        )
    else:
        print(f"pygame-ce {pygame.version.ver} | window closed before 1s")


if __name__ == "__main__":
    main()
