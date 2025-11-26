# FILE: environment/rendering.py
import pygame
import math

GRAY = (200, 200, 200)
NEUTRAL_COLOR = (240, 240, 240)

ZONE_COLORS = {
    0: NEUTRAL_COLOR,
    1: (80, 120, 240),   # private (blue)
    2: (80, 200, 120),   # medical (green)
    3: (255, 165, 0),    # finance (orange)
    4: (240, 100, 100)   # engineering (red)
}

AGENT_COLOR = (0, 0, 0)
SIDEBAR_COLOR = (255, 255, 255)
TEXT_COLOR = (10, 10, 10)
ZONE_BORDER_COLOR = (30, 30, 30)

GOLD = (255, 215, 0)
PURPLE = (160, 0, 200)
FAIL_RED = (200, 20, 20)

def draw_star(surface, cx, cy, radius, color):
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        px = cx + r * math.cos(angle)
        py = cy - r * math.sin(angle)
        pts.append((px, py))
    pygame.draw.polygon(surface, color, pts)

def draw_up_arrow(surface, rect, color):
    x, y, w, h = rect
    tip = (x + w * 0.5, y + h * 0.13)
    left = (x + w * 0.26, y + h * 0.56)
    right = (x + w * 0.74, y + h * 0.56)
    body_top = (x + w * 0.44, y + h * 0.56)
    body_bottom = (x + w * 0.44, y + h * 0.86)
    body_bottom_r = (x + w * 0.56, y + h * 0.86)
    body_top_r = (x + w * 0.56, y + h * 0.56)
    pygame.draw.polygon(surface, color, [tip, left, right])
    pygame.draw.polygon(surface, color, [body_top, body_bottom, body_bottom_r, body_top_r])

def draw_big_x(surface, x, y, w, h, color, thickness=6):
    pygame.draw.line(surface, color, (x + 5, y + 5), (x + w - 5, y + h - 5), thickness)
    pygame.draw.line(surface, color, (x + w - 5, y + 5), (x + 5, y + h - 5), thickness)

def render_pygame(env, scale=56):
    pygame.init()
    grid_size = env.grid_size
    size = grid_size * scale
    screen = pygame.display.set_mode((size + 320, size))
    pygame.display.set_caption("Career Path Environment")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    bold = pygame.font.SysFont('Arial', 18, bold=True)

    def draw():
        screen.fill(GRAY)

        # determine agent zone (only highlight the zone agent currently is in)
        agent_zone = env._get_zone(int(env.agent_pos[0]), int(env.agent_pos[1]))

        # draw grid cells
        for gx in range(grid_size):
            for gy in range(grid_size):
                zone = env._get_zone(gx, gy)
                color = ZONE_COLORS.get(zone, NEUTRAL_COLOR)
                # dim other non-neutral zones
                if zone != agent_zone and zone != 0:
                    color = tuple(max(0, c - 40) for c in color)
                rect = pygame.Rect(gx * scale, gy * scale, scale, scale)
                pygame.draw.rect(screen, color, rect)
                # border
                if zone == agent_zone:
                    pygame.draw.rect(screen, ZONE_BORDER_COLOR, rect, 2)
                else:
                    pygame.draw.rect(screen, (180, 180, 180), rect, 1)

        # draw training arrows
        for zid, cells in env.training_tiles.items():
            for tx, ty in cells:
                draw_up_arrow(screen, (tx * scale, ty * scale, scale, scale), PURPLE)

        # draw stars for each goal cell (large gold star)
        for zid, (gx, gy) in env.goal_tiles.items():
            cx = gx * scale + scale / 2
            cy = gy * scale + scale / 2
            star_radius = scale * 0.42

            failed = False
            last_info = getattr(env, "last_info", {}) or {}
            if isinstance(last_info, dict):
                if last_info.get("result") == "failure" and last_info.get("pos") == (gx, gy):
                    failed = True

            # if agent is on star but insufficient skill, show X
            if (int(env.agent_pos[0]) == gx and int(env.agent_pos[1]) == gy
                    and env.skill_level < env.skill_thresholds.get(zid, 1.0)):
                failed = True

            if failed:
                draw_big_x(screen, gx * scale, gy * scale, scale, scale, FAIL_RED, thickness=max(4, scale // 10))
            else:
                draw_star(screen, cx, cy, star_radius, GOLD)

        # draw agent
        ax, ay = env.agent_pos
        agent_rect = pygame.Rect(int(ax) * scale + scale * 0.22, int(ay) * scale + scale * 0.22,
                                 int(scale * 0.56), int(scale * 0.56))
        pygame.draw.ellipse(screen, AGENT_COLOR, agent_rect)

        # sidebar
        sidebar_x = size
        pygame.draw.rect(screen, (255, 255, 255), (sidebar_x, 0, 320, size))
        screen.blit(bold.render("Career Path Environment", True, TEXT_COLOR), (sidebar_x + 10, 8))

        field_names = {0: "Neutral Zone", 1: "Private", 2: "Medical", 3: "Finance", 4: "Engineering"}
        screen.blit(font.render(f"Field: {field_names.get(env._get_zone(int(env.agent_pos[0]), int(env.agent_pos[1])), 'Unknown')}", True, TEXT_COLOR), (sidebar_x + 10, 48))
        screen.blit(font.render(f"Skill: {env.skill_level:.2f}", True, TEXT_COLOR), (sidebar_x + 10, 88))

        thr = env.skill_thresholds.get(env.target_zone, 0.0) if hasattr(env, "skill_thresholds") else 0.0
        screen.blit(font.render(f"Threshold: {thr:.2f}", True, TEXT_COLOR), (sidebar_x + 10, 118))

        steps_left = max(0, env.steps - env.current_step) if hasattr(env, "current_step") else 0
        screen.blit(font.render(f"Steps Left: {steps_left}", True, TEXT_COLOR), (sidebar_x + 10, 158))

        status = "In Progress"
        last_info = getattr(env, "last_info", {}) or {}
        if isinstance(last_info, dict):
            if last_info.get("result") == "success":
                status = "Success"
            elif last_info.get("result") == "failure":
                status = "Failure"
        screen.blit(font.render(f"Status: {status}", True, TEXT_COLOR), (sidebar_x + 10, 198))

        # legend
        y0 = 240
        legend = {1: "Private", 2: "Medical", 3: "Finance", 4: "Engineering"}
        for zid in [1,2,3,4]:
            c = ZONE_COLORS[zid]
            pygame.draw.rect(screen, c, (sidebar_x + 10, y0, 18, 18))
            screen.blit(font.render(legend[zid], True, TEXT_COLOR), (sidebar_x + 35, y0))
            y0 += 26

        pygame.display.flip()

    return screen, draw, clock
