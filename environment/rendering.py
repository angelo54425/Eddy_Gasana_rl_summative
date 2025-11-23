import pygame
import math

GRAY = (200, 200, 200)
NEUTRAL_COLOR = (220, 220, 220)

ZONE_COLORS = {
    -1: NEUTRAL_COLOR,
    0: (80, 120, 240),
    1: (80, 200, 120),
    2: (255, 165, 0),
    3: (240, 100, 100)
}

AGENT_COLOR = (0, 0, 0)
SIDEBAR_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
ZONE_BORDER_COLOR = (50, 50, 50)

GOLD = (255, 215, 0)
PURPLE = (160, 0, 200)
FAIL_RED = (200, 0, 0)


def draw_star(surface, x, y, size, color):
    """Draw a 5-point star centered at (x, y)."""
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = size if i % 2 == 0 else size * 0.45
        px = x + radius * math.cos(angle)
        py = y - radius * math.sin(angle)
        points.append((px, py))
    pygame.draw.polygon(surface, color, points)


def draw_up_arrow(surface, rect, color):
    """Draw a bold upward arrow inside a cell rectangle."""
    x, y, w, h = rect
    tip = (x + w / 2, y + h * 0.15)
    left = (x + w * 0.25, y + h * 0.55)
    right = (x + w * 0.75, y + h * 0.55)
    body_top = (x + w * 0.45, y + h * 0.55)
    body_bottom = (x + w * 0.45, y + h * 0.85)
    body_bottom_r = (x + w * 0.55, y + h * 0.85)
    body_top_r = (x + w * 0.55, y + h * 0.55)

    pygame.draw.polygon(surface, color, [tip, left, right])
    pygame.draw.polygon(surface, color, [body_top, body_bottom, body_bottom_r, body_top_r])


def render_pygame(env, scale=50):
    pygame.init()

    grid_size = env.grid_size
    size = grid_size * scale

    screen = pygame.display.set_mode((size + 300, size))
    pygame.display.set_caption("Career Path Environment")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)

    def draw():
        screen.fill(GRAY)

        # ---- DRAW ZONES ----
        for x in range(grid_size):
            for y in range(grid_size):
                zone = env.zone_map[(x, y)]
                color = ZONE_COLORS.get(zone, NEUTRAL_COLOR)

                if env.field_id != -1 and zone != env.field_id:
                    color = tuple(max(0, c - 40) for c in color)

                rect = pygame.Rect(x * scale, y * scale, scale, scale)
                pygame.draw.rect(screen, color, rect)

                if env.field_id != -1 and zone == env.field_id:
                    pygame.draw.rect(screen, ZONE_BORDER_COLOR, rect, 2)
                else:
                    pygame.draw.rect(screen, (180, 180, 180), rect, 1)

        # ---- DRAW TRAINING ARROWS (PURPLE UP ARROWS) ----
        for field, cells in env.train_cells.items():
            for (tx, ty) in cells:
                cell_rect = (tx * scale, ty * scale, scale, scale)
                draw_up_arrow(screen, cell_rect, PURPLE)

        # ---- DRAW ANIMATED GOLD STAR ----
        if env.field_id in [0, 1, 2, 3] and env.status == "In Progress":
            sx, sy = env.star_cells[env.field_id]

            size_scale = env.star_animation_scale
            center_x = sx * scale + scale / 2
            center_y = sy * scale + scale / 2
            star_size = (scale * 0.45) * size_scale

            draw_star(screen, center_x, center_y, star_size, GOLD)

        # ---- DRAW FAILURE (Big Red X) ----
        if env.failed_star is not None:
            fx, fy = env.failed_star
            x = fx * scale
            y = fy * scale
            thickness = max(4, scale // 10)

            pygame.draw.line(screen, FAIL_RED, (x+5, y+5), (x+scale-5, y+scale-5), thickness)
            pygame.draw.line(screen, FAIL_RED, (x+scale-5, y+5), (x+5, y+scale-5), thickness)

        # ---- DRAW AGENT ----
        ax, ay = env.agent_pos
        rect = pygame.Rect(
            ax * scale + scale * 0.2,
            ay * scale + scale * 0.2,
            scale * 0.6,
            scale * 0.6
        )
        pygame.draw.ellipse(screen, AGENT_COLOR, rect)

        # ---- SIDEBAR ----
        sidebar_x = size
        pygame.draw.rect(screen, SIDEBAR_COLOR, (sidebar_x, 0, 300, size))

        def write(text, y):
            txt = font.render(text, True, TEXT_COLOR)
            screen.blit(txt, (sidebar_x + 10, y))

        field_names = {
            -1: "Neutral Zone",
             0: "Private Sector",
             1: "Medical Field",
             2: "Finance Sector",
             3: "Engineering"
        }

        write(f"Field: {field_names[env.field_id]}", 20)
        write(f"Skill: {env.skill_level:.2f}", 60)

        threshold = 0.0 if env.field_id == -1 else env.skill_thresholds[env.field_id]
        write(f"Threshold: {threshold:.2f}", 90)

        write(f"Steps Left: {env.steps_remaining}", 130)
        write(f"Status: {env.status}", 180)

        pygame.display.flip()

    return screen, draw, clock
