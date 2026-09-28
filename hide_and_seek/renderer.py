import math
import random
import pygame


COLORS = {
    "bg_outer": (188, 190, 200),
    "floor_a": (232, 232, 237),
    "floor_b": (226, 226, 231),
    "grid": (218, 218, 225),
    "wall_top": (185, 190, 200),
    "wall_front": (150, 155, 168),
    "wall_right": (165, 170, 180),
    "wall_outline": (135, 140, 152),
    "shadow": (0, 0, 0),
    "hider": (91, 192, 235),
    "hider_light": (140, 215, 248),
    "hider_dark": (55, 145, 195),
    "hider_base": (65, 160, 210),
    "seeker": (247, 108, 94),
    "seeker_light": (255, 150, 135),
    "seeker_dark": (200, 70, 60),
    "seeker_base": (220, 85, 72),
    "eye_white": (255, 255, 255),
    "eye_pupil": (25, 25, 35),
    "box_top": (245, 190, 72),
    "box_front": (215, 160, 42),
    "box_right": (225, 170, 52),
    "box_outline": (190, 140, 30),
    "box_locked_top": (210, 165, 55),
    "box_locked_front": (180, 138, 35),
    "ramp_top": (140, 200, 170),
    "ramp_front": (105, 165, 135),
    "ramp_locked_top": (110, 170, 145),
    "ramp_locked_front": (85, 140, 115),
    "ramp_outline": (75, 130, 105),
    "lock_body": (240, 240, 245),
    "lock_outline": (120, 100, 50),
    "goal": (100, 220, 140),
    "goal_ring": (60, 190, 110),
    "ui_bg": (248, 248, 251),
    "ui_border": (215, 215, 222),
    "ui_text": (120, 125, 140),
    "ui_value": (40, 42, 55),
    "prep_color": (91, 192, 235),
    "play_color": (247, 108, 94),
    "timeline_bg": (215, 215, 222),
    "particle": (180, 175, 165),
}


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "radius", "color")

    def __init__(self, x, y, vx, vy, life=20, radius=3, color=None):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.radius = radius
        self.color = color or COLORS["particle"]

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.88
        self.vy *= 0.88
        self.life -= 1

    @property
    def alive(self):
        return self.life > 0

    @property
    def alpha(self):
        return max(0, int(80 * (self.life / self.max_life)))

    @property
    def current_radius(self):
        return max(1, int(self.radius * (self.life / self.max_life)))


class GameRenderer:
    def __init__(self, width=700, height=700, hud_height=80, title="Hide & Seek"):
        pygame.init()
        self.width = width
        self.height = height
        self.arena_size = min(width, height) - 80
        self.arena_x = (width - self.arena_size) // 2
        self.arena_y = 30
        self.hud_height = hud_height
        self.total_height = height + hud_height
        self.screen = pygame.display.set_mode((self.width, self.total_height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("Helvetica", 12)
        self.font_md = pygame.font.SysFont("Helvetica", 15, bold=True)
        self.font_phase = pygame.font.SysFont("Helvetica", 12, bold=True)
        self.font_label = pygame.font.SysFont("Helvetica", 10)

        self.frame = 0
        self.particles = []
        self._prev_positions = {}

        self._floor_cache = None
        self._vignette_cache = None

    # --- coordinate helpers ---

    def _to_screen(self, pos, world_size=600):
        scale = self.arena_size / world_size
        return (self.arena_x + pos[0] * scale, self.arena_y + pos[1] * scale)

    def _scale(self, val, world_size=600):
        return val * (self.arena_size / world_size)

    # --- floor ---

    def _build_floor(self):
        surf = pygame.Surface((self.arena_size, self.arena_size))
        tile_count = 15
        tile_size = self.arena_size / tile_count
        for row in range(tile_count):
            for col in range(tile_count):
                color = COLORS["floor_a"] if (row + col) % 2 == 0 else COLORS["floor_b"]
                rect = pygame.Rect(int(col * tile_size), int(row * tile_size),
                                   int(tile_size) + 1, int(tile_size) + 1)
                pygame.draw.rect(surf, color, rect)

        for i in range(tile_count + 1):
            p = int(i * tile_size)
            pygame.draw.line(surf, COLORS["grid"], (p, 0), (p, self.arena_size), 1)
            pygame.draw.line(surf, COLORS["grid"], (0, p), (self.arena_size, p), 1)
        self._floor_cache = surf

    def _draw_floor(self, surface):
        if self._floor_cache is None:
            self._build_floor()
        surface.blit(self._floor_cache, (self.arena_x, self.arena_y))

    # --- vignette ---

    def _build_vignette(self):
        surf = pygame.Surface((self.arena_size, self.arena_size), pygame.SRCALPHA)
        for i in range(40):
            alpha = int(18 * (1 - i / 40))
            if alpha <= 0:
                continue
            rect = pygame.Rect(i, i, self.arena_size - i * 2, self.arena_size - i * 2)
            if rect.width <= 0:
                break
            pygame.draw.rect(surf, (0, 0, 0, alpha), rect, 1)
        self._vignette_cache = surf

    def _draw_vignette(self, surface):
        if self._vignette_cache is None:
            self._build_vignette()
        surface.blit(self._vignette_cache, (self.arena_x, self.arena_y))

    # --- shadows ---

    def _draw_shadow_ellipse(self, surface, center, rx, ry, alpha=22):
        s = pygame.Surface((int(rx * 2 + 4), int(ry * 2 + 4)), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (0, 0, 0, alpha), (2, 2, int(rx * 2), int(ry * 2)))
        surface.blit(s, (int(center[0] - rx - 2), int(center[1] - ry + 2)))

    # --- walls ---

    def _draw_walls(self, surface, walls):
        wall_w = self._scale(7)
        depth = 8

        for start, end in walls:
            s = self._to_screen(start)
            e = self._to_screen(end)
            dx, dy = e[0] - s[0], e[1] - s[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1:
                continue
            nx = -dy / length * (wall_w / 2)
            ny = dx / length * (wall_w / 2)

            top = [
                (s[0] + nx, s[1] + ny),
                (e[0] + nx, e[1] + ny),
                (e[0] - nx, e[1] - ny),
                (s[0] - nx, s[1] - ny),
            ]

            shadow = [(p[0] + 4, p[1] + 6) for p in top]
            ss = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(ss, (0, 0, 0, 18), shadow)
            surface.blit(ss, (0, 0))

            front = [
                (s[0] - nx, s[1] - ny),
                (e[0] - nx, e[1] - ny),
                (e[0] - nx, e[1] - ny + depth),
                (s[0] - nx, s[1] - ny + depth),
            ]
            pygame.draw.polygon(surface, COLORS["wall_front"], front)
            pygame.draw.polygon(surface, COLORS["wall_outline"], front, 1)

            right_side = [
                (e[0] + nx, e[1] + ny),
                (e[0] - nx, e[1] - ny),
                (e[0] - nx, e[1] - ny + depth),
                (e[0] + nx, e[1] + ny + depth),
            ]
            pygame.draw.polygon(surface, COLORS["wall_right"], right_side)

            pygame.draw.polygon(surface, COLORS["wall_top"], top)
            pygame.draw.polygon(surface, COLORS["wall_outline"], top, 1)

    # --- goal ---

    def _draw_goal(self, surface, pos, radius=12):
        if pos is None:
            return
        sp = self._to_screen(pos)
        sr = self._scale(radius)
        pulse = math.sin(self.frame * 0.05) * 0.15 + 1.0

        glow = pygame.Surface((int(sr * 8), int(sr * 8)), pygame.SRCALPHA)
        c = int(sr * 4)
        for i in range(5):
            r = int(sr * (2 + i * 0.6) * pulse)
            a = 18 - i * 3
            if a > 0 and r > 0:
                pygame.draw.circle(glow, (*COLORS["goal_ring"], a), (c, c), r, 2)
        surface.blit(glow, (int(sp[0] - sr * 4), int(sp[1] - sr * 4)))

        pygame.draw.circle(surface, COLORS["goal"], (int(sp[0]), int(sp[1])), int(sr * pulse), 3)
        pygame.draw.circle(surface, COLORS["goal"], (int(sp[0]), int(sp[1])), max(2, int(sr * 0.35)), 0)

    # --- boxes ---

    def _draw_box(self, surface, pos, size=30, locked=False):
        sp = self._to_screen(pos)
        ss = self._scale(size)
        half = ss / 2
        depth = 7

        self._draw_shadow_ellipse(surface, (sp[0], sp[1] + half + 2), half * 1.1, half * 0.4, alpha=20)

        top_c = COLORS["box_locked_top"] if locked else COLORS["box_top"]
        front_c = COLORS["box_locked_front"] if locked else COLORS["box_front"]
        right_c = COLORS["box_right"]
        outline_c = COLORS["box_outline"]

        front = [
            (sp[0] - half, sp[1] + half),
            (sp[0] + half, sp[1] + half),
            (sp[0] + half, sp[1] + half + depth),
            (sp[0] - half, sp[1] + half + depth),
        ]
        pygame.draw.polygon(surface, front_c, front)

        right = [
            (sp[0] + half, sp[1] - half),
            (sp[0] + half, sp[1] + half),
            (sp[0] + half, sp[1] + half + depth),
            (sp[0] + half, sp[1] - half + depth),
        ]
        pygame.draw.polygon(surface, right_c, right)

        top = [
            (sp[0] - half, sp[1] - half),
            (sp[0] + half, sp[1] - half),
            (sp[0] + half, sp[1] + half),
            (sp[0] - half, sp[1] + half),
        ]
        pygame.draw.polygon(surface, top_c, top)
        pygame.draw.polygon(surface, outline_c, top, 1)
        pygame.draw.polygon(surface, outline_c, front, 1)
        pygame.draw.polygon(surface, outline_c, right, 1)

        hl = pygame.Surface((int(ss), int(ss)), pygame.SRCALPHA)
        pygame.draw.rect(hl, (255, 255, 255, 40), (2, 2, int(ss * 0.3), int(ss * 0.5)), border_radius=2)
        surface.blit(hl, (int(sp[0] - half), int(sp[1] - half)))

        if locked:
            self._draw_lock_glyph(surface, int(sp[0]), int(sp[1] - 1), max(4, int(ss * 0.18)))

    def _draw_lock_glyph(self, surface, lx, ly, ls):
        shackle_rect = (lx - ls + 2, ly - ls - 3, (ls - 2) * 2, ls + 2)
        pygame.draw.arc(surface, COLORS["lock_outline"], shackle_rect, 0, math.pi, 2)

        body = pygame.Rect(lx - ls, ly - 2, ls * 2, int(ls * 1.4))
        pygame.draw.rect(surface, COLORS["lock_body"], body, border_radius=2)
        pygame.draw.rect(surface, COLORS["lock_outline"], body, 1, border_radius=2)

        dot_r = max(1, ls // 3)
        pygame.draw.circle(surface, COLORS["lock_outline"], (lx, ly + int(ls * 0.3)), dot_r)

    # --- ramp (Stage 7): a wedge, drawn box-like so it sits in the same visual language ---

    def _draw_ramp(self, surface, pos, size=40, locked=False):
        sp = self._to_screen(pos)
        ss = self._scale(size)
        half = ss / 2
        depth = 7

        self._draw_shadow_ellipse(surface, (sp[0], sp[1] + half + 2), half * 1.1, half * 0.4, alpha=20)

        top_c = COLORS["ramp_locked_top"] if locked else COLORS["ramp_top"]
        front_c = COLORS["ramp_locked_front"] if locked else COLORS["ramp_front"]
        outline_c = COLORS["ramp_outline"]

        front = [
            (sp[0] - half, sp[1] + half),
            (sp[0] + half, sp[1] + half),
            (sp[0] + half, sp[1] + half + depth),
            (sp[0] - half, sp[1] + half + depth),
        ]
        pygame.draw.polygon(surface, front_c, front)

        # Wedge: low at the left edge, rising to a high right face.
        wedge = [
            (sp[0] - half, sp[1] + half),
            (sp[0] + half, sp[1] - half),
            (sp[0] + half, sp[1] + half),
        ]
        pygame.draw.polygon(surface, top_c, wedge)
        pygame.draw.polygon(surface, outline_c, wedge, 1)
        pygame.draw.polygon(surface, outline_c, front, 1)

        # Tread lines up the slope, for readability.
        for f in (0.3, 0.55, 0.8):
            x = sp[0] - half + ss * f
            y_top = sp[1] + half - ss * f
            pygame.draw.line(surface, outline_c, (x, sp[1] + half), (x, y_top), 1)

        if locked:
            self._draw_lock_glyph(surface, int(sp[0] + half * 0.35), int(sp[1] + half * 0.3),
                                  max(4, int(ss * 0.18)))

    # --- agents ---

    def _draw_agent(self, surface, pos, radius=18, role="hider", facing=(0, 0), agent_id=0):
        sp = self._to_screen(pos)
        sr = self._scale(radius)

        if role == "hider":
            color, light, dark, base_c = COLORS["hider"], COLORS["hider_light"], COLORS["hider_dark"], COLORS["hider_base"]
        else:
            color, light, dark, base_c = COLORS["seeker"], COLORS["seeker_light"], COLORS["seeker_dark"], COLORS["seeker_base"]

        bob = math.sin(self.frame * 0.08 + agent_id * 2.0) * 2
        cy = sp[1] + bob

        self._draw_shadow_ellipse(surface, (sp[0], sp[1] + sr + 2), sr * 0.9, sr * 0.35, alpha=25)

        base_h = 5
        base_w = sr * 1.1
        for i in range(3):
            ring_y = sp[1] + sr - i * (base_h // 3) + 2
            ring_r = base_w - i * 2
            if ring_r > 0:
                alpha = 180 - i * 40
                rs = pygame.Surface((int(ring_r * 2 + 4), int(base_h + 4)), pygame.SRCALPHA)
                pygame.draw.ellipse(rs, (*base_c, alpha), (2, 2, int(ring_r * 2), base_h))
                surface.blit(rs, (int(sp[0] - ring_r - 2), int(ring_y - base_h // 2)))

        pygame.draw.circle(surface, dark, (int(sp[0]), int(cy) + 2), int(sr))
        pygame.draw.circle(surface, color, (int(sp[0]), int(cy)), int(sr))

        hl = pygame.Surface((int(sr * 3), int(sr * 3)), pygame.SRCALPHA)
        hl_r = int(sr * 0.6)
        pygame.draw.circle(hl, (255, 255, 255, 55), (int(sr * 1.1), int(sr * 0.9)), hl_r)
        surface.blit(hl, (int(sp[0] - sr * 1.5), int(cy - sr * 1.5)))

        rim = pygame.Surface((int(sr * 2 + 4), int(sr * 2 + 4)), pygame.SRCALPHA)
        pygame.draw.circle(rim, (*light, 60), (int(sr + 2), int(sr + 2)), int(sr), 2)
        surface.blit(rim, (int(sp[0] - sr - 2), int(cy - sr - 2)))

        speed = math.sqrt(facing[0] ** 2 + facing[1] ** 2)
        if speed > 1:
            lx, ly = facing[0] / speed, facing[1] / speed
        else:
            lx, ly = 0, -0.3

        eye_spread = sr * 0.38
        eye_fwd = sr * 0.15
        eye_r = sr * 0.26
        pupil_r = sr * 0.14

        for side in [-1, 1]:
            px_dir = -ly * side
            py_dir = lx * side
            ex = sp[0] + lx * eye_fwd + px_dir * eye_spread
            ey = cy + ly * eye_fwd + py_dir * eye_spread

            pygame.draw.circle(surface, COLORS["eye_white"], (int(ex), int(ey)), int(eye_r))

            look_offset = pupil_r * 0.5
            ppx = ex + lx * look_offset
            ppy = ey + ly * look_offset
            pygame.draw.circle(surface, COLORS["eye_pupil"], (int(ppx), int(ppy)), int(pupil_r))

            gr = max(1, int(pupil_r * 0.35))
            pygame.draw.circle(surface, (255, 255, 255),
                               (int(ppx - pupil_r * 0.3), int(ppy - pupil_r * 0.35)), gr)

        speed_key = f"agent_{agent_id}"
        prev = self._prev_positions.get(speed_key, pos)
        dx = pos[0] - prev[0]
        dy = pos[1] - prev[1]
        move_speed = math.sqrt(dx * dx + dy * dy)
        self._prev_positions[speed_key] = pos

        if move_speed > 1.5 and self.frame % 3 == 0:
            for _ in range(2):
                angle = math.atan2(-dy, -dx) + random.uniform(-0.8, 0.8)
                spd = random.uniform(0.5, 2.0)
                self.particles.append(Particle(
                    sp[0] + random.uniform(-sr * 0.3, sr * 0.3),
                    sp[1] + sr * 0.5,
                    math.cos(angle) * spd,
                    math.sin(angle) * spd,
                    life=random.randint(12, 22),
                    radius=random.randint(2, 4),
                ))

    # --- particles ---

    def _update_and_draw_particles(self, surface):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

        for p in self.particles:
            ps = pygame.Surface((p.current_radius * 4, p.current_radius * 4), pygame.SRCALPHA)
            r = p.current_radius
            pygame.draw.circle(ps, (*p.color, p.alpha), (r * 2, r * 2), r)
            surface.blit(ps, (int(p.x - r * 2), int(p.y - r * 2)))

    # --- hud ---

    def _draw_hud(self, info):
        hy = self.height
        pygame.draw.rect(self.screen, COLORS["ui_bg"], (0, hy, self.width, self.hud_height))
        pygame.draw.line(self.screen, COLORS["ui_border"], (0, hy), (self.width, hy), 2)

        phase = info.get("phase", "PLAY")
        pc = COLORS["prep_color"] if phase == "PREP" else COLORS["play_color"]
        pill = pygame.Rect(20, hy + 12, 65, 22)
        pygame.draw.rect(self.screen, pc, pill, border_radius=11)
        pt = self.font_phase.render(phase, True, (255, 255, 255))
        self.screen.blit(pt, pt.get_rect(center=pill.center))

        stats = [
            ("EPISODE", str(info.get("episode", 0))),
            ("STEP", f"{info.get('step', 0)}/{info.get('max_steps', 0)}"),
            ("HIDERS", str(info.get("hiders", 0))),
            ("SEEKERS", str(info.get("seekers", 0))),
        ]
        x = 110
        for label, val in stats:
            lt = self.font_label.render(label, True, COLORS["ui_text"])
            vt = self.font_md.render(val, True, COLORS["ui_value"])
            self.screen.blit(lt, (x, hy + 8))
            self.screen.blit(vt, (x, hy + 24))
            x += 105

        reward = info.get("reward", 0.0)
        rt = self.font_md.render(f"R: {reward:+.1f}", True, COLORS["ui_value"])
        self.screen.blit(rt, (self.width - 85, hy + 18))

        bar_x, bar_y = 20, hy + 54
        bar_w, bar_h = self.width - 40, 5
        pygame.draw.rect(self.screen, COLORS["timeline_bg"], (bar_x, bar_y, bar_w, bar_h), border_radius=2)

        prep_frac = info.get("prep_fraction", 0.25)
        prep_w = int(bar_w * prep_frac)
        if prep_w > 0:
            pygame.draw.rect(self.screen, (*COLORS["prep_color"], 50),
                             (bar_x, bar_y, prep_w, bar_h), border_radius=2)

        max_s = max(1, info.get("max_steps", 1))
        progress = min(1.0, info.get("step", 0) / max_s)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            fc = COLORS["prep_color"] if progress < prep_frac else COLORS["play_color"]
            pygame.draw.rect(self.screen, fc, (bar_x, bar_y, fill_w, bar_h), border_radius=2)

        dot_x = bar_x + fill_w
        pygame.draw.circle(self.screen, COLORS["play_color"], (int(dot_x), bar_y + bar_h // 2), 5)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(dot_x), bar_y + bar_h // 2), 3)

        prep_x = bar_x + prep_w
        pygame.draw.line(self.screen, (*COLORS["prep_color"], 150),
                         (int(prep_x), bar_y - 3), (int(prep_x), bar_y + bar_h + 3), 1)

    # --- main render ---

    def render(self, agents, walls, boxes=None, ramp=None, goal_pos=None, info=None):
        self.frame += 1
        if info is None:
            info = {}

        self.screen.fill(COLORS["bg_outer"])
        self._draw_floor(self.screen)
        self._draw_vignette(self.screen)
        self._draw_goal(self.screen, goal_pos)

        if boxes:
            sorted_boxes = sorted(boxes, key=lambda b: b["pos"][1])
            for box in sorted_boxes:
                self._draw_box(self.screen, box["pos"], box.get("size", 30), box.get("locked", False))

        if ramp:
            self._draw_ramp(self.screen, ramp["pos"], ramp.get("size", 40), ramp.get("locked", False))

        self._draw_walls(self.screen, walls)

        sorted_agents = sorted(enumerate(agents), key=lambda a: a[1]["pos"][1])
        for idx, agent in sorted_agents:
            self._draw_agent(
                self.screen,
                pos=agent["pos"],
                radius=agent.get("radius", 18),
                role=agent.get("role", "hider"),
                facing=agent.get("vel", (0, 0)),
                agent_id=idx,
            )

        self._update_and_draw_particles(self.screen)
        self._draw_hud(info)
        pygame.display.flip()
        self.clock.tick(60)

    def get_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                return None
        return events

    def get_keys(self):
        return pygame.key.get_pressed()

    def close(self):
        pygame.quit()
