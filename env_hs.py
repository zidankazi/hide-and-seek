import numpy as np
import pymunk
from gymnasium import spaces
from pettingzoo import ParallelEnv


class HideAndSeekEnv(ParallelEnv):
    """
    Stage 5 hide-and-seek, 1v1. A hider starts inside a walled room with a single doorway
    and some movable boxes; a seeker starts outside. The paper-faithful goal (Stage 5b) is
    line-of-sight: the seeker is rewarded when it can SEE the hider, the hider when unseen,
    so the hider learns to barricade the doorway with boxes and lock them.

    THIS FILE IS STAGE 5a: the env skeleton. Room geometry, prep phase, movable boxes, and
    the full 21-dim observation / 3-dim action layout are in place. The lock action and the
    line-of-sight reward/masking are stubbed (lock is inert, visibility is always 1) and get
    wired up in 5b. Reward here is a TEMPORARY touch-tag (same as TagEnv) purely so we can
    smoke-test that agents move, push boxes, and respect the prep freeze before adding LOS.

    PettingZoo Parallel API: both agents act each step; obs/rewards/etc. come back as dicts
    keyed by "hider"/"seeker".
    """

    metadata = {"render_modes": ["human"], "name": "hide_and_seek_v0", "render_fps": 60}

    # --- arena / agent constants (shared with TagEnv) ---
    ARENA_SIZE = 600
    AGENT_RADIUS = 18
    FORCE_SCALE = 1500
    MAX_VEL = 300
    MAX_STEPS = 240
    TAG_DIST = 2 * AGENT_RADIUS + 2      # only used by the temporary 5a touch-tag reward

    # --- Stage 5 additions ---
    PREP_FRACTION = 0.4                  # first 40% of the episode: seeker frozen, no reward
    N_BOXES = 2
    BOX_SIZE = 44                        # full side length (matches renderer's box `size`)
    BOX_MASS = 3                         # pushable by an agent but heavy enough to stack a wall
    LOCK_DIST = AGENT_RADIUS + BOX_SIZE  # (5b) reach for locking the nearest box

    # --- fixed map: a room in the top-left corner with one doorway ---
    # Outer walls come from the arena edge; these interior segments close off the room,
    # leaving a 60px doorway in the bottom wall (x in [150, 210] at y=240).
    ROOM_WALLS = [
        [(240, 16), (240, 240)],        # right wall of room (solid)
        [(16, 240), (150, 240)],        # bottom wall, left of doorway
        [(210, 240), (240, 240)],       # bottom wall, right of doorway
    ]
    # Regions used for spawning (kept well clear of the walls).
    ROOM_LO, ROOM_HI = 40, 210          # hider spawns inside this square
    DOORWAY = ((150, 240), (210, 240))  # for reference / rendering debug

    @property
    def PREP_STEPS(self):
        return int(self.MAX_STEPS * self.PREP_FRACTION)

    @property
    def PLAY_STEPS(self):
        return self.MAX_STEPS - self.PREP_STEPS

    def __init__(self, render_mode=None):
        self.possible_agents = ["hider", "seeker"]
        self.agents = []
        self.episode = 0

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.5

        # --- agents ---
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, self.AGENT_RADIUS)
        self.bodies = {}
        self.shapes = {}
        for name in self.possible_agents:
            body = pymunk.Body(mass, moment)
            shape = pymunk.Circle(body, self.AGENT_RADIUS)
            shape.elasticity = 0.6
            shape.friction = 0.4
            self.bodies[name] = body
            self.shapes[name] = shape
            self.space.add(body, shape)

        # --- walls: arena edge + interior room walls ---
        edge = self.ARENA_SIZE - 10
        self.walls = [
            [(10, 10), (edge, 10)],
            [(edge, 10), (edge, edge)],
            [(edge, edge), (10, edge)],
            [(10, edge), (10, 10)],
        ] + [list(w) for w in self.ROOM_WALLS]
        for start, end in self.walls:
            seg = pymunk.Segment(self.space.static_body, start, end, 6)
            seg.elasticity = 0.4
            seg.friction = 0.5
            self.space.add(seg)

        # --- movable boxes ---
        # Created once and repositioned each reset (same pattern as the agents).
        # lock_owner tracks which team has locked a box: None / "hider" / "seeker" (used in 5b).
        self.box_bodies = []
        self.box_shapes = []
        self.box_lock_owner = [None] * self.N_BOXES
        box_moment = pymunk.moment_for_box(self.BOX_MASS, (self.BOX_SIZE, self.BOX_SIZE))
        for _ in range(self.N_BOXES):
            body = pymunk.Body(self.BOX_MASS, box_moment)
            shape = pymunk.Poly.create_box(body, (self.BOX_SIZE, self.BOX_SIZE))
            shape.elasticity = 0.1
            shape.friction = 0.7
            self.box_bodies.append(body)
            self.box_shapes.append(shape)
            self.space.add(body, shape)

        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            from renderer import GameRenderer
            self.renderer = GameRenderer(title="Hide & Seek")
        self.steps = 0

    # obs: self(4) + opp(4+visible) + N_BOXES*(4+lock+visible)
    def _obs_dim(self):
        return 4 + 5 + self.N_BOXES * 6

    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim(),), dtype=np.float32)

    def action_space(self, agent):
        # [fx, fy, lock]. fx,fy are thrust in [-1,1]; lock>0.5 toggles a box lock (inert in 5a).
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _visible(self, agent, target_body):
        """
        Line-of-sight from `agent` to a target body. STUB for 5a: always visible.
        5b replaces this with a raycast (space.segment_query_first) that returns False when a
        wall or box occludes the straight line between them.
        """
        return True

    def _get_obs(self, agent):
        half = self.ARENA_SIZE / 2
        mv = self.MAX_VEL
        opponent = "hider" if agent == "seeker" else "seeker"
        sb = self.bodies[agent]
        ob = self.bodies[opponent]

        parts = [
            (sb.position.x - half) / half,
            (sb.position.y - half) / half,
            sb.velocity.x / mv,
            sb.velocity.y / mv,
        ]

        # Opponent block, masked by line-of-sight. When not visible, zero pos/vel + flag 0.
        opp_visible = self._visible(agent, ob)
        if opp_visible:
            parts += [
                (ob.position.x - half) / half,
                (ob.position.y - half) / half,
                ob.velocity.x / mv,
                ob.velocity.y / mv,
                1.0,
            ]
        else:
            parts += [0.0, 0.0, 0.0, 0.0, 0.0]

        # Boxes: pos/vel + lock_state (+1 self / -1 other / 0 none) + visible flag.
        for i, body in enumerate(self.box_bodies):
            visible = self._visible(agent, body)
            owner = self.box_lock_owner[i]
            lock_state = 0.0 if owner is None else (1.0 if owner == agent else -1.0)
            if visible:
                parts += [
                    (body.position.x - half) / half,
                    (body.position.y - half) / half,
                    body.velocity.x / mv,
                    body.velocity.y / mv,
                    lock_state,
                    1.0,
                ]
            else:
                parts += [0.0, 0.0, 0.0, 0.0, lock_state, 0.0]

        return np.clip(np.array(parts, dtype=np.float32), -1.0, 1.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.episode += 1
        self.box_lock_owner = [None] * self.N_BOXES

        # Hider inside the room.
        hx = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
        hy = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
        self.bodies["hider"].position = (hx, hy)
        self.bodies["hider"].velocity = (0, 0)

        # Seeker outside the room (clearly past the room's bottom-right extent).
        margin = self.AGENT_RADIUS + 20
        hi = self.ARENA_SIZE - margin
        while True:
            sx = self.np_random.uniform(300, hi)
            sy = self.np_random.uniform(300, hi)
            if sx > 270 or sy > 270:  # outside the 240x240 room
                break
        self.bodies["seeker"].position = (sx, sy)
        self.bodies["seeker"].velocity = (0, 0)

        # Boxes spread inside the room, near the hider, non-overlapping-ish.
        for i, body in enumerate(self.box_bodies):
            bx = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
            by = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
            body.position = (bx, by)
            body.velocity = (0, 0)
            body.angular_velocity = 0.0
            body.angle = 0.0
        self.space.reindex_shapes_for_body(self.space.static_body)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        in_prep = self.steps < self.PREP_STEPS

        for name in self.agents:
            action = actions[name]
            # Prep phase: the seeker is frozen (no force, velocity pinned to zero) so the
            # hider gets a head start to set up. The hider moves freely the whole time.
            if in_prep and name == "seeker":
                self.bodies[name].velocity = (0, 0)
                continue
            fx = float(action[0]) * self.FORCE_SCALE
            fy = float(action[1]) * self.FORCE_SCALE
            self.bodies[name].apply_force_at_local_point((fx, fy))
            # action[2] is the lock signal; inert in 5a, handled in 5b.

        self.space.step(1 / 60)
        self.steps += 1

        # Keep the seeker pinned during prep even after the physics step (contacts could nudge it).
        if in_prep:
            self.bodies["seeker"].velocity = (0, 0)

        # Clamp agent velocities (boxes keep their own physics).
        for name in self.agents:
            body = self.bodies[name]
            vx, vy = body.velocity
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > self.MAX_VEL:
                scale = self.MAX_VEL / speed
                body.velocity = (vx * scale, vy * scale)

        # --- TEMPORARY 5a reward: touch-tag, active only in the play phase. ---
        # 5b replaces this whole block with the line-of-sight visibility reward.
        hp = self.bodies["hider"].position
        sp = self.bodies["seeker"].position
        dist = ((hp.x - sp.x) ** 2 + (hp.y - sp.y) ** 2) ** 0.5
        tagged = (not in_prep) and dist < self.TAG_DIST

        if in_prep:
            rewards = {"hider": 0.0, "seeker": 0.0}
        else:
            rewards = {"hider": 0.01, "seeker": -0.01}
            if tagged:
                rewards["hider"] -= 1.0
                rewards["seeker"] += 1.0

        time_up = self.steps >= self.MAX_STEPS
        terminations = {a: tagged for a in self.agents}
        truncations = {a: time_up for a in self.agents}

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if tagged or time_up:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self.renderer is None:
            return
        agents = []
        for name in self.possible_agents:
            body = self.bodies[name]
            agents.append({
                "pos": (body.position.x, body.position.y),
                "vel": (body.velocity.x, body.velocity.y),
                "role": name,
                "radius": self.AGENT_RADIUS,
            })
        boxes = []
        for i, body in enumerate(self.box_bodies):
            boxes.append({
                "pos": (body.position.x, body.position.y),
                "size": self.BOX_SIZE,
                "locked": self.box_lock_owner[i] is not None,
            })
        in_prep = self.steps < self.PREP_STEPS
        info = {
            "phase": "PREP" if in_prep else "PLAY",
            "episode": self.episode,
            "step": self.steps,
            "max_steps": self.MAX_STEPS,
            "prep_fraction": self.PREP_FRACTION,
            "hiders": 1,
            "seekers": 1,
            "reward": 0.0,
        }
        self.renderer.render(agents=agents, walls=self.walls, boxes=boxes, goal_pos=None, info=info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
