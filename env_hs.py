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
    MIN_SPAWN_DIST = 250                  # min hider/seeker separation at spawn (open layout)

    # --- Stage 5 additions ---
    PREP_FRACTION = 0.4                  # first 40% of the episode: seeker frozen, no reward
    N_BOXES = 2
    BOX_SIZE = 44                        # full side length (matches renderer's box `size`)
    BOX_MASS = 3                         # pushable by an agent but heavy enough to stack a wall
    LOCK_DIST = AGENT_RADIUS + BOX_SIZE  # reach for locking the nearest box

    # Collision/query filter categories. LOS raycasts use a mask of OCCLUDER_CAT so they only
    # "see" walls and boxes and pass straight through agents (we don't want an agent's own body
    # blocking its view). Physics collisions are unaffected (masks stay all-ones).
    AGENT_CAT = 0b01
    OCCLUDER_CAT = 0b10

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

    def __init__(self, render_mode=None, layout="room"):
        """
        layout="room" (Stage 5b): corner room with a doorway, 2 boxes. The room hides the
            hider passively, so tool-use is optional.
        layout="open" (Stage 5b-ii): open arena, no interior walls, 4 boxes, hider spawns in a
            random corner. Nothing hides the hider for free — it must push+lock boxes to close
            a corner pocket, forcing genuine fort-building (closest to the paper).
        """
        assert layout in ("room", "open")
        self.layout = layout
        self.N_BOXES = 2 if layout == "room" else 4
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
            shape.filter = pymunk.ShapeFilter(categories=self.AGENT_CAT)
            self.bodies[name] = body
            self.shapes[name] = shape
            self.space.add(body, shape)

        # --- walls: arena edge (+ interior room walls only in the "room" layout) ---
        edge = self.ARENA_SIZE - 10
        self.walls = [
            [(10, 10), (edge, 10)],
            [(edge, 10), (edge, edge)],
            [(edge, edge), (10, edge)],
            [(10, edge), (10, 10)],
        ]
        if self.layout == "room":
            self.walls += [list(w) for w in self.ROOM_WALLS]
        for start, end in self.walls:
            seg = pymunk.Segment(self.space.static_body, start, end, 6)
            seg.elasticity = 0.4
            seg.friction = 0.5
            seg.filter = pymunk.ShapeFilter(categories=self.OCCLUDER_CAT)
            self.space.add(seg)

        # --- movable boxes ---
        # Created once and repositioned each reset (same pattern as the agents).
        # lock_owner tracks which team has locked a box: None / "hider" / "seeker" (used in 5b).
        self.box_bodies = []
        self.box_shapes = []
        self.box_lock_owner = [None] * self.N_BOXES
        self._box_moment = pymunk.moment_for_box(self.BOX_MASS, (self.BOX_SIZE, self.BOX_SIZE))
        for _ in range(self.N_BOXES):
            body = pymunk.Body(self.BOX_MASS, self._box_moment)
            shape = pymunk.Poly.create_box(body, (self.BOX_SIZE, self.BOX_SIZE))
            shape.elasticity = 0.1
            shape.friction = 0.7
            shape.filter = pymunk.ShapeFilter(categories=self.OCCLUDER_CAT)
            self.box_bodies.append(body)
            self.box_shapes.append(shape)
            self.space.add(body, shape)

        # Rising-edge tracking for the lock action: an agent only toggles a lock when its lock
        # signal crosses from <=0.5 to >0.5, so holding the signal high doesn't thrash lock/unlock.
        self._prev_lock = {name: False for name in self.possible_agents}

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

    def _visible(self, agent, target_body, target_shape=None):
        """
        Line-of-sight from `agent` to a target body: cast a ray between the two centers and
        report whether a wall or box occludes it. The ray uses OCCLUDER_CAT as its mask so it
        ignores both agents' bodies entirely (only walls/boxes can block).

        `target_shape` is the target's own shape when the target is itself an occluder (a box):
        the ray ends at the box center and would otherwise register a hit on the box itself, so
        a hit *on the target* counts as visible. For the opponent (not an occluder) pass None.
        """
        start = self.bodies[agent].position
        end = target_body.position
        query_filter = pymunk.ShapeFilter(mask=self.OCCLUDER_CAT)
        hit = self.space.segment_query_first(start, end, 1, query_filter)
        if hit is None:
            return True
        return hit.shape is target_shape

    def _nearest_box(self, agent):
        """Index of the box whose center is nearest the agent, and that distance."""
        ap = self.bodies[agent].position
        best_i, best_d = 0, float("inf")
        for i, body in enumerate(self.box_bodies):
            d = ((body.position.x - ap.x) ** 2 + (body.position.y - ap.y) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        return best_i, best_d

    def _set_box_dynamic(self, i):
        body = self.box_bodies[i]
        body.body_type = pymunk.Body.DYNAMIC
        body.mass = self.BOX_MASS
        body.moment = self._box_moment
        self.space.reindex_shapes_for_body(body)

    def _set_box_static(self, i):
        body = self.box_bodies[i]
        body.velocity = (0, 0)
        body.angular_velocity = 0.0
        body.body_type = pymunk.Body.STATIC
        self.space.reindex_shapes_for_body(body)

    def _handle_lock(self, agent, lock_signal):
        """
        Rising-edge lock toggle. On the frame the signal crosses >0.5, toggle the nearest box
        within LOCK_DIST: unlocked -> lock to self; locked-by-self -> unlock; locked-by-other ->
        no-op (only the team that locked a box can unlock it).
        """
        pressed = lock_signal > 0.5
        rising = pressed and not self._prev_lock[agent]
        self._prev_lock[agent] = pressed
        if not rising:
            return
        i, d = self._nearest_box(agent)
        if d > self.LOCK_DIST:
            return
        owner = self.box_lock_owner[i]
        if owner is None:
            self._set_box_static(i)
            self.box_lock_owner[i] = agent
        elif owner == agent:
            self._set_box_dynamic(i)
            self.box_lock_owner[i] = None
        # owner == other: no-op

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
            visible = self._visible(agent, body, self.box_shapes[i])
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
        # Release any boxes left locked (STATIC) from the previous episode, then clear state.
        for i in range(self.N_BOXES):
            if self.box_lock_owner[i] is not None:
                self._set_box_dynamic(i)
        self.box_lock_owner = [None] * self.N_BOXES
        self._prev_lock = {name: False for name in self.possible_agents}

        margin = self.AGENT_RADIUS + 20
        lo, hi = margin, self.ARENA_SIZE - margin

        if self.layout == "room":
            # Hider inside the room; seeker outside; boxes inside near the hider.
            hx = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
            hy = self.np_random.uniform(self.ROOM_LO, self.ROOM_HI)
            while True:
                sx = self.np_random.uniform(300, hi)
                sy = self.np_random.uniform(300, hi)
                if sx > 270 or sy > 270:  # outside the 240x240 room
                    break
            box_spawn = lambda: (self.np_random.uniform(self.ROOM_LO, self.ROOM_HI),
                                 self.np_random.uniform(self.ROOM_LO, self.ROOM_HI))
        else:
            # Open arena. Hider spawns in a random corner pocket (near two arena walls it can
            # complete into cover); boxes scatter mid-arena; seeker spawns anywhere far off.
            corner = self.np_random.integers(4)
            cx = lo if corner in (0, 2) else hi
            cy = lo if corner in (0, 1) else hi
            hx = cx + (1 if cx == lo else -1) * self.np_random.uniform(0, 90)
            hy = cy + (1 if cy == lo else -1) * self.np_random.uniform(0, 90)
            while True:
                sx = self.np_random.uniform(lo, hi)
                sy = self.np_random.uniform(lo, hi)
                if ((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5 >= self.MIN_SPAWN_DIST:
                    break
            mid_lo, mid_hi = self.ARENA_SIZE * 0.25, self.ARENA_SIZE * 0.75
            box_spawn = lambda: (self.np_random.uniform(mid_lo, mid_hi),
                                 self.np_random.uniform(mid_lo, mid_hi))

        self.bodies["hider"].position = (hx, hy)
        self.bodies["hider"].velocity = (0, 0)
        self.bodies["seeker"].position = (sx, sy)
        self.bodies["seeker"].velocity = (0, 0)

        for i, body in enumerate(self.box_bodies):
            bx, by = box_spawn()
            body.position = (bx, by)
            body.velocity = (0, 0)
            body.angular_velocity = 0.0
            body.angle = 0.0
            # Boxes were teleported directly (not via physics), so their broadphase index
            # entries are stale. Reindex each so the very first LOS raycast in this episode's
            # reset obs sees them at their new positions (step() reindexes via space.step()).
            self.space.reindex_shapes_for_body(body)
        self.space.reindex_static()

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        in_prep = self.steps < self.PREP_STEPS

        for name in self.agents:
            action = actions[name]
            # Prep phase: the seeker is frozen (no force, velocity pinned to zero) so the
            # hider gets a head start to set up. The hider moves and locks freely the whole time.
            if in_prep and name == "seeker":
                self.bodies[name].velocity = (0, 0)
                self._prev_lock[name] = action[2] > 0.5  # track edge so it can't lock on unfreeze
                continue
            fx = float(action[0]) * self.FORCE_SCALE
            fy = float(action[1]) * self.FORCE_SCALE
            self.bodies[name].apply_force_at_local_point((fx, fy))
            self._handle_lock(name, float(action[2]))

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

        # --- line-of-sight reward (paper-faithful), play phase only ---
        # Seeker gets +r when it can see the hider, hider gets +r when unseen. Per-step
        # magnitude is 1/PLAY_STEPS so a full episode totals at most ±1. No reward in prep,
        # no tag/termination on contact: the game is pure visibility over a fixed horizon.
        seeker_sees = self._visible("seeker", self.bodies["hider"])
        if in_prep:
            rewards = {"hider": 0.0, "seeker": 0.0}
        else:
            r = 1.0 / self.PLAY_STEPS
            if seeker_sees:
                rewards = {"hider": -r, "seeker": +r}
            else:
                rewards = {"hider": +r, "seeker": -r}

        time_up = self.steps >= self.MAX_STEPS
        terminations = {a: False for a in self.agents}  # no contact-termination anymore
        truncations = {a: time_up for a in self.agents}

        obs = {a: self._get_obs(a) for a in self.agents}
        # Expose visibility so eval/rendering can track the hidden-fraction metric.
        infos = {a: {"seeker_sees_hider": seeker_sees, "in_prep": in_prep} for a in self.agents}

        if time_up:
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
