import numpy as np
import pymunk
from gymnasium import spaces
from pettingzoo import ParallelEnv


class HideAndSeekEnv(ParallelEnv):
    """
    Stage 5 hide-and-seek. Hiders get a prep-phase head start to push and lock movable boxes
    into cover; the reward is line-of-sight (paper-faithful): seekers are rewarded when they
    can SEE a hider, hiders when unseen, so hiders learn to barricade with boxes and lock them.

    Supports 1v1 (Stage 5a/5b, agents "hider"/"seeker") and NvN teams (the scale-up run,
    agents "hider_0".."seeker_N-1") via team_size — see __init__ for the team semantics.

    PettingZoo Parallel API: all agents act each step; obs/rewards/etc. come back as dicts
    keyed by agent name.
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

    def __init__(self, render_mode=None, layout="room", team_size=1, n_boxes=None):
        """
        layout="room" (Stage 5b): corner room with a doorway, 2 boxes. The room hides the
            hider passively, so tool-use is optional.
        layout="open" (Stage 5b-ii): open arena, no interior walls, 4 boxes, hider spawns in a
            random corner. Nothing hides the hider for free — it must push+lock boxes to close
            a corner pocket, forcing genuine fort-building (closest to the paper).

        team_size=1 keeps the original 1v1 game unchanged (agents "hider"/"seeker").
        team_size>=2 (the scale-up run) fields teams ("hider_0".."seeker_N-1"): obs gain
        teammate blocks, box locks are owned per-TEAM (either teammate can unlock), and the
        LOS reward is team-level — seekers score when ANY seeker sees ANY hider, hiders score
        only when ALL of them are unseen (paper-faithful team reward).
        n_boxes overrides the layout default (room 2 / open 4).
        """
        assert layout in ("room", "open")
        assert team_size >= 1
        self.layout = layout
        self.team_size = team_size
        self.N_BOXES = n_boxes if n_boxes is not None else (2 if layout == "room" else 4)
        if team_size == 1:
            self.possible_agents = ["hider", "seeker"]
        else:
            self.possible_agents = [f"hider_{i}" for i in range(team_size)] + \
                                   [f"seeker_{i}" for i in range(team_size)]
        # team[name] -> "hider"/"seeker"; teams[team] -> member names. With team_size=1 the
        # member names ARE the team names, which is what keeps all 1v1 code paths identical.
        self.team = {n: ("hider" if n.startswith("hider") else "seeker")
                     for n in self.possible_agents}
        self.teams = {t: [n for n in self.possible_agents if self.team[n] == t]
                      for t in ("hider", "seeker")}
        # Curriculum knob (set per-episode via reset(options={"active_seekers": k})):
        # seekers beyond the first k are DORMANT — frozen in place for the whole episode
        # and excluded from the team line-of-sight check. Obs layout is unaffected, so
        # policies transfer across curriculum phases. Default: everyone active.
        self.active_seekers = team_size
        self._dormant = set()
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

    # obs: self(4) + each other agent (teammates first, then opponents) (4+visible)
    #      + N_BOXES*(4+lock+visible)
    def _obs_dim(self):
        return 4 + (2 * self.team_size - 1) * 5 + self.N_BOXES * 6

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
        within LOCK_DIST: unlocked -> lock to own team; locked-by-own-team -> unlock (either
        teammate can); locked-by-other-team -> no-op. lock_owner stores the TEAM name.
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
        team = self.team[agent]
        if owner is None:
            self._set_box_static(i)
            self.box_lock_owner[i] = team
        elif owner == team:
            self._set_box_dynamic(i)
            self.box_lock_owner[i] = None
        # owner == other team: no-op

    def _get_obs(self, agent):
        half = self.ARENA_SIZE / 2
        mv = self.MAX_VEL
        my_team = self.team[agent]
        opp_team = "hider" if my_team == "seeker" else "seeker"
        sb = self.bodies[agent]

        parts = [
            (sb.position.x - half) / half,
            (sb.position.y - half) / half,
            sb.velocity.x / mv,
            sb.velocity.y / mv,
        ]

        # Every other agent (teammates first, then opponents), each masked by line-of-sight:
        # when not visible, zero pos/vel + flag 0. Teammates are masked too — agents share a
        # policy, not a radio.
        others = [n for n in self.teams[my_team] if n != agent] + self.teams[opp_team]
        for other in others:
            ob = self.bodies[other]
            if self._visible(agent, ob):
                parts += [
                    (ob.position.x - half) / half,
                    (ob.position.y - half) / half,
                    ob.velocity.x / mv,
                    ob.velocity.y / mv,
                    1.0,
                ]
            else:
                parts += [0.0, 0.0, 0.0, 0.0, 0.0]

        # Boxes: pos/vel + lock_state (+1 own team / -1 other team / 0 none) + visible flag.
        for i, body in enumerate(self.box_bodies):
            visible = self._visible(agent, body, self.box_shapes[i])
            owner = self.box_lock_owner[i]
            lock_state = 0.0 if owner is None else (1.0 if owner == my_team else -1.0)
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

        if options and "active_seekers" in options:
            k = int(options["active_seekers"])
            assert 1 <= k <= self.team_size
            self.active_seekers = k
        self._dormant = set(self.teams["seeker"][self.active_seekers:])

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
        positions = {}
        sep = 2 * self.AGENT_RADIUS + 4  # min separation between agents placed together

        def clear_of(names, p, min_d):
            return all(((p[0] - positions[n][0]) ** 2 +
                        (p[1] - positions[n][1]) ** 2) ** 0.5 >= min_d
                       for n in names if n in positions)

        if self.layout == "room":
            # Hiders inside the room; seekers outside; boxes inside near the hiders.
            for name in self.teams["hider"]:
                while True:
                    p = (self.np_random.uniform(self.ROOM_LO, self.ROOM_HI),
                         self.np_random.uniform(self.ROOM_LO, self.ROOM_HI))
                    if clear_of(list(positions), p, sep):
                        break
                positions[name] = p
            for name in self.teams["seeker"]:
                while True:
                    p = (self.np_random.uniform(300, hi), self.np_random.uniform(300, hi))
                    if clear_of(self.teams["seeker"], p, sep):  # outside the 240x240 room by construction
                        break
                positions[name] = p
            box_spawn = lambda: (self.np_random.uniform(self.ROOM_LO, self.ROOM_HI),
                                 self.np_random.uniform(self.ROOM_LO, self.ROOM_HI))
        else:
            # Open arena. The hider team spawns in one random corner pocket (near two arena
            # walls it can complete into cover); boxes scatter mid-arena; seekers spawn
            # anywhere far off from every hider.
            corner = self.np_random.integers(4)
            cx = lo if corner in (0, 2) else hi
            cy = lo if corner in (0, 1) else hi
            for name in self.teams["hider"]:
                while True:
                    p = (cx + (1 if cx == lo else -1) * self.np_random.uniform(0, 90),
                         cy + (1 if cy == lo else -1) * self.np_random.uniform(0, 90))
                    if clear_of(list(positions), p, sep):
                        break
                positions[name] = p
            for name in self.teams["seeker"]:
                while True:
                    p = (self.np_random.uniform(lo, hi), self.np_random.uniform(lo, hi))
                    if clear_of(self.teams["hider"], p, self.MIN_SPAWN_DIST) and \
                       clear_of(self.teams["seeker"], p, sep):
                        break
                positions[name] = p
            mid_lo, mid_hi = self.ARENA_SIZE * 0.25, self.ARENA_SIZE * 0.75
            box_spawn = lambda: (self.np_random.uniform(mid_lo, mid_hi),
                                 self.np_random.uniform(mid_lo, mid_hi))

        for name, p in positions.items():
            self.bodies[name].position = p
            self.bodies[name].velocity = (0, 0)

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
            # Prep phase: seekers are frozen (no force, velocity pinned to zero) so the
            # hiders get a head start to set up. Hiders move and lock freely the whole time.
            # Dormant seekers (curriculum) stay frozen the entire episode.
            if (in_prep and self.team[name] == "seeker") or name in self._dormant:
                self.bodies[name].velocity = (0, 0)
                self._prev_lock[name] = action[2] > 0.5  # track edge so it can't lock on unfreeze
                continue
            fx = float(action[0]) * self.FORCE_SCALE
            fy = float(action[1]) * self.FORCE_SCALE
            self.bodies[name].apply_force_at_local_point((fx, fy))
            self._handle_lock(name, float(action[2]))

        self.space.step(1 / 60)
        self.steps += 1

        # Keep seekers pinned during prep (and dormant ones always) even after the physics
        # step (contacts could nudge them).
        for name in self.teams["seeker"]:
            if in_prep or name in self._dormant:
                self.bodies[name].velocity = (0, 0)

        # Clamp agent velocities (boxes keep their own physics).
        for name in self.agents:
            body = self.bodies[name]
            vx, vy = body.velocity
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > self.MAX_VEL:
                scale = self.MAX_VEL / speed
                body.velocity = (vx * scale, vy * scale)

        # --- line-of-sight reward (paper-faithful), play phase only ---
        # Team-level: seekers get +r when ANY seeker sees ANY hider, hiders get +r only when
        # ALL hiders are unseen. Per-step magnitude is 1/PLAY_STEPS so a full episode totals
        # at most ±1. No reward in prep, no tag/termination on contact: the game is pure
        # visibility over a fixed horizon. (With team_size=1 this is the original 1v1 reward.)
        seeker_sees = any(self._visible(s, self.bodies[h])
                          for s in self.teams["seeker"][:self.active_seekers]
                          for h in self.teams["hider"])
        if in_prep:
            rewards = {a: 0.0 for a in self.agents}
        else:
            r = 1.0 / self.PLAY_STEPS
            seen_sign = +1.0 if seeker_sees else -1.0
            rewards = {a: (seen_sign if self.team[a] == "seeker" else -seen_sign) * r
                       for a in self.agents}

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
                "role": self.team[name],  # renderer colors by "hider"/"seeker"
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
            "hiders": self.team_size,
            "seekers": self.team_size,
            "reward": 0.0,
        }
        self.renderer.render(agents=agents, walls=self.walls, boxes=boxes, goal_pos=None, info=info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
