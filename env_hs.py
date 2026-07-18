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
    # Stage 7 splits occluders into two heights: WALL_CAT (arena edge, blocks sight always)
    # and LOW_CAT (interior room walls + boxes — an ELEVATED agent standing on the ramp sees
    # over these, within ELEV_RANGE). OCCLUDER_CAT stays the union, so every pre-ramp layout
    # raycasts exactly as before.
    AGENT_CAT = 0b0001
    WALL_CAT = 0b0010
    LOW_CAT = 0b0100
    OCCLUDER_CAT = WALL_CAT | LOW_CAT
    RAMP_CAT = 0b1000                    # the ramp itself never blocks sight

    # --- Stage 7: the ramp (2D translation of the paper's climb-over-the-wall ramp) ---
    # A pushable, lockable object. Any agent within RAMP_USE_DIST of its center is "elevated":
    # its line-of-sight ignores LOW_CAT occluders (boxes, interior walls) out to ELEV_RANGE.
    # Range is what makes ramp POSITION matter (as in the paper): a seeker must transport the
    # ramp near the room to peek inside, and hiders can counter by locking it far away (locked
    # objects can't be pushed; locks are per-team) or stealing it into the room.
    RAMP_SIZE = 40
    RAMP_MASS = 2                        # lighter than a box: quick to reposition
    RAMP_USE_DIST = 55                   # ~touching the ramp = standing on it
    ELEV_RANGE = 400                     # elevated sight only beats LOW occluders this far
                                         # (400 = whole room visible from the near band, so
                                         # in-room distance-evasion can't neutralize the ramp;
                                         # denying the ramp becomes the hider's only counter)
    RAMP_PARK = (540, 540)               # inactive ramp sits here, static and inert

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

    def __init__(self, render_mode=None, layout="room", team_size=1, n_boxes=None,
                 ramp=False, max_steps=None, lock_mode="toggle",
                 n_hiders=None, n_seekers=None, box_mass=None, door_box_size=None):
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

        ramp=True (Stage 7) adds the ramp object: +7 obs dims (ramp block + own elevated
        flag), elevation-aware LOS, and the reset(options={"ramp_active": bool}) curriculum
        knob — inactive episodes park the ramp at RAMP_PARK, static, with elevation disabled,
        so a run can anneal the mechanic in without changing the obs layout.
        max_steps overrides MAX_STEPS (Stage 7 uses 360: prep must fit ramp defense AND
        barricading). Reward scale adapts (per-step r = 1/PLAY_STEPS).

        lock_mode="toggle" (default, Stages 5-6): rising edge of action[2]>0.5 toggles
        the nearest lockable — but under exploration noise an agent dwelling near its own
        locked box keeps re-crossing the threshold and randomly UNDOES its own lock, so
        "build and hold" can't be reinforced. lock_mode="level" (Stage 7 runs 10+):
        action[2]>0.5 locks (level-triggered, idempotent — holding it keeps the lock);
        unlocking one's own lock is a separate EDGE-triggered press below -0.5.
        """
        assert layout in ("room", "open")
        assert lock_mode in ("toggle", "level")
        assert team_size >= 1
        self.layout = layout
        self.lock_mode = lock_mode
        self.team_size = team_size
        # Asymmetric teams (the paper's actual pressure: several seekers pincer, so
        # evasion stops paying and construction becomes the hider's only refuge).
        # Defaults preserve the symmetric team_size behavior exactly.
        self.n_hiders = n_hiders if n_hiders is not None else team_size
        self.n_seekers = n_seekers if n_seekers is not None else team_size
        assert self.n_hiders >= 1 and self.n_seekers >= 1
        self.ramp = ramp
        if max_steps is not None:
            self.MAX_STEPS = int(max_steps)
        if box_mass is not None:
            # Stage 7 uses 2: a lighter box shortens the contact-push needed to place it,
            # which is what makes precise barricade construction learnable under noise.
            self.BOX_MASS = float(box_mass)
        self.N_BOXES = n_boxes if n_boxes is not None else (2 if layout == "room" else 4)
        if self.n_hiders == 1 and self.n_seekers == 1:
            self.possible_agents = ["hider", "seeker"]
        else:
            self.possible_agents = [f"hider_{i}" for i in range(self.n_hiders)] + \
                                   [f"seeker_{i}" for i in range(self.n_seekers)]
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
        self.active_seekers = self.n_seekers
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
        n_edge = len(self.walls)  # arena-edge walls = WALL_CAT; interior room walls = LOW_CAT
        if self.layout == "room":
            self.walls += [list(w) for w in self.ROOM_WALLS]
        for k, (start, end) in enumerate(self.walls):
            seg = pymunk.Segment(self.space.static_body, start, end, 6)
            seg.elasticity = 0.4
            seg.friction = 0.5
            seg.filter = pymunk.ShapeFilter(
                categories=self.WALL_CAT if k < n_edge else self.LOW_CAT)
            self.space.add(seg)

        # --- movable boxes ---
        # Created once and repositioned each reset (same pattern as the agents).
        # lock_owner tracks which team has locked a box: None / "hider" / "seeker" (used in 5b).
        self.box_bodies = []
        self.box_shapes = []
        self.box_lock_owner = [None] * self.N_BOXES
        # Per-box sizes: box 0 can be enlarged (door_box_size, ramp+room only) so it seals
        # the 60px doorway from a much wider placement range — this lowers the PRECISION the
        # rung-1 barricade needs (a rough shove seals it), attacking the diagnosed bottleneck
        # without touching the reward. Every other box, and all non-ramp layouts, unchanged.
        self._door_box_size = (door_box_size if (door_box_size and ramp and layout == "room")
                               else None)
        self._box_sizes = [self._door_box_size if (i == 0 and self._door_box_size) else self.BOX_SIZE
                           for i in range(self.N_BOXES)]
        for i in range(self.N_BOXES):
            sz = self._box_sizes[i]
            moment = pymunk.moment_for_box(self.BOX_MASS, (sz, sz))
            body = pymunk.Body(self.BOX_MASS, moment)
            shape = pymunk.Poly.create_box(body, (sz, sz))
            shape.elasticity = 0.1
            shape.friction = 0.7
            shape.filter = pymunk.ShapeFilter(categories=self.LOW_CAT)
            self.box_bodies.append(body)
            self.box_shapes.append(shape)
            self.space.add(body, shape)
        self._box_moment = pymunk.moment_for_box(self.BOX_MASS, (self.BOX_SIZE, self.BOX_SIZE))

        # --- the ramp (Stage 7) ---
        self.ramp_body = None
        self.ramp_shape = None
        self.ramp_lock_owner = None
        self.ramp_active = True     # curriculum knob; set per-episode via reset options
        self._elevated_now = set()  # agents currently within RAMP_USE_DIST of the ramp
        if self.ramp:
            self._ramp_moment = pymunk.moment_for_box(
                self.RAMP_MASS, (self.RAMP_SIZE, self.RAMP_SIZE))
            body = pymunk.Body(self.RAMP_MASS, self._ramp_moment)
            shape = pymunk.Poly.create_box(body, (self.RAMP_SIZE, self.RAMP_SIZE))
            shape.elasticity = 0.1
            shape.friction = 0.7
            shape.filter = pymunk.ShapeFilter(categories=self.RAMP_CAT)
            self.ramp_body = body
            self.ramp_shape = shape
            self.space.add(body, shape)

        # Rising-edge tracking for the lock action: an agent only toggles a lock when its lock
        # signal crosses from <=0.5 to >0.5, so holding the signal high doesn't thrash lock/unlock.
        self._prev_lock = {name: False for name in self.possible_agents}
        self._prev_unlock = {name: False for name in self.possible_agents}  # level mode only

        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            from renderer import GameRenderer
            self.renderer = GameRenderer(title="Hide & Seek")
        self.steps = 0

    # obs: self(4) + each other agent (teammates first, then opponents) (4+visible)
    #      + N_BOXES*(4+lock+visible) [+ ramp(4+lock+visible) + own-elevated flag]
    def _obs_dim(self):
        return 4 + (self.n_hiders + self.n_seekers - 1) * 5 + self.N_BOXES * 6 + (7 if self.ramp else 0)

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

        An ELEVATED observer (standing on the ramp) sees over LOW_CAT occluders — boxes and
        interior walls — but only within ELEV_RANGE; beyond that, normal rules. Arena-edge
        walls block sight regardless.
        """
        start = self.bodies[agent].position
        end = target_body.position
        mask = self.OCCLUDER_CAT
        if agent in self._elevated_now and (end - start).length <= self.ELEV_RANGE:
            mask = self.WALL_CAT
        hit = self.space.segment_query_first(start, end, 1, pymunk.ShapeFilter(mask=mask))
        if hit is None:
            return True
        return hit.shape is target_shape

    def _compute_elevated(self):
        """Set of agents currently standing on (within RAMP_USE_DIST of) an active ramp."""
        if not (self.ramp and self.ramp_active):
            return set()
        rp = self.ramp_body.position
        return {n for n in self.possible_agents
                if (self.bodies[n].position - rp).length <= self.RAMP_USE_DIST}

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
        sz = self._box_sizes[i]
        body.body_type = pymunk.Body.DYNAMIC
        body.mass = self.BOX_MASS
        body.moment = pymunk.moment_for_box(self.BOX_MASS, (sz, sz))
        self.space.reindex_shapes_for_body(body)

    def _set_box_static(self, i):
        body = self.box_bodies[i]
        body.velocity = (0, 0)
        body.angular_velocity = 0.0
        body.body_type = pymunk.Body.STATIC
        self.space.reindex_shapes_for_body(body)

    def _set_ramp_dynamic(self):
        body = self.ramp_body
        body.body_type = pymunk.Body.DYNAMIC
        body.mass = self.RAMP_MASS
        body.moment = self._ramp_moment
        self.space.reindex_shapes_for_body(body)

    def _set_ramp_static(self):
        body = self.ramp_body
        body.velocity = (0, 0)
        body.angular_velocity = 0.0
        body.body_type = pymunk.Body.STATIC
        self.space.reindex_shapes_for_body(body)

    def _handle_lock(self, agent, lock_signal):
        """
        Acts on the nearest lockable (box, or the ramp when active) within LOCK_DIST.
        Owners store the TEAM name; locked-by-other-team is always a no-op.

        toggle mode: rising edge of signal>0.5 toggles (unlocked -> lock, own -> unlock).
        level mode:  signal>0.5 locks (idempotent, noise can't undo a held lock);
                     rising edge of signal<-0.5 unlocks one's own lock.
        """
        if self.lock_mode == "toggle":
            pressed = lock_signal > 0.5
            rising = pressed and not self._prev_lock[agent]
            self._prev_lock[agent] = pressed
            if not rising:
                return
            do_lock = do_unlock = True
        else:
            do_lock = lock_signal > 0.5
            unlock_pressed = lock_signal < -0.5
            do_unlock = unlock_pressed and not self._prev_unlock[agent]
            self._prev_unlock[agent] = unlock_pressed
            if not (do_lock or do_unlock):
                return
        team = self.team[agent]
        i, d = self._nearest_box(agent)
        if self.ramp and self.ramp_active:
            rd = (self.ramp_body.position - self.bodies[agent].position).length
            if rd < d and rd <= self.LOCK_DIST:
                owner = self.ramp_lock_owner
                if owner is None and do_lock:
                    self._set_ramp_static()
                    self.ramp_lock_owner = team
                elif owner == team and do_unlock:
                    self._set_ramp_dynamic()
                    self.ramp_lock_owner = None
                return
        if d > self.LOCK_DIST:
            return
        owner = self.box_lock_owner[i]
        if owner is None and do_lock:
            self._set_box_static(i)
            self.box_lock_owner[i] = team
        elif owner == team and do_unlock:
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

        # Ramp block (same shape as a box block) + own-elevated flag. The ramp never occludes,
        # but something between the agent and the ramp can hide it (target_shape=None works:
        # the ray can't hit the ramp itself).
        if self.ramp:
            rb = self.ramp_body
            owner = self.ramp_lock_owner
            lock_state = 0.0 if owner is None else (1.0 if owner == my_team else -1.0)
            if self._visible(agent, rb):
                parts += [
                    (rb.position.x - half) / half,
                    (rb.position.y - half) / half,
                    rb.velocity.x / mv,
                    rb.velocity.y / mv,
                    lock_state,
                    1.0,
                ]
            else:
                parts += [0.0, 0.0, 0.0, 0.0, lock_state, 0.0]
            parts.append(1.0 if agent in self._elevated_now else 0.0)

        return np.clip(np.array(parts, dtype=np.float32), -1.0, 1.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        if options and "active_seekers" in options:
            k = int(options["active_seekers"])
            assert 1 <= k <= self.n_seekers
            self.active_seekers = k
        self._dormant = set(self.teams["seeker"][self.active_seekers:])
        # Ramp curriculum knob (persists across resets until changed, like active_seekers).
        if self.ramp and options and "ramp_active" in options:
            self.ramp_active = bool(options["ramp_active"])
        # Discovery assists (per-episode, do NOT persist — evals never see them unless
        # asked). Training-only spawn-state tweaks; the reward is never touched.
        #   seeker_on_ramp: seekers start standing on the ramp — elevated vision is
        #     experienced without first discovering transport+standing (ignited rung 2).
        #   ramp_locked:    the ramp starts already hider-locked where it spawned — the
        #     hider experiences "ramp denied -> seeker grounded" payoff directly (rung 3).
        #   doorway_sealed: box 0 starts hider-locked in the doorway — the hider
        #     experiences the barricade payoff directly (rung 1).
        #   hider_on_ramp:  hiders start beside the ramp during prep — locking it is one
        #     press away, transporting it a short push (reverse-chained rung 3).
        #   doorway_box:    "sealed" = box 0 hider-locked in the doorway (full payoff
        #     state); "placed" = box 0 sitting in the doorway UNLOCKED — the barricade
        #     needs only the lock press; "near" = box 0 a short push (25-50px) above the
        #     doorway — push + press (graded reverse chain for rung 1).
        #   hider_at_door:  (with doorway_box="placed") hiders start beside the unlocked
        #     doorway box — the barricade lock-press is immediately reachable. The evade
        #     prior keeps hiders AWAY from the doorway, so without this the placed state
        #     is never dwelled in and the press never fires (phase A lesson).
        seeker_on_ramp = bool(options.get("seeker_on_ramp")) if options else False
        ramp_locked = bool(options.get("ramp_locked")) if options else False
        hider_on_ramp = bool(options.get("hider_on_ramp")) if options else False
        hider_at_door = bool(options.get("hider_at_door")) if options else False
        doorway_box = options.get("doorway_box") if options else None
        if options and options.get("doorway_sealed"):  # back-compat alias
            doorway_box = "sealed"

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.episode += 1
        # Release any boxes left locked (STATIC) from the previous episode, then clear state.
        for i in range(self.N_BOXES):
            if self.box_lock_owner[i] is not None:
                self._set_box_dynamic(i)
        self.box_lock_owner = [None] * self.N_BOXES
        if self.ramp and self.ramp_lock_owner is not None:
            self._set_ramp_dynamic()
            self.ramp_lock_owner = None
        self._prev_lock = {name: False for name in self.possible_agents}
        self._prev_unlock = {name: False for name in self.possible_agents}

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
            # Ramp-era room spawns: box 0 starts ABOVE the doorway (same x band, ~80px
            # push to seal it) — near enough to make rung 1 short, but clear of the
            # doorway's sightlines and entry path. (Run 2 spawned it at y 170-210,
            # directly behind the doorway: a passive plug that killed all hunting
            # pressure and with it every gradient in the ladder.)
            if self.ramp and self.layout == "room" and i == 0:
                bx, by = (self.np_random.uniform(150, 205),
                          self.np_random.uniform(140, 190))
            else:
                bx, by = box_spawn()
            body.position = (bx, by)
            body.velocity = (0, 0)
            body.angular_velocity = 0.0
            body.angle = 0.0
            # Boxes were teleported directly (not via physics), so their broadphase index
            # entries are stale. Reindex each so the very first LOS raycast in this episode's
            # reset obs sees them at their new positions (step() reindexes via space.step()).
            self.space.reindex_shapes_for_body(body)

        if self.ramp:
            if self.ramp_active:
                # Ramp spawns outside the room: half the time in a NEAR band (within
                # elevation range of the room interior — standing on it where it lies
                # already pays, the discovery gradient), otherwise anywhere mid-arena
                # (transport required). The first 30M run showed a uniform [250,480]
                # band leaves stand-on-ramp reward too rare to ever be discovered.
                self._set_ramp_dynamic()
                while True:
                    if self.np_random.random() < 0.5:
                        p = (self.np_random.uniform(240, 330), self.np_random.uniform(240, 330))
                    else:
                        p = (self.np_random.uniform(240, 480), self.np_random.uniform(240, 480))
                    if all(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 >= 60
                           for q in positions.values()):
                        break
            else:
                p = self.RAMP_PARK
            self.ramp_body.position = p
            self.ramp_body.velocity = (0, 0)
            self.ramp_body.angular_velocity = 0.0
            self.ramp_body.angle = 0.0
            if self.ramp_active:
                self.space.reindex_shapes_for_body(self.ramp_body)
                for team, flag in (("seeker", seeker_on_ramp), ("hider", hider_on_ramp)):
                    if not flag:
                        continue
                    rp = self.ramp_body.position
                    for name in self.teams[team]:
                        ang = self.np_random.uniform(0, 2 * np.pi)
                        d = self.np_random.uniform(40, 50)  # inside RAMP_USE_DIST, outside the body
                        self.bodies[name].position = (rp.x + d * np.cos(ang),
                                                      rp.y + d * np.sin(ang))
                        self.bodies[name].velocity = (0, 0)
            else:
                self._set_ramp_static()  # parked: inert scenery until the curriculum turns it on

        if self.ramp and self.layout == "room":
            if doorway_box in ("placed", "sealed", "near"):
                b = self.box_bodies[0]
                by = self.np_random.uniform(185, 212) if doorway_box == "near" else 228.0
                b.position = (self.np_random.uniform(172, 188), by)
                b.velocity = (0, 0)
                b.angular_velocity = 0.0
                b.angle = 0.0
                self.space.reindex_shapes_for_body(b)
                if doorway_box == "sealed":
                    self._set_box_static(0)
                    self.box_lock_owner[0] = "hider"
                if hider_at_door:
                    n_h = len(self.teams["hider"])
                    for j, name in enumerate(self.teams["hider"]):
                        # just inside the room, above the box, within LOCK_DIST of it
                        self.bodies[name].position = (b.position.x + (j - (n_h - 1) / 2) * 40,
                                                      b.position.y - 55)
                        self.bodies[name].velocity = (0, 0)
            if ramp_locked and self.ramp_active:
                self._set_ramp_static()
                self.ramp_lock_owner = "hider"
        self.space.reindex_static()

        self._elevated_now = self._compute_elevated()
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
                self._prev_unlock[name] = action[2] < -0.5
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

        self._elevated_now = self._compute_elevated()

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
        # Stage 7 rung metrics: is the doorway sealed by a hider-locked box, is any seeker
        # elevated, and who holds the ramp lock — the emergence-timeline signals.
        if self.ramp:
            barricaded = any(
                self.box_lock_owner[i] == "hider"
                and 145 <= b.position.x <= 215 and 210 <= b.position.y <= 270
                for i, b in enumerate(self.box_bodies))
            seeker_elev = any(s in self._elevated_now for s in self.teams["seeker"])
            for a in self.agents:
                infos[a]["doorway_barricaded"] = barricaded
                infos[a]["seeker_elevated"] = seeker_elev
                infos[a]["ramp_lock_owner"] = self.ramp_lock_owner
                infos[a]["ramp_active"] = self.ramp_active

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
                "size": self._box_sizes[i],
                "locked": self.box_lock_owner[i] is not None,
            })
        ramp = None
        if self.ramp:
            ramp = {
                "pos": (self.ramp_body.position.x, self.ramp_body.position.y),
                "size": self.RAMP_SIZE,
                "locked": self.ramp_lock_owner is not None,
                "active": self.ramp_active,
            }
        in_prep = self.steps < self.PREP_STEPS
        info = {
            "phase": "PREP" if in_prep else "PLAY",
            "episode": self.episode,
            "step": self.steps,
            "max_steps": self.MAX_STEPS,
            "prep_fraction": self.PREP_FRACTION,
            "hiders": self.n_hiders,
            "seekers": self.n_seekers,
            "reward": 0.0,
        }
        self.renderer.render(agents=agents, walls=self.walls, boxes=boxes, ramp=ramp,
                             goal_pos=None, info=info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
