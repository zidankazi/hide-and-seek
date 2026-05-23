import numpy as np
import pymunk
from gymnasium import spaces
from pettingzoo import ParallelEnv


class TagEnv(ParallelEnv):
    """
    Two-agent tag in an empty arena. The hider tries to survive a fixed number of steps,
    the seeker tries to touch the hider before time runs out.

    PettingZoo Parallel API: both agents act simultaneously each step, observations and
    rewards come back as dicts keyed by agent name ("hider", "seeker").

    Rewards are roughly zero-sum:
        Per step:  hider +1, seeker -1
        On tag:    hider -5, seeker +5 (one-shot bonus on top of the per-step)
    Episode ends on tag (termination) or after MAX_STEPS (truncation).
    """

    metadata = {"render_modes": ["human"], "name": "tag_v0", "render_fps": 60}

    # Constants. Copied from env_nav.py plus a few new ones for the tag mechanic
    ARENA_SIZE = 600
    AGENT_RADIUS = 18
    FORCE_SCALE = 1500
    MAX_STEPS = 240                   # paper uses ~240
    MIN_SPAWN_DIST = 250              # min initial distance between hider and seeker
    TAG_DIST = 2 * AGENT_RADIUS + 2   # touch threshold

    def __init__(self, render_mode=None):
        """
        Build the physics world once and reuse it across episodes. We make one Body+Circle
        per agent and store them in dicts keyed by name so we can look up by "hider"/"seeker"
        instead of carrying around two separate attributes.
        """
        self.possible_agents = ["hider", "seeker"]
        self.agents = []  # populated in reset()
        self.episode = 0

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.5

        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, self.AGENT_RADIUS)

        self.bodies = {}
        self.shapes = {}  # optional, handy for rendering / collision groups later

        for name in self.possible_agents:
            body = pymunk.Body(mass, moment)
            shape = pymunk.Circle(body, self.AGENT_RADIUS)
            shape.elasticity = 0.6
            shape.friction = 0.4
            self.bodies[name] = body
            self.shapes[name] = shape
            self.space.add(body, shape)

        # Walls: four static segments around the arena edge
        self.walls = [
            [(10, 10), (self.ARENA_SIZE - 10, 10)],                                       # top
            [(self.ARENA_SIZE - 10, 10), (self.ARENA_SIZE - 10, self.ARENA_SIZE - 10)],   # right
            [(self.ARENA_SIZE - 10, self.ARENA_SIZE - 10), (10, self.ARENA_SIZE - 10)],   # bottom
            [(10, self.ARENA_SIZE - 10), (10, 10)],                                       # left
        ]
        for start, end in self.walls:
            seg = pymunk.Segment(self.space.static_body, start, end, 6)
            seg.elasticity = 0.6
            seg.friction = 0.5
            self.space.add(seg)

        # Renderer is optional, skip it for headless training and build it when watching
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            from renderer import GameRenderer
            self.renderer = GameRenderer(title="Tag")
        self.steps = 0

    def observation_space(self, agent):
        """
        PettingZoo treats spaces as per-agent (a method, not an attribute) because in general
        different agents can have different obs shapes. Here both agents are symmetric so we
        return the same Box for either name.
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def action_space(self, agent):
        """Same Box for both agents: 2D continuous thrust in [-1, 1], scaled by FORCE_SCALE in step()."""
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_obs(self, agent):
        """
        Builds the 8-number observation that one agent sees this step:
            [self_x, self_y, self_vx, self_vy, opp_x, opp_y, opp_vx, opp_vy]
        Positions are normalized to [-1, 1] (offset by half-arena, divided by half-arena),
        velocities are scaled by a rough max so they usually land in [-1, 1] too.
        Clip at the end as a safety net for the rare frames where physics nudges values past 1.
        """
        opponent = "hider" if agent == "seeker" else "seeker"
        self_body = self.bodies[agent]
        opp_body = self.bodies[opponent]
        
        half = self.ARENA_SIZE / 2
        max_vel = 300 # rough cap so velocities land in [-1, 1] most of the time
        obs = np.array([
            (self_body.position.x - half) / half,
            (self_body.position.y - half) / half,
            self_body.velocity.x / max_vel,
            self_body.velocity.y / max_vel,
            (opp_body.position.x - half) / half,
            (opp_body.position.y - half) / half,
            opp_body.velocity.x / max_vel,
            opp_body.velocity.y / max_vel,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def reset(self, seed=None, options=None):
        """
        Start a new episode. Returns (obs_dict, info_dict), each keyed by agent name.
        Picks random spawn points for both agents and rejects pairs that spawn too close
        so the seeker can't instantly tag the hider on step 0.
        """
        # PettingZoo ParallelEnv has no super().reset() so we manage the rng ourselves
        # Seed if given, otherwise keep the one we have (or make one on first call)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, "np_random"):
            self.np_random = np.random.default_rng()

        # self.agents is the live list PettingZoo reads. Refill it from possible_agents
        self.agents = self.possible_agents[:]
        self.steps = 0
        self.episode += 1

        # Keep spawns off the walls so agents don't start clipping into them
        margin = self.AGENT_RADIUS + 20
        lo, hi = margin, self.ARENA_SIZE - margin

        # Re-roll spawn positions until they're far enough apart.
        # Otherwise the seeker sometimes spawns on top of the hider and tags in 1 step,
        # which trashes the reward signal early in training
        while True:
            pos_h = (self.np_random.uniform(lo, hi), self.np_random.uniform(lo, hi))
            pos_s = (self.np_random.uniform(lo, hi), self.np_random.uniform(lo, hi))
            dx, dy = pos_h[0] - pos_s[0], pos_h[1] - pos_s[1]
            if (dx * dx + dy * dy) ** 0.5 >= self.MIN_SPAWN_DIST:
                break

        # Teleport bodies to spawn points, zero velocities
        self.bodies["hider"].position = pos_h
        self.bodies["hider"].velocity = (0, 0)
        self.bodies["seeker"].position = pos_s
        self.bodies["seeker"].velocity = (0, 0)

        # Return obs and info as dicts keyed by agent name
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        """
        Advance the world one frame. Returns 5 dicts (obs, rewards, terminations, truncations, infos),
        all keyed by agent name.

        Termination vs truncation matters for PPO: on truncation we still bootstrap a value estimate
        for the last state, on termination we treat that value as 0 (the episode really ended).
        Tag = termination, hit MAX_STEPS = truncation.
        """
        # actions is a dict: {"hider": np.array([fx, fy]), "seeker": np.array([fx, fy])}
        # Each component is in [-1, 1] and gets scaled by FORCE_SCALE below

        # Apply each agent's thrust before stepping the physics so the forces hit this frame
        for name in self.agents:
            action = actions[name]
            fx = float(action[0]) * self.FORCE_SCALE
            fy = float(action[1]) * self.FORCE_SCALE
            self.bodies[name].apply_force_at_local_point((fx, fy))

        # Advance physics one frame (1/60s). Both agents' forces are applied at once, that's the parallel bit
        self.space.step(1 / 60)
        self.steps += 1

        # Tag check: distance between the two body centers, measured after the physics step
        hp = self.bodies["hider"].position
        sp = self.bodies["seeker"].position
        dist = ((hp.x - sp.x) ** 2 + (hp.y - sp.y) ** 2) ** 0.5
        tagged = dist < self.TAG_DIST

        # Zero-sum per-step reward: +1 hider, -1 seeker every frame the hider is still alive.
        # On tag, add an extra +5/-5 so the catch event has a bigger gradient than a normal step
        rewards = {"hider": 1.0, "seeker": -1.0}
        if tagged:
            rewards["hider"] -= 5.0
            rewards["seeker"] += 5.0

        # Terminated = task ended (got tagged). Truncated = ran out of time.
        # Both flip together for both agents since the game is symmetric
        time_up = self.steps >= self.MAX_STEPS
        terminations = {a: tagged for a in self.agents}
        truncations = {a: time_up for a in self.agents}

        # Build obs/info before emptying self.agents (otherwise the comprehensions have nothing to iterate)
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Empty self.agents tells PettingZoo the episode is over
        if tagged or time_up:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        """
        Draw the current frame using the shared GameRenderer.
        Builds a list of agent dicts (pos, vel, role, radius) and hands it off along with
        the walls and a small info dict for the HUD.
        """
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

        info = {
            "phase": "PLAY",
            "episode": self.episode,
            "step": self.steps,
            "max_steps": self.MAX_STEPS,
            "prep_fraction": 0.0,
            "hiders": 1,
            "seekers": 1,
            "reward": 0.0,
        }
        self.renderer.render(agents=agents, walls=self.walls, goal_pos=None, info=info)

    def close(self):
        """Tear down the renderer if we built one."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None