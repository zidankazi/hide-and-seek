import numpy as np
import pymunk
import gymnasium as gym
from gymnasium import spaces

# A Gymnasium environment is the "game" PPO plays.
# Standard contract: reset() to start, step(action) each frame,
# observation_space + action_space describe the shapes.


class HideAndSeekEnv(gym.Env):
    """
    Single-agent navigation env.
    One circle agent applies continuous 2D thrust to reach a goal.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # Constants
    ARENA_SIZE = 600    # 600x600 Arena
    AGENT_RADIUS = 18
    FORCE_SCALE = 1500  # Max force when action component = 1.0
    MAX_STEPS = 500     # Episode ends after this many steps

    def __init__(self, render_mode=None):
        super().__init__()

        # Observation: 6-number vector [agent_x, agent_y, agent_vx, agent_vy, goal_x, goal_y]
        # Normalized to [-1, 1] so the network deals with small numbers
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Action: 2D continuous thrust, each in [-1, 1]. Multiplied by FORCE_SCALE in step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # The physics world. Top-down so no gravity. Damping bleeds off velocity over time
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.5

        # Agent: invisible Body (mass + velocity) attached to a Circle Shape (collision)
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, self.AGENT_RADIUS) # rotational inertia, pymunk needs it
        self.agent_body = pymunk.Body(mass, moment)
        self.agent_body.position = (self.ARENA_SIZE / 2, self.ARENA_SIZE / 2)
        agent_shape = pymunk.Circle(self.agent_body, self.AGENT_RADIUS)
        agent_shape.elasticity = 0.6 # bounces a bit off walls
        agent_shape.friction = 0.4
        self.space.add(self.agent_body, agent_shape)

        # Walls: four static segments around the arena edge. Kept around for the renderer
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

        # Goal position the agent is trying to reach
        self.goal_pos = (self.ARENA_SIZE - 100, 100)

        # Renderer is optional. Skip it for headless training, build it when watching
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            from renderer import GameRenderer
            self.renderer = GameRenderer(title="Hide & Seek")

        # Step counter, used to cap episode length at MAX_STEPS
        self.steps = 0
