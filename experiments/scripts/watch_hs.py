"""View random hide-and-seek policies, nudging the hider toward a box during preparation."""

import numpy as np

from hide_and_seek.env_hs import HideAndSeekEnv

env = HideAndSeekEnv(render_mode="human")
obs, _ = env.reset(seed=0)
rng = np.random.default_rng(0)

ep_reward_h = 0.0
ep_reward_s = 0.0

while True:
    if env.renderer.get_events() is None:
        break

    if not env.agents:
        print(f"Ep {env.episode} | hider={ep_reward_h:+.2f} | seeker={ep_reward_s:+.2f}")
        ep_reward_h = 0.0
        ep_reward_s = 0.0
        obs, _ = env.reset()

    # Hider: steer toward the nearest box so we can see boxes move. Seeker: random.
    hp = np.array(tuple(env.bodies["hider"].position))
    boxes = [np.array(tuple(b.position)) for b in env.box_bodies]
    nearest = min(boxes, key=lambda b: np.linalg.norm(b - hp))
    direction = nearest - hp
    n = np.linalg.norm(direction)
    steer = (direction / n) if n > 1e-6 else np.zeros(2)
    hider_act = np.array([steer[0], steer[1], 0.0], dtype=np.float32)

    actions = {
        "hider": hider_act,
        "seeker": env.action_space("seeker").sample(),
    }
    obs, rewards, terms, truncs, infos = env.step(actions)
    ep_reward_h += rewards["hider"]
    ep_reward_s += rewards["seeker"]
    env.render()

env.close()
