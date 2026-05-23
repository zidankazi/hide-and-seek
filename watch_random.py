"""
Watch random policies play tag. Useful for eyeballing that the env looks right
before we start training. Quit by closing the window.
"""

from env import TagEnv

env = TagEnv(render_mode="human")
obs, _ = env.reset(seed=0)

ep_reward_h = 0.0
ep_reward_s = 0.0

while True:
    if env.renderer.get_events() is None:
        break

    if not env.agents:
        print(f"Ep {env.episode} | hider={ep_reward_h:+.1f} | seeker={ep_reward_s:+.1f}")
        ep_reward_h = 0.0
        ep_reward_s = 0.0
        obs, _ = env.reset()

    actions = {
        "hider": env.action_space("hider").sample(),
        "seeker": env.action_space("seeker").sample(),
    }
    obs, rewards, terms, truncs, infos = env.step(actions)
    ep_reward_h += rewards["hider"]
    ep_reward_s += rewards["seeker"]
    env.render()

env.close()
