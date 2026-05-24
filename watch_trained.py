"""
Watch the trained hider and seeker policies play each other.
Loads hider.pt and seeker.pt (best-saved policies from ppo_multi.py training) and renders
them playing the tag env. Quit by closing the window.

Uses the mean of each policy's distribution instead of sampling, so behavior is deterministic.
"""

import torch

from env import TagEnv
from ppo_continuous import ActorCritic


env = TagEnv(render_mode="human")
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

# Load each agent's best saved network
nets = {}
for name in env.possible_agents:
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load(f"{name}.pt"))
    ac.eval()
    nets[name] = ac

obs, _ = env.reset(seed=0)

ep_reward = {"hider": 0.0, "seeker": 0.0}
ep_count = 0
tag_count = 0

while True:
    if env.renderer.get_events() is None:
        break

    if not env.agents:
        ep_count += 1
        was_tagged = ep_reward["seeker"] > -2.4 + 0.01  # seeker scored above random-baseline = caught
        if was_tagged:
            tag_count += 1
        print(f"Ep {ep_count:3d} | hider={ep_reward['hider']:+.2f} | seeker={ep_reward['seeker']:+.2f} | tags so far: {tag_count}/{ep_count}")
        ep_reward = {"hider": 0.0, "seeker": 0.0}
        obs, _ = env.reset()

    # Pick deterministic actions (mean of the Normal distribution, not a sample)
    actions = {}
    for name in env.possible_agents:
        obs_t = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = nets[name](obs_t)
        actions[name] = mean.squeeze(0).numpy()

    obs, rewards, terms, truncs, _ = env.step(actions)
    ep_reward["hider"] += rewards["hider"]
    ep_reward["seeker"] += rewards["seeker"]
    env.render()

env.close()
