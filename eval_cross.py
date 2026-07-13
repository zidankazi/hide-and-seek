"""
Cross-matchup eval: load hider from one dir and seeker from another so we can isolate
each policy's contribution. Usage: python eval_cross.py <hider_dir> <seeker_dir> [N]
"""
import sys
import numpy as np
import torch

from env import TagEnv
from ppo_continuous import ActorCritic

hider_dir = sys.argv[1]
seeker_dir = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200

env = TagEnv()
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

dirs = {"hider": hider_dir, "seeker": seeker_dir}
nets = {}
for name in env.possible_agents:
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load(f"{dirs[name]}/{name}.pt"))
    ac.eval()
    nets[name] = ac

catches, catch_steps = 0, []
for ep in range(N):
    obs, _ = env.reset(seed=ep)
    steps = 0
    while env.agents:
        actions = {}
        for name in env.possible_agents:
            obs_t = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, _ = nets[name](obs_t)
            actions[name] = mean.squeeze(0).numpy()
        obs, rewards, terms, truncs, _ = env.step(actions)
        steps += 1
        if any(terms.values()):
            catches += 1
            catch_steps.append(steps)
            break
env.close()

rate = catches / N
surv = np.mean([s if s else 240 for s in catch_steps] + [240] * (N - catches))
tail = f"mean steps-to-catch {np.mean(catch_steps):.0f}" if catch_steps else "no catches"
print(f"hider={hider_dir:14s} seeker={seeker_dir:14s} | catch {rate:5.1%} ({catches:3d}/{N}) | {tail}")
