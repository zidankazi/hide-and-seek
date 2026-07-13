"""
Headless evaluation: run trained hider vs seeker for N episodes (deterministic actions),
report catch rate and mean steps-to-catch. Pass a directory to load {hider,seeker}.pt from
a different location (e.g. stage3_backup) for A/B comparison.
"""
import sys
import numpy as np
import torch

from env import TagEnv
from ppo_continuous import ActorCritic

load_dir = sys.argv[1] if len(sys.argv) > 1 else "."
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200

env = TagEnv()
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

nets = {}
for name in env.possible_agents:
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load(f"{load_dir}/{name}.pt"))
    ac.eval()
    nets[name] = ac

catches = 0
catch_steps = []
for ep in range(N):
    obs, _ = env.reset(seed=ep)  # fixed seeds so both models see the same spawns
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
        if any(terms.values()):  # tagged
            catches += 1
            catch_steps.append(steps)
            break
env.close()

rate = catches / N
print(f"[{load_dir}] {N} episodes | catch rate: {rate:.1%} ({catches}/{N})", end="")
if catch_steps:
    print(f" | mean steps-to-catch: {np.mean(catch_steps):.0f}/240 (median {int(np.median(catch_steps))})")
else:
    print(" | no catches")
