"""
Quick sanity check for env.py. Runs random policies for both agents over a handful of
episodes and prints what happened. We're not training, just checking that:
    - obs / reward / termination dicts are shaped right
    - episodes actually end (by tag or by timeout)
    - rewards are zero-sum like we designed
    - both ending paths fire at least sometimes
"""

import numpy as np
from env import TagEnv

env = TagEnv()
n_episodes = 20

n_tagged = 0
n_truncated = 0
episode_lengths = []
hider_returns = []
seeker_returns = []

for ep in range(n_episodes):
    obs, infos = env.reset(seed=ep)

    # Reset return shape checks
    assert set(obs.keys()) == {"hider", "seeker"}, f"obs keys: {obs.keys()}"
    assert obs["hider"].shape == (8,), f"hider obs shape: {obs['hider'].shape}"
    assert obs["seeker"].shape == (8,), f"seeker obs shape: {obs['seeker'].shape}"

    hider_total = 0.0
    seeker_total = 0.0
    steps = 0
    last_terms = {}
    last_truncs = {}

    while env.agents:
        actions = {
            "hider": env.action_space("hider").sample(),
            "seeker": env.action_space("seeker").sample(),
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        hider_total += rewards["hider"]
        seeker_total += rewards["seeker"]
        steps += 1
        last_terms = terms
        last_truncs = truncs

    if any(last_terms.values()):
        n_tagged += 1
        ending = "TAG"
    else:
        n_truncated += 1
        ending = "TIMEOUT"

    episode_lengths.append(steps)
    hider_returns.append(hider_total)
    seeker_returns.append(seeker_total)

    print(f"Ep {ep:2d} | {ending:8s} | steps={steps:3d} | hider={hider_total:+7.1f} | seeker={seeker_total:+7.1f}")

print()
print(f"Tagged:    {n_tagged}/{n_episodes}")
print(f"Timed out: {n_truncated}/{n_episodes}")
print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
print(f"Mean hider return:   {np.mean(hider_returns):+.1f}")
print(f"Mean seeker return:  {np.mean(seeker_returns):+.1f}")
print(f"Zero-sum check (should be ~0): {np.mean(hider_returns) + np.mean(seeker_returns):+.3f}")
