"""
Watch the trained Stage 7 policies (room 1v1 + ramp) play deterministically.
Renders the full game: prep phase, boxes, the ramp (green wedge), locks.

Usage: python watch_hs7.py [--final] [--best]   (default --final)
Close the window to quit.
"""
import sys
import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_continuous import ActorCritic

SUFFIX = "" if "--best" in sys.argv else "_final"

env = HideAndSeekEnv(layout="room", ramp=True, max_steps=360, lock_mode="level",
                     n_hiders=1, n_seekers=2, box_mass=2, door_box_size=72, render_mode="human")
obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
act_dim = env.action_space(env.possible_agents[0]).shape[0]

nets = {}
for role in ("hider", "seeker"):
    sd = torch.load(f"hs_ramp_{role}{SUFFIX}.pt")
    ac = ActorCritic(obs_dim, act_dim, hidden=sd["shared.0.weight"].shape[0])
    ac.load_state_dict(sd)
    ac.eval()
    nets[role] = ac


def act(name, o):
    with torch.no_grad():
        mean, _ = nets[env.team[name]](torch.tensor(o, dtype=torch.float32).unsqueeze(0))
    return mean.squeeze(0).numpy()


obs, _ = env.reset()
ep_h = 0.0
while True:
    if env.renderer.get_events() is None:
        break
    if not env.agents:
        print(f"Ep {env.episode} | hider return {ep_h:+.2f} | "
              f"ramp_lock={env.ramp_lock_owner} | box_locks={env.box_lock_owner}")
        ep_h = 0.0
        obs, _ = env.reset()
    actions = {n: act(n, obs[n]) for n in env.possible_agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    ep_h += rewards[env.teams["hider"][0]]
    env.render()

env.close()
