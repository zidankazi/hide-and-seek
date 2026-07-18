"""
Behavioral eval for Stage 7 (room 1v2 + ramp). Loads hs_ramp_hider.pt / hs_ramp_seeker.pt
(one shared policy per team) and measures each rung of the emergent arc over N
deterministic episodes with NO training assists:
  rung 1  barricade: doorway sealed by a hider-locked box at episode end
  rung 2  ramp use: fraction of play steps any seeker spends elevated
  rung 3  ramp defense: ramp locked by the hider at episode end (+ where it ended up)
plus hidden-fraction, a single-seeker comparison (is the second seeker what forces the
tools?), and the hider-disabled counterfactual (geometry-alone baseline).

Usage: python eval_hs7.py [N] [--final]
"""
import sys
import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_continuous import ActorCritic

digits = [a for a in sys.argv[1:] if a.isdigit()]
N = int(digits[0]) if digits else 200
SUFFIX = "_final" if "--final" in sys.argv else ""
PREFIX = "hs_ramp"
DOORWAY_CENTER = np.array([180.0, 240.0])

env = HideAndSeekEnv(layout="room", ramp=True, max_steps=360, lock_mode="level",
                     n_hiders=1, n_seekers=2, box_mass=2, door_box_size=72)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]
H0 = env.teams["hider"][0]

nets = {}
for team in ("hider", "seeker"):
    sd = torch.load(f"{PREFIX}_{team}{SUFFIX}.pt")
    ac = ActorCritic(obs_dim, act_dim, hidden=sd["shared.0.weight"].shape[0])
    ac.load_state_dict(sd)
    ac.eval()
    nets[team] = ac


def act(name, obs):
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = nets[env.team[name]](t)
    return mean.squeeze(0).numpy()


def run(active_seekers=2, disable_hider=False):
    hidden, barricade, elev, rlock, ramp_dist = [], [], [], [], []
    for ep in range(N):
        obs, _ = env.reset(seed=ep, options={"ramp_active": True,
                                             "active_seekers": active_seekers})
        play, hid, ele = 0, 0, 0
        info = None
        while env.agents:
            actions = {}
            for name in env.possible_agents:
                a = np.zeros(act_dim, dtype=np.float32) \
                    if (disable_hider and env.team[name] == "hider") else act(name, obs[name])
                actions[name] = a
            obs, rewards, terms, truncs, infos = env.step(actions)
            info = infos[H0]
            if not info["in_prep"]:
                play += 1
                if not info["seeker_sees_hider"]:
                    hid += 1
                if info["seeker_elevated"]:
                    ele += 1
        hidden.append(hid / play if play else 0.0)
        barricade.append(info["doorway_barricaded"])
        elev.append(ele / play if play else 0.0)
        rlock.append(info["ramp_lock_owner"] == "hider")
        rp = np.array([env.ramp_body.position.x, env.ramp_body.position.y])
        ramp_dist.append(float(np.linalg.norm(rp - DOORWAY_CENTER)))
    return (np.mean(hidden), np.mean(barricade), np.mean(elev),
            np.mean(rlock), np.mean(ramp_dist))


hf, barr, elev, rlock, rdist = run(active_seekers=2)
hf1, barr1, elev1, rlock1, _ = run(active_seekers=1)
hf_static, _, _, _, _ = run(active_seekers=2, disable_hider=True)
env.close()

print(f"Stage 7 eval ({N} eps, room 1v2 + ramp, weights={PREFIX}_*{SUFFIX}.pt):")
print(f"BOTH SEEKERS:")
print(f"  hidden-fraction (play phase):          {hf:.1%}")
print(f"  rung 1 - doorway barricaded at end:    {barr:.1%}")
print(f"  rung 2 - seeker elevated (play steps): {elev:.1%}")
print(f"  rung 3 - ramp hider-locked at end:     {rlock:.1%}")
print(f"           ramp dist from doorway at end: {rdist:.0f}px")
print(f"SINGLE SEEKER (same policies):")
print(f"  hidden-fraction:                       {hf1:.1%}")
print(f"  barricaded / elevated / ramp-locked:   {barr1:.1%} / {elev1:.1%} / {rlock1:.1%}")
print(f"COUNTERFACTUAL (hider disabled, both seekers):")
print(f"  hidden-fraction from geometry alone:   {hf_static:.1%}")
print(f"  => hider's active contribution:        {hf - hf_static:+.1%}")
