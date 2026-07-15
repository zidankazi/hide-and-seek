"""
Behavioral eval for the 2v2 scale-up run. Loads hs_2v2_hider.pt / hs_2v2_seeker.pt
(one shared policy per team, best-vs-best) and, over N episodes, measures whether the
hider team actually builds cover:
  - team hidden-fraction during the play phase (hidden = NO seeker sees ANY hider)
  - episodes where the hider team locked at least one box
  - mean number of hider-locked boxes at episode end (fort size proxy)
  - counterfactual: both hiders disabled (boxes never move) -> hidden-fraction from
    geometry alone; the delta is the hider team's active contribution

Usage: python eval_hs2.py [N] [--final] [--prefix=hs_2v2c]
(default 200 eps, prefix hs_2v2; --final loads the end-of-run *_final.pt weights
instead of the save-best ones, which can saturate early)
"""
import sys
import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_continuous import ActorCritic

digits = [a for a in sys.argv[1:] if a.isdigit()]
N = int(digits[0]) if digits else 200
SUFFIX = "_final" if "--final" in sys.argv else ""
PREFIX = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--prefix=")), "hs_2v2")
TEAM_SIZE = 2
N_BOXES = 6

env = HideAndSeekEnv(layout="open", team_size=TEAM_SIZE, n_boxes=N_BOXES)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

nets = {}
for team in ("hider", "seeker"):
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load(f"{PREFIX}_{team}{SUFFIX}.pt"))
    ac.eval()
    nets[team] = ac


def act(name, obs):
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = nets[env.team[name]](t)
    return mean.squeeze(0).numpy()


def run(disable_hiders):
    hidden_fracs, locked_any, locked_count = [], [], []
    for ep in range(N):
        obs, _ = env.reset(seed=ep)
        play, hidden = 0, 0
        locked_this_ep = False
        while env.agents:
            actions = {}
            for name in env.possible_agents:
                a = act(name, obs[name])
                if env.team[name] == "hider" and disable_hiders:
                    a = np.zeros(act_dim, dtype=np.float32)  # hiders do nothing, boxes stay
                actions[name] = a
            obs, rewards, terms, truncs, infos = env.step(actions)
            info = infos["hider_0"]
            if not info["in_prep"]:
                play += 1
                if not info["seeker_sees_hider"]:
                    hidden += 1
            if any(o == "hider" for o in env.box_lock_owner):
                locked_this_ep = True
        hidden_fracs.append(hidden / play if play else 0.0)
        locked_any.append(locked_this_ep)
        locked_count.append(sum(o == "hider" for o in env.box_lock_owner))
    return (np.mean(hidden_fracs), np.mean(locked_any), np.mean(locked_count))


hf, lock_rate, lock_n = run(disable_hiders=False)
hf_static, _, _ = run(disable_hiders=True)
env.close()

print(f"Trained hider team vs seeker team ({N} eps, 2v2 open, {N_BOXES} boxes):")
print(f"  team hidden-fraction (play phase):     {hf:.1%}")
print(f"  episodes with a hider-locked box:      {lock_rate:.1%}")
print(f"  hider-locked boxes at episode end:     {lock_n:.2f} (of {N_BOXES})")
print(f"Counterfactual (both hiders disabled, boxes never move):")
print(f"  hidden-fraction from geometry alone:   {hf_static:.1%}")
print(f"  => hider team's active contribution:   {hf - hf_static:+.1%}")
