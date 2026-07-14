"""
Behavioral eval for Stage 5b. Loads hs_hider.pt / hs_seeker.pt and, over N episodes,
measures not just hidden-fraction but WHETHER the hider is actually using tools:
  - lock events (did the hider lock a box during prep?)
  - min box->doorway distance at end of prep (did a box get pushed into the doorway?)
  - hidden-fraction during the play phase

Also runs a counterfactual: the same seeker vs a "frozen-box" hider (hider policy disabled,
boxes never move) to see how much of the hiding is the walls alone vs the hider's doing.
"""
import sys
import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_continuous import ActorCritic

# Usage: python eval_hs.py [layout=room|open] [N]
LAYOUT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "room"
N = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 200
PREFIX = "hs" if LAYOUT == "room" else "hs_open"
DOOR = np.array([180.0, 240.0])  # room doorway center (gap x in [150,210] at y=240)

env = HideAndSeekEnv(layout=LAYOUT)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

nets = {}
for name in env.possible_agents:
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load(f"{PREFIX}_{name}.pt"))
    ac.eval()
    nets[name] = ac


def act(name, obs):
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, _ = nets[name](t)
    return mean.squeeze(0).numpy()


def run(disable_hider):
    hidden_fracs, locked_any, door_dists, ever_locked_door = [], [], [], []
    for ep in range(N):
        obs, _ = env.reset(seed=ep)
        play, hidden = 0, 0
        locked_this_ep = False
        door_at_prep_end = None
        while env.agents:
            actions = {}
            for name in env.possible_agents:
                a = act(name, obs[name])
                if name == "hider" and disable_hider:
                    a = np.zeros(act_dim, dtype=np.float32)  # hider does nothing, boxes stay
                actions[name] = a
            obs, rewards, terms, truncs, infos = env.step(actions)
            info = infos["hider"]
            if not info["in_prep"]:
                play += 1
                if not info["seeker_sees_hider"]:
                    hidden += 1
            if any(o is not None for o in env.box_lock_owner):
                locked_this_ep = True
            if env.steps == env.PREP_STEPS:  # just crossed into play
                dmin = min(np.linalg.norm(np.array(tuple(b.position)) - DOOR)
                           for b in env.box_bodies)
                door_at_prep_end = dmin
        hidden_fracs.append(hidden / play if play else 0.0)
        locked_any.append(locked_this_ep)
        if door_at_prep_end is not None:
            door_dists.append(door_at_prep_end)
    return (np.mean(hidden_fracs), np.mean(locked_any), np.mean(door_dists), np.min(door_dists))


hf, lock_rate, door_mean, door_min = run(disable_hider=False)
hf_static, _, door_mean_s, _ = run(disable_hider=True)
env.close()

print(f"Trained hider vs seeker ({N} eps, layout={LAYOUT}):")
print(f"  hidden-fraction (play phase):     {hf:.1%}")
print(f"  episodes where a box got locked:  {lock_rate:.1%}")
if LAYOUT == "room":
    print(f"  min box->doorway dist @ prep end: mean {door_mean:.0f}px, best {door_min:.0f}px  (doorway gap ~30px)")
print(f"Counterfactual (hider disabled, boxes never move):")
print(f"  hidden-fraction from walls alone: {hf_static:.1%}")
print(f"  => hider's active contribution:   {hf - hf_static:+.1%}")
