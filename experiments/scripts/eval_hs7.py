"""Evaluate saved feed-forward or LSTM policies with optional climbing mechanics.

Reports ramp proximity, elevation-dependent sightlines, wall-entry events, and
ramp displacement. These are separate measurements, not a test of a complete strategy.
Use the root evaluate.py for the current result tables.

Usage: python eval_hs7.py [episodes] [--prefix=hs_ramp] [--final]
       [--layout=room|roomt] [--nh=1] [--ns=2] [--speed=1.0]
       [--climb] [--no-xray] [--climb-steps=45]"""
import os
import sys

import numpy as np
import torch

from hide_and_seek.env_hs import HideAndSeekEnv


def arg(name, default, cast=str):
    hit = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return cast(hit) if hit is not None else default


digits = [a for a in sys.argv[1:] if a.isdigit()]
N = int(digits[0]) if digits else 200
SUFFIX = "_final" if "--final" in sys.argv else ""
PREFIX = arg("prefix", "hs_ramp")
LAYOUT = arg("layout", "room")
NH, NS = arg("nh", 1, int), arg("ns", 2, int)
SPEED = arg("speed", 1.0, float)
CLIMB = "--climb" in sys.argv
XRAY = "--no-xray" not in sys.argv
CLIMB_STEPS = arg("climb-steps", 45, int)

env = HideAndSeekEnv(layout=LAYOUT, ramp=True, max_steps=360, lock_mode="level",
                     n_hiders=NH, n_seekers=NS, box_mass=2, door_box_size=72,
                     seeker_speed_mult=SPEED, ramp_climb=CLIMB, ramp_xray=XRAY,
                     climb_teams=("seeker",), climb_steps=CLIMB_STEPS)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]
HIDERS = env.teams["hider"]
H0 = HIDERS[0]
# derive the doorway from the layout instead of hardcoding the "room" values, so roomt's
# ramp-distance number is right too
DOORWAY_CENTER = np.array([float(env._door_cx), float(env._door_y)])


# load policies, auto-detecting architecture from the checkpoint keys
def ckpt(team):
    p = f"{PREFIX}_{team}{SUFFIX}.pt"
    if SUFFIX and not os.path.exists(p):
        p = f"{PREFIX}_{team}.pt"          # fall back to save-best if no _final exists
    return p


sd0 = torch.load(ckpt("hider"))
IS_LSTM = "enc.0.weight" in sd0
if IS_LSTM:
    from hide_and_seek.ppo_recurrent import ActorCriticLSTM

    def build(sd):
        ac = ActorCriticLSTM(obs_dim, act_dim, hidden=sd["enc.0.weight"].shape[0])
        ac.load_state_dict(sd); ac.eval(); return ac
else:
    from hide_and_seek.ppo_continuous import ActorCritic

    def build(sd):
        ac = ActorCritic(obs_dim, act_dim, hidden=sd["shared.0.weight"].shape[0])
        ac.load_state_dict(sd); ac.eval(); return ac


nets = {"hider": build(sd0), "seeker": build(torch.load(ckpt("seeker")))}
_hc = {}


def reset_hidden():
    for n in env.possible_agents:
        _hc[n] = nets[env.team[n]].init_hidden() if IS_LSTM else None


def act(name, obs):
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if IS_LSTM:
            mean, _, _hc[name] = nets[env.team[name]].step(t, _hc[name])
        else:
            mean, _ = nets[env.team[name]](t)
    return mean.squeeze(0).numpy()


def sightline_now():
    """Is a seeker's elevation what reveals a hider right now?

    `seeker_elevated` only means "within RAMP_USE_DIST of the ramp". This asks the stricter
    question the rung-2 claim actually makes: the seeker sees a hider, and would not see it
    standing on the ground. Same LOS query run twice, once with elevation suppressed.
    """
    saved = env._elevated_now
    for s in env.teams["seeker"][:env.active_seekers]:
        if s not in saved:
            continue
        for h in HIDERS:
            hb = env.bodies[h]
            if not env._visible(s, hb):
                continue
            env._elevated_now = set()
            try:
                blocked = not env._visible(s, hb)
            finally:
                env._elevated_now = saved
            if blocked:
                return True
    return False


def run(active_seekers=2, disable_hider=False):
    acc = {k: [] for k in ("hidden", "barricade", "near", "sight", "climbed",
                           "rlock", "ramp_dist", "ramp_moved")}
    for ep in range(N):
        obs, _ = env.reset(seed=ep, options={"ramp_active": True,
                                             "active_seekers": active_seekers})
        reset_hidden()
        ramp0 = np.array([env.ramp_body.position.x, env.ramp_body.position.y])
        play = hid = near = sight = 0
        info = None
        while env.agents:
            actions = {}
            for name in env.possible_agents:
                actions[name] = (np.zeros(act_dim, dtype=np.float32)
                                 if (disable_hider and env.team[name] == "hider")
                                 else act(name, obs[name]))
            obs, rewards, terms, truncs, infos = env.step(actions)
            info = infos[H0]
            if not info["in_prep"]:
                play += 1
                hid += (not info["seeker_sees_hider"])
                near += info["seeker_elevated"]
                sight += sightline_now()
        rampf = np.array([env.ramp_body.position.x, env.ramp_body.position.y])
        acc["hidden"].append(hid / play if play else 0.0)
        acc["barricade"].append(info["doorway_barricaded"])
        acc["near"].append(near / play if play else 0.0)
        acc["sight"].append(sight / play if play else 0.0)
        acc["climbed"].append(info.get("seeker_climbed", False))
        acc["rlock"].append(info["ramp_lock_owner"] == "hider")
        acc["ramp_dist"].append(float(np.linalg.norm(rampf - DOORWAY_CENTER)))
        acc["ramp_moved"].append(float(np.linalg.norm(rampf - ramp0)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


both = run(active_seekers=NS)
single = run(active_seekers=1)
static = run(active_seekers=NS, disable_hider=True)
env.close()

arch = "LSTM" if IS_LSTM else "FF"
mech = f"climb={'on' if CLIMB else 'off'} xray={'on' if XRAY else 'off'}"
if CLIMB:
    mech += f" climb_steps={CLIMB_STEPS}"
print(f"Stage 7 eval ({N} eps, {LAYOUT} {NH}v{NS}, speed={SPEED}x, {mech}, "
      f"arch={arch}, weights={PREFIX}_*{SUFFIX}.pt):")
print(f"BOTH SEEKERS:")
print(f"  hidden-fraction (play phase):          {both['hidden']:.1%}")
print(f"  rung 1 - doorway barricaded at end:    {both['barricade']:.1%}")
print(f"  rung 2 - near-ramp (legacy proxy):     {both['near']:.1%}")
if not XRAY:
    print(f"           sightline (ramp bought it):   n/a (ramp_xray off)")
elif both["near"] > 0:
    conv = both["sight"] / both["near"]
    print(f"           sightline (ramp bought it):   {both['sight']:.1%}"
          f"   [{conv:.0%} of near-ramp is real]")
else:
    print(f"           sightline (ramp bought it):   {both['sight']:.1%}")
if CLIMB:
    print(f"           climbed a wall (episodes):    {both['climbed']:.1%}")
print(f"  rung 3 - ramp hider-locked at end:     {both['rlock']:.1%}")
print(f"           ramp dist from doorway at end: {both['ramp_dist']:.0f}px")
print(f"           ramp moved from spawn:         {both['ramp_moved']:.0f}px")
print(f"SINGLE SEEKER (same policies):")
print(f"  hidden-fraction:                       {single['hidden']:.1%}")
print(f"  barricaded / near-ramp / ramp-locked:  "
      f"{single['barricade']:.1%} / {single['near']:.1%} / {single['rlock']:.1%}")
if CLIMB:
    print(f"  climbed:                               {single['climbed']:.1%}")
print(f"COUNTERFACTUAL (hider disabled, both seekers):")
print(f"  hidden-fraction from geometry alone:   {static['hidden']:.1%}")
print(f"  => hider's active contribution:        {both['hidden'] - static['hidden']:+.1%}")
