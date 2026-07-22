"""
Reverse-curriculum vectorized recurrent trainer — the one untried lever for rung 1.

Every prior rung-1 attempt used the doorway assist at a FIXED, low mixture (8% sealed /
12% placed / 20% near / 60% nothing), constant for the whole run. That is a static
mixture, not a curriculum: 60% of episodes get zero help from step one, and the precise
box-into-doorway push is never randomly sampled, so PPO gets ~zero gradient toward it.

This trainer instead ANNEALS the assist. Early on ~90% of episodes spawn box 0 in or just
above the doorway (the barricade's endpoint), and the hider only has to press lock / push a
little. As training proceeds, BOTH the assist probability AND the push distance anneal to
zero, so the hider must learn the push chain backward — from "just lock it" to "push it all
the way from the default spawn and lock" — until it barricades with no help at all. That is
the honest-eval condition, so if backward chaining works, rung 1 shows up unassisted.

Warm-started from the 120M policies (strong evasion + rung 2 already in place), so the
reverse curriculum only has to add the construction skill on top.

Usage: python train_hs7_lstm_vec_rc.py [total] [rc_end] [_unused]
                                       [--envs=N] [--steps=per_env] [--hidden=N] [--load=prefix]
  total  = total steps per team (default 50M)
  rc_end = step at which the door assist fully anneals to zero (default 30M)
"""

import copy
import random
import sys

import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_recurrent import RecurrentPPO, ActorCriticLSTM, EpisodeBuffer

torch.set_num_threads(max(1, __import__("os").cpu_count() - 1))


# ---- config ----
SAVE_PREFIX = "hs_lstmrc"                     # NOTE: distinct prefix — never clobbers shipped hs_lstm_*
N_ENVS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--envs=")), 10))
STEPS_PER_ENV = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--steps=")), 800))
_ints = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
TOTAL_TIMESTEPS = _ints[0] if len(_ints) > 0 else 50_000_000
RC_END = _ints[1] if len(_ints) > 1 else 30_000_000   # door assist anneals 1->0 over [0, RC_END]
HIDDEN = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--hidden=")), 256))
LOAD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--load=")), None)
ENTROPY_COEF = 0.005
TEAMS = ("hider", "seeker")
P2 = 1.0                                       # both seekers active from step 1 (warm-started base already is)

# reverse-curriculum spawn geometry (room+ramp): doorway is at y~228; the box's default
# unassisted spawn sits at y~155. Assisted push_y interpolates door(222) -> default(155).
DOOR_Y = 222.0
FAR_Y = 155.0
HONEST_FRAC = 0.18                              # always-unassisted episodes (honest practice, keeps evasion/rung2 alive)


def make_env():
    return HideAndSeekEnv(layout="room", ramp=True, max_steps=360, lock_mode="level",
                          n_hiders=1, n_seekers=2, box_mass=2, door_box_size=72)


envs = [make_env() for _ in range(N_ENVS)]
e0 = envs[0]
possible = e0.possible_agents
obs_dim = e0.observation_space(possible[0]).shape[0]
act_dim = e0.action_space(possible[0]).shape[0]
members = {t: e0.teams[t] for t in TEAMS}
h0 = members["hider"][0]
ZERO = np.zeros(act_dim, dtype=np.float32)
print(f"[rc] {N_ENVS} envs x {STEPS_PER_ENV} steps = {N_ENVS*STEPS_PER_ENV}/update | "
      f"obs={obs_dim} steps={TOTAL_TIMESTEPS} rc_end={RC_END} hidden={HIDDEN}")

live = {t: RecurrentPPO(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
if LOAD:
    for t in TEAMS:
        live[t].ac.load_state_dict(torch.load(f"{LOAD}_{t}.pt"))
    print(f"[rc] warm-started both teams from {LOAD}_*.pt")
frozen = {t: ActorCriticLSTM(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def rc_phase(steps):
    """Training progress p in [0,1] over [0, RC_END] and a phase label, for logging."""
    p = steps / RC_END if RC_END > 0 else 1.0
    if p >= 1.0:
        return p, "honest"
    return p, ("A/prelock" if p < 0.35 else "B/lock" if p < 0.60 else "C/push")


def reset_options(steps):
    on_ramp = random.random() < 0.10
    r = random.random()
    opts = {"active_seekers": 2 if random.random() < P2 else 1, "ramp_active": True,
            "seeker_on_ramp": on_ramp,
            "ramp_locked": (not on_ramp) and r < 0.05,
            "hider_on_ramp": (not on_ramp) and 0.05 <= r < 0.10}
    # 3-PHASE reverse curriculum on the barricade — decouple the two atoms the fixed-mixture
    # runs never taught: (A) sealed-state VALUE + lock-press with the box prelocked; (B) the
    # LOCK-PRESS itself, box handed to the hider in the doorway unlocked, its only job to lock
    # it; (C) the PUSH, box distance grown door->far. Assist fades to zero by RC_END, so the
    # end state is the honest no-assist condition. A constant HONEST_FRAC of episodes are
    # unassisted throughout so evasion/rung 2 don't decay and honest play stays in-distribution.
    p, _ = rc_phase(steps)
    if p >= 1.0 or random.random() < HONEST_FRAC:
        return opts
    if p < 0.35:                                  # Phase A: prelock warmup
        opts["door_push_y"] = DOOR_Y
        opts["hider_at_door"] = True
        opts["door_prelock"] = random.random() < 0.70
    elif p < 0.60:                                # Phase B: lock-press drill (the missing atom)
        opts["door_push_y"] = DOOR_Y
        opts["hider_at_door"] = random.random() < 0.85
    else:                                         # Phase C: grow the push distance
        q = (p - 0.60) / 0.40                     # 0 at p=0.60 -> 1 at p=1.0
        opts["door_push_y"] = DOOR_Y - (DOOR_Y - FAR_Y) * q
        opts["hider_at_door"] = random.random() < (1.0 - q) * 0.7
    return opts


def zero_hidden_slots(hc, rows):
    for row in rows:
        hc[0][:, row, :] = 0.0
        hc[1][:, row, :] = 0.0


# ---- main loop ----
best_mean_return = {t: float("-inf") for t in TEAMS}
steps_done = {t: 0 for t in TEAMS}
episode_returns = {t: [] for t in TEAMS}
hidden_frac = {t: [] for t in TEAMS}
barricade_eps, elev_fracs, ramp_lock_eps = [], [], []
iteration = 0

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in TEAMS:
        opponent = "seeker" if learner == "hider" else "hider"
        Ml, Mo = members[learner], members[opponent]
        nl, no = len(Ml), len(Mo)
        frozen[opponent].load_state_dict(random.choice(pools[opponent]))

        buffers = [[EpisodeBuffer() for _ in Ml] for _ in range(N_ENVS)]
        ep_ret = [0.0] * N_ENVS
        play = [0] * N_ENVS
        hid = [0] * N_ENVS
        ele = [0] * N_ENVS

        obs = []
        for e in range(N_ENVS):
            o, _ = envs[e].reset(options=reset_options(steps_done[learner]))
            obs.append(o)
        hc_l = live[learner].ac.init_hidden(N_ENVS * nl)
        hc_o = frozen[opponent].init_hidden(N_ENVS * no)

        for _ in range(STEPS_PER_ENV):
            lob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Ml]).astype(np.float32)
            la, llp, lval, hc_l = live[learner].ac.act_batch(torch.from_numpy(lob), hc_l)
            la = la.numpy(); llp = llp.numpy(); lval = lval.numpy()
            oob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Mo]).astype(np.float32)
            oa, _, _, hc_o = frozen[opponent].act_batch(torch.from_numpy(oob), hc_o)
            oa = oa.numpy()

            done_rows = []
            for e in range(N_ENVS):
                env = envs[e]
                acts = {}
                for mi, m in enumerate(Ml):
                    row = e * nl + mi
                    acts[m] = ZERO if m in env._dormant else np.clip(la[row], -1.0, 1.0)
                for mi, m in enumerate(Mo):
                    row = e * no + mi
                    acts[m] = ZERO if m in env._dormant else np.clip(oa[row], -1.0, 1.0)
                nobs, rew, terms, truncs, infos = env.step(acts)
                done = any(terms.values()) or any(truncs.values())

                for mi, m in enumerate(Ml):
                    if m not in env._dormant:
                        row = e * nl + mi
                        buffers[e][mi].store(obs[e][m], la[row], float(llp[row]),
                                             rew[m], float(lval[row]))
                info = infos[h0]
                ep_ret[e] += rew[Ml[0]]
                if not info["in_prep"]:
                    play[e] += 1
                    hid[e] += (not info["seeker_sees_hider"])
                    ele[e] += info["seeker_elevated"]

                if done:
                    for mi in range(nl):
                        if buffers[e][mi].has_open():
                            buffers[e][mi].end_episode(0.0)
                    episode_returns[learner].append(ep_ret[e])
                    if play[e] > 0:
                        hidden_frac[learner].append(hid[e] / play[e])
                        elev_fracs.append(ele[e] / play[e])
                    barricade_eps.append(info["doorway_barricaded"])
                    ramp_lock_eps.append(info["ramp_lock_owner"] == "hider")
                    ep_ret[e] = 0.0; play[e] = hid[e] = ele[e] = 0
                    nobs, _ = env.reset(options=reset_options(steps_done[learner]))
                    done_rows.append(e)
                obs[e] = nobs
                steps_done[learner] += 1

            if done_rows:
                zero_hidden_slots(hc_l, [e * nl + mi for e in done_rows for mi in range(nl)])
                zero_hidden_slots(hc_o, [e * no + mi for e in done_rows for mi in range(no)])

        for e in range(N_ENVS):
            for mi, m in enumerate(Ml):
                if not buffers[e][mi].has_open():
                    continue
                if m in envs[e]._dormant:
                    buffers[e][mi].end_episode(0.0)
                else:
                    row = e * nl + mi
                    hc_slot = (hc_l[0][:, row:row + 1, :].contiguous(),
                               hc_l[1][:, row:row + 1, :].contiguous())
                    ot = torch.tensor(obs[e][m], dtype=torch.float32).unsqueeze(0)
                    lv = live[learner].ac.value_only(ot, hc_slot).item()
                    buffers[e][mi].end_episode(lv)

        merged = EpisodeBuffer()
        for e in range(N_ENVS):
            for mi in range(nl):
                merged.episodes.extend(buffers[e][mi].episodes)
        live[learner].update(merged, entropy_coef=ENTROPY_COEF, episodes_per_batch=8)
        with torch.no_grad():
            lo = -3.0 if learner == "hider" else -1.4
            live[learner].ac.log_std.clamp_(min=lo, max=0.0)

    # ---- logging + save-best + snapshot ----
    lsteps = min(steps_done.values())
    _p, _ph = rc_phase(lsteps)
    parts = [f"Iter {iteration}", f"Steps {lsteps}",
             f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}",
             f"ph={_ph}({_p:.2f})",
             "std " + " ".join(f"{t[0]}={float(live[t].ac._std().mean().item()):.2f}" for t in TEAMS)]
    for t in TEAMS:
        if not episode_returns[t]:
            continue
        mean_r = float(np.mean(episode_returns[t]))
        improved = mean_r > best_mean_return[t]
        if improved:
            best_mean_return[t] = mean_r
            torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}.pt")
        hf = float(np.mean(hidden_frac[t])) if hidden_frac[t] else float("nan")
        parts.append(f"{t}={mean_r:+6.3f}{'*' if improved else ' '} hid={hf:.2f} ({len(episode_returns[t])}ep)")
        episode_returns[t] = []
        hidden_frac[t] = []
    barr = float(np.mean(barricade_eps)) if barricade_eps else float("nan")
    elev = float(np.mean(elev_fracs)) if elev_fracs else float("nan")
    rlock = float(np.mean(ramp_lock_eps)) if ramp_lock_eps else float("nan")
    parts.append(f"barr={barr:.2f} elev={elev:.2f} rlock={rlock:.2f}")
    barricade_eps, elev_fracs, ramp_lock_eps = [], [], []

    if iteration % 20 == 0:
        for t in TEAMS:
            pools[t].append(copy.deepcopy(live[t].ac.state_dict()))
            if len(pools[t]) > 30:
                pools[t].pop(0)

    print(" | ".join(parts))

for t in TEAMS:
    torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}_final.pt")
for e in envs:
    e.close()
