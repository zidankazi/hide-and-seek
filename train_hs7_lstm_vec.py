"""
Vectorized recurrent Stage 7 trainer — same game/curriculum/reward as train_hs7_lstm.py,
but runs N env copies in ONE process and does a single BATCHED policy forward per step.

Why this is the speedup: profiling showed the bottleneck is the per-step LSTM forward at
batch-1 (~84% of rollout wall-clock); physics is cheap (~17k steps/s). Batching N envs'
observations into one forward amortizes that cost N-fold. No multiprocessing — just a list
of envs, batched act, and per-env hidden state carried as (1, N*|members|, H) tensors.

One simplification vs the single-env trainer: all N envs face the SAME frozen opponent for
a given rollout (resampled from the live pool each rollout), so the opponent forward also
batches. Opponent diversity is per-rollout instead of per-episode — the pool ladder still
covers the space over training.

Usage: python train_hs7_lstm_vec.py [total_steps] [s2_start] [s2_end]
                                    [--envs=N] [--steps=per_env] [--hidden=N]
"""

import copy
import random
import sys

import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_recurrent import RecurrentPPO, ActorCriticLSTM, EpisodeBuffer

torch.set_num_threads(max(1, __import__("os").cpu_count() - 1))  # use the cores for the update matmuls


# ---- config ----
SAVE_PREFIX = "hs_lstm"
N_ENVS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--envs=")), 12))
STEPS_PER_ENV = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--steps=")), 1000))
_ints = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
TOTAL_TIMESTEPS = _ints[0] if len(_ints) > 0 else 40_000_000   # per team (summed over envs)
S2_START = _ints[1] if len(_ints) > 1 else 6_000_000
S2_END = _ints[2] if len(_ints) > 2 else 12_000_000
HIDDEN = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--hidden=")), 256))
LOAD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--load=")), None)
ENTROPY_COEF = 0.005
TEAMS = ("hider", "seeker")


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
print(f"[vec] {N_ENVS} envs x {STEPS_PER_ENV} steps = {N_ENVS*STEPS_PER_ENV}/update | "
      f"obs={obs_dim} steps={TOTAL_TIMESTEPS} s2={S2_START}-{S2_END} hidden={HIDDEN}")

live = {t: RecurrentPPO(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
if LOAD:
    for t in TEAMS:
        live[t].ac.load_state_dict(torch.load(f"{LOAD}_{t}.pt"))
    print(f"[vec] warm-started both teams from {LOAD}_*.pt")
frozen = {t: ActorCriticLSTM(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def second_seeker_prob(steps):
    if S2_END <= S2_START:
        return 1.0
    return min(1.0, max(0.0, (steps - S2_START) / (S2_END - S2_START)))


def reset_options(p2):
    active = 2 if random.random() < p2 else 1
    on_ramp = random.random() < 0.10
    r = random.random()
    d = random.random()
    if d < 0.08:
        door, at_door = "sealed", False
    elif d < 0.20:
        door, at_door = "placed", True
    elif d < 0.40:
        door, at_door = "near", True
    else:
        door, at_door = None, False
    return {"active_seekers": active, "ramp_active": True, "seeker_on_ramp": on_ramp,
            "ramp_locked": not on_ramp and r < 0.05,
            "hider_on_ramp": not on_ramp and 0.05 <= r < 0.10,
            "doorway_box": door, "hider_at_door": at_door}


def zero_hidden_slots(hc, rows):
    """Zero the (h, c) columns for the given batch rows (envs that just reset)."""
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
        p2 = second_seeker_prob(steps_done[learner])
        Ml, Mo = members[learner], members[opponent]
        nl, no = len(Ml), len(Mo)
        frozen[opponent].load_state_dict(random.choice(pools[opponent]))  # one opponent / rollout

        # per-(env, member) episode buffers + rolling episode accumulators
        buffers = [[EpisodeBuffer() for _ in Ml] for _ in range(N_ENVS)]
        ep_ret = [0.0] * N_ENVS
        play = [0] * N_ENVS
        hid = [0] * N_ENVS
        ele = [0] * N_ENVS

        # reset all envs; per-team batched hidden (1, N*|M|, H)
        obs = []
        for e in range(N_ENVS):
            o, _ = envs[e].reset(options=reset_options(p2))
            obs.append(o)
        hc_l = live[learner].ac.init_hidden(N_ENVS * nl)
        hc_o = frozen[opponent].init_hidden(N_ENVS * no)

        for _ in range(STEPS_PER_ENV):
            # --- batched learner forward over (env, member) ---
            lob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Ml]).astype(np.float32)
            la, llp, lval, hc_l = live[learner].ac.act_batch(torch.from_numpy(lob), hc_l)
            la = la.numpy(); llp = llp.numpy(); lval = lval.numpy()
            # --- batched opponent forward ---
            oob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Mo]).astype(np.float32)
            oa, _, _, hc_o = frozen[opponent].act_batch(torch.from_numpy(oob), hc_o)
            oa = oa.numpy()

            # --- assemble action dicts, step each env ---
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
                    nobs, _ = env.reset(options=reset_options(p2))
                    done_rows.append(e)
                obs[e] = nobs
                steps_done[learner] += 1

            if done_rows:
                zero_hidden_slots(hc_l, [e * nl + mi for e in done_rows for mi in range(nl)])
                zero_hidden_slots(hc_o, [e * no + mi for e in done_rows for mi in range(no)])

        # close episodes still open at rollout end (bootstrap with value of last obs)
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
    parts = [f"Iter {iteration}", f"Steps {lsteps}",
             f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}",
             f"p2={second_seeker_prob(lsteps):.2f}",
             "std " + " ".join(f"{t[0]}={float(live[t].ac._std().mean()):.2f}" for t in TEAMS)]
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
