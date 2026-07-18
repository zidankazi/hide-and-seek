"""
Stage 7 recurrent trainer: 1v2 room+ramp with an LSTM policy — the paper's rung-1 recipe.

Same game, env, curriculum, assists, and live opponent ladder as train_hs7.py; the only
change is the policy is recurrent (ppo_recurrent.RecurrentPPO) so it carries memory across
the episode. Rollouts are collected per-episode with a fresh hidden state, and the update
backprops through each whole episode (see ppo_recurrent for the correctness argument).

Both learner and frozen-opponent nets are recurrent; hidden state is maintained per agent
and reset at every episode boundary (new_episode) and whenever a new frozen opponent is
sampled.

Usage: python train_hs7_lstm.py [total_steps] [s2_start] [s2_end] [--hidden=N] [--rollout=N]
       (fresh nets by default; long run recommended, e.g. 40M+)
"""

import copy
import random
import sys

import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_recurrent import RecurrentPPO, ActorCriticLSTM, EpisodeBuffer


# ---- config ----
SAVE_PREFIX = "hs_lstm"
ROLLOUT_STEPS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rollout=")), 8192))
SNAPSHOT_EVERY = 100
POOL_CAP = 30
_ints = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
TOTAL_TIMESTEPS = _ints[0] if len(_ints) > 0 else 40_000_000
S2_START = _ints[1] if len(_ints) > 1 else 6_000_000
S2_END = _ints[2] if len(_ints) > 2 else 12_000_000
HIDDEN = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--hidden=")), 256))
ENTROPY_COEF = 0.005
TEAMS = ("hider", "seeker")


# ---- setup ----
env = HideAndSeekEnv(layout="room", ramp=True, max_steps=360, lock_mode="level",
                     n_hiders=1, n_seekers=2, box_mass=2, door_box_size=72)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]
members = {t: env.teams[t] for t in TEAMS}
h0 = members["hider"][0]
print(f"[train_hs7_lstm] 1v2 room+ramp LSTM obs={obs_dim} steps={TOTAL_TIMESTEPS} "
      f"s2={S2_START}-{S2_END} hidden={HIDDEN} rollout={ROLLOUT_STEPS} -> {SAVE_PREFIX}_*.pt")

live = {t: RecurrentPPO(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
frozen = {t: ActorCriticLSTM(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def second_seeker_prob(steps):
    if S2_END <= S2_START:
        return 1.0
    return min(1.0, max(0.0, (steps - S2_START) / (S2_END - S2_START)))


def load_frozen(opponent):
    frozen[opponent].load_state_dict(random.choice(pools[opponent]))


# ---- main loop ----
best_mean_return = {t: float("-inf") for t in TEAMS}
steps_done = {t: 0 for t in TEAMS}
episode_returns = {t: [] for t in TEAMS}
hidden_frac = {t: [] for t in TEAMS}
barricade_eps, elev_fracs, ramp_lock_eps = [], [], []
iteration = 0
ZERO = np.zeros(act_dim, dtype=np.float32)

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in TEAMS:
        opponent = "seeker" if learner == "hider" else "hider"
        p2 = second_seeker_prob(steps_done[learner])
        buffers = {m: EpisodeBuffer() for m in members[learner]}
        # per-agent LSTM hidden state (reset each episode)
        hc_learn = {m: live[learner].ac.init_hidden() for m in members[learner]}
        hc_frozen = {m: frozen[opponent].init_hidden() for m in members[opponent]}

        def new_episode():
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
            o, _ = env.reset(options={
                "active_seekers": active, "ramp_active": True, "seeker_on_ramp": on_ramp,
                "ramp_locked": not on_ramp and r < 0.05,
                "hider_on_ramp": not on_ramp and 0.05 <= r < 0.10,
                "doorway_box": door, "hider_at_door": at_door,
            })
            load_frozen(opponent)
            for m in members[learner]:
                hc_learn[m] = live[learner].ac.init_hidden()
            for m in members[opponent]:
                hc_frozen[m] = frozen[opponent].init_hidden()
            return o

        obs = new_episode()
        ep_ret = 0.0
        play_steps = hidden_steps = elev_steps = 0

        for _ in range(ROLLOUT_STEPS):
            actions, cached = {}, {}
            for m in members[learner]:
                if m in env._dormant:
                    actions[m] = ZERO
                    continue
                ot = torch.tensor(obs[m], dtype=torch.float32).unsqueeze(0)
                a, lp, v, hc_learn[m] = live[learner].ac.act(ot, hc_learn[m])
                a_np = a.squeeze(0).numpy()
                actions[m] = np.clip(a_np, -1.0, 1.0)
                cached[m] = (obs[m], a_np, lp.item(), v.item())
            for m in members[opponent]:
                if m in env._dormant:
                    actions[m] = ZERO
                    continue
                ot = torch.tensor(obs[m], dtype=torch.float32).unsqueeze(0)
                a, _, _, hc_frozen[m] = frozen[opponent].act(ot, hc_frozen[m])
                actions[m] = np.clip(a.squeeze(0).numpy(), -1.0, 1.0)

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())

            for m, (o_m, a_m, lp_m, v_m) in cached.items():
                buffers[m].store(o_m, a_m, lp_m, rewards[m], v_m)

            ep_ret += rewards[members[learner][0]]
            info = infos[h0]
            if not info["in_prep"]:
                play_steps += 1
                hidden_steps += (not info["seeker_sees_hider"])
                elev_steps += info["seeker_elevated"]

            obs = next_obs
            steps_done[learner] += 1

            if done:
                for m in members[learner]:
                    if buffers[m].has_open():
                        buffers[m].end_episode(0.0)   # truncation at horizon -> bootstrap 0
                episode_returns[learner].append(ep_ret)
                if play_steps > 0:
                    hidden_frac[learner].append(hidden_steps / play_steps)
                barricade_eps.append(info["doorway_barricaded"])
                if play_steps > 0:
                    elev_fracs.append(elev_steps / play_steps)
                ramp_lock_eps.append(info["ramp_lock_owner"] == "hider")
                ep_ret = 0.0
                play_steps = hidden_steps = elev_steps = 0
                obs = new_episode()

        # close any episode still open at rollout end: bootstrap with value of last obs
        for m in members[learner]:
            if buffers[m].has_open() and m not in env._dormant:
                ot = torch.tensor(obs[m], dtype=torch.float32).unsqueeze(0)
                lv = live[learner].ac.value_only(ot, hc_learn[m]).item()
                buffers[m].end_episode(lv)
            elif buffers[m].has_open():
                buffers[m].end_episode(0.0)

        # merge per-member episode lists into one buffer and update the team net
        merged = EpisodeBuffer()
        for m in members[learner]:
            merged.episodes.extend(buffers[m].episodes)
        live[learner].update(merged, entropy_coef=ENTROPY_COEF, episodes_per_batch=4)
        with torch.no_grad():
            lo = -3.0 if learner == "hider" else -1.4
            live[learner].ac.log_std.clamp_(min=lo, max=0.0)

    # ---- logging + save-best + periodic snapshot ----
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

    if iteration % SNAPSHOT_EVERY == 0:
        for t in TEAMS:
            pools[t].append(copy.deepcopy(live[t].ac.state_dict()))
            if len(pools[t]) > POOL_CAP:
                pools[t].pop(0)

    print(" | ".join(parts))

for t in TEAMS:
    torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}_final.pt")
env.close()
