"""
Stage 5c self-play trainer: 2v2 hide-and-seek at scale (open arena, 6 boxes, 10M env
steps per team, pure LOS reward — no shaping).

Same quality-ladder structure as train_hs.py, generalized to teams:
  - ONE policy per team, shared by both teammates (paper-style parameter sharing). Each
    teammate acts from its own masked obs; both teammates' transitions train the policy.
  - GAE must walk a single agent's trajectory, so each teammate gets its OWN rollout
    buffer; per-member advantages/returns are computed separately and concatenated for
    the minibatch update (update_team below — the loop body matches PPO.update).
  - Opponent pool is per-TEAM: one frozen net drives both opponent members, sampled from
    the opponent team's pool each episode.

Reward reminder (team LOS, play phase only): seekers get +1/PLAY_STEPS when ANY seeker
sees ANY hider, hiders get it when ALL are unseen. Teammates share the reward exactly, so
per-member episode returns are identical and we track member 0's.

Saves best to hs_2v2_hider.pt / hs_2v2_seeker.pt. Stage 4/5b policies untouched.
Usage: python train_hs2.py [total_env_steps_per_team]   (default 10M; small value = smoke test)
"""

import copy
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from env_hs import HideAndSeekEnv
from ppo_continuous import PPO, ActorCritic, RolloutBuffer


# ---- config ----
TEAM_SIZE = 2
N_BOXES = 6
LAYOUT = "open"
SAVE_PREFIX = "hs_2v2"
ROLLOUT_STEPS = 2048          # env steps per rollout (each yields TEAM_SIZE transitions)
TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000  # env steps per team
ENTROPY_COEF = 0.02
TEAMS = ("hider", "seeker")


# ---- setup ----
env = HideAndSeekEnv(layout=LAYOUT, team_size=TEAM_SIZE, n_boxes=N_BOXES)
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]
members = {t: env.teams[t] for t in TEAMS}
print(f"[train_hs2] {TEAM_SIZE}v{TEAM_SIZE} layout={LAYOUT} boxes={N_BOXES} "
      f"obs={obs_dim} steps={TOTAL_TIMESTEPS} saving to {SAVE_PREFIX}_*.pt")

live = {t: PPO(obs_dim, act_dim) for t in TEAMS}
team_buffers = {t: {m: RolloutBuffer() for m in members[t]} for t in TEAMS}
frozen_net = ActorCritic(obs_dim, act_dim)
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def sample_opponent(learner):
    opponent = "seeker" if learner == "hider" else "hider"
    frozen_net.load_state_dict(random.choice(pools[opponent]))


def frozen_action(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = frozen_net.act(obs_t)
    return action.squeeze(0).numpy()


def update_team(team, last_obs, last_done, gamma=0.99, lam=0.95, clip_eps=0.2,
                entropy_coef=ENTROPY_COEF, value_coef=0.5, update_epochs=4, batch_size=64):
    """PPO.update generalized to a team: GAE per member buffer, one concatenated update."""
    ppo = live[team]
    obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
    for m in members[team]:
        buf = team_buffers[team][m]
        last_obs_t = torch.tensor(last_obs[m], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, last_value = ppo.ac(last_obs_t)
        last_value = 0.0 if last_done else last_value.item()
        adv, ret = buf.compute_returns(last_value, gamma, lam)
        obs_l.append(np.array(buf.obs))
        act_l.append(np.array(buf.actions))
        logp_l.append(np.array(buf.log_probs))
        adv_l.append(adv)
        ret_l.append(ret)
        buf.clear()

    obs = torch.tensor(np.concatenate(obs_l), dtype=torch.float32)
    actions = torch.tensor(np.concatenate(act_l), dtype=torch.float32)
    old_log_probs = torch.tensor(np.concatenate(logp_l), dtype=torch.float32)
    advantages = torch.cat(adv_l)
    returns = torch.cat(ret_l)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(obs)
    for _ in range(update_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            log_probs, entropy, values = ppo.ac.evaluate(obs[idx], actions[idx])
            ratio = torch.exp(log_probs - old_log_probs[idx])
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(ratio * advantages[idx], clip_adv).mean()
            value_loss = nn.functional.mse_loss(values, returns[idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            ppo.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ppo.ac.parameters(), 0.5)
            ppo.optimizer.step()


# ---- main loop ----
best_mean_return = {t: float("-inf") for t in TEAMS}
steps_done = {t: 0 for t in TEAMS}
episode_returns = {t: [] for t in TEAMS}
hidden_frac = {t: [] for t in TEAMS}
iteration = 0

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in TEAMS:
        opponent = "seeker" if learner == "hider" else "hider"
        obs, _ = env.reset()
        sample_opponent(learner)
        episode_return = 0.0
        play_steps = 0
        hidden_steps = 0

        for _ in range(ROLLOUT_STEPS):
            actions, cached = {}, {}
            for m in members[learner]:
                a, log_prob, value = live[learner].select_action(obs[m])
                actions[m] = a
                cached[m] = (a, log_prob, value)
            for m in members[opponent]:
                actions[m] = frozen_action(obs[m])

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())

            for m in members[learner]:
                a, log_prob, value = cached[m]
                team_buffers[learner][m].store(obs[m], a, log_prob, rewards[m], done, value)

            m0 = members[learner][0]
            episode_return += rewards[m0]
            info = infos[m0]
            if not info["in_prep"]:
                play_steps += 1
                if not info["seeker_sees_hider"]:
                    hidden_steps += 1

            obs = next_obs
            steps_done[learner] += 1

            if done:
                episode_returns[learner].append(episode_return)
                if play_steps > 0:
                    hidden_frac[learner].append(hidden_steps / play_steps)
                episode_return = 0.0
                play_steps = 0
                hidden_steps = 0
                obs, _ = env.reset()
                sample_opponent(learner)

        update_team(learner, obs, done)

    # ---- logging + save-best + paired snapshot ----
    parts = [f"Iter {iteration}", f"Steps {min(steps_done.values())}"]
    parts.append(f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}")

    improved_any = False
    for t in TEAMS:
        if not episode_returns[t]:
            continue
        mean_r = float(np.mean(episode_returns[t]))
        improved = mean_r > best_mean_return[t]
        if improved:
            best_mean_return[t] = mean_r
            torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}.pt")
            improved_any = True
        hf = float(np.mean(hidden_frac[t])) if hidden_frac[t] else float("nan")
        tag = "*" if improved else " "
        parts.append(f"{t}={mean_r:+6.3f}{tag} hid={hf:.2f} ({len(episode_returns[t])}ep)")
        episode_returns[t] = []
        hidden_frac[t] = []

    if improved_any:
        for t in TEAMS:
            pools[t].append(copy.deepcopy(live[t].ac.state_dict()))

    print(" | ".join(parts))

# Also save the FINAL live weights. Save-best can saturate early (a team that scores a
# perfect mean vs the weak early opponent pool can never "improve" again — the 10M run's
# seeker hit +1.000 at iter 4 and froze there), so the end-of-run policies are the only
# faithful "fully trained" snapshot.
for t in TEAMS:
    torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}_final.pt")

env.close()
