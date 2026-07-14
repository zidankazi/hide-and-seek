"""
Stage 5b self-play trainer for the hide-and-seek env (HideAndSeekEnv, 1v1).

Structurally identical to train_selfplay.py (quality-ladder opponent pool, snapshot on
improvement, paired snapshots) — the only changes are the env and the save filenames.
We keep the Stage 4 tag policies (hider.pt / seeker.pt) untouched and save here to
hs_hider.pt / hs_seeker.pt.

Reward reminder (LOS, play phase only): each play step the seeker gets +1/PLAY_STEPS when it
can see the hider, the hider gets it when unseen, so a full episode return is in [-1, +1]:
  hider return  = fraction of the play phase spent HIDDEN, minus fraction seen
  seeker return = the negation
So "save best hider" = most-hidden, "save best seeker" = most-seeing. The hider metric
saturates near +1 while the seeker is weak (the room walls hide it by default), so paired
snapshotting captures "the hider that survives against this generation of seekers."
"""

import copy
import random

import numpy as np
import torch

from env_hs import HideAndSeekEnv
from ppo_continuous import PPO, ActorCritic


# ---- config ----
ROLLOUT_STEPS = 2048
TOTAL_TIMESTEPS = 2_000_000   # per agent
ENTROPY_COEF = 0.02


# ---- setup ----
env = HideAndSeekEnv()
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

live = {name: PPO(obs_dim, act_dim) for name in env.possible_agents}
frozen_net = ActorCritic(obs_dim, act_dim)
pools = {
    name: [copy.deepcopy(live[name].ac.state_dict())] for name in env.possible_agents
}


def sample_opponent(learner):
    opponent = "seeker" if learner == "hider" else "hider"
    frozen_net.load_state_dict(random.choice(pools[opponent]))


def frozen_action(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = frozen_net.act(obs_t)
    return action.squeeze(0).numpy()


# ---- main loop ----
best_mean_return = {name: float("-inf") for name in env.possible_agents}
steps_done = {name: 0 for name in env.possible_agents}
episode_returns = {name: [] for name in env.possible_agents}
# Track the hider's hidden-fraction directly (the headline Stage 5b metric): over the play
# phase, the fraction of steps the seeker could NOT see the hider.
hidden_frac = {name: [] for name in env.possible_agents}
iteration = 0

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in env.possible_agents:
        opponent = "seeker" if learner == "hider" else "hider"
        obs, _ = env.reset()
        sample_opponent(learner)
        episode_return = 0.0
        play_steps = 0
        hidden_steps = 0

        for _ in range(ROLLOUT_STEPS):
            learner_action, log_prob, value = live[learner].select_action(obs[learner])
            opponent_action = frozen_action(obs[opponent])
            actions = {learner: learner_action, opponent: opponent_action}
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())

            live[learner].store_transition(
                obs[learner], learner_action, log_prob, rewards[learner], done, value
            )
            episode_return += rewards[learner]
            info = infos[learner]
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

        live[learner].update(obs[learner], done, entropy_coef=ENTROPY_COEF)

    # ---- logging + save-best + paired snapshot ----
    parts = [f"Iter {iteration}", f"Steps {min(steps_done.values())}"]
    parts.append(f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}")

    improved_any = False
    for name in env.possible_agents:
        if not episode_returns[name]:
            continue
        mean_r = float(np.mean(episode_returns[name]))
        improved = mean_r > best_mean_return[name]
        if improved:
            best_mean_return[name] = mean_r
            torch.save(live[name].ac.state_dict(), f"hs_{name}.pt")
            improved_any = True
        hf = float(np.mean(hidden_frac[name])) if hidden_frac[name] else float("nan")
        tag = "*" if improved else " "
        parts.append(f"{name}={mean_r:+6.3f}{tag} hid={hf:.2f} ({len(episode_returns[name])}ep)")
        episode_returns[name] = []
        hidden_frac[name] = []

    if improved_any:
        for name in env.possible_agents:
            pools[name].append(copy.deepcopy(live[name].ac.state_dict()))

    print(" | ".join(parts))

env.close()
