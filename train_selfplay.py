"""
Self-play training loop for the tag env. Fixes Stage 3's non-stationarity collapse
by training each live agent against a *frozen* opponent sampled from a pool of past
snapshots, instead of against the live (constantly-changing) opposite agent.

Per iteration:
  Rollout A: live hider plays vs a sampled-frozen seeker. Only the hider stores
             transitions and updates. New frozen seeker is sampled at each env reset.
  Rollout B: mirror - live seeker vs sampled-frozen hider. Only the seeker updates.
  Every SNAPSHOT_EVERY iterations, deep-copy each live policy's state_dict and append
  to its own pool so future opponents have access to this version.
"""

import copy
import random

import numpy as np
import torch

from env import TagEnv
from ppo_continuous import PPO, ActorCritic


# ---- config ----
ROLLOUT_STEPS = 2048
TOTAL_TIMESTEPS = 2_000_000   # per agent (so total env steps is ~2x this)
ENTROPY_COEF = 0.02
SNAPSHOT_EVERY = 10           # iterations between snapshots


# ---- setup ----
env = TagEnv()
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]
act_dim = env.action_space(sample).shape[0]

# Two live learners (same as train_multi.py)
live = {name: PPO(obs_dim, act_dim) for name in env.possible_agents}

# A single ActorCritic instance we re-use as "the frozen opponent" - load whichever
# snapshot state_dict we want into it before a rollout / episode. Saves allocating
# a new network every time we sample.
frozen_net = ActorCritic(obs_dim, act_dim)

# Pools of past snapshots, keyed by role. Each entry is a deep-copied state_dict.
# Seed each pool with the random-init weights so iteration 0 has something to sample.
pools = {
    name: [copy.deepcopy(live[name].ac.state_dict())] for name in env.possible_agents
}


# ---- helpers ----
def sample_opponent(learner):
    """
    Pick a random snapshot from the opponent's pool and load it into frozen_net.
    `learner` is the LIVE learner's role; we sample from the OTHER role's pool.
    """
    opponent = "seeker" if learner == "hider" else "hider"
    rand = random.choice(pools[opponent])
    frozen_net.load_state_dict(rand)

def frozen_action(obs):
    """Run frozen_net on one observation under no_grad. Return just the action."""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = frozen_net.act(obs_t)
    return action.squeeze(0).numpy()


# ---- main loop ----
best_mean_return = {name: float("-inf") for name in env.possible_agents}
steps_done = {name: 0 for name in env.possible_agents}  # per-agent counters
episode_returns = {name: [] for name in env.possible_agents}  # cleared each iteration
iteration = 0

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    # One rollout per role per iteration. Inside a rollout, `learner` is the live PPO
    # being trained and `opponent` plays via a frozen snapshot sampled from the pool.
    # Then we flip roles and do it again. Two rollouts = one iteration.
    for learner in env.possible_agents:
        opponent = "seeker" if learner == "hider" else "hider"

        # Fresh reset at the start of each rollout. The previous rollout had a different
        # opponent loaded into frozen_net, so carrying over mid-episode state would mean
        # mixing two opponents inside one episode, which breaks the "stationary opponent
        # per episode" guarantee. Cheap to reset, so just do it.
        obs, _ = env.reset()
        sample_opponent(learner)  # loads a random snapshot into frozen_net
        done = False
        episode_return = 0.0

        for _ in range(ROLLOUT_STEPS):
            # Learner action: goes through PPO, log_prob + value get cached so we can
            # train on this transition later.
            # Opponent action: forward pass through frozen_net under no_grad. Nothing
            # is recorded because the frozen opponent isn't learning, it's part of
            # the environment from the learner's POV.
            learner_action, log_prob, value = live[learner].select_action(obs[learner])
            opponent_action = frozen_action(obs[opponent])

            # PettingZoo expects actions as a dict keyed by agent name.
            actions = {learner: learner_action, opponent: opponent_action}
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            # Symmetric env: terms/truncs flip together for both agents, single flag is fine
            done = any(terms.values()) or any(truncs.values())

            # Only store the learner's transition. The opponent's actions, rewards,
            # log_probs are all thrown away. If we stored both, we'd be training two
            # policies again and Stage 4 would collapse exactly like Stage 3.
            live[learner].store_transition(
                obs[learner], learner_action, log_prob, rewards[learner], done, value
            )
            episode_return += rewards[learner]

            obs = next_obs
            steps_done[learner] += 1  # per-agent counter, used as the loop termination check

            # Episode boundary: log the learner's total return, reset env, and pick a
            # new frozen opponent. Resampling at episode boundaries (instead of holding
            # one opponent for the whole rollout) means the learner sees ~8-10 different
            # snapshots per rollout - good for generalization across the pool.
            if done:
                episode_returns[learner].append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
                sample_opponent(learner)

        # Same PPO update as Stage 3. `obs[learner]` and `done` are the last-step state,
        # used by update() to bootstrap a value estimate for the trailing partial episode
        # (or zero it out if the rollout ended exactly on a done).
        live[learner].update(obs[learner], done, entropy_coef=ENTROPY_COEF)

    # ---- snapshot ----
    if iteration % SNAPSHOT_EVERY == 0:
        for name in env.possible_agents:
            pools[name].append(copy.deepcopy(live[name].ac.state_dict()))

    # ---- logging + save-best (per role, same shape as train_multi.py) ----
    # Build the log line piece by piece so we can skip roles that didn't finish any
    # episodes this iteration (rare, but possible if rollouts somehow ran with zero
    # terminations or truncations).
    parts = [f"Iter {iteration}", f"Steps {min(steps_done.values())}"]
    # Pool sizes climb by 1 every SNAPSHOT_EVERY iterations. Eyeballing this in the log
    # is a quick sanity check that snapshotting is firing.
    parts.append(f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}")
    for name in env.possible_agents:
        if not episode_returns[name]:
            continue
        mean_r = float(np.mean(episode_returns[name]))
        # Save-best, same idea as Stage 2/3: only overwrite the .pt when we beat the
        # all-time best mean return for this role. Note that in self-play this is a bit
        # noisy because difficulty changes as the opponent pool grows - a high score
        # vs early easy opponents can lock in the .pt and rarely get overwritten later.
        # Good enough for now, can revisit if it bites.
        improved = mean_r > best_mean_return[name]
        if improved:
            best_mean_return[name] = mean_r
            torch.save(live[name].ac.state_dict(), f"{name}.pt")
        tag = "*" if improved else " "
        parts.append(f"{name}={mean_r:+6.3f}{tag} ({len(episode_returns[name])}ep)")
        # Reset the list so next iteration only logs its own episodes
        episode_returns[name] = []
    print(" | ".join(parts))

env.close()
