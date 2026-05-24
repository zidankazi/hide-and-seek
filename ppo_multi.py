"""
Multi-agent PPO training loop for the tag env. Two completely independent PPO
instances (hider and seeker), each with its own network, buffer, and optimizer.

From each agent's perspective the other agent is just part of the environment, so
the PPO class from ppo_continuous.py works as-is. The only new code is the
orchestration: query both networks each step, store into both buffers, update both
networks after each rollout, and save each one to its own file.
"""

import numpy as np
import torch

from env import TagEnv
from ppo_continuous import PPO


env = TagEnv()

# Both agents are symmetric so we can read shapes from either one.
# observation_space and action_space are methods now (PettingZoo), not attributes.
sample = env.possible_agents[0]
obs_dim = env.observation_space(sample).shape[0]   # 8
act_dim = env.action_space(sample).shape[0]        # 2

# One PPO instance per role. They share zero weights, zero optimizer state, zero anything.
ppos = {name: PPO(obs_dim, act_dim) for name in env.possible_agents}

# Training loop
rollout_steps = 2048        # Steps collected per agent before each update
total_timesteps = 1_000_000 # Total env steps to train for (per agent)
steps_done = 0

obs, _ = env.reset()
done = False  # Both agents share episode boundaries in our symmetric env, so one flag is enough

# Per-agent running totals for the current episode
episode_return = {name: 0.0 for name in env.possible_agents}
# Per-agent list of finished episode returns this rollout
episode_returns = {name: [] for name in env.possible_agents}
# Per-agent best mean return seen so far, used for save-best logic
best_mean_return = {name: float("-inf") for name in env.possible_agents}

while steps_done < total_timesteps:
    # Collect rollout_steps of experience for both agents at once
    for _ in range(rollout_steps):
        # Query each policy for its action. Cache log_prob and value so we can store them after env.step
        actions = {}
        cached = {}
        for name in env.possible_agents:
            action, log_prob, value = ppos[name].select_action(obs[name])
            actions[name] = action
            cached[name] = (log_prob, value)

        # Step the env once with both agents' actions at the same time
        next_obs, rewards, terms, truncs, _ = env.step(actions)

        # Symmetric env: terms/truncs flip together for both agents, so a single done flag works
        done = any(terms.values()) or any(truncs.values())

        # Store each agent's transition into its own buffer
        for name in env.possible_agents:
            log_prob, value = cached[name]
            ppos[name].store_transition(obs[name], actions[name], log_prob, rewards[name], done, value)
            episode_return[name] += rewards[name]

        obs = next_obs
        steps_done += 1

        # If the episode ended, log per-agent returns and reset
        if done:
            for name in env.possible_agents:
                episode_returns[name].append(episode_return[name])
                episode_return[name] = 0.0
            obs, _ = env.reset()

    # Update each PPO independently with its own last-obs (the value bootstrap is per-agent).
    # If the rollout ended exactly on a done step, last_done=True makes update() zero out the
    # bootstrap value so we don't accidentally credit the next episode's reset state.
    for name in env.possible_agents:
        ppos[name].update(obs[name], done)

    # Log progress and save best policies per agent
    n_ep = len(episode_returns["hider"])
    if n_ep > 0:
        parts = [f"Steps: {steps_done}", f"Eps: {n_ep}"]
        for name in env.possible_agents:
            mean_r = float(np.mean(episode_returns[name]))
            improved = mean_r > best_mean_return[name]
            if improved:
                best_mean_return[name] = mean_r
                torch.save(ppos[name].ac.state_dict(), f"{name}.pt")
            tag = "*" if improved else " "
            parts.append(f"{name}={mean_r:+7.1f}{tag}")
        print(" | ".join(parts))
        episode_returns = {name: [] for name in env.possible_agents}

env.close()
