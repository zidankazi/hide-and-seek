import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical 
import torch.optim as optim

# PPO needs a brain that does two things when it sees a state:
# Actor: "What should I do?" -> Outputs a probability for each action
# Critic: "How good is this state/action pair?" -> Outputs a single number (value)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        # Takes the game state, runs it through the brain, returns a decision and assessment
        features = self.shared(x)
        logits = self.policy_head(features) # Scores for each action
        value = self.value_head(features) # 
        return logits, value

    def act(self, obs): 
        # Forward gives raw logits (scores for each action)
        # Act picks an action from those scores at random
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits) # creates a probability distribution from raw scores
        action = dist.sample() # Randomly picks an action based on the probabilities
        log_prob = dist.log_prob(action) # Tells us the log probability of the picked action (needed later for PPO math)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        # Re-evaluates old decisions with current network weights
        # "What do I think about those past actions now?"
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy() # entropy = uniformity, high entropy = high uniformity = info is spread out evenly and randomly
        return log_probs, entropy, values

# Before the network can learn, it needs experience. The agent plays the game for a bunch of steps, and 
# The RolloutBuffer is a recording of what it saw, what it did, what reward it got, etc.

class RolloutBuffer():
    def __init__(self): #
        self.obs = []          # What the agent saw each step 
        self.actions = []      # What action it picked
        self.log_probs = []    # Log-probability of the chosen action (needed for PPO math)
        self.rewards = []      # Reward received
        self.dones = []        # Boolean of Did the episode end this step?
        self.values = []       # Network's estimate of future reward from this state

    def store(self, obs, action, log_prob, reward, done, value): # appends each item to its list
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self): # resets all lists to empty
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """
        Walks backwards through steps to figure out which actions actually deserve credit
        Computes advantages (was this action better than expected?) and returns (total future reward)
        Gamma = discount factor, with gamma = 0.99:
            Step 0 reward is worth: 1
            Step 1 reward is worth: 1 * 0.99 = 0.99
            Step 2 reward is worth: 1 * 0.99 * 0.99 = 0.98
        Lambda = how far back do I spread credit?
            It mostly credits recent actions but still gives some credit to actions further back
            If Lambda was 1, spread credit way back and trust the full process
            If Lambda was 0, only credit the most recent step
            Lambda is 0.95, good balance that mostly credits recent actions
        """
        gae = 0 # Generalized Advantage Estimation
        advantages = [] 

        for t in reversed(range(len(self.rewards))): # iterate through list backwards
            if t == len(self.rewards) - 1: 
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # delta = was this step better or worse than expected?
            # Reward you got + discounted future - what you expected (positive = good)
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t] 

            # Accumulate GAE through loop then insert into advantages
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        # Return advantages and returns as tensors
        adv = torch.tensor(advantages, dtype=torch.float32)
        returns = []
        for i in range(len(advantages)):
            returns.append(advantages[i] + self.values[i])
        ret = torch.tensor(returns, dtype=torch.float32)
        
        return adv, ret

class PPO():
    def __init__(self, obs_dim, act_dim, learning_rate=3e-4):
        self.ac = ActorCritic(obs_dim, act_dim) # Create instance of ActorCritic
        self.buffer = RolloutBuffer() # Create instance of RolloutBuffer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)  # Create optimizer

    def select_action(self, obs): # Agent recieves an observation and needs to pick an action
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # Convert the observations numpy array to a tensor for PyTorch
        action, log_prob, value = self.ac.act(obs_t)
        return action.item(), log_prob.item(), value.item() # .item() strips the tensor wrapper so it returns a plain Python number

    def store_transition(self, obs, action, log_prob, reward, done, value): # Stores the transition in the buffer
        self.buffer.store(obs, action, log_prob, reward, done, value)

    def update(self, last_obs, gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, update_epochs=4, batch_size=64): 
        """
        Docstring TODO
        """
        last_obs_t = torch.tensor(last_obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): 
            _, last_value = self.ac(last_obs_t) # Calls forward, puts values in last_value
            last_value = last_value.item() # Convert tensor to python float
        advantages, returns = self.buffer.compute_returns(last_value, gamma, lam)

        # Convert the buffer lists to PyTorch tensors
        obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.long)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32)
        
        # Normalize advantages to help training - Mean = 0, Std = 1
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        n = len(obs)
        for _ in range(update_epochs):
            indices = np.random.permutation(n)    # Shuffle all indices randomly
            for start in range(0, n, batch_size): # Step through in chunks of 64 
                end = start + batch_size
                idx = indices[start:end]          # grab one mini-batch of random indices

                # Get mini-batch of data
                obs_batch = obs[idx]
                actions_batch = actions[idx]
                old_log_probs_batch = old_log_probs[idx]
                advantages_batch = advantages[idx]
                returns_batch = returns[idx]

                # Re-evaluate old actions with current network weights
                log_probs, entropy, values = self.ac.evaluate(obs_batch, actions_batch)

                # Calculate PPO loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                # Clip the ratio to be between 1 - clip_eps and 1 + clip_eps
                clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_batch
                # Policy loss = -min(ratio * advantages, clip_adv)
                policy_loss = -torch.min(ratio * advantages_batch, clip_adv).mean()
                # Value loss = MSE loss between network's value estimate and the expected return
                value_loss = nn.functional.mse_loss(values, returns_batch)

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                # Optimize the network by calculating the gradients and updating the weights
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5) # Cap the gradients at 0.5 to prevent the network from exploding
                # Update the weights
                self.optimizer.step()

        self.buffer.clear()
