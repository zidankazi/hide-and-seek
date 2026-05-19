import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical 

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

