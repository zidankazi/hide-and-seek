"""
Recurrent (LSTM) PPO for Stage 7 — the paper's actual rung-1 recipe: memory.

Why memory helps where the feed-forward policy walled: hide-and-seek is partially
observed (boxes, ramp, and opponents drop out of line-of-sight and their obs slots
zero), and the barricade is a *multi-step* construction whose value only pays off many
steps later. An LSTM lets the policy carry "I am mid-build; the box is just below me even
though I can't see it this frame" across the sequence, which is exactly the credit-
assignment the feed-forward net couldn't do (its exploration std never annealed → no
gradient toward the multi-step skill).

Design notes (correctness of recurrent PPO in this codebase):
- Every episode in a rollout starts from a FRESH env reset (train_hs7 calls new_episode()
  at rollout start and after every done), so each episode is an independent sequence with
  zero initial hidden state. We therefore store rollouts as a list of per-episode sequences
  and, in the update, re-run the LSTM over each whole episode from zero hidden — no stored
  hidden states, no cross-episode leakage, no BPTT-through-reset hazards.
- Minibatching is over EPISODES (not shuffled timesteps), because shuffling timesteps would
  destroy the sequences the LSTM needs. With ~22 episodes per 8192-step rollout that is a
  fine granularity; advantages are normalized across the whole rollout first.
- log_std uses the same clamp discipline as the feed-forward version ([-4, 0] → std in
  [0.018, 1.0]); the ratio and finite-loss guards are carried over verbatim.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _std(self):
        return torch.exp(torch.clamp(self.log_std, -4.0, 0.0))

    def init_hidden(self, batch=1):
        return (torch.zeros(1, batch, self.hidden), torch.zeros(1, batch, self.hidden))

    def step(self, obs_t, hc):
        """One timestep. obs_t: (1, obs_dim); hc: ((1,1,H),(1,1,H)). Returns mean, value, hc2."""
        x = self.enc(obs_t).unsqueeze(1)          # (1, 1, H)
        out, hc2 = self.lstm(x, hc)               # out: (1, 1, H)
        feat = out.squeeze(1)                     # (1, H)
        mean = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return mean, value, hc2

    @torch.no_grad()
    def act(self, obs_t, hc):
        mean, value, hc2 = self.step(obs_t, hc)
        dist = Normal(mean, self._std())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value, hc2

    @torch.no_grad()
    def value_only(self, obs_t, hc):
        _, value, _ = self.step(obs_t, hc)
        return value

    def eval_episode(self, obs_seq, act_seq):
        """Re-run the whole episode from zero hidden (grad-tracked). obs_seq: (T, obs_dim).
        Returns per-step log_probs (T,), entropy (T,), values (T,)."""
        x = self.enc(obs_seq).unsqueeze(0)        # (1, T, H)
        out, _ = self.lstm(x)                     # zero init hidden
        feat = out.squeeze(0)                     # (T, H)
        mean = self.policy_head(feat)
        dist = Normal(mean, self._std())
        log_probs = dist.log_prob(act_seq).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value_head(feat).squeeze(-1)
        return log_probs, entropy, values


class EpisodeBuffer:
    """Collects one rollout as a list of complete episodes (each a fresh sequence)."""

    def __init__(self):
        self.episodes = []          # finalized episodes: dicts of np arrays
        self._cur = self._blank()

    def _blank(self):
        return {"obs": [], "actions": [], "log_probs": [], "rewards": [], "values": []}

    def store(self, obs, action, log_prob, reward, value):
        self._cur["obs"].append(obs)
        self._cur["actions"].append(action)
        self._cur["log_probs"].append(log_prob)
        self._cur["rewards"].append(reward)
        self._cur["values"].append(value)

    def end_episode(self, last_value):
        """Close the current episode. last_value = bootstrap value (0.0 if terminal)."""
        if self._cur["obs"]:
            ep = {k: np.array(v, dtype=np.float32) for k, v in self._cur.items()}
            ep["last_value"] = float(last_value)
            self.episodes.append(ep)
        self._cur = self._blank()

    def has_open(self):
        return len(self._cur["obs"]) > 0

    def clear(self):
        self.episodes = []
        self._cur = self._blank()


def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """GAE over one episode (no dones inside — an episode is one uninterrupted sequence)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret


class RecurrentPPO:
    def __init__(self, obs_dim, act_dim, hidden=256, learning_rate=3e-4):
        self.ac = ActorCriticLSTM(obs_dim, act_dim, hidden=hidden)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)

    def update(self, buffer, gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.005,
               value_coef=0.5, update_epochs=4, episodes_per_batch=4, log_std_floor=-4.0):
        eps = buffer.episodes
        if not eps:
            return
        # GAE per episode, then normalize advantages across the whole rollout.
        for ep in eps:
            adv, ret = compute_gae(ep["rewards"], ep["values"], ep["last_value"], gamma, lam)
            ep["adv"], ep["ret"] = adv, ret
        all_adv = np.concatenate([ep["adv"] for ep in eps])
        a_mean, a_std = all_adv.mean(), all_adv.std() + 1e-8
        for ep in eps:
            ep["adv"] = (ep["adv"] - a_mean) / a_std

        idx = np.arange(len(eps))
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), episodes_per_batch):
                batch = idx[start:start + episodes_per_batch]
                self.optimizer.zero_grad()
                total_loss = 0.0
                n_steps = 0
                for j in batch:
                    ep = eps[j]
                    obs = torch.tensor(ep["obs"], dtype=torch.float32)
                    act = torch.tensor(ep["actions"], dtype=torch.float32)
                    old_lp = torch.tensor(ep["log_probs"], dtype=torch.float32)
                    adv = torch.tensor(ep["adv"], dtype=torch.float32)
                    ret = torch.tensor(ep["ret"], dtype=torch.float32)
                    log_probs, entropy, values = self.ac.eval_episode(obs, act)
                    ratio = torch.exp(torch.clamp(log_probs - old_lp, -20.0, 20.0))
                    clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                    policy_loss = -torch.min(ratio * adv, clip_adv).sum()
                    value_loss = ((values - ret) ** 2).sum()
                    ent = entropy.sum()
                    total_loss = total_loss + policy_loss + value_coef * value_loss - entropy_coef * ent
                    n_steps += len(ret)
                if n_steps == 0:
                    continue
                loss = total_loss / n_steps
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
        # per-team exploration clamp is applied by the trainer after update()
