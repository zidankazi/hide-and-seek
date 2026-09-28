"""Recurrent PPO with per-agent LSTM state.

Rollouts store whole episodes starting from zero hidden state. Updates batch episode
sequences, recomputing hidden state rather than using states from an older policy.
Exploration standard deviation is clamped, as in the feed-forward implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn  # noqa: F401  (pack_padded_sequence used in ActorCriticLSTM.eval_batch)
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

    @torch.no_grad()
    def act_batch(self, obs_b, hc):
        """Sample actions for a batch of agents while keeping their hidden states separate.

        obs_b has shape (B, obs_dim); hidden and cell states have shape (1, B, H)."""
        x = self.enc(obs_b).unsqueeze(1)          # (B, 1, H)
        out, hc2 = self.lstm(x, hc)               # (B, 1, H)
        feat = out.squeeze(1)                     # (B, H)
        mean = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        dist = Normal(mean, self._std())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value, hc2

    @torch.no_grad()
    def value_batch(self, obs_b, hc):
        x = self.enc(obs_b).unsqueeze(1)
        out, _ = self.lstm(x, hc)
        return self.value_head(out.squeeze(1)).squeeze(-1)

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

    def eval_batch(self, obs_pad, act_pad, lengths):
        """Evaluate padded episode batches with packed LSTM sequences.

        obs_pad and act_pad have shape (B, Tmax, features); lengths has shape (B,).
        Each episode starts from zero hidden state. Returned tensors have shape
        (B, Tmax); the caller masks padding before computing the loss."""
        B, Tmax, _ = obs_pad.shape
        x = self.enc(obs_pad)                     # (B, Tmax, H)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_p, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True, total_length=Tmax)
        mean = self.policy_head(out)              # (B, Tmax, act)
        dist = Normal(mean, self._std())
        log_probs = dist.log_prob(act_pad).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value_head(out).squeeze(-1)
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
    """Compute GAE over one complete episode."""
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
               value_coef=0.5, update_epochs=4, episodes_per_batch=8):
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

        obs_dim = eps[0]["obs"].shape[1]
        act_dim = eps[0]["actions"].shape[1]
        idx = np.arange(len(eps))
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), episodes_per_batch):
                batch = idx[start:start + episodes_per_batch]
                B = len(batch)
                lens = np.array([len(eps[j]["rewards"]) for j in batch])
                Tmax = int(lens.max())
                obs_pad = torch.zeros(B, Tmax, obs_dim)
                act_pad = torch.zeros(B, Tmax, act_dim)
                oldlp = torch.zeros(B, Tmax)
                adv = torch.zeros(B, Tmax)
                ret = torch.zeros(B, Tmax)
                mask = torch.zeros(B, Tmax)
                for bi, j in enumerate(batch):
                    ep = eps[j]
                    T = len(ep["rewards"])
                    obs_pad[bi, :T] = torch.from_numpy(ep["obs"])
                    act_pad[bi, :T] = torch.from_numpy(ep["actions"])
                    oldlp[bi, :T] = torch.from_numpy(ep["log_probs"])
                    adv[bi, :T] = torch.from_numpy(ep["adv"])
                    ret[bi, :T] = torch.from_numpy(ep["ret"])
                    mask[bi, :T] = 1.0
                log_probs, entropy, values = self.ac.eval_batch(
                    obs_pad, act_pad, torch.from_numpy(lens))
                ratio = torch.exp(torch.clamp(log_probs - oldlp, -20.0, 20.0))
                clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -(torch.min(ratio * adv, clip_adv) * mask).sum()
                value_loss = (((values - ret) ** 2) * mask).sum()
                ent = (entropy * mask).sum()
                n = mask.sum().clamp(min=1.0)
                loss = (policy_loss + value_coef * value_loss - entropy_coef * ent) / n
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
        # per-team exploration clamp is applied by the trainer after update()
