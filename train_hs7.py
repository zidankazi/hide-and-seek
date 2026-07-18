"""
Stage 7 trainer: room + ramp hide-and-seek, engineered for the full emergent arc:
  rung 1  hiders barricade the doorway with a locked box
  rung 2  seekers transport the ramp to the room and peek over the wall
  rung 3  hiders lock the ramp away (or steal it) during prep

Now 1 hider vs 2 seekers (the paper's actual pressure): a lone seeker is dodgeable, so
1v1 always converged to evasion (the 40M fresh-256 mega-run ended at 94.7% hidden with
no construction). Two seekers pincer — camp the doorway + chase — which is what makes a
sealed room the hider's only refuge. The second seeker anneals in (dormant-seeker
curriculum, Stage 6 pattern) so the hider isn't crushed before it can learn.

Carries every accumulated fix: level-triggered locks, std ceiling 1.0 + floor 0.25,
entropy 0.005, clipped env actions, live opponent ladder (unconditional snapshots),
graded doorway curriculum (sealed / placed+at-door / near+at-door), ramp discovery
assists, ELEV_RANGE 400, near-band ramp spawns, tight box-0 spawn. Reward stays pure
team LOS.

Per-iteration logs carry the rung metrics (barr/elev/rlock) — the emergence timeline.

Saves hs_ramp_hider.pt / hs_ramp_seeker.pt (save-best) + *_final.pt (end-of-run).
Usage: python train_hs7.py [total_steps] [s2_start] [s2_end]
                           [--load=prefix] [--train=hider|seeker] [--fresh=hider|seeker]
                           [--hidden=N]
  s2_start..s2_end: P(second seeker active) anneals 0 -> 1 over this env-step range
  --load  warm-starts teams from prefix_{hider,seeker}.pt (arch inferred per side)
  --train iterated-best-response mode: only that team learns, the other plays fixed
  --fresh don't warm-start that team even under --load (fresh net at --hidden width)
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
SAVE_PREFIX = "hs_ramp"
# 8192 (was 2048): the paper's biggest lever I hadn't pulled. Hider construction shows
# the classic near-zero-gradient hard-exploration signature (std never anneals); a 4x
# batch is the standard variance-reduction fix. Overridable via --rollout=N.
ROLLOUT_STEPS = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rollout=")), 8192))
SNAPSHOT_EVERY = 100  # iterations between unconditional pool snapshots
POOL_CAP = 40         # rolling window of snapshots per team
_ints = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
TOTAL_TIMESTEPS = _ints[0] if len(_ints) > 0 else 40_000_000  # per team
S2_START = _ints[1] if len(_ints) > 1 else 4_000_000
S2_END = _ints[2] if len(_ints) > 2 else 8_000_000
LOAD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--load=")), None)
TRAIN_ONLY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--train=")), None)
assert TRAIN_ONLY in (None, "hider", "seeker")
FRESH = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--fresh=")), None)
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
print(f"[train_hs7] 1v2 room+ramp obs={obs_dim} steps={TOTAL_TIMESTEPS} "
      f"s2={S2_START}-{S2_END} hidden={HIDDEN} saving to {SAVE_PREFIX}_*.pt")

live = {}
for t in TEAMS:
    if LOAD and t != FRESH:
        sd = torch.load(f"{LOAD}_{t}.pt")
        live[t] = PPO(obs_dim, act_dim, hidden=sd["shared.0.weight"].shape[0])
        live[t].ac.load_state_dict(sd)
        print(f"[train_hs7] {t}: warm-started from {LOAD}_{t}.pt "
              f"(hidden={sd['shared.0.weight'].shape[0]})")
    else:
        live[t] = PPO(obs_dim, act_dim, hidden=HIDDEN)
        print(f"[train_hs7] {t}: fresh net (hidden={HIDDEN})")
team_buffers = {t: {m: RolloutBuffer() for m in members[t]} for t in TEAMS}
_frozen_cache = {}
frozen_net = None
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def second_seeker_prob(steps):
    if S2_END <= S2_START:
        return 1.0
    return min(1.0, max(0.0, (steps - S2_START) / (S2_END - S2_START)))


def sample_opponent(learner):
    # 50% newest snapshot, 50% uniform history (live ladder; improvement-gated pools
    # froze and made agents overfit stale opponents — run 6 lesson).
    global frozen_net
    opponent = "seeker" if learner == "hider" else "hider"
    pool = pools[opponent]
    sd = pool[-1] if random.random() < 0.5 else random.choice(pool)
    h = sd["shared.0.weight"].shape[0]
    if h not in _frozen_cache:
        _frozen_cache[h] = ActorCritic(obs_dim, act_dim, hidden=h)
    frozen_net = _frozen_cache[h]
    frozen_net.load_state_dict(sd)


def frozen_action(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _ = frozen_net.act(obs_t)
    return action.squeeze(0).numpy()


def update_team(team, last_obs, last_done, gamma=0.99, lam=0.95, clip_eps=0.2,
                entropy_coef=ENTROPY_COEF, value_coef=0.5, update_epochs=4, batch_size=64):
    """PPO.update generalized to a team: GAE per member buffer, one concatenated update.
    Members with empty buffers (a seeker dormant for the whole rollout) are skipped."""
    ppo = live[team]
    obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
    for m in members[team]:
        buf = team_buffers[team][m]
        if not buf.obs:
            continue
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
    if not obs_l:
        return

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
            ratio = torch.exp(torch.clamp(log_probs - old_log_probs[idx], -20.0, 20.0))
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(ratio * advantages[idx], clip_adv).mean()
            value_loss = nn.functional.mse_loss(values, returns[idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                continue
            ppo.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ppo.ac.parameters(), 0.5)
            ppo.optimizer.step()


# ---- main loop ----
best_mean_return = {t: float("-inf") for t in TEAMS}
steps_done = {t: 0 for t in TEAMS}
episode_returns = {t: [] for t in TEAMS}
hidden_frac = {t: [] for t in TEAMS}
barricade_eps = []
elev_fracs = []
ramp_lock_eps = []
iteration = 0
LEARNERS = [TRAIN_ONLY] if TRAIN_ONLY else list(TEAMS)
if TRAIN_ONLY:
    print(f"[train_hs7] best-response mode: training {TRAIN_ONLY} only, opponent fixed")

while min(steps_done[t] for t in LEARNERS) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in LEARNERS:
        opponent = "seeker" if learner == "hider" else "hider"
        p2 = second_seeker_prob(steps_done[learner])

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
                "active_seekers": active,
                "ramp_active": True,
                "seeker_on_ramp": on_ramp,
                "ramp_locked": not on_ramp and r < 0.05,
                "hider_on_ramp": not on_ramp and 0.05 <= r < 0.10,
                "doorway_box": door,
                "hider_at_door": at_door,
            })
            sample_opponent(learner)
            return o

        obs = new_episode()
        episode_return = 0.0
        play_steps = 0
        hidden_steps = 0
        elev_steps = 0

        for _ in range(ROLLOUT_STEPS):
            actions, cached = {}, {}
            for m in members[learner]:
                if m in env._dormant:
                    actions[m] = np.zeros(act_dim, dtype=np.float32)
                    continue
                a, log_prob, value = live[learner].select_action(obs[m])
                actions[m] = np.clip(a, -1.0, 1.0)
                cached[m] = (a, log_prob, value)
            for m in members[opponent]:
                if m in env._dormant:
                    actions[m] = np.zeros(act_dim, dtype=np.float32)
                else:
                    actions[m] = np.clip(frozen_action(obs[m]), -1.0, 1.0)

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())

            for m, (a, log_prob, value) in cached.items():
                team_buffers[learner][m].store(obs[m], a, log_prob, rewards[m], done, value)

            m0 = members[learner][0]
            episode_return += rewards[m0]
            info = infos[h0]
            if not info["in_prep"]:
                play_steps += 1
                if not info["seeker_sees_hider"]:
                    hidden_steps += 1
                if info["seeker_elevated"]:
                    elev_steps += 1

            obs = next_obs
            steps_done[learner] += 1

            if done:
                episode_returns[learner].append(episode_return)
                if play_steps > 0:
                    hidden_frac[learner].append(hidden_steps / play_steps)
                barricade_eps.append(info["doorway_barricaded"])
                if play_steps > 0:
                    elev_fracs.append(elev_steps / play_steps)
                ramp_lock_eps.append(info["ramp_lock_owner"] == "hider")
                episode_return = 0.0
                play_steps = 0
                hidden_steps = 0
                elev_steps = 0
                obs = new_episode()

        update_team(learner, obs, done)
        # Exploration floor AND ceiling on the parameter itself, PER TEAM: seekers need
        # a noise floor to keep searching (0.25), but that same floor makes the hider's
        # precision task — pushing a 44px box into a ~20px doorway window — physically
        # unlearnable, so hiders may anneal nearly deterministic (0.05). Ceiling 1.0
        # because clipped [-1,1] actions make mu irrelevant beyond that (and a one-sided
        # clamp would strand the param above the ceiling with zero gradient).
        with torch.no_grad():
            lo = -3.0 if learner == "hider" else -1.4
            live[learner].ac.log_std.clamp_(min=lo, max=0.0)

    # ---- logging + save-best + periodic snapshot ----
    lsteps = min(steps_done[t] for t in LEARNERS)
    parts = [f"Iter {iteration}", f"Steps {lsteps}"]
    parts.append(f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}")
    parts.append(f"p2={second_seeker_prob(lsteps):.2f}")
    with torch.no_grad():
        parts.append("std " + " ".join(
            f"{t[0]}={float(live[t].ac._std().mean()):.2f}" for t in TEAMS))

    for t in TEAMS:
        if not episode_returns[t]:
            continue
        mean_r = float(np.mean(episode_returns[t]))
        improved = mean_r > best_mean_return[t]
        if improved:
            best_mean_return[t] = mean_r
            torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}.pt")
        hf = float(np.mean(hidden_frac[t])) if hidden_frac[t] else float("nan")
        tag = "*" if improved else " "
        parts.append(f"{t}={mean_r:+6.3f}{tag} hid={hf:.2f} ({len(episode_returns[t])}ep)")
        episode_returns[t] = []
        hidden_frac[t] = []

    barr = float(np.mean(barricade_eps)) if barricade_eps else float("nan")
    elev = float(np.mean(elev_fracs)) if elev_fracs else float("nan")
    rlock = float(np.mean(ramp_lock_eps)) if ramp_lock_eps else float("nan")
    parts.append(f"barr={barr:.2f} elev={elev:.2f} rlock={rlock:.2f}")
    barricade_eps = []
    elev_fracs = []
    ramp_lock_eps = []

    if iteration % SNAPSHOT_EVERY == 0:
        for t in TEAMS:
            pools[t].append(copy.deepcopy(live[t].ac.state_dict()))
            if len(pools[t]) > POOL_CAP:
                pools[t].pop(0)

    print(" | ".join(parts))

for t in TEAMS:
    torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}_final.pt")

env.close()
