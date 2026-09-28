"""
Climb variant — the ramp as an ENTRY tool instead of a surveillance one.

Every prior Stage 7 run gave elevation one effect: it rewrote the LOS mask, so a seeker
standing on the ramp SAW over interior walls. It could never cross one. Two consequences,
both measured on 2026-07-28 and written up in NOTES:

  1. the ramp is never transported (1-5px moved per episode) — it spawns inside ELEV_RANGE,
     so standing where it lies already pays and fetching it buys nothing;
  2. "rung 2" was scored by proximity to the ramp, which is a ~74-85% proxy for genuinely
     seeing over a wall, and not a claim about climbing at all.

This run turns the mechanic into the paper's: ramp_climb=True drops LOW_CAT from a seeker's
COLLISION mask while elevated, so it can cross interior walls and boxes; ramp_xray=False
removes the see-through effect entirely. The ramp now buys NOTHING until a seeker climbs it,
which is the sharpest possible test of the project's thesis — a tool emerges iff the
environment makes it necessary.

The necessity chain this sets up:
  rung 1 (hider seals the doorway)  ->  the doorway is the only walk-in route, so sealing it
                                        leaves the seekers with no way in at all
  rung 2 (seeker climbs the ramp)   ->  the ONLY remaining route, and worth real reward
  rung 3 (hider locks the ramp away) -> now denies ENTRY, not just sight, so it is finally
                                        load-bearing (it was ~1.5-2% when it only cost sight)

CLIMB_STEPS is the lever that decides how much ramp POSITION matters: it is how long
elevation persists after leaving the ramp, so seeker_speed * CLIMB_STEPS/60 is roughly how
far a seeker can carry a climb. Short window => the ramp must be near the wall => hiders
have a real reason to lock it far away (rung 3). 25 steps at 1.4x speed is ~175px.

Warm-starts from the 2v2 arc lineage (hs_2v2), which already has rung 1 at ~23% and a weak
rung 3, so this run only has to add the climb on top.

Usage: python train_hs7_climb.py [total] [rc_end]
           [--envs=N] [--steps=per_env] [--hidden=N] [--save=prefix]
           [--load=prefix | --load-hider=p --load-seeker=p]
           [--layout=roomt|room] [--speed=1.4] [--climb-steps=25] [--xray]
  total  = total steps per team (default 50M)
  rc_end = step at which the climb curriculum anneals to zero (default 30M)
  --xray = also keep the old see-through effect on (default off for this run)
"""

import copy
import random
import sys

import numpy as np
import torch

from hide_and_seek.env_hs import HideAndSeekEnv
from hide_and_seek.ppo_recurrent import RecurrentPPO, ActorCriticLSTM, EpisodeBuffer

torch.set_num_threads(max(1, __import__("os").cpu_count() - 1))


def _arg(name, default, cast=str):
    hit = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), None)
    return cast(hit) if hit is not None else default


# ---- config ----
SAVE_PREFIX = _arg("save", "hs_climb")
LAYOUT = _arg("layout", "roomt")
SPEED = _arg("speed", 1.4, float)
N_ENVS = _arg("envs", 10, int)
STEPS_PER_ENV = _arg("steps", 800, int)
HIDDEN = _arg("hidden", 256, int)
CLIMB_STEPS = _arg("climb-steps", 25, int)
XRAY = "--xray" in sys.argv
LOAD = _arg("load", "hs_2v2")
_ints = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
TOTAL_TIMESTEPS = _ints[0] if len(_ints) > 0 else 50_000_000
RC_END = _ints[1] if len(_ints) > 1 else 30_000_000
ENTROPY_COEF = 0.005
TEAMS = ("hider", "seeker")

_GEO = {"room": (222.0, 155.0), "roomt": (148.0, 105.0)}
DOOR_Y, FAR_Y = _GEO[LAYOUT]
HONEST_FRAC = 0.18          # always-unassisted episodes, so honest play stays in-distribution
OFF_NEAR, OFF_FAR = 55.0, 130.0


def make_env():
    # 2v2 as in the arc run: two hiders let one barricade while the other denies the ramp,
    # the division of labor rung 3 needs. climb_teams stays seeker-only — a hider that could
    # climb out of its own sealed room would make rung 1 pointless.
    return HideAndSeekEnv(layout=LAYOUT, ramp=True, max_steps=360, lock_mode="level",
                          n_hiders=2, n_seekers=2, box_mass=2, door_box_size=72,
                          seeker_speed_mult=SPEED,
                          ramp_climb=True, ramp_xray=XRAY,
                          climb_teams=("seeker",), climb_steps=CLIMB_STEPS)


envs = [make_env() for _ in range(N_ENVS)]
e0 = envs[0]
possible = e0.possible_agents
obs_dim = e0.observation_space(possible[0]).shape[0]
act_dim = e0.action_space(possible[0]).shape[0]
members = {t: e0.teams[t] for t in TEAMS}
h0 = members["hider"][0]
ZERO = np.zeros(act_dim, dtype=np.float32)
print(f"[climb] layout={LAYOUT} speed={SPEED}x climb_steps={CLIMB_STEPS} xray={XRAY} | "
      f"{N_ENVS} envs x {STEPS_PER_ENV} = {N_ENVS*STEPS_PER_ENV}/update | obs={obs_dim} "
      f"steps={TOTAL_TIMESTEPS} rc_end={RC_END} hidden={HIDDEN}")

live = {t: RecurrentPPO(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
LOAD_T = {t: _arg(f"load-{t}", LOAD) for t in TEAMS}
for t in TEAMS:
    if LOAD_T[t]:
        live[t].ac.load_state_dict(torch.load(f"{LOAD_T[t]}_{t}.pt"))
        print(f"[climb] warm-started {t} from {LOAD_T[t]}_{t}.pt")
frozen = {t: ActorCriticLSTM(obs_dim, act_dim, hidden=HIDDEN) for t in TEAMS}
pools = {t: [copy.deepcopy(live[t].ac.state_dict())] for t in TEAMS}


def rc_phase(steps):
    p = steps / RC_END if RC_END > 0 else 1.0
    if p >= 1.0:
        return p, "honest"
    return p, ("A/onramp" if p < 0.35 else "B/fetch" if p < 0.65 else "C/unaided")


def reset_options(steps):
    """Climb curriculum: back-chain the crossing, then fade every assist to zero.

    A seeker only discovers the climb if it is elevated AND near a wall at the same time, and
    neither happens by chance — the ramp spawns mid-arena and the window is 25 steps. So phase
    A hands it both (sealed door, ramp in the near band, seeker starting on the ramp) and the
    only thing left to learn is "move toward the room". B removes the free elevation so it has
    to reach the ramp first; C removes the ramp placement too.
    """
    opts = {"active_seekers": 2, "ramp_active": True}
    p, _ = rc_phase(steps)
    if p >= 1.0 or random.random() < HONEST_FRAC:
        return opts

    c = random.random()
    # --- rung 2: the climb itself (the point of this run) ---
    if c < 0.45:
        opts["doorway_box"] = "sealed"      # no walk-in route, so climbing is the only way
        opts["ramp_band"] = "near"          # ramp within a climb window of the wall
        if p < 0.35:
            opts["seeker_on_ramp"] = True                       # A: start elevated
        elif p < 0.65:
            opts["seeker_on_ramp"] = random.random() < 0.5      # B: usually fetch it yourself
        else:
            opts["seeker_on_ramp"] = False                      # C: unaided
            opts.pop("ramp_band")
        return opts
    # --- rung 3: denying the ramp, which now denies ENTRY rather than sight ---
    if c < 0.62:
        opts["doorway_box"] = "sealed"
        opts["hider_on_ramp"] = True        # hider starts at the ramp so it can lock it
        opts["ramp_band"] = "near"
        return opts
    # --- rung 1: keep the barricade alive; without it nothing above is necessary ---
    if c < 0.80:
        opts["door_push_y"] = DOOR_Y
        opts["split_roles"] = True
        return opts
    q = max(0.0, (p - 0.60) / 0.40)
    opts["door_push_y"] = DOOR_Y - (DOOR_Y - FAR_Y) * q
    opts["hider_at_door"] = True
    opts["hider_door_offset"] = OFF_NEAR + (OFF_FAR - OFF_NEAR) * q
    return opts


def zero_hidden_slots(hc, rows):
    for row in rows:
        hc[0][:, row, :] = 0.0
        hc[1][:, row, :] = 0.0


# ---- main loop ----
best_mean_return = {t: float("-inf") for t in TEAMS}
steps_done = {t: 0 for t in TEAMS}
episode_returns = {t: [] for t in TEAMS}
hidden_frac = {t: [] for t in TEAMS}
barricade_eps, climb_eps, ramp_lock_eps = [], [], []
iteration = 0

while min(steps_done.values()) < TOTAL_TIMESTEPS:
    iteration += 1

    for learner in TEAMS:
        opponent = "seeker" if learner == "hider" else "hider"
        Ml, Mo = members[learner], members[opponent]
        nl, no = len(Ml), len(Mo)
        frozen[opponent].load_state_dict(random.choice(pools[opponent]))

        buffers = [[EpisodeBuffer() for _ in Ml] for _ in range(N_ENVS)]
        ep_ret = [0.0] * N_ENVS
        play = [0] * N_ENVS
        hid = [0] * N_ENVS

        obs = []
        for e in range(N_ENVS):
            o, _ = envs[e].reset(options=reset_options(steps_done[learner]))
            obs.append(o)
        hc_l = live[learner].ac.init_hidden(N_ENVS * nl)
        hc_o = frozen[opponent].init_hidden(N_ENVS * no)

        for _ in range(STEPS_PER_ENV):
            lob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Ml]).astype(np.float32)
            la, llp, lval, hc_l = live[learner].ac.act_batch(torch.from_numpy(lob), hc_l)
            la = la.numpy(); llp = llp.numpy(); lval = lval.numpy()
            oob = np.stack([obs[e][m] for e in range(N_ENVS) for m in Mo]).astype(np.float32)
            oa, _, _, hc_o = frozen[opponent].act_batch(torch.from_numpy(oob), hc_o)
            oa = oa.numpy()

            done_rows = []
            for e in range(N_ENVS):
                env = envs[e]
                acts = {}
                for mi, m in enumerate(Ml):
                    row = e * nl + mi
                    acts[m] = ZERO if m in env._dormant else np.clip(la[row], -1.0, 1.0)
                for mi, m in enumerate(Mo):
                    row = e * no + mi
                    acts[m] = ZERO if m in env._dormant else np.clip(oa[row], -1.0, 1.0)
                nobs, rew, terms, truncs, infos = env.step(acts)
                done = any(terms.values()) or any(truncs.values())

                for mi, m in enumerate(Ml):
                    if m not in env._dormant:
                        row = e * nl + mi
                        buffers[e][mi].store(obs[e][m], la[row], float(llp[row]),
                                             rew[m], float(lval[row]))
                info = infos[h0]
                ep_ret[e] += rew[Ml[0]]
                if not info["in_prep"]:
                    play[e] += 1
                    hid[e] += (not info["seeker_sees_hider"])

                if done:
                    for mi in range(nl):
                        if buffers[e][mi].has_open():
                            buffers[e][mi].end_episode(0.0)
                    episode_returns[learner].append(ep_ret[e])
                    if play[e] > 0:
                        hidden_frac[learner].append(hid[e] / play[e])
                    barricade_eps.append(info["doorway_barricaded"])
                    # the direct rung-2 signal: a seeker physically crossed a wall this episode
                    climb_eps.append(info["seeker_climbed"])
                    ramp_lock_eps.append(info["ramp_lock_owner"] == "hider")
                    ep_ret[e] = 0.0; play[e] = hid[e] = 0
                    nobs, _ = env.reset(options=reset_options(steps_done[learner]))
                    done_rows.append(e)
                obs[e] = nobs
                steps_done[learner] += 1

            if done_rows:
                zero_hidden_slots(hc_l, [e * nl + mi for e in done_rows for mi in range(nl)])
                zero_hidden_slots(hc_o, [e * no + mi for e in done_rows for mi in range(no)])

        for e in range(N_ENVS):
            for mi, m in enumerate(Ml):
                if not buffers[e][mi].has_open():
                    continue
                if m in envs[e]._dormant:
                    buffers[e][mi].end_episode(0.0)
                else:
                    row = e * nl + mi
                    hc_slot = (hc_l[0][:, row:row + 1, :].contiguous(),
                               hc_l[1][:, row:row + 1, :].contiguous())
                    ot = torch.tensor(obs[e][m], dtype=torch.float32).unsqueeze(0)
                    lv = live[learner].ac.value_only(ot, hc_slot).item()
                    buffers[e][mi].end_episode(lv)

        merged = EpisodeBuffer()
        for e in range(N_ENVS):
            for mi in range(nl):
                merged.episodes.extend(buffers[e][mi].episodes)
        live[learner].update(merged, entropy_coef=ENTROPY_COEF, episodes_per_batch=8)
        with torch.no_grad():
            lo = -3.0 if learner == "hider" else -1.4
            live[learner].ac.log_std.clamp_(min=lo, max=0.0)

    # ---- logging + save-best + snapshot ----
    lsteps = min(steps_done.values())
    _p, _ph = rc_phase(lsteps)
    parts = [f"Iter {iteration}", f"Steps {lsteps}",
             f"Pools h={len(pools['hider'])} s={len(pools['seeker'])}",
             f"ph={_ph}({_p:.2f})",
             "std " + " ".join(f"{t[0]}={float(live[t].ac._std().mean().item()):.2f}" for t in TEAMS)]
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
    climb = float(np.mean(climb_eps)) if climb_eps else float("nan")
    rlock = float(np.mean(ramp_lock_eps)) if ramp_lock_eps else float("nan")
    parts.append(f"barr={barr:.2f} climb={climb:.2f} rlock={rlock:.2f}")
    barricade_eps, climb_eps, ramp_lock_eps = [], [], []

    if iteration % 20 == 0:
        for t in TEAMS:
            pools[t].append(copy.deepcopy(live[t].ac.state_dict()))
            if len(pools[t]) > 30:
                pools[t].pop(0)

    print(" | ".join(parts))

for t in TEAMS:
    torch.save(live[t].ac.state_dict(), f"{SAVE_PREFIX}_{t}_final.pt")
for e in envs:
    e.close()
