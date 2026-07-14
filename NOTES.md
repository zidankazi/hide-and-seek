# Hide & Seek — Project Journal

Building toward reproducing emergent tool-use from OpenAI's 2019 multi-agent hide-and-seek paper. Six-stage roadmap: continuous PPO → custom env → multi-agent → self-play → full hide-and-seek → analysis.

---

## 2026-05-21 — Stage 1 complete: continuous PPO

Adapted the existing discrete PPO (CartPole, LunarLander) to continuous action spaces. Core change: swapped `Categorical` distribution for `Normal` (Gaussian: mean + log_std), so the actor outputs continuous action values instead of action probabilities. Tested on Pendulum-v1. ~30-50 lines of changes.

The trick that took the longest to internalize: `log_std` is a learnable `nn.Parameter`, not a network output. Starts at 0 (std=1, max exploration) and shrinks via gradient descent as the policy gets more confident.

---

## 2026-05-22 — Stage 2 begins: custom env + renderer

### Renderer iterations

Started with a pygame demo that drew a single white circle bouncing around in a black arena. Worked but looked like a 1990s screensaver.

**Pivot 1: dark cyberpunk theme.** Added neon-blue glowing agent, particle trails, force/velocity arrows, animated goal pulse, dark background with concentric ring overlays. Looked cool in isolation but didn't match the friendly aesthetic of the OpenAI paper.

**Pivot 2: clean OpenAI-style aesthetic.** User shared a screenshot from OpenAI's hide-and-seek video — bright, charming, cute character agents. Rebuilt the renderer:
- Light grey arena floor with subtle alternating-tile pattern
- Walls with 3D depth (top + front + right faces, drop shadows)
- Agents as colored circles with tracking eyes, base "pedestal" rings (matching OpenAI's avatar style), gentle bobbing animation
- Boxes rendered as proper 3D objects with lock icons when locked
- Dust particles trailing fast-moving agents
- Edge vignette
- Bottom HUD: phase pill (PREP/PLAY), episode/step/reward counters, timeline progress bar
- Y-sorting so closer objects draw on top

Took ~3 full rewrites of `renderer.py` to land on the right look. Worth it — the visual quality is now demo-ready.

### Env design

Built `HideAndSeekEnv(gym.Env)` for single-agent navigation. Choices:
- **Observation (6-dim, normalized [-1,1]):** `[agent_x, agent_y, agent_vx, agent_vy, goal_x, goal_y]`. Normalized because NNs train way faster on small numbers.
- **Action (2-dim, [-1,1]):** continuous 2D thrust, multiplied by `FORCE_SCALE=1500` inside `step()`.
- **Reward:** dense `-dist/ARENA_SIZE` per step + `10` bonus on reaching goal. Dense signal so PPO has gradient even when the agent never reaches the goal.
- **Termination:** within `AGENT_RADIUS + 12` pixels of goal. Truncation: `MAX_STEPS=500`.
- **Physics:** Pymunk space, gravity=(0,0) (top-down), damping=0.5, agent elasticity=0.6 (mild bounce).
- **Pipeline split:** `env.py` owns physics + game logic, `renderer.py` only knows how to draw. Renderer doesn't import env, env imports renderer only when `render_mode="human"`.

---

## 2026-05-22 — Training run 1: policy collapse

Hooked the env into `ppo_continuous.py` with `env = HideAndSeekEnv()`. Saved `policy.pt` every PPO update. Trained 1M steps.

**Trajectory:**
- 0–340k steps: random-policy baseline (~-200 mean return)
- 340k–460k: **learned!** Episodes per rollout jumped from 5 to 20+, returns hit -13 to -45
- 460k–1M: **diverged** — returns crashed to -1000, -3000, even -5000

Classic PPO failure mode: `log_std` collapsed too far, policy got overconfident in bad actions, value function diverged, each update made it worse.

**The save logic made it worse:** by saving every update, the collapsed late-training policy *overwrote* the good mid-training one. By the end, `policy.pt` held the broken version.

---

## 2026-05-22 — Fix: save-best logic + run 2

Changed the save logic to track `best_mean_return` and only save when beaten. Re-ran from scratch.

**Trajectory:**
- Best saved at step 147k, mean return **-60.6**
- Never improved over the remaining 850k steps
- `policy.pt` locked in at peak

Worse peak than run 1 (-60 vs -13) because PPO is stochastic — different random seeds give different trajectories. But the policy is *saved* this time, even though training continued to diverge afterward.

**Lesson learned:** always save best, not last. Single line fix, but it's the difference between "we trained an agent" and "we trained an agent and then erased it."

### Where -60 puts us
- ~-235: random baseline
- ~-60: reaches goal occasionally but slowly *(current)*
- ~0: reaches consistently in 30-60 steps
- +2 to +5: reaches in <20 steps (bonus outweighs dense cost)

### Stage 2 status
Pipeline is proven end-to-end: custom Gymnasium env + Pymunk physics + continuous PPO + clean renderer + save/load. The task itself is simple (point-to-point navigation) and the agent half-learned it. Could push higher with reward shaping or relative observations, but diminishing returns vs moving on to multi-agent.

---

## 2026-05-23 - Stage 3 begins: multi-agent tag

Two agents in an empty arena. Hider tries to survive a fixed step count, seeker tries to touch the hider before time runs out. Reproducing the 0:18 mark of the OpenAI video: "they've already learned to chase and run away."

### Design choices

- **PettingZoo Parallel API** instead of single-agent Gymnasium. Observation/action/reward/term/trunc all come back as dicts keyed by agent name. The whole multi-agent abstraction is just "everything is a dict now."
- **Equal speed for both agents + 240-step time limit.** Hider wins by surviving the full episode, seeker wins by tagging before then. No speed asymmetry to tune.
- **Observations: 8 dims, absolute coords.** Self (x,y,vx,vy) + opponent (x,y,vx,vy), all normalized to [-1,1]. Will swap to ego-centric in Stage 5 when we add lidar.
- **Paper-faithful sparse reward** (rescaled small to keep value targets manageable):
  - Per step: hider +0.01, seeker -0.01
  - On tag: extra +/-0.05 swing
  - Zero-sum, episode totals around +/-2.4
- **Two PPO policies, simultaneous training.** Each agent gets its own ActorCritic, RolloutBuffer, optimizer. From each agent's POV the other is just part of the environment. Reused the PPO class from ppo_continuous.py unchanged, only the orchestration in ppo_multi.py is new.
- **Single env, both agents step at once.** Each iteration: query both networks, env.step(actions_dict), store into both buffers, update both after each rollout, save best per agent.

### Files

- env.py is now the multi-agent TagEnv (renamed Stage 2 nav env to env_nav.py).
- ppo_multi.py for the dual-agent training loop.
- watch_random.py and watch_trained.py for eyeballing the env and the trained policies.

---

## 2026-05-23 - Training run 1: predicted collapse

2M steps, ~1.5 hours. Followed the expected naive-2-policy trajectory exactly:

- 0-100k: hider +2.4, seeker -2.4 (random play baseline).
- 100k-524k: seeker steadily learns to chase. Hider returns drop to +1.2 (catch rate ~50%).
- 524k-2M: regression. By end of training both policies are back at random-baseline returns.

**Why it collapses:** non-stationarity. Seeker learns to chase pattern A, hider learns to evade pattern A. Once hider's evasion is good enough that the seeker stops getting tag signals, the seeker's policy drifts and forgets. Both regress to random.

Save-best logic kept the peak: seeker.pt is from the 524k iteration. hider.pt is from iteration 1 (where it won by default against the clueless seeker), so it never actually learned anything.

This is the failure mode the OpenAI paper explicitly fixes with self-play. Stage 4 territory.

---

## 2026-05-23 - Speed bug + velocity cap

Watched seeker.pt and noticed the seeker was crossing the arena in ~0.3 seconds. Physics wasn't broken, just nothing capping velocity.

Math: FORCE_SCALE=1500, damping=0.5/sec, mass=1. Per-step velocity gain = 25, per-step decay = 1.15%. Terminal velocity = 25/0.0115 ~ 2000 px/sec. Arena is 600 px wide.

The seeker had learned to mash max thrust constantly (locally optimal: catch fast = less per-step penalty). Added a velocity clamp inside env.step() after the physics step:

```
for name in self.agents:
    body = self.bodies[name]
    vx, vy = body.velocity
    speed = (vx*vx + vy*vy) ** 0.5
    if speed > MAX_VEL:
        scale = MAX_VEL / speed
        body.velocity = (vx*scale, vy*scale)
```

MAX_VEL=300 matches the obs normalization exactly, so velocity components are now precisely in [-1, 1] without relying on the np.clip safety net.

**Unexpected side effect: training got 5-6x faster.** Pymunk seems to burn a lot of cycles resolving collisions when bodies hit walls at 2000 px/sec. With cap=300, the same 2M-step run took 14 minutes instead of 1.5 hours.

---

## 2026-05-23 - Retrain + Stage 3 done

Same training setup, just the velocity-capped env. Same collapse pattern but with a lower peak:
- Peak: hider +1.7, seeker -1.7 at step 251k (catch rate ~29%).
- Final state: returns drifted back to +2.4 / -2.4 (random baseline).

Lower peak because random catch rate dropped from ~15% to ~5% with capped velocity. Less serendipity = less reward signal = harder bootstrap. Expected tradeoff for the visual win.

### Watching the trained policies

Loaded hider.pt + seeker.pt in watch_trained.py (deterministic actions = mean of the policy distribution):
- Two runs of ~20-30 episodes gave wildly different catch rates: 5/11 (45%) then 1/30 (3%).
- Across both, ~15% catch rate. Brittle policy, depends heavily on spawn config.
- Visually: seeker clearly chases (moves with intent toward the hider), hider drifts without real evasion (since hider.pt is iteration 1).

### Stage 3 status

Reproduced the 0:18 mark of the OpenAI video: a seeker that chases. Hider evasion is the next thing to fix, and that's exactly what self-play (Stage 4) addresses. The non-stationarity in Stage 3 is what prevents the hider from ever getting a stable training signal in the first place.

Calling Stage 3 done. Next: self-play with policy snapshotting and opponent sampling.

---

## 2026-07-13 — Stage 4: self-play (fixes the collapse)

Built `train_selfplay.py` to kill Stage 3's non-stationarity collapse. Core idea: never train a live agent against the live opponent. Each iteration:
- **Rollout A:** live hider vs a *frozen* seeker sampled from a pool of past seeker snapshots. Only the hider stores transitions and updates.
- **Rollout B:** mirror — live seeker vs a frozen hider from the hider pool.

A frozen opponent is stationary within an episode, so each learner sees a stable target instead of a co-adapting one. A new frozen opponent is resampled at every env reset (~8–10 per rollout) for generalization across the pool.

### Two design decisions that mattered

1. **Snapshot on improvement, not on a timer (v2).** v1 snapshotted every K iters; the pool filled with ~97% copies of barely-trained noise, so opponents were mostly random and self-play plateaued. Switched to snapshotting only when the live policy beats its all-time best — same trigger as save-best — so the pool is a *quality ladder* of progressively stronger past selves.
2. **Catch bonus ±0.05 → ±1.0.** With the old scaling a catch was ~2% of total episode signal vs ~98% per-step ±0.01, so PPO's gradient barely saw the catch event. ±1.0 makes a catch worth ~30% of max per-episode return, so the policy actually gets pushed toward/away from catches.

### Run: 977 iterations, 2M steps/agent

- Pools grew hider 1→7, seeker 1→7 (paired snapshotting: if either role improves, both get snapshotted, since the hider metric saturates at +2.4 and can't visibly "improve" on its own).
- Seeker best return climbed −2.4 (random) → −0.942, **with real late gains at iter 420 and iter 827 (near the very end)**. No regression to baseline.

**The headline: no collapse.** Stage 3 peaked at ~251k steps then drifted back to random by 2M. Stage 4 held — and improved — its gains across the full run. That's the whole point of self-play, confirmed.

### Eval: catch rate + cross-matchups (200 fixed-seed episodes, deterministic actions)

Head-to-head trained-vs-trained: **30.5% catch rate** vs Stage 3's 12.5%. But catch rate alone conflates "seeker got better" with "hider got worse." Cross-matchups isolate it:

| hider | seeker | catch rate | steps-to-catch |
|-------|--------|-----------|----------------|
| Stage 3 | **Stage 4** | 29.0% | 124 |
| Stage 4 | **Stage 4** | 30.5% | **149** |
| Stage 3 | Stage 3 | 12.5% | 132 |
| Stage 4 | Stage 3 | 13.5% | 129 |

Reading it:
- **Seeker learned a lot** — ~doubled its catch rate (12.5% → ~29–30%) against *any* hider. This is the big winner.
- **Hider learned modest, real evasion — delay, not escape.** Against the strong Stage 4 seeker it's caught at the same rate as the clueless Stage 3 hider but survives ~20% longer when caught (149 vs 124 steps). It learned to drag out the chase, not to get away.

### Why the hider only delays

In an empty arena with equal speeds, the hider fundamentally *cannot* escape a competent seeker — geometry doesn't allow it. It can only prolong survival. Decisive evasion requires **tools**: walls to break line-of-sight, movable boxes to build cover. That's Stage 5. So Stage 4 fixed the training-stability problem (the actual goal), and the hider hit the ceiling of what's achievable in an empty box.

### Stage 4 status

Done. Self-play with a quality-ladder opponent pool eliminates the non-stationarity collapse: both policies now improve and hold. Eval scripts added (`eval_headless.py`, `eval_cross.py`). Next: Stage 5 — the real hide-and-seek environment (walls, movable + lockable boxes, lidar, ego-centric observations) where tool-use can actually emerge.

---

## 2026-07-14 — Stage 5a: hide-and-seek env skeleton

Scoped Stage 5 (see `STAGE5_PLAN.md`): line-of-sight reward, boxes + walls + lock, 1v1, ramps deferred. Built into four sub-stages; this is 5a — the env skeleton, before LOS/lock.

New `env_hs.py` (`HideAndSeekEnv`), leaving `TagEnv` in `env.py` untouched. Key realization: the renderer was already built for this (3D boxes with lock icons, arbitrary interior walls, PREP/PLAY phase pill), so Stage 5 is almost entirely env-side.

What's in the skeleton:
- **Fixed map:** a 240×240 room in the top-left corner (arena edge walls + 3 interior segments) with a single 60px doorway in the bottom wall. Hider spawns inside, seeker outside.
- **Prep phase:** first 40% of the episode (96 of 240 steps) the seeker is frozen (force suppressed + velocity pinned to zero), hider moves freely, no reward accrues.
- **Movable boxes:** `N_BOXES=2` dynamic pymunk box bodies (mass 3 — pushable by an agent but heavy enough to build a wall), repositioned inside the room each reset.
- **Obs (21 dims):** self(4) + opponent(4 + visible flag) + 2 boxes×(4 + lock_state + visible flag). Lock/visible are stubbed (always visible, lock_state 0) until 5b.
- **Action (3 dims):** `[fx, fy, lock]`; the lock signal is inert in 5a.
- **Reward (temporary):** touch-tag, play-phase only — just to smoke-test motion/boxes/prep before LOS replaces it in 5b.

Verified headless: obs/act dims correct, seeker displacement during prep = 0.0, boxes get pushed, prep reward 0/0, episodes truncate at 240 cleanly. Also ran a 3-iteration self-play integration test — `train_selfplay.py`'s loop wires up to the new env (21/3 dims) with no changes needed. Added `watch_hs.py` (random-ish viewer) to eyeball geometry + prep freeze.

Next: 5b — the real game (LOS visibility reward + obs masking + lock mechanic), where hiding behind the barricade finally pays off.
