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
