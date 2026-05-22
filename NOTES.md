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
