# Stage 5 Plan — Hide & Seek with tools (LOS reward, boxes + lock, 1v1)

Goal: reproduce the paper's **first** emergent behavior — a hider that pushes boxes to
barricade a doorway and locks them, shutting the seeker out. Locked-in decisions:

- **Perception/reward:** line-of-sight visibility (paper-faithful), not touch-tag.
- **Tools:** movable boxes + interior walls + lock. No ramps yet (Stage 5c).
- **Team size:** 1v1 (reuse Stage 4 self-play loop unchanged).

New env `env_hs.py` (`HideAndSeekEnv`); leave `TagEnv` in `env.py` intact. The renderer
already supports boxes (3D, lockable, lock icons), arbitrary interior walls, and the
PREP/PLAY phase pill + prep-timeline — **Stage 5 is almost entirely env-side work.**

---

## Sub-stages (each = one NOTES entry, matching project workflow)

### 5a — env skeleton, no lock/LOS yet (get physics + obs right)
- **Map:** 600×600 arena + interior walls forming one room in a corner with a single
  ~60px doorway. Fixed layout for v1 (absolute-coord obs implicitly encodes the fixed
  geometry, so no lidar needed yet — lidar comes when we randomize maps in a later stage).
- **Spawns:** hider inside the room, seeker outside. Enforce min distance as today.
- **Boxes:** `N_BOXES = 2` movable pymunk box bodies (dynamic, moderate mass so agents
  push them by contact). Spawn near the hider / inside the room.
- **Prep phase:** `PREP_FRACTION = 0.4`. During prep (steps `0 .. 0.4*MAX_STEPS`) the
  seeker's action is zeroed (frozen); hider moves freely. No reward accrues during prep.
- **Action:** grow to 3D `[fx, fy, lock]` (lock unused until 5b). `act_dim = 3`.
- **Obs (fixed map, N_BOXES=2 → 21 dims), all normalized [-1,1], masked by LOS in 5b:**
  - self: `x, y, vx, vy` (4)
  - opponent: `x, y, vx, vy, visible_flag` (5)
  - each box: `x, y, vx, vy, lock_state, visible_flag` (6) × 2 = 12
  - `lock_state`: 0 none / +1 locked-by-self / -1 locked-by-other
- **Reward for 5a only:** keep touch-tag temporarily so we can smoke-test that agents
  move and push boxes before wiring up LOS. Retrain briefly, eyeball in the renderer.
- **Exit criterion:** agents move, boxes get shoved around, prep phase visibly freezes
  the seeker, nothing crashes over a full training run.

### 5b — lock mechanic + line-of-sight reward (the real game)
- **LOS (v1):** unobstructed raycast, 360°, unlimited range. `space.segment_query_first`
  from seeker→hider filtered to wall+box shapes; if it hits something with fraction < 1,
  the hider is hidden. (View cone + range come later with agent orientation.)
- **Reward (main phase only):** per step, `+r` to seeker / `-r` to hider if seeker sees
  hider, flipped if not. `r = 1 / PLAY_STEPS` so episode total ∈ [-1, +1]. No tag reward.
  Zero reward during prep. This is what makes hiding behind cover pay off.
- **Obs masking:** when the opponent (or a box) is not in LOS, zero its pos/vel and set
  its `visible_flag = 0`. Agents literally cannot see through walls.
- **Lock action:** `lock > 0.5` → toggle the nearest box within `LOCK_DIST`. Unlocked box
  → lock to self. Self-locked → unlock. Other-locked → no-op (only the locking team can
  unlock, so a hider-locked box is permanently fixed vs the seeker).
- **Lock physics:** on lock, set the box body to STATIC (and `reindex_shapes_for_body`);
  on unlock, restore DYNAMIC. **Known risk:** pymunk body_type toggling needs a space
  reindex — verify a locked box both stops moving AND still occludes raycasts.
- **Exit criterion / the payoff:** over a self-play run, the hider learns to push boxes
  into the doorway during prep and lock them, and its main-phase "unseen" fraction climbs.
  Eval: fraction of episode the hider stays hidden, trained vs a random-box baseline.

### 5c — ramps (second arms-race: seekers ramp over walls, hiders lock ramps)
- Deferred. Needs agent orientation + a climb/vault mechanic. Scope after 5b lands.

### 5d — analysis & write-up
- Visibility-over-training curves, emergence timeline, the arms-race narrative vs the paper.

---

## What carries over unchanged
- `train_selfplay.py` — 1v1, so the quality-ladder self-play loop needs no changes; it just
  instantiates `HideAndSeekEnv` and reads `obs_dim=21, act_dim=3` off the space.
- `ppo_continuous.py` — continuous PPO already handles arbitrary obs/act dims.
- `renderer.py` — already draws boxes, walls, locks, prep phase. May add a seeker→hider
  LOS line as a debug nicety.
- `eval_headless.py` / `eval_cross.py` — adapt the "success" metric from catch-rate to
  hidden-fraction.

## Open risks to watch
1. pymunk body_type STATIC↔DYNAMIC toggle + reindex (lock mechanic).
2. Box mass tuning: too heavy = unpushable, too light = agents can't build a stable wall.
3. LOS reward can be noisy frame-to-frame; may need to reward "hidden for a contiguous
   stretch" rather than per-frame if learning is unstable.
4. Fixed map risks overfitting one layout; randomize + add lidar once 5b works.
