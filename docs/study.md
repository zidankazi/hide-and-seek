# Hide & Seek: findings and limitations

## Outcome

I set out to reproduce a sequence of tool-use behaviors in a small, custom hide-and-seek
environment. I built the environment and learning pipeline, trained several policy families,
and observed partial object interaction. I did not establish the complete intended sequence.
The study ends here with saved policies, replayable examples, and a documented evaluation.
Further training is outside this release's scope.

The engineering work includes continuous PPO, LSTM policies, frozen-opponent self-play,
vectorized rollouts, and a Pymunk environment using the PettingZoo parallel interface. The
historical journal records failed runs and changes to the task, including a run extended to
roughly 120 million steps. That history describes the development process; it is not a
controlled comparison of compute budgets.

## Evaluation protocol

The closeout evaluation runs the historical save-best checkpoint pairs in three configurations:

| Preset | Checkpoint prefix | Layout | Hiders / seekers | Seeker speed multiplier |
|---|---|---|---|---|
| `room` | `hs_lstm` | room | 1 / 2 | 1.0 |
| `tight-room` | `hs_roomt` | tight room | 1 / 2 | 1.0 |
| `two-hiders` | `hs_2v2` | tight room | 2 / 2 | 1.4 |

Each configuration uses 200 seeds (0–199) in each of three conditions: trained policies,
one active seeker, and zero-action hiders. The two comparison conditions reuse the same
weights. The inactive seeker remains a physical body in the scene; it does not contribute
to visibility or proximity metrics. Zero-action hiders can still be moved by collisions.
These conditions are diagnostics, not separately trained baselines or pure geometry controls.

Each episode starts with a fresh environment and zeroed recurrent state, runs for 360 physics
steps, and has 144 preparation steps followed by 216 play steps. All actions are policy means
on CPU with one Torch thread. Training spawn assists are absent; the preparation phase remains
part of the game. Climbing is off and the original elevation-based visibility mechanic is on.

The JSON reports store each episode, aggregate means, settings, runtime versions, and SHA-256
hashes for weights and source files. The run used Python 3.13.13, Torch 2.12.0, and Pymunk 7.2.0
on macOS ARM64. Reproduction means rerunning this recorded protocol; bitwise agreement on other
platforms is not promised. Only the save-best pairs were evaluated for the published table.
The CLI also supports final checkpoints, but those are not additional evidence for this table.

## Measurements

| Configuration | Doorway box (% episodes) | Ramp sightline (% play steps) | Hider ramp lock (% episodes) |
|---|---:|---:|---:|
| `room` | 0.0% | 12.7% | 0.0% |
| `tight-room` | 14.5% | 8.2% | 0.0% |
| `two-hiders` | 26.5% | 10.6% | 3.0% |

- **Doorway box:** percentage of episodes ending with a hider-locked box center inside the
  environment's designated doorway region. This is a placement proxy. It does not verify
  an impassable doorway, a safe shelter, or a complete construction strategy.
- **Ramp sightline:** mean fraction of play steps where at least one active seeker sees a
  hider with elevation and cannot see that same hider when elevation is temporarily removed
  from the visibility query. Another seeker may already see a hider, so this is not necessarily
  an improvement in the team's reward. The query does not advance physics or rerun the episode.
- **Hider ramp lock:** percentage of episodes ending with the hider team owning the ramp lock.
  It does not establish that the lock denied access or improved survival.
- **Hidden fraction:** mean fraction of play steps where every hider is out of sight of all
  active seekers. There is no contact capture in this version of the environment.
- **Ramp displacement:** distance from spawn to final position, plus the greatest such distance
  at any point in the episode. Neither metric identifies which agent moved it or why.

The table mixes episode frequencies and time fractions because the behaviors require different
measurements. It does not report three stages of one successful sequence.

## Comparison conditions

| Configuration | Hidden: trained | Hidden: zero-action hiders | Sightline: trained | Sightline: one active seeker |
|---|---:|---:|---:|---:|
| `room` | 83.7% | 78.0% | 12.7% | 10.4% |
| `tight-room` | 75.2% | 71.7% | 8.2% | 4.1% |
| `two-hiders` | 66.3% | 58.5% | 10.6% | 6.3% |

The trained hiders spend more time unseen than zero-action hiders in these evaluations. The
sightline frequency is lower with one active seeker. These comparisons describe the saved
policies under changed play conditions. They do not isolate why a behavior emerged during
training: layouts, curricula, checkpoint histories, team sizes, and speeds also differ.

## Corrections to the earlier interpretation

**Proximity was too broad a tool-use measure.** Earlier evaluation counted steps near the ramp.
The current evaluator also asks whether elevation changes a seeker-to-hider sightline. Tests
cover an occluding wall, a clear sightline, a second hider, and an inactive seeker.

**Visibility is not traversal.** In the evaluated task, ramp proximity changes the line-of-sight
mask within a range. It does not change collision rules. Earlier descriptions of seekers
climbing over walls were incorrect for these checkpoints' evaluated mechanics.

**Movement is not purposeful transport.** The new two-hider evaluation measures mean final ramp
displacement of 44.62 pixels and mean maximum displacement of 46.14 pixels in a 600-pixel arena.
The zero-action-hider condition has even larger mean final displacement (58.25 pixels). The
older journal's blanket claim that the ramp never moves does not describe this evaluation.
The measurements do not establish transport toward a useful destination or its strategic value.

**Rare locking is not proof of coordinated defense.** Six of 200 two-hider episodes end with a
hider ramp lock. In seed 14, both the doorway-box and ramp-lock flags are true, yet the hiders
are visible for every play step. Nonzero event counts do not establish successful cooperation.

**The experiments do not prove a general law about emergence.** The journal's claim that
necessity, rather than compute, determines tool use is too strong. Several interventions
changed at once, curricula provided scaffolding, and no multi-seed retraining study isolated
those effects. Failing to obtain a behavior in the attempted runs does not show that scale
cannot help. Likewise, changing team size does not prove a single agent is incapable of
learning a sequence of actions.

## Why these numbers differ from the journal

This is a fresh evaluation of the current saved files, not a recreation of every historical
run. Earlier scripts reused a physics environment between episodes, reported a looser ramp
metric, and did not save a complete runtime and source manifest alongside each table. The
closeout protocol creates a fresh environment for each seed so a selected episode can be
replayed by itself. I have not attributed each numerical difference to a specific cause.
Use the JSON reports for current figures and the journal for historical context.

## Evidence and limits

The [doorway example](../media/barricade.gif) uses `tight-room`, seed 10. The
[sightline example](../media/sightline.gif) uses `room`, seed 136, selected for a high sightline
fraction. The [ramp-lock example](../media/ramp-lock.gif) uses `two-hiders`, seed 25, selected
from the six lock-positive episodes. Matching JSON files record the selection and measurements.
The examples illustrate behaviors; the 200-episode results provide their frequencies.

Two hundred evaluation seeds sample spawn variation for one saved policy pair per configuration.
They are not 200 independent training runs. Save-best selection, engineered spawn curricula,
manual environment changes, and the absence of independent retraining limit any causal claim.
The study did not test generalization to unseen layouts or validate the experimental climbing
extension. The result is a working learning system with partial behaviors and measured limits.

## Reproduce

From the repository root after `uv sync --locked --python 3.13`:

```bash
uv run python evaluate.py --preset room --episodes 200 --output results/room-rerun.json
uv run python evaluate.py --preset tight-room --episodes 200 --output results/tight-room-rerun.json
uv run python evaluate.py --preset two-hiders --episodes 200 --output results/two-hiders-rerun.json
uv run python demo.py --preset tight-room --seed 10 --output media/barricade-rerun.gif
uv run python demo.py --preset room --seed 136 --output media/sightline-rerun.gif
uv run python demo.py --preset two-hiders --seed 25 --output media/ramp-lock-rerun.gif
```

The original inspiration is [Baker et al.'s hide-and-seek research](https://openai.com/index/emergent-tool-use/).
This project uses its own simplified environment and training implementation; it is a partial
reproduction attempt, not a replication of that paper's results.
