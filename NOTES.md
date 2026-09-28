# Experiment notes

This is a shortened record of the training runs. Numbers in the dated sections come from
the original evaluations and may use older metrics. The current evaluation is in
[walkthrough.ipynb](walkthrough.ipynb), with raw data in `results/`.

## May 21–22: continuous control

Started with continuous PPO on Pendulum, then a point-to-point navigation environment in
Pymunk. Actions were two-dimensional thrust; the policy used a Gaussian distribution with
a learned `log_std`.

The first navigation run improved around 340k–460k steps, then diverged. Saving on every
update overwrote the useful policy. Added save-best logic. A second run kept its peak
checkpoint, though its performance was worse than the first run's peak.

The renderer went through several versions before settling on the current top-down view.
`env_nav.py` and `ppo_continuous.py` contain this early work.

## May 23: two-agent tag

Added a hider and seeker with separate policies. Training both live policies at once was
unstable: the seeker improved, the hider changed behavior, and the seeker lost its gains.

A separate physics issue made agents cross the arena too fast. Capping velocity at 300 pixels
per second improved the display and reduced training time, likely by reducing collision work.
The retrained seeker still regressed later in the run.

Files: `env.py`, `train_multi.py`.

## July 13: self-play

Changed training so each live policy faced a frozen opponent sampled from earlier snapshots.
The first pool filled with too many similar weak policies, so snapshots were added on
improvement rather than at a fixed interval. Also increased the tag reward.

The historical 200-episode evaluation reported a 30.5% catch rate, compared with 12.5% for
the earlier pair. Cross-matchups suggested most of the improvement came from the seeker.
The new hider delayed catches but did not escape more often.

Files: `train_selfplay.py`, `eval_headless.py`, `eval_cross.py`.

## July 14: walls, boxes, and visibility

Replaced tag with a visibility reward. Seekers score when any seeker sees any hider;
hiders score when all hiders are hidden. Added movable boxes, locks, masked observations,
and a preparation phase with frozen seekers.

One reset bug came from moving boxes without updating Pymunk's spatial index. The first
observation after reset used stale positions until `reindex_shapes_for_body` was added.

The room run often ended with a locked box, but most boxes stayed away from the doorway.
The hider spent 91.9% of play unseen; a zero-action hider still managed 83.9%. The room itself
provided substantial cover.

An open-arena run removed that shelter. Hidden time fell to 32.8%, compared with 31.3% for
the zero-action hider. Locking boxes did not produce much useful cover in that setup.

Files: `train_hs.py`, `eval_hs.py`; checkpoint prefixes `hs` and `hs_open`.

## July 14–15: more agents and a seeker curriculum

Tried two hiders, two seekers, six boxes, and 10M steps per team. Hiders still performed poorly.
The save-best rule also caused trouble: the seeker reached its maximum score against weak
early opponents, so its saved checkpoint stopped updating. Added final snapshots as a separate
output. The original run's final weights were not retained.

The next run began with one active seeker and introduced the second between 4M and 6M steps.
Training returns improved, but final head-to-head evaluation still showed little effective
cover-building. Mean return against a changing opponent pool was not a reliable way to
compare checkpoint quality.

Files: `train_hs2.py`, `train_hs2c.py`, `eval_hs2.py`.
The `hs_2v2` prefix was reused later; the current files with that name belong to the recurrent
room experiment, not this early open-arena run.

## July 17: ramp experiments and unstable exploration

Added a movable ramp. Nearby agents could see over interior walls and boxes within a fixed
range. At this point the ramp affected visibility only.

A 30M-step attempt failed with NaNs around 19.9M steps. The exploration standard deviation
had grown too large. Clamping `log_std` exposed a problem with earlier interpretations:
failed training could not simply be attributed to a lack of compute.

Other attempts changed doorway geometry, box mass and size, locking behavior, entropy,
network capacity, rollout size, and curriculum spawns. With two seekers, the policies spent
more time near the ramp than with one seeker. The metric at the time counted proximity;
it did not establish that the ramp helped them see a hider.

Files: `train_hs7.py`; logs `train_hs7_run*.log` and the later named runs.

## July 18–20: recurrence and longer training

Added an LSTM and kept hidden state separately for each agent. Updates use whole episode
sequences so minibatches do not scramble the recurrent state.

Profiling pointed to small, repeated policy forward passes as a bottleneck. Running several
environments with a batched forward pass increased throughput from roughly 201 to 881
steps per second in the original benchmark.

Extended the recurrent room run to roughly 120M total steps. The doorway-box metric remained
zero in the saved evaluations. Ramp proximity varied across checkpoints instead of increasing
steadily. This run did not produce the intended construction behavior; it does not establish
that more compute could never help.

Files: `ppo_recurrent.py`, `train_hs7_lstm.py`, `train_hs7_lstm_vec.py`.
Checkpoints: `hs_lstm20m`, `hs_lstm`.

## July 21–22: reverse curriculum and a smaller room

Tried teaching the doorway action backward. Early episodes started with a box already in
place, then gradually required more pushing. One recurring failure was locking the box
before moving it into the doorway.

A later version spawned the hider farther behind the box to encourage pushing before locking.
The smaller `roomt` layout also put seekers closer to the doorway. The original evaluations
reported doorway-box rates around 20–25% for that run.

Several things changed together, including geometry and the curriculum. Those results did
not isolate a single cause. The current evaluation of the saved pair gives 14.5%; see the
notebook for the current protocol.

Files: `train_hs7_lstm_vec_rc.py`, `train_hs7_lstm_vec_rc3.py`, `train_hs7_roomt.py`.
Checkpoints: `hs_lstmrc`, `hs_lstmrc3`, `hs_roomt`.

## July 23–27: combined curricula and two hiders

Combined the smaller room with faster seekers and spawn curricula for doorway placement
and ramp interaction. Then added a second hider with a curriculum that placed one near the
door and one near the ramp.

The saved two-hider policies sometimes place a box at the doorway and occasionally lock
the ramp. Calling this a complete tool-use sequence was premature. Aggregate event counts
do not show that the actions happened in order or formed a successful strategy.

Files: `train_hs7_arc.py`, `train_hs7_arc2v2.py`.
Checkpoints: `hs_arc*`, `hs_2v2`.

## July 28: checking what the ramp actually does

While preparing demos, the measured behavior did not match the description. Being near the
ramp was counted as tool use even when it provided no useful sightline. The code also had
no wall traversal in the evaluated configuration.

Added a stricter visibility check: query the same seeker-hider pair with elevation enabled
and disabled. Also added experimental collision changes for climbing, but no verified
climbing result came from that work.

The old note that the ramp never moves was too broad. In the current two-hider evaluation,
mean final displacement is 44.62 pixels, and the zero-action-hider comparison reaches 58.25.
Movement by itself doesn't show who moved the ramp or whether it helped.

## Final evaluation

`evaluate.py` uses the saved pairs with seeds 0–199, a fresh physics environment per episode,
and reset LSTM state. Each episode has 144 preparation steps and 216 play steps. There are
three conditions per setup: both seekers active, one active seeker, and zero-action hiders.

The inactive seeker remains a physical body. Zero-action hiders can still be pushed. These
comparisons help inspect the saved policies; they are not independent training baselines.
Older evaluations reused environments between episodes and used different metrics, so the
new numbers are not an exact reconstruction of every earlier table.

A few limits matter when reading the results:

- A locked box in the doorway region may still leave the hiders exposed.
- A ramp-assisted sightline may be redundant if another seeker already sees a hider.
- A ramp lock does not establish successful defense. In two-hider seed 14, both lock and
  doorway-box flags are true, yet the hiders are visible throughout play.
- The 200 seeds vary the starting positions for one policy pair. They are not independent
  training runs, and these experiments do not separate the effects of compute, geometry,
  team size, and curricula.

The project ends with partial behaviors and a working training and evaluation pipeline.
Purposeful ramp transport, climbing, and a reliable sequence of strategies remain unproven.
