# Checkpoints

| Evaluation preset | File prefix |
|---|---|
| `room` | `hs_lstm` |
| `tight-room` | `hs_roomt` |
| `two-hiders` | `hs_2v2` |

Each prefix has a hider and seeker file. The default pair uses the trainer's save-best
rule; `_final.pt` files hold the last snapshot. Neither is necessarily best at every behavior.
The JSON files in `results/` record the hashes of the evaluated weights.

Other files belong to earlier experiments and may use different network or observation
sizes. See [NOTES.md](../NOTES.md). New `.pt` files are ignored by default.
