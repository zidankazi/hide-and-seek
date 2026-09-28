# Saved checkpoints

The supported evaluation presets select these pairs automatically:

| Preset | Save-best files |
|---|---|
| `room` | `hs_lstm_hider.pt`, `hs_lstm_seeker.pt` |
| `tight-room` | `hs_roomt_hider.pt`, `hs_roomt_seeker.pt` |
| `two-hiders` | `hs_2v2_hider.pt`, `hs_2v2_seeker.pt` |

The reports in `results/` identify the exact files by SHA-256 hash. Files ending in `_final.pt`
contain final snapshots and can be selected with `evaluate.py --final`. The published results
use the save-best pairs only; the names do not imply a ranking across every behavior.

Other checkpoints preserve earlier experiments. They can differ in architecture, observation
size, and environment assumptions; see the [experiment index](../docs/experiments.md). Moving
these files did not change their contents. Additional arc snapshots are archived
but are not used in the evaluated study.

New `.pt` files in this directory are ignored by default to avoid accumulating training
outputs in version control. The existing study and historical weights are tracked.
