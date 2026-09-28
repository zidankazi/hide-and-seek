# Experiment index

The concluded study uses `evaluate.py`, `demo.py`, and the three configurations in
[the report](study.md). Core modules live in `hide_and_seek/`, historical scripts in
`experiments/scripts/`, logs in `experiments/logs/`, and saved weights in `checkpoints/`.
The [file map](../experiments/file-map.json) maps names in the journal to their new paths.
Historical scripts retain their original behavior and assumptions; their printed labels and
metrics are not the current results interface.

| Files / prefixes | Role |
|---|---|
| `env_nav.py`, `ppo_continuous.py`, `policy.pt` | Early continuous-control navigation work |
| `env.py`, `train_multi.py`, `train_selfplay.py`, `hider.pt`, `seeker.pt` | Tag and self-play experiments |
| `train_hs.py`, `train_hs2.py`, `train_hs2c.py` | Early hide-and-seek training |
| `train_hs7.py`, `hs_ramp_*`, `hs_mega1v1_*` | Feed-forward room and ramp experiments |
| `train_hs7_lstm_vec.py`, `hs_lstm_*` | Recurrent room experiments; `room` preset |
| `train_hs7_lstm_vec_rc*.py`, `hs_lstmrc*` | Reverse-curriculum attempts |
| `train_hs7_roomt.py`, `hs_roomt_*` | Tight-room experiment; `tight-room` preset |
| `train_hs7_arc.py`, `hs_arc*` | Combined-curriculum experiments |
| `train_hs7_arc2v2.py`, `hs_2v2_*` | Two-hider experiment; `two-hiders` preset |
| `eval_hs7*.py`, `timeline_hs7.py`, `watch_hs7.py` | Historical evaluation and inspection tools |
| `*.log` | Training records, not current evaluation reports |

The ordinary checkpoint filenames are the trainers' historical save-best snapshots. A
`_final.pt` suffix means the last saved snapshot, not a stronger or more reliable policy.
The final study records exact hashes rather than relying on names as evidence of quality.

`hide_and_seek/env_hs.py` also contains optional climbing mechanics added during later exploration. The
study presets explicitly disable them. There is no verified climbing result in this release.
`experiments/scripts/train_hs7_climb.py` preserves the unfinished training attempt. Older
demo GIFs and their generator remain under `experiments/` as historical material, not current
evidence. The earlier visual report was withdrawn because its claims were superseded.

See [experiments/README.md](../experiments/README.md) for historical script invocation.
