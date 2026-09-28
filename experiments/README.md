# Experiments

`scripts/` contains the earlier trainers and evaluators, `logs/` their training output,
and `media/` the older demo GIFs. The filenames match those used in [NOTES.md](../NOTES.md).

| Scripts | Experiment |
|---|---|
| `train_multi.py`, `train_selfplay.py` | Tag and frozen-opponent self-play |
| `train_hs.py`, `train_hs2.py`, `train_hs2c.py` | Box locking, open arena, and seeker curriculum |
| `train_hs7.py` | Feed-forward policies with a ramp |
| `train_hs7_lstm.py`, `train_hs7_lstm_vec.py` | Recurrent policies and batched rollouts |
| `train_hs7_lstm_vec_rc*.py` | Reverse curricula for doorway placement |
| `train_hs7_roomt.py` | Smaller room |
| `train_hs7_arc.py`, `train_hs7_arc2v2.py` | Combined curricula and two hiders |
| `train_hs7_climb.py` | Unfinished climbing experiment |

Use the root `evaluate.py` for current results. The older evaluators retain their original
metrics and often load weights relative to the working directory. For example:

```bash
# From the repository root:
cd checkpoints
uv run --project .. python ../experiments/scripts/eval_hs7_lstm.py hs_2v2 2 --layout=roomt --speed=1.4 --nh=2 --ns=2
```

The trainers also save weights in the working directory. For a new run, use a separate
`runs/` directory and copy any warm-start weights there. The old training scripts have not
all been rerun during the cleanup. `timeline_hs7.py` reads the files in `logs/`.
