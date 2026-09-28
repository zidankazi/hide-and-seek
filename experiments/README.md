# Experiment history

This folder preserves earlier scripts, training logs, and demo material. Their original
filenames match the development journal. [file-map.json](file-map.json) records the moves.

- `scripts/`: training, historical evaluation, and viewing scripts. Imports now point to
  `hide_and_seek`; the research algorithms and historical command arguments remain intact.
- `logs/`: saved training output.
- `media/`: earlier selected GIFs. Their original captions and interpretations are superseded
  by the [study report](../docs/study.md).

Use `evaluate.py` and `demo.py` at the repository root for the supported study workflow.
These older scripts often choose checkpoints relative to the working directory. For example,
to run the historical LSTM evaluator on the two-hider checkpoint pair:

```bash
# From the repository root, after uv sync --locked:
cd checkpoints
uv run --project .. python ../experiments/scripts/eval_hs7_lstm.py hs_2v2 2 --layout=roomt --speed=1.4 --nh=2 --ns=2
```

This prints the legacy proximity metric, not the current sightline metric. No training runs
are required to reproduce the final study. Training scripts retain their original convention
of writing weights in the working directory; use a separate `runs/` directory for a new
experiment and copy any required warm-start weights into it. Training all historical scripts
has not been revalidated as part of the closeout.

`timeline_hs7.py` reads log files from its working directory; its inputs are now under `logs/`.
The optional climbing trainer remains an unfinished experiment, outside the final results.
