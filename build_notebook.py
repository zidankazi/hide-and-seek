"""Build the walkthrough from the study's current prose and recorded results."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build():
    report = (ROOT / "docs/study.md").read_text()
    metrics = report.split("## Measurements\n\n", 1)[1].split("## Comparison conditions", 1)[0]
    corrections = report.split("## Corrections to the earlier interpretation\n\n", 1)[1].split(
        "## Why these numbers differ", 1
    )[0]
    cells = [
        nbf.v4.new_markdown_cell("""# Hide & Seek: a completed study with partial results

I built a 2D hide-and-seek environment, PPO, self-play, and recurrent policies to explore
object use. Some saved agents place and lock boxes near a doorway or gain visibility near
a ramp. I did not establish the intended sequence of construction, purposeful ramp transport,
wall traversal, and effective ramp defense.

This walkthrough uses existing checkpoints. None of its cells starts training.
The [study report](docs/study.md) contains the protocol and limitations.
The [journal](NOTES.md) preserves the development history, including superseded interpretations.
"""),
        nbf.v4.new_markdown_cell("""## Selected episodes

Blue agents hide, red agents seek, orange objects are boxes, and the green wedge is a ramp.
The red line marks a sightline that depends on elevation. Ramp proximity changes visibility;
climbing is disabled in these evaluations. These episodes illustrate behaviors and are not
representative success rates.

### Doorway box: tight-room, seed 10
![Selected doorway-box episode](media/barricade.gif)

### Ramp-assisted sightline: room, seed 136
![Selected ramp-sightline episode](media/sightline.gif)

### Ramp lock: two-hiders, seed 25
![Selected ramp-lock episode](media/ramp-lock.gif)

The ramp-lock example is one of six lock-positive episodes in 200. A lock does not establish
successful defense. Matching JSON files beside the GIFs record their seeds and measurements.
"""),
        nbf.v4.new_markdown_cell("## Recorded evaluation\n\n" + metrics),
        nbf.v4.new_code_cell("""# Inspect recorded results without rerunning the simulations.
import json
from pathlib import Path

root = Path.cwd()
assert (root / "evaluate.py").exists(), "Open the notebook from the repository root."
for preset in ("room", "tight-room", "two-hiders"):
    report = json.loads((root / "results" / f"{preset}.json").read_text())
    print(preset, report["protocol"]["episodes_per_condition"], "episodes per condition")
    print(json.dumps(report["conditions"]["trained"]["mean"], indent=2))"""),
        nbf.v4.new_markdown_cell("## Correcting the interpretation\n\n" + corrections),
        nbf.v4.new_markdown_cell("""## Run a small check

Install dependencies with `uv sync --locked --python 3.13 --extra notebook` and choose
the repository's `.venv` as the kernel. The next cell evaluates two episodes in each condition.
It checks that the pipeline runs; it does not estimate a rare behavior's frequency.
"""),
        nbf.v4.new_code_cell("""import subprocess
import sys

subprocess.run(
    [sys.executable, "evaluate.py", "--preset", "two-hiders", "--episodes", "2"],
    cwd=root,
    check=True,
)"""),
        nbf.v4.new_markdown_cell("""## Reproduce the full results

Run `evaluate.py` with `--episodes 200` for each preset. Save new output files so you can compare
them with the recorded reports. Reports include per-episode rows, runtime versions, source
hashes, and checkpoint hashes. Use `demo.py --preset room --seed 136 --output media/replay.gif`
to reproduce the sightline example.

These are evaluations of one saved policy pair per configuration. They do not measure variance
across independent training runs or isolate the effects of compute, curricula, and geometry.
The project is concluded at this scope. See [README.md](README.md) for setup and
[docs/experiments.md](docs/experiments.md) for the historical file index.
"""),
    ]
    # Stable IDs make regeneration deterministic and keep notebook diffs readable.
    for index, cell in enumerate(cells):
        cell["id"] = f"study-{index:02d}"
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "language_info": {"name": "python"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / "walkthrough.ipynb")
    print(f"Wrote walkthrough.ipynb ({len(cells)} cells)")


if __name__ == "__main__":
    build()
