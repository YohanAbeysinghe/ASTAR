# ASTAR

**Action-Label-Free Spatial Trajectory Aggregation and Refinement via
Energy-Based Self-Training**

ASTAR trains a pi0.5-based navigation model without using recorded robot
actions as targets. It constructs goal-biased waypoint priors, obtains local
trajectory-improvement directions from a differentiable geometric energy,
aggregates those states in replay, and trains the model to predict the
refinement field. Geometry and ESDF maps supervise training but are not model
inputs at inference time.

This repository contains the standalone training and offline evaluation stack.
It deliberately excludes datasets, generated visualizations, deployment/ROS
code, W&B caches, and historical experiment checkpoints.

## Method

For a fixed pool of observations and goals, ASTAR repeatedly:

1. initializes dynamically feasible, goal-biased waypoint paths;
2. evaluates a geometric trajectory energy;
3. computes a projected negative-energy direction;
4. stores the state and direction in replay;
5. trains the pi0.5 field to imitate all retained refinements; and
6. advances paths with the learned field before the next aggregation round.

The current objective combines footprint clearance, all-waypoint goal
progress, retreat prevention, early heading, and smoothness terms. It targets
0.25 m beyond the unchanged robot radius, while physical collision
(`footprint clearance < 0`) remains available as a separate safety diagnostic.
Goal-progress weights rise linearly from zero at the fixed origin to one at the
endpoint. Goal-aware segment caps use straight-line spacing with an explicit
1.25x detour allowance, bounded by the physical step cap.

## Repository layout

```text
ASTAR/
├── src/astar/                 # model, data, training, and evaluation code
├── configs/splits/            # versioned train/evaluation clip splits
├── slurm/                     # thin cluster launchers
├── tests/                     # CPU/unit and distributed integration tests
├── docs/                      # training notes
├── checkpoints/               # ignored models; README is tracked
├── data/                      # ignored dataset; README is tracked
└── third_party/openpi/        # pinned pi0.5/OpenPI Git submodule
```

## Installation

ASTAR requires Linux, Python 3.11, and an NVIDIA GPU for real training. Install
[`uv`](https://docs.astral.sh/uv/) and clone recursively:

```bash
git clone --recurse-submodules https://github.com/YohanAbeysinghe/ASTAR.git
cd ASTAR
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra dev
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The OpenPI submodule is pinned to the working fork and revision used by this
implementation:

```text
https://github.com/gershom96/openpi
a324ec445007031ed5b11d708ad764406b1f55ee
```

Do not replace it with an unpinned OpenPI `main` checkout when reproducing a
run.

## Download the pi0.5 checkpoint

The 12 GB base checkpoint is an external model artifact, not a Git object. The
submodule provides the exact OpenPI code that downloads and loads it:

```bash
uv run astar-download-pi05
```

This fetches the official parameters from:

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

and makes them available at:

```text
checkpoints/pi05_base/params
```

By default the local path is a symlink to OpenPI's cache, avoiding a second
12 GB copy. Set `OPENPI_DATA_HOME=/path/to/cache` before downloading to choose
the cache location, or use `--copy` for a physical copy. See
[`checkpoints/README.md`](checkpoints/README.md) for details.

Verify the installation and checkpoint:

```bash
uv run astar-verify --require-checkpoint
```

## Dataset

ASTAR does not distribute the training dataset. Prepare the manifest, images,
path arrays, and ESDF arrays using the schema in
[`data/README.md`](data/README.md), then expose the location explicitly:

```bash
export ASTAR_DATA_ROOT=/absolute/path/to/dataset
```

The checked-in training and evaluation clip lists are disjoint and live under
`configs/splits/`.

## Training

The canonical launcher targets two Slurm nodes with eight GPUs each:

```bash
ASTAR_DATA_ROOT=/absolute/path/to/dataset \
  sbatch slurm/train_2node_16gpu.sbatch
```

Override the generated experiment name if desired:

```bash
ASTAR_DATA_ROOT=/absolute/path/to/dataset \
EXP_NAME=astar_clean_v1 \
  sbatch slurm/train_2node_16gpu.sbatch
```

For direct single-process use, inspect all supported settings with:

```bash
uv run astar-train --help
```

A clean experiment must use a new name and must not pass `--resume` or any
migration/relabel flags. Training outputs are stored under:

```text
checkpoints/astar_pi05_energy_aggregation/<experiment-name>/
```

Only the latest committed full checkpoint and its matching aggregation replay
are retained.

## Evaluation

Evaluate a completed experiment with:

```bash
uv run astar-eval \
  --checkpoint-root checkpoints/astar_pi05_energy_aggregation/astar_clean_v1 \
  --data-root "$ASTAR_DATA_ROOT" \
  --eval-split-ids configs/splits/eval_clip_ids.txt \
  --steps <checkpoint-step>
```

The evaluator automatically reads the training configuration saved inside the
checkpoint directory. Evaluation outputs go to `outputs/evaluation/` and are
not committed.

## Tests

```bash
uv run pytest
```

The multi-process distributed integration test is more expensive than the
pure unit tests but runs on CPU with simulated JAX controllers.

## Reproducibility

Each run records its resolved training arguments, goal-conditioning counts,
dataset split paths, JAX process/device counts, and aggregation algorithm
version. Exact resumption requires the full checkpoint and its matching
`aggregation_state_<step>.npz` replay archive.

## License

ASTAR is released under the MIT License. The OpenPI submodule and pi0.5 model
artifacts retain their own upstream licenses and terms.
