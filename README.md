# GEAR-Nav

GEAR-Nav trains a vision-language navigation policy to refine local waypoint
trajectories using an energy-based geometric oracle. The trained policy does
not require an ESDF as input.

## Setup

Requires Linux, Python 3.11, `uv`, and NVIDIA GPUs for training.

```bash
git clone https://github.com/YohanAbeysinghe/GEAR-Nav.git
cd GEAR-Nav
git submodule update --init --recursive third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra dev
uv run gear-nav-download-pi05
uv run gear-nav-verify --require-checkpoint
```

The pi0.5 download uses the OpenPI cache and links the parameters into
`checkpoints/pi05_base/params` by default. Checkpoints and datasets are not
included in this repository.

## Data and training

Set `GEAR_NAV_DATA_ROOT` to the prepared dataset directory. The dataset needs a
`dataset_manifest.json` and its referenced sample, image, ESDF, goal and path
files. Train and evaluation splits are in `configs/splits/`.

```bash
export GEAR_NAV_DATA_ROOT=/path/to/dataset
uv run gear-nav-train --help
sbatch slurm/train_1node_4gpu_a6000.sbatch
```

Adjust the Slurm account, partition and GPU settings for your cluster before
submitting. Other launchers are in `slurm/`; run `uv run gear-nav-eval --help`
for evaluation options. Use a new experiment name for a fresh run; resume only
with the matching experiment and configuration.

## Tests

```bash
uv run pytest -m 'not manual'
```

## Compatibility

The project was previously named ASTAR. Existing `astar` Python imports,
`astar-*` commands and `ASTAR_DATA_ROOT` remain supported. Use either
`gear_nav` or `astar` consistently within a Python process. The original machine
also has an `ASTAR` workspace symlink pointing to `GEAR-Nav`; this link is not
created by cloning the repository.

Licensed under the MIT License. OpenPI and the pi0.5 checkpoint have their own
licenses.
