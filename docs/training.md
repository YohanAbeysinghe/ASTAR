# Training notes

## Fresh run

A clean run loads `checkpoints/pi05_base/params`, initializes ASTAR's action
and goal adapters, and creates a new experiment directory. Do not pass
`--resume`, `--relabel-*`, or `--migrate-*` to a fresh run.

The checked-in launchers use a fixed 1,024-example multimodal condition pool:
1/6 sampled goals and 5/6 object goals, with independent text, image, and
waypoint masks for object goals. Startup fails if any required modality is
absent. They force aggregation toward 64 rounds while retaining convergence
stopping, and ignore valid-map obstacle samples at or above 0.2 m for learning.
The adapter-only optimization, 16-waypoint output, and energy-term weights are
unchanged.

For a single node with four RTX A6000 GPUs, use
`slurm/train_1node_4gpu_a6000.sbatch`. It keeps the global batch at 16 and uses
four-way FSDP, giving four examples per 48 GB GPU.

## Weights & Biases

W&B remains opt-in for Slurm jobs. Enable it with `WANDB_ENABLED=1`. Training
defaults to the `yohanab/astar` project; `WANDB_ENTITY` and `WANDB_PROJECT` can
override that destination for Slurm jobs. Authentication uses `WANDB_API_KEY`
or credentials previously stored by `wandb login`. The run configuration
records the requested modality probabilities, the realized modality counts,
the 1,024 example pool size, the ESDF cutoff, and the forced round target.

## Resume

Pass `--resume` only with the same experiment name and resume-critical
configuration. ASTAR restores the latest committed model, optimizer, and
matching aggregation replay. Objective or conditioning migrations are guarded
by explicit flags in the trainer and should not be used for the first clean
ASTAR experiment.

## Checkpoint retention

Only the latest committed full checkpoint is retained. Replay archives and
trainable-only exports are pruned to the same retained step. This prevents an
experiment from accumulating historical multi-gigabyte checkpoints.
