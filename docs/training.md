# Training notes

## Fresh run

A clean run loads `checkpoints/pi05_base/params`, initializes ASTAR's action
and goal adapters, and creates a new experiment directory. Do not pass
`--resume`, `--relabel-*`, or `--migrate-*` to a fresh run.

The checked-in two-node launcher reproduces the current waypoint-only setup:
`sampled_goal_fraction=1.0`, adapter-only optimization, 16 waypoints, and the
clearance/goal/progress/heading/smoothness energy objective.

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
