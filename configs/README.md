# Experiment configuration

ASTAR keeps stable dataset splits in this directory. Training hyperparameters
are currently explicit in the canonical Slurm launcher so the exact command is
captured in the scheduler log and in each run's
`monitoring/configuration.json`.

`train_clip_ids.txt` is the full usable training split and
`eval_clip_ids.txt` is the clip-disjoint validation/evaluation split.
