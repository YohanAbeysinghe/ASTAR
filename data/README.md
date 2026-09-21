# Dataset

The dataset is not stored in Git. Pass its root with `--data-root` or set
`ASTAR_DATA_ROOT`.

The root must contain `dataset_manifest.json` (or pass `--manifest-path`). The
manifest points to a JSONL samples file. Each record must provide:

- `clip_id`, `sample_id`, and `image_path`;
- `esdf_path` plus the clip's ESDF bounds, resolution, and robot radius;
- `object_goals` and/or `sampled_goals` with `goal_xy_m`;
- each usable goal's `path_plan.path_data_path`;
- path arrays for `path_xytheta_m`, `path_xy_m`, and `actions_vw`.

The repository-owned split files are in `configs/splits/`. They are passed
explicitly by the Slurm launchers and are kept in Git for reproducibility.

Example:

```text
/path/to/dataset/
├── dataset_manifest.json
├── samples.jsonl
├── <clip-id>/
│   ├── manifest.json
│   ├── images/
│   ├── esdf/
│   └── paths/
└── ...
```

Paths inside the manifest may be absolute or relative to the dataset root.
