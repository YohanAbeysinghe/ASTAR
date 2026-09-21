# Checkpoints

Checkpoint payloads are intentionally excluded from Git.

Install the official Physical Intelligence pi0.5 base parameters with:

```bash
uv run astar-download-pi05
```

ASTAR downloads `gs://openpi-assets/checkpoints/pi05_base/params` through
OpenPI's downloader and creates this local layout:

```text
checkpoints/
└── pi05_base/
    └── params/            # symlink to the OpenPI cache by default
```

Use `uv run astar-download-pi05 --copy` only when a physical 12 GB copy is
required. Set `OPENPI_DATA_HOME` before downloading to move OpenPI's cache.

Training outputs are written under:

```text
checkpoints/astar_pi05_energy_aggregation/<experiment-name>/
```

ASTAR retains one committed full checkpoint and its matching replay/archive
state. A trainable-parameter-only export may also be written at the final
step. The pi0.5 base checkpoint is immutable and never used as an output path.
