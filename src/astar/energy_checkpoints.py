"""Checkpoint helpers shared by ASTAR energy training entry points."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
import time

import etils.epath as epath
import jax
from jax.experimental import multihost_utils
import numpy as np
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.utils as training_utils
import orbax.checkpoint as ocp

FULL_CHECKPOINT_MAX_TO_KEEP = 1
_AUXILIARY_CHECKPOINT_PATTERN = re.compile(
    r"(?:aggregation_state_(\d+)\.npz|trainable_(\d+))"
)


def _wait_for_checkpoint_dir(checkpoint_dir: epath.Path, timeout_secs: float = 120.0) -> bool:
    """Allow shared-filesystem metadata caches to catch up after rank zero's mkdir."""
    deadline = time.monotonic() + timeout_secs
    while True:
        if checkpoint_dir.is_dir():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1.0, remaining))


def save_trainable_params_checkpoint(
    state: training_utils.TrainState,
    config: _config.TrainConfig,
    step: int,
) -> epath.Path:
    """Save only trainable parameters, avoiding a duplicate frozen backbone."""
    output_path = config.checkpoint_dir / f"trainable_{step}"
    trainable_params = state.params.filter(config.trainable_filter)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        output_path,
        {
            # Preserve the globally replicated scalar for multi-host Orbax saves.
            "step": state.step,
            "params": trainable_params,
        },
        force=True,
    )
    logging.info("Saved trainable-only checkpoint to %s", output_path)
    return output_path


def prune_auxiliary_checkpoints(
    checkpoint_dir: epath.Path | str,
    retained_steps: set[int],
) -> None:
    """Keep replay and adapter exports only for committed full checkpoints."""
    root = Path(str(checkpoint_dir))
    if not root.is_dir():
        return
    for child in root.iterdir():
        match = _AUXILIARY_CHECKPOINT_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        step = int(match.group(1) or match.group(2))
        if step in retained_steps:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        logging.info("Removed stale auxiliary checkpoint %s", child)


def initialize_bounded_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    """Create a resume manager that cannot retain an unbounded checkpoint set."""
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    error = ""
    if jax.process_index() == 0:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Wiped checkpoint directory %s", checkpoint_dir)
            elif resume:
                resuming = True
            else:
                error = (
                    f"Checkpoint directory {checkpoint_dir} already exists. "
                    "Use --overwrite for a new run or --resume to continue it."
                )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if jax.process_count() > 1:
        state = multihost_utils.broadcast_one_to_all(
            np.asarray([bool(error), resuming], dtype=np.int8)
        )
        multihost_utils.sync_global_devices("checkpoint_directory_prepared")
        if state[0]:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. "
                "Use --overwrite for a new run or --resume to continue it."
            )
        resuming = bool(state[1])
    elif error:
        raise FileExistsError(error)

    if jax.process_count() > 1:
        # A JAX barrier does not invalidate another host's negative NFS lookup.
        # All workers must see the directory before Orbax opens it with create=False.
        visible = _wait_for_checkpoint_dir(checkpoint_dir)
        visibility = np.asarray(
            multihost_utils.process_allgather(np.asarray(visible, dtype=np.bool_))
        ).reshape(-1)
        missing_ranks = np.flatnonzero(~visibility).tolist()
        if missing_ranks:
            raise TimeoutError(
                f"Checkpoint directory {checkpoint_dir} was not visible within 120 seconds "
                f"on JAX processes {missing_ranks}."
            )
        logging.info(
            "Checkpoint directory visible on all %d JAX processes: %s",
            visibility.size,
            checkpoint_dir,
        )

    manager = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=FULL_CHECKPOINT_MAX_TO_KEEP,
            keep_period=None,
            keep_checkpoints_without_metrics=False,
            cleanup_tmp_directories=True,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    if resuming and tuple(manager.all_steps()) in [(), (0,)]:
        logging.info(
            "Checkpoint directory exists but contains no committed checkpoint; "
            "starting without restore."
        )
        resuming = False
    return manager, resuming
