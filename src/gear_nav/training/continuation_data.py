"""Deterministic full-partition batches for bounded-depth continuation training.

The caller supplies the length of an already filtered training dataset. This
module neither reads examples nor chooses clips. Every real training index is
visited exactly once per epoch; a validity mask identifies shape-only padding.
Only the permutation for the most recently requested epoch remains in memory.
"""

from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path
from typing import Any

import numpy as np

SAMPLER_VERSION = 1
_PERMUTATION_DOMAIN = 0x41535450
_BATCH_RANDOM_DOMAIN = 0x41535442


def _nonnegative_integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if result < (1 if positive else 0):
        comparison = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {comparison}.")
    return int(result)


def deterministic_batch_seed(seed: int, batch_index: int, *, stream: int = 0) -> int:
    """Return a restart-independent seed for one batch and named numeric stream.

    Use different stream numbers for modality masks and other random choices.
    Reconstructing an epoch or accessing another batch cannot change this seed.
    The result fits in uint32, so it also works with JAX's integer key API.
    """
    seed = _nonnegative_integer(seed, "seed")
    batch_index = _nonnegative_integer(batch_index, "batch_index")
    stream = _nonnegative_integer(stream, "stream")
    sequence = np.random.SeedSequence([_BATCH_RANDOM_DOMAIN, seed, batch_index, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class FullPartitionBatches:
    """Random-access shuffled epochs with fixed-size, explicitly padded batches.

    ``batch_index`` is absolute across epochs and starts at zero. Save the next
    batch index in a training checkpoint to resume without replaying the loader.
    The final batch of each epoch repeats the beginning of that epoch's
    permutation as needed; those positions have a false validity mask and must
    have zero weight in the training loss and coverage counters.
    """

    def __init__(self, dataset_size: int, batch_size: int, seed: int):
        self.dataset_size = _nonnegative_integer(dataset_size, "dataset_size", positive=True)
        self.batch_size = _nonnegative_integer(batch_size, "batch_size", positive=True)
        self.seed = _nonnegative_integer(seed, "seed")
        if self.dataset_size > np.iinfo(np.int64).max:
            raise ValueError("dataset_size exceeds the supported int64 index range.")
        self.steps_per_epoch = (self.dataset_size + self.batch_size - 1) // self.batch_size
        self._cached_epoch: int | None = None
        self._permutation: np.ndarray | None = None

    def epoch_for_batch(self, batch_index: int) -> tuple[int, int]:
        """Return the zero-based epoch and batch offset for an absolute index."""
        batch_index = _nonnegative_integer(batch_index, "batch_index")
        return divmod(batch_index, self.steps_per_epoch)

    def indices(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return int64 dataset indices and a bool mask of real examples."""
        epoch, batch_in_epoch = self.epoch_for_batch(batch_index)
        if self._cached_epoch != epoch:
            sequence = np.random.SeedSequence([_PERMUTATION_DOMAIN, self.seed, epoch])
            rng = np.random.Generator(np.random.PCG64(sequence))
            self._permutation = rng.permutation(self.dataset_size).astype(np.int64, copy=False)
            self._cached_epoch = epoch
        assert self._permutation is not None

        start = batch_in_epoch * self.batch_size
        real_count = min(self.batch_size, self.dataset_size - start)
        indices = np.empty(self.batch_size, dtype=np.int64)
        indices[:real_count] = self._permutation[start : start + real_count]
        if real_count < self.batch_size:
            padding_positions = np.arange(self.batch_size - real_count) % self.dataset_size
            indices[real_count:] = self._permutation[padding_positions]
        valid = np.arange(self.batch_size) < real_count
        return indices, valid


def _file_metadata(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Dataset identity requires a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    compared_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in compared_fields):
        raise RuntimeError(f"Dataset identity file changed while being hashed: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def dataset_resume_metadata(
    *,
    manifest_path: str | Path,
    train_split_ids_path: str | Path,
    eval_split_ids_path: str | Path,
    dataset_size: int,
) -> dict[str, Any]:
    """Fingerprint the indexed manifest and both split files for exact resume.

    This reads files sequentially in bounded memory, without loading a dataset.
    The manifest must be the actual manifest from which the supplied dataset was
    indexed. Save this metadata plus the sampler seed/batch size, generation
    settings, and the next absolute batch index in the continuation checkpoint.
    """
    dataset_size = _nonnegative_integer(dataset_size, "dataset_size", positive=True)
    return {
        "sampler_version": SAMPLER_VERSION,
        "dataset_size": dataset_size,
        "manifest": _file_metadata(manifest_path),
        "train_split": _file_metadata(train_split_ids_path),
        "eval_split": _file_metadata(eval_split_ids_path),
    }


def metadata_signature(metadata: dict[str, Any]) -> str:
    """Hash JSON-compatible resume metadata without depending on key order."""
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
