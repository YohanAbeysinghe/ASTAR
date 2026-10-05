"""Coverage, restart, and provenance guarantees for continuation data."""

import hashlib

import numpy as np
import pytest

from astar.training.continuation_data import FullPartitionBatches
from astar.training.continuation_data import dataset_resume_metadata
from astar.training.continuation_data import deterministic_batch_seed
from astar.training.continuation_data import metadata_signature


@pytest.mark.parametrize("dataset_size,batch_size", [(37, 8), (32, 8), (3, 8), (1, 1)])
def test_every_epoch_covers_full_partition_once(dataset_size, batch_size):
    sampler = FullPartitionBatches(dataset_size, batch_size, seed=9)
    for epoch in range(3):
        real_indices = []
        for offset in range(sampler.steps_per_epoch):
            batch_index = epoch * sampler.steps_per_epoch + offset
            indices, valid = sampler.indices(batch_index)
            assert sampler.epoch_for_batch(batch_index) == (epoch, offset)
            assert indices.shape == valid.shape == (batch_size,)
            assert indices.dtype == np.int64
            assert valid.dtype == np.bool_
            assert np.all((indices >= 0) & (indices < dataset_size))
            real_indices.extend(indices[valid].tolist())
        assert sorted(real_indices) == list(range(dataset_size))


def test_tail_padding_cycles_from_same_epoch_and_is_marked_invalid():
    sampler = FullPartitionBatches(dataset_size=3, batch_size=8, seed=4)
    indices, valid = sampler.indices(0)
    np.testing.assert_array_equal(valid, [True, True, True, False, False, False, False, False])
    np.testing.assert_array_equal(indices[3:], np.resize(indices[:3], 5))


def test_random_access_resume_reconstructs_same_batches_and_modality_draws():
    sequential = FullPartitionBatches(dataset_size=23, batch_size=7, seed=109)
    expected = {index: sequential.indices(index) for index in range(16)}
    resumed = FullPartitionBatches(dataset_size=23, batch_size=7, seed=109)
    for index in [11, 12, 4, 15, 0, 7, 11]:
        indices, valid = resumed.indices(index)
        np.testing.assert_array_equal(indices, expected[index][0])
        np.testing.assert_array_equal(valid, expected[index][1])
        left = np.random.default_rng(deterministic_batch_seed(109, index, stream=2))
        right = np.random.default_rng(deterministic_batch_seed(109, index, stream=2))
        np.testing.assert_array_equal(left.integers(2, size=(7, 3)), right.integers(2, size=(7, 3)))


def test_epoch_shuffling_and_streams_are_distinct():
    sampler = FullPartitionBatches(dataset_size=100, batch_size=100, seed=55)
    epoch_zero, _ = sampler.indices(0)
    epoch_one, _ = sampler.indices(1)
    another_seed, _ = FullPartitionBatches(100, 100, seed=56).indices(0)
    assert not np.array_equal(epoch_zero, epoch_one)
    assert not np.array_equal(epoch_zero, another_seed)
    seeds = {deterministic_batch_seed(55, batch, stream=stream) for batch in range(8) for stream in range(3)}
    assert len(seeds) == 24
    assert all(0 <= seed <= np.iinfo(np.uint32).max for seed in seeds)


def test_returned_indices_cannot_mutate_future_batches():
    sampler = FullPartitionBatches(dataset_size=4, batch_size=4, seed=1)
    original, _ = sampler.indices(0)
    expected = original.copy()
    original[:] = -1
    reconstructed, _ = sampler.indices(0)
    np.testing.assert_array_equal(reconstructed, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_size": 0},
        {"dataset_size": -1},
        {"dataset_size": 1.5},
        {"dataset_size": True},
        {"batch_size": 0},
        {"batch_size": -2},
        {"seed": -1},
    ],
)
def test_invalid_sampler_configuration_fails(kwargs):
    values = {"dataset_size": 3, "batch_size": 2, "seed": 1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        FullPartitionBatches(**values)


@pytest.mark.parametrize("batch_index", [-1, 0.5, True])
def test_invalid_batch_index_fails(batch_index):
    with pytest.raises(ValueError):
        FullPartitionBatches(3, 2, 1).indices(batch_index)
    with pytest.raises(ValueError):
        deterministic_batch_seed(1, batch_index)


def test_resume_metadata_tracks_manifest_and_both_splits(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    train_split = tmp_path / "train.txt"
    eval_split = tmp_path / "eval.txt"
    manifest.write_text('{"clip_id":"train_a"}\n')
    train_split.write_text("train_a\n")
    eval_split.write_text("eval_a\n")
    kwargs = {
        "manifest_path": manifest,
        "train_split_ids_path": train_split,
        "eval_split_ids_path": eval_split,
        "dataset_size": 7,
    }
    initial = dataset_resume_metadata(**kwargs)
    assert initial["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert initial["manifest"]["size_bytes"] == manifest.stat().st_size
    assert initial["manifest"]["path"] == str(manifest.resolve())
    assert metadata_signature(initial) == metadata_signature(dataset_resume_metadata(**kwargs))
    assert metadata_signature(initial) == metadata_signature(dict(reversed(list(initial.items()))))
    for changed_path in [manifest, train_split, eval_split]:
        previous = metadata_signature(dataset_resume_metadata(**kwargs))
        changed_path.write_text(changed_path.read_text() + "changed\n")
        assert metadata_signature(dataset_resume_metadata(**kwargs)) != previous
    previous = metadata_signature(dataset_resume_metadata(**kwargs))
    kwargs["dataset_size"] = 8
    assert metadata_signature(dataset_resume_metadata(**kwargs)) != previous


def test_resume_metadata_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_resume_metadata(
            manifest_path=tmp_path / "absent",
            train_split_ids_path=tmp_path / "train.txt",
            eval_split_ids_path=tmp_path / "eval.txt",
            dataset_size=1,
        )
