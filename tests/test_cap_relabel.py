import argparse

import numpy as np
import pytest

import astar.training.aggregation as trainer
from astar.training.replay import AggregationReplay


def test_cap_migration_rejects_unrelated_training_changes(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--exp-name", "original"])
    old = trainer.parse_args()
    source = {**vars(old), "aggregation_algorithm_version": trainer.AGGREGATION_ALGORITHM_VERSION,
              "process_count": trainer.jax.process_count()}
    new = argparse.Namespace(**vars(old))
    new.exp_name = "cap2"
    new.max_step_length_m = 2.0
    new.manifest_path = "/pinned/original/manifest.json"
    assert trainer.max_step_source_signature(new, source) == trainer._resume_signature(old)
    new.peak_lr *= 2
    with pytest.raises(ValueError, match="only"):
        trainer.max_step_source_signature(new, source)


def test_cap_relabel_preserves_source_on_invalid_labels():
    paths = np.zeros((2, 1, 4, 2), np.float32)
    replay = AggregationReplay(config_signature="old", condition_fingerprints=["a", "b"], current_paths=paths)
    replay.append_current_round(paths.copy(), np.ones((2, 1), np.float32), {"energy": 1.0})
    replay.finish_round(paths.copy(), {"round": 0}, convergence_streak=0)
    def invalid(batch, values):
        return np.full_like(values, np.nan), np.ones(1, np.float32), {"energy": 2.0}
    with pytest.raises(ValueError, match="NaN"):
        trainer.relabel_max_step_replay(replay, invalid, "new")
    assert replay.config_signature == "old"
    np.testing.assert_array_equal(replay.oracle_directions[0], paths)
