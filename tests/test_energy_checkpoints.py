"""Shared checkpoint directories must be visible before either worker opens Orbax."""

from unittest.mock import Mock

import numpy as np
import pytest

from astar import energy_checkpoints as checkpoints


def _mock_workers(monkeypatch, rank, visibility):
    monkeypatch.setattr(checkpoints.jax, "process_index", lambda: rank)
    monkeypatch.setattr(checkpoints.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        checkpoints.multihost_utils, "broadcast_one_to_all", lambda value: value
    )
    monkeypatch.setattr(checkpoints.multihost_utils, "sync_global_devices", lambda name: None)
    gather = Mock(return_value=np.asarray(visibility, dtype=np.bool_))
    monkeypatch.setattr(checkpoints.multihost_utils, "process_allgather", gather)
    manager = Mock()
    monkeypatch.setattr(checkpoints.ocp, "CheckpointManager", manager)
    return gather, manager


def test_secondary_waits_for_directory_before_opening_manager(tmp_path, monkeypatch):
    directory = tmp_path / "new_run"
    gather, manager = _mock_workers(monkeypatch, 1, [True, True])
    # Model the directory becoming visible after an initially negative lookup.
    sleep = Mock(side_effect=lambda seconds: directory.mkdir())
    monkeypatch.setattr(checkpoints.time, "sleep", sleep)

    def open_manager(path, **kwargs):
        assert path.is_dir()
        gather.assert_called_once()
        return "manager"

    manager.side_effect = open_manager
    result = checkpoints.initialize_bounded_checkpoint_dir(
        directory, overwrite=False, resume=False
    )
    assert result == ("manager", False)
    sleep.assert_called_once()
    assert bool(gather.call_args.args[0])
    manager.assert_called_once()


def test_visibility_wait_has_a_deadline(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(checkpoints.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        checkpoints.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    assert not checkpoints._wait_for_checkpoint_dir(tmp_path / "missing", timeout_secs=2.5)
    assert now[0] == 2.5


@pytest.mark.parametrize("rank", [0, 1])
def test_any_missing_worker_prevents_manager_creation(tmp_path, monkeypatch, rank):
    gather, manager = _mock_workers(monkeypatch, rank, [True, False])
    monkeypatch.setattr(checkpoints, "_wait_for_checkpoint_dir", lambda path: rank == 0)
    with pytest.raises(TimeoutError, match=r"JAX processes \[1\]"):
        checkpoints.initialize_bounded_checkpoint_dir(
            tmp_path / "new_run", overwrite=False, resume=False
        )
    gather.assert_called_once()
    manager.assert_not_called()


def test_existing_run_is_protected(tmp_path, monkeypatch):
    _, manager = _mock_workers(monkeypatch, 0, [True, True])
    with pytest.raises(FileExistsError):
        checkpoints.initialize_bounded_checkpoint_dir(tmp_path, overwrite=False, resume=False)
    manager.assert_not_called()
