from __future__ import annotations

import numpy as np
import pytest

from astar.training.replay import AggregationReplay
from astar.training.replay import ConvergenceConfig
from astar.training.replay import ConvergenceTracker
from astar.training.replay import deterministic_replay_slot
from astar.training.replay import stable_config_signature


def _paths(value: float = 0.0) -> np.ndarray:
    return np.full((2, 6, 4, 2), value, dtype=np.float32)


def test_replay_caches_labels_and_round_trips_without_condition_duplication(
    tmp_path,
) -> None:
    replay = AggregationReplay(
        config_signature="config",
        condition_fingerprints=["condition-a", "condition-b"],
        current_paths=_paths(),
    )
    labels = _paths(0.25)
    gradients = np.full((2, 6), 0.4, dtype=np.float32)
    replay.append_current_round(labels, gradients, {"oracle_acceptance_rate": 1.0})

    labels.fill(99.0)
    gradients.fill(99.0)
    assert np.all(replay.oracle_directions[0] == np.float32(0.25))
    assert np.all(replay.oracle_gradient_rms[0] == np.float32(0.4))
    assert replay.labelled_rounds == 1

    replay.update_in_round = 7
    archive = replay.save(tmp_path / "replay.npz")
    restored = AggregationReplay.load(
        archive,
        expected_config_signature="config",
        expected_condition_fingerprints=["condition-a", "condition-b"],
    )
    assert restored.round_index == 0
    assert restored.update_in_round == 7
    np.testing.assert_array_equal(restored.current_paths, replay.current_paths)
    np.testing.assert_array_equal(restored.visited_paths[0], replay.visited_paths[0])
    np.testing.assert_array_equal(
        restored.oracle_directions[0], replay.oracle_directions[0]
    )

    restored.finish_round(
        _paths(0.1),
        {"round": 0, "status": "running"},
        convergence_streak=0,
    )
    assert restored.round_index == 1
    assert restored.update_in_round == 0
    assert restored.labelled_rounds == 1


def test_resume_guards_configuration_and_condition_identity(tmp_path) -> None:
    replay = AggregationReplay(
        config_signature="expected",
        condition_fingerprints=["condition-a", "condition-b"],
        current_paths=_paths(),
    )
    archive = replay.save(tmp_path / "replay.npz")
    with pytest.raises(ValueError, match="configuration"):
        AggregationReplay.load(
            archive,
            expected_config_signature="changed",
            expected_condition_fingerprints=["condition-a", "condition-b"],
        )
    with pytest.raises(ValueError, match="Condition pool"):
        AggregationReplay.load(
            archive,
            expected_config_signature="expected",
            expected_condition_fingerprints=["different", "condition-b"],
        )


def test_replay_selection_is_deterministic_and_covers_valid_slots() -> None:
    first = [
        deterministic_replay_slot(
            seed=17,
            global_step=step,
            num_rounds=4,
            num_condition_batches=5,
        )
        for step in range(30)
    ]
    second = [
        deterministic_replay_slot(
            seed=17,
            global_step=step,
            num_rounds=4,
            num_condition_batches=5,
        )
        for step in range(30)
    ]
    assert first == second
    assert all(
        0 <= round_index < 4 and 0 <= batch_index < 5
        for round_index, batch_index in first
    )
    assert len(set(first)) > 8


def test_config_signature_is_order_independent() -> None:
    assert stable_config_signature({"b": 2, "a": 1}) == stable_config_signature(
        {"a": 1, "b": 2}
    )
    assert stable_config_signature({"a": 1}) != stable_config_signature({"a": 2})


def test_convergence_rejects_zero_field_collapse_and_uses_patience() -> None:
    tracker = ConvergenceTracker(
        ConvergenceConfig(
            min_rounds=2,
            patience=2,
            median_path_change_m=0.01,
            p95_path_change_m=0.03,
            relative_energy_change=1.0e-3,
            oracle_grad_rms=1.0e-4,
            max_collision_rate=0.05,
            min_progress_ratio=0.9,
        )
    )
    stable = {
        "path_change_median": 0.001,
        "path_change_p95": 0.002,
        "relative_energy_improvement": 0.0001,
        "oracle_grad_rms": 1.0e-5,
        "collision_rate": 0.0,
        "progress_ratio": 1.0,
    }
    collapsed = stable | {"oracle_grad_rms": 0.2}

    converged, status = tracker.observe(2, collapsed)
    assert not converged
    assert status == "stalled"
    assert tracker.stable_rounds == 0

    assert tracker.observe(2, stable) == (False, "running")
    assert tracker.observe(3, stable) == (True, "converged")


def test_replay_rejects_duplicate_oracle_query() -> None:
    replay = AggregationReplay(
        config_signature="config",
        condition_fingerprints=["condition-a", "condition-b"],
        current_paths=_paths(),
    )
    replay.append_current_round(_paths(0.1), np.ones((2, 6), np.float32), {})
    with pytest.raises(RuntimeError, match="already labelled"):
        replay.append_current_round(_paths(0.2), np.ones((2, 6), np.float32), {})


def test_replay_rejects_updates_before_oracle_labels() -> None:
    replay = AggregationReplay(
        config_signature="config",
        condition_fingerprints=["condition-a", "condition-b"],
        current_paths=_paths(),
    )
    replay.update_in_round = 1
    with pytest.raises(ValueError, match="unlabelled frontier"):
        replay.validate()


def test_completed_converged_replay_is_terminal() -> None:
    replay = AggregationReplay(
        config_signature="config",
        condition_fingerprints=["condition-a", "condition-b"],
        current_paths=_paths(),
    )
    replay.append_current_round(_paths(0.1), np.ones((2, 6), np.float32), {})
    replay.finish_round(
        _paths(0.05),
        {"round": 0, "status": "converged"},
        convergence_streak=2,
    )
    assert replay.converged


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_collision_rate": 1.01}, "max_collision_rate"),
        ({"min_progress_ratio": 1.01}, "min_progress_ratio"),
    ],
)
def test_convergence_rate_thresholds_are_probabilities(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ConvergenceConfig(**kwargs)
