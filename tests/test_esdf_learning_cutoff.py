"""Cutoff boundaries, raw safety reporting, and reversible resume migration."""
import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import astar.training.aggregation as trainer
from astar.training.replay import AggregationReplay
from astar.waypoint_energy import ObstacleEnergyConfig
from astar.waypoint_energy import compute_obstacle_energy


def fixture():
    paths = jnp.asarray([[[0., 0.], [1., 0.], [2., 0.]]])
    metrics = dict(esdf=jnp.full((1, 64, 64), .75), esdf_x_min=jnp.asarray([0.]),
                   esdf_y_min=jnp.asarray([-3.2]), esdf_resolution=jnp.asarray([.1]),
                   robot_radius_m=jnp.asarray([.5]), goal_xy=jnp.asarray([[3., 0.]]))
    config = ObstacleEnergyConfig(
        collision_weight=0., clearance_weight=1., goal_weight=0., progress_weight=0.,
        early_heading_weight=0., smoothness_weight=0., safety_margin_m=.25,
        segment_samples=1, max_step_length_m=2., strict_esdf_coverage=True,
        esdf_ramp_start_m=0., esdf_min_x_m=.1, esdf_learning_cutoff_m=.25)
    return paths, metrics, config


@pytest.mark.parametrize(
    "distance,expected_loss,active",
    [(.4, .35**2, True), (.69, .06**2, True), (.75, 0., False), (.76, 0., False)],
)
def test_footprint_clearance_hinge_and_cutoff_boundary(distance, expected_loss, active):
    paths, metrics, config = fixture()
    def energy(d):
        return compute_obstacle_energy(paths, metrics | {"esdf": jnp.full((1, 64, 64), d)}, config)[0]
    loss, gradient = jax.jit(jax.value_and_grad(energy))(jnp.asarray(distance))
    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
    if active:
        assert float(gradient) < 0.
    else:
        assert float(loss) == 0.
        assert float(gradient) == 0.


def test_cutoff_preserves_physical_collision_and_margin_metrics():
    paths, metrics, config = fixture()
    loss, info = compute_obstacle_energy(paths, metrics, config)
    old_loss, old_info = compute_obstacle_energy(paths, metrics, dataclasses.replace(config, esdf_learning_cutoff_m=None))
    assert float(loss) == 0. and float(old_loss) == 0.
    for key in (
        "collision_rate",
        "unsafe_rate",
        "clearance_violation_rate",
        "min_esdf_m",
        "invalid_esdf_rate",
    ):
        np.testing.assert_array_equal(info[key], old_info[key])
    assert float(info["collision_rate"]) == 0.
    assert float(info["clearance_violation_rate"]) == 0.


def test_out_of_map_retains_recovery_gradient_with_cutoff():
    paths, metrics, config = fixture()
    paths = paths.at[0, 2, 0].set(3.)
    metrics = metrics | {"esdf": jnp.full((1, 16, 16), 5.), "esdf_y_min": jnp.asarray([-.8])}
    loss, info = compute_obstacle_energy(paths, metrics, config)
    gradient = jax.grad(lambda x: compute_obstacle_energy(x, metrics, config)[0])(paths)
    assert float(loss) > 0. and float(info["invalid_esdf_rate"]) > 0.
    assert np.isfinite(gradient).all()
    # Projection may route the endpoint recovery gradient through its
    # predecessor when the final raw segment is at the dynamic cap.
    assert float(jnp.sum(gradient[0, 1:, 0])) > 0.


def test_goal_learning_is_not_disabled_by_esdf_cutoff():
    paths, metrics, config = fixture()
    config = dataclasses.replace(config, goal_weight=1.)
    gradient = jax.grad(lambda x: compute_obstacle_energy(x, metrics, config)[0])(paths)
    assert np.all(np.asarray(gradient[0, 1:, 0]) < 0.)
    assert abs(float(gradient[0, -1, 0])) > abs(float(gradient[0, 1, 0]))


def test_disabled_signature_is_backward_compatible(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--exp-name", "original"])
    args = trainer.parse_args()
    legacy = vars(args).copy()
    legacy.pop("esdf_learning_cutoff_m")
    legacy.pop("relabel_esdf_cutoff_from_config")
    assert trainer._resume_signature(args) == trainer._resume_signature(argparse.Namespace(**legacy))


def test_cutoff_migration_can_enable_adjust_and_disable_only_cutoff(monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--exp-name", "original"])
    args = trainer.parse_args()
    for old_cutoff, new_cutoff in ((None, .25), (.25, .3), (.25, None)):
        args.esdf_learning_cutoff_m = old_cutoff
        source = {**vars(args), "aggregation_algorithm_version": trainer.AGGREGATION_ALGORITHM_VERSION,
                  "process_count": jax.process_count()}
        new = argparse.Namespace(**vars(args))
        new.exp_name = "branch"
        new.manifest_path = "/pinned/manifest.json"
        new.esdf_learning_cutoff_m = new_cutoff
        assert trainer.esdf_cutoff_source_signature(new, source) == trainer._resume_signature(args)
        new.max_step_length_m *= 2
        with pytest.raises(ValueError, match="only"):
            trainer.esdf_cutoff_source_signature(new, source)


def test_relabel_preserves_all_aggregated_paths():
    paths = np.ones((2, 1, 4, 2), np.float32)
    replay = AggregationReplay(config_signature="old", condition_fingerprints=["a", "b"], current_paths=paths)
    replay.append_current_round(paths.copy(), np.ones((2, 1), np.float32), {"energy": 1.})
    replay.finish_round(paths * 2, {"round": 0}, convergence_streak=1)
    refreshed = trainer.relabel_max_step_replay(replay, lambda batch, values: (values * 0, np.zeros(1), {"energy": 0.}), "new")
    np.testing.assert_array_equal(refreshed.current_paths, replay.current_paths)
    np.testing.assert_array_equal(refreshed.visited_paths, replay.visited_paths)
    np.testing.assert_array_equal(replay.oracle_directions[0], paths)
    assert refreshed.round_index == replay.round_index and refreshed.convergence_streak == 0
