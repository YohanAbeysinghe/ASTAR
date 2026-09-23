from __future__ import annotations

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astar.path_sampler import DynamicWaypointPriorConfig
import astar.training.aggregation as aggregation_trainer
from astar.training.aggregation import GROUND_TRUTH_METRIC_KEYS
from astar.training.aggregation import ObstacleEnergyConfig
from astar.training.aggregation import OracleOnlyNavigationDataset
from astar.training.aggregation import _validate_args
from astar.training.aggregation import backtracking_projected_step
from astar.training.aggregation import compute_obstacle_energy
from astar.training.aggregation import condition_fingerprint
from astar.training.aggregation import decode_bounded_waypoint_path
from astar.training.aggregation import parse_args
from astar.training.aggregation import project_bounded_actions
from astar.training.aggregation import query_energy_oracle
from astar.training.aggregation import sanitize_metric_tensors
from astar.training.replay import AggregationReplay


def _energy_fixture():
    esdf = jnp.full((1, 64, 64), 5.0, dtype=jnp.float32)
    metrics = {
        "esdf": esdf,
        "esdf_x_min": jnp.asarray([0.0], dtype=jnp.float32),
        "esdf_y_min": jnp.asarray([-3.2], dtype=jnp.float32),
        "esdf_resolution": jnp.asarray([0.1], dtype=jnp.float32),
        "goal_xy": jnp.asarray([[1.2, 0.0]], dtype=jnp.float32),
    }
    config = ObstacleEnergyConfig(
        collision_weight=0.0,
        clearance_weight=0.0,
        goal_weight=1.0,
        progress_weight=0.0,
        early_heading_weight=0.0,
        smoothness_weight=0.0,
        segment_samples=3,
        max_step_length_m=0.4,
        required_progress_fraction=0.65,
        train_particles=1,
        diversity_direction_weight=0.0,
    )
    paths = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    return paths, metrics, config


def test_quarter_prior_scale_preserves_shape_and_scales_distance() -> None:
    _, metrics, base_config = _energy_fixture()
    full_config = dataclasses.replace(
        base_config,
        prior=DynamicWaypointPriorConfig(
            dt_s=0.5,
            min_v_mps=1.0,
            max_v_mps=1.0,
            max_omega_radps=0.5,
            length_scale=1.0,
            forward_only=True,
        ),
        max_step_length_m=1.0,
    )
    short_config = dataclasses.replace(
        full_config,
        prior=dataclasses.replace(full_config.prior, length_scale=0.25),
    )
    rng = jax.random.key(123)

    def initialize(config):
        encoded = aggregation_trainer.initialize_prior_paths(
            rng,
            metrics,
            config,
            base_batch_size=1,
            particles=1,
            action_horizon=4,
            action_dim=2,
        )
        physical, _ = decode_bounded_waypoint_path(encoded, metrics, config)
        return np.asarray(physical)

    full = initialize(full_config)
    short = initialize(short_config)
    np.testing.assert_allclose(short, 0.25 * full, atol=1.0e-6)
    full_length = np.linalg.norm(np.diff(full, axis=1), axis=-1).sum()
    short_length = np.linalg.norm(np.diff(short, axis=1), axis=-1).sum()
    assert short_length == pytest.approx(0.25 * full_length, abs=1.0e-6)


def test_goal_aware_segment_cap_uses_divided_length_and_detour_allowance() -> None:
    _, metrics, config = _energy_fixture()
    metrics = metrics | {"goal_xy": jnp.asarray([[0.6, 0.0]], dtype=jnp.float32)}
    config = dataclasses.replace(
        config,
        max_step_length_m=1.0,
        path_detour_factor=1.25,
    )
    raw_path = jnp.asarray(
        [[[0.0, 0.0], [0.4, 0.0], [0.8, 0.0], [1.2, 0.0]]],
        dtype=jnp.float32,
    )

    decoded, _ = decode_bounded_waypoint_path(raw_path, metrics, config)
    segment_lengths = np.linalg.norm(np.diff(np.asarray(decoded), axis=1), axis=-1)

    # 0.6 m / 3 segments * 1.25 detour allowance = 0.25 m per segment.
    np.testing.assert_allclose(segment_lengths, 0.25, atol=1.0e-6)


def test_goal_progress_has_zero_origin_weight_and_linearly_stronger_future_weights() -> None:
    _, metrics, config = _energy_fixture()
    path = jnp.asarray(
        [[[0.0, 0.0], [0.05, 0.0], [0.10, 0.0], [0.15, 0.0]]],
        dtype=jnp.float32,
    )

    gradient = jax.grad(
        lambda candidate: compute_obstacle_energy(candidate, metrics, config)[0]
    )(path)
    x_gradient = np.asarray(gradient[0, :, 0])

    assert x_gradient[0] == pytest.approx(0.0, abs=1.0e-8)
    assert np.all(x_gradient[1:] < 0.0)
    assert abs(x_gradient[1]) < abs(x_gradient[2]) < abs(x_gradient[3])

    zero_path = jnp.zeros_like(path)
    _, info = compute_obstacle_energy(zero_path, metrics, config)
    fractions = np.arange(1, 4, dtype=np.float32) / 3.0
    np.testing.assert_allclose(
        info["goal_energy"],
        np.sum(fractions**3),
        atol=1.0e-5,
    )


@pytest.mark.parametrize(
    ("raw_esdf_m", "expected_collision", "expected_clearance_violation"),
    [
        (0.40, 1.0, 1.0),  # 0.40 - 0.50 = -0.10 m: physical collision.
        (0.60, 0.0, 1.0),  # 0.10 m footprint clearance: below the 0.25 m target.
        (0.75, 0.0, 0.0),  # Exactly 0.25 m outside the footprint.
    ],
)
def test_clearance_target_is_applied_after_robot_radius(
    raw_esdf_m: float,
    expected_collision: float,
    expected_clearance_violation: float,
) -> None:
    paths = jnp.asarray([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    metrics = {
        "esdf": jnp.full((1, 64, 64), raw_esdf_m),
        "esdf_x_min": jnp.asarray([0.0]),
        "esdf_y_min": jnp.asarray([-3.2]),
        "esdf_resolution": jnp.asarray([0.1]),
        "robot_radius_m": jnp.asarray([0.5]),
        "goal_xy": jnp.asarray([[3.0, 0.0]]),
    }
    config = ObstacleEnergyConfig(
        clearance_weight=1.0,
        goal_weight=0.0,
        progress_weight=0.0,
        early_heading_weight=0.0,
        smoothness_weight=0.0,
        safety_margin_m=0.25,
        segment_samples=1,
        max_step_length_m=2.0,
        esdf_ramp_start_m=0.0,
        esdf_min_x_m=0.1,
        strict_esdf_coverage=True,
    )

    energy, info = compute_obstacle_energy(paths, metrics, config)

    assert float(info["collision_rate"]) == expected_collision
    assert float(info["clearance_violation_rate"]) == expected_clearance_violation
    assert (float(energy) > 0.0) is bool(expected_clearance_violation)


def test_oracle_label_is_projection_aware_and_non_energy_increasing() -> None:
    paths, metrics, config = _energy_fixture()
    direction, info = query_energy_oracle(
        paths,
        metrics,
        config,
        base_batch_size=1,
        particles=1,
        gradient_floor=1.0e-8,
        max_direction_rms=1.0,
        diversity_direction_weight=0.0,
        step_size=0.1,
        backtracks=6,
        backtrack_factor=0.5,
        energy_tolerance=1.0e-7,
    )
    oracle_next = project_bounded_actions(paths + 0.1 * direction, metrics, config)
    energy_before, _ = compute_obstacle_energy(paths, metrics, config)
    energy_after, _ = compute_obstacle_energy(oracle_next, metrics, config)

    assert float(energy_after) <= float(energy_before) + 1.0e-6
    assert float(info["oracle_acceptance_rate"]) == 1.0
    direction_rms = jnp.sqrt(jnp.mean(jnp.square(direction[:, 1:, :])))
    assert float(direction_rms) <= 1.0 + 1.0e-6
    np.testing.assert_array_equal(
        np.asarray(direction[:, 0, :]), np.zeros((1, 2), np.float32)
    )
    assert float(direction[0, -1, 0]) > 0.0

    decoded, _ = decode_bounded_waypoint_path(oracle_next, metrics, config)
    segment_lengths = jnp.linalg.norm(decoded[:, 1:] - decoded[:, :-1], axis=-1)
    assert float(jnp.max(segment_lengths)) <= config.max_step_length_m + 1.0e-6


def test_bad_learned_direction_is_rejected_instead_of_increasing_energy() -> None:
    paths, metrics, config = _energy_fixture()
    away_from_goal = jnp.zeros_like(paths).at[:, 1:, 0].set(-10.0)
    next_paths, acceptance = backtracking_projected_step(
        paths,
        away_from_goal,
        metrics,
        config,
        step_size=0.1,
        backtracks=5,
        backtrack_factor=0.5,
        energy_tolerance=0.0,
    )
    assert not bool(acceptance["accepted"][0])
    np.testing.assert_array_equal(np.asarray(next_paths), np.asarray(paths))
    assert float(acceptance["energy_after_per_path"][0]) <= float(
        acceptance["energy_before_per_path"][0]
    )


@pytest.mark.parametrize("endpoint", [(-0.2, 0.0), (1.0, 0.0), (0.2, -0.6), (0.2, 0.6)])
def test_out_of_map_is_penalized_even_beside_high_clearance(endpoint) -> None:
    _, metrics, config = _energy_fixture()
    metrics = metrics | {
        "esdf": jnp.full((1, 8, 8), 5.0),
        "esdf_y_min": jnp.asarray([-0.4]),
    }
    config = dataclasses.replace(
        config, strict_esdf_coverage=True, collision_weight=0.0,
        clearance_weight=1.0, goal_weight=0.0, max_step_length_m=1.0,
        safety_margin_m=0.25,
    )
    paths = jnp.asarray([[[0.0, 0.0], list(endpoint)]], dtype=jnp.float32)
    energy, info = compute_obstacle_energy(paths, metrics, config)
    assert float(info["invalid_esdf_rate"]) > 0.0
    assert float(info["collision_rate"]) == 1.0
    assert float(info["unsafe_rate"]) == 1.0
    assert float(energy) > 0.0
    gradient = jax.grad(lambda x: compute_obstacle_energy(x, metrics, config)[0])(paths)
    assert np.isfinite(np.asarray(gradient)).all()
    # Descent must oppose escape at each edge of the map.
    axis = 0 if endpoint[0] < 0.0 or endpoint[0] > 0.8 else 1
    assert float(gradient[0, 1, axis]) * endpoint[axis] > 0.0


def test_strict_safety_scores_occupied_near_field_despite_training_ramp() -> None:
    _, metrics, config = _energy_fixture()
    metrics = metrics | {"esdf": jnp.full((1, 64, 64), -1.0)}
    config = dataclasses.replace(config, strict_esdf_coverage=True)
    paths = jnp.asarray([[[0., 0.], [.2, 0.], [.3, 0.], [.4, 0.]]])
    _, info = compute_obstacle_energy(paths, metrics, config)
    assert float(info["invalid_esdf_rate"]) == 0.0
    assert float(info["collision_rate"]) == 1.0
    assert float(info["unsafe_rate"]) == 1.0


def test_strict_coverage_does_not_mark_observed_free_near_field_unsafe() -> None:
    _, metrics, config = _energy_fixture()
    config = dataclasses.replace(config, strict_esdf_coverage=True)
    paths = jnp.asarray([[[0., 0.], [.2, 0.], [.3, 0.], [.4, 0.]]])
    _, info = compute_obstacle_energy(paths, metrics, config)
    assert float(info["invalid_esdf_rate"]) == 0.0
    assert float(info["collision_rate"]) == 0.0
    assert float(info["unsafe_rate"]) == 0.0


def test_ground_truth_metrics_are_structurally_removed() -> None:
    _, metrics, _ = _energy_fixture()
    placeholders = metrics | {
        "actions_vw": jnp.zeros((1, 4, 2)),
        "path_xy": jnp.zeros((1, 4, 2)),
        "path_xytheta": jnp.zeros((1, 4, 3)),
        "path_step_mask": jnp.zeros((1, 4), dtype=bool),
    }
    sanitized = sanitize_metric_tensors(placeholders)
    assert not GROUND_TRUTH_METRIC_KEYS.intersection(sanitized)
    assert set(metrics).issubset(sanitized)

    contaminated = placeholders | {"path_xy": jnp.ones((1, 4, 2))}
    try:
        sanitize_metric_tensors(contaminated)
    except AssertionError as error:
        assert "Nonzero ground-truth" in str(error)
    else:
        raise AssertionError("A nonzero trajectory label was silently accepted.")


def test_oracle_only_dataset_accepts_goal_without_path_plan() -> None:
    dataset = object.__new__(OracleOnlyNavigationDataset)
    assert dataset._usable_goal({"goal_xy_m": [1.0, 2.0]})
    assert not dataset._usable_goal({"goal_xy_m": [1.0]})
    assert not dataset._usable_goal({"goal_xy_m": [float("nan"), 2.0]})
    assert not dataset._usable_goal({"goal_xy_m": ["not-a-number", 2.0]})
    assert not dataset._usable_goal({"path_plan": {"path_data_path": "unused.npz"}})


@pytest.mark.parametrize("radius", [-0.1, float("nan"), float("inf")])
def test_invalid_robot_radius_is_rejected_before_oracle_use(radius) -> None:
    _, metrics, _ = _energy_fixture()
    with pytest.raises(ValueError, match="robot_radius_m"):
        sanitize_metric_tensors(metrics | {"robot_radius_m": jnp.asarray([radius])})


def test_oracle_query_jits_on_cpu() -> None:
    paths, metrics, config = _energy_fixture()
    query = jax.jit(
        lambda x, m: query_energy_oracle(
            x,
            m,
            config,
            base_batch_size=1,
            particles=1,
            gradient_floor=1.0e-8,
            max_direction_rms=1.0,
            diversity_direction_weight=0.0,
            step_size=0.1,
            backtracks=2,
            backtrack_factor=0.5,
            energy_tolerance=1.0e-7,
        )
    )
    direction, info = query(paths, metrics)
    assert bool(jnp.all(jnp.isfinite(direction)))
    assert bool(jnp.isfinite(info["diagnostic_energy"]))


def test_denominator_arguments_must_be_positive(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["trainer", "--exp-name", "validation-test"])
    args = parse_args()
    assert args.prior_length_scale == pytest.approx(0.25)
    _, _, config = _energy_fixture()
    for field_name, option in (
        ("energy_temperature", "--energy-temperature"),
        ("min_step_scale_m", "--min-step-scale-m"),
    ):
        invalid = dataclasses.replace(config, **{field_name: 0.0})
        try:
            _validate_args(args, invalid)
        except ValueError as error:
            assert option in str(error)
        else:
            raise AssertionError(f"{option} accepted zero.")
    invalid_prior = dataclasses.replace(
        config.prior,
        length_scale=0.0,
    )
    with pytest.raises(ValueError, match="--prior-length-scale"):
        _validate_args(args, dataclasses.replace(config, prior=invalid_prior))


def test_condition_fingerprint_covers_image_and_esdf_contents() -> None:
    observation = {
        "image": {"base": np.zeros((1, 3, 4, 3), dtype=np.uint8)},
        "state": np.zeros((1, 4), dtype=np.float32),
    }
    metrics = {"esdf": np.zeros((1, 5, 6), dtype=np.float32)}
    baseline = condition_fingerprint(observation, metrics)
    assert baseline == condition_fingerprint(observation, metrics)

    changed_observation = {
        "image": {"base": observation["image"]["base"].copy()},
        "state": observation["state"].copy(),
    }
    changed_observation["image"]["base"][0, 0, 0, 0] = 1
    assert condition_fingerprint(changed_observation, metrics) != baseline

    changed_metrics = {"esdf": metrics["esdf"].copy()}
    changed_metrics["esdf"][0, 0, 0] = 1.0
    assert condition_fingerprint(observation, changed_metrics) != baseline


def test_incremental_transition_plot_labels_depths_in_order(monkeypatch) -> None:
    paths, metrics, config = _energy_fixture()
    replay = AggregationReplay(
        config_signature="config",
        condition_fingerprints=["condition"],
        current_paths=np.asarray(paths)[None],
    )
    replay.append_current_round(
        np.zeros_like(replay.current_paths),
        np.zeros(replay.current_paths.shape[:2], dtype=np.float32),
        {},
    )
    next_paths = replay.current_paths.copy()
    next_paths[..., 1:, 0] = 0.1
    record = {
        "round": 1,
        "source_depth": 0,
        "target_depth": 1,
        "energy_before": 1.0,
        "energy_after": 0.8,
        "path_change_median": 0.1,
        "status": "running",
    }
    replay.finish_round(next_paths, record, convergence_streak=0)

    captured = {}

    def fake_wandb_image(_figure, *, caption):
        captured["caption"] = caption
        return caption

    monkeypatch.setattr(aggregation_trainer.wandb, "Image", fake_wandb_image)
    result = aggregation_trainer.aggregation_transition_figure(
        replay,
        metrics,
        config,
        record,
        particles=1,
    )
    assert result == captured["caption"]
    assert "x0 to x1" in captured["caption"]
