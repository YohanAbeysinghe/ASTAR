"""Shared bounded-path geometry and ESDF energy for waypoint learning and evaluation.

This module defines the horizon-aware energy used by aggregation and the
one-step goal-progress trainer. It does not depend on either training entry point.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.shared import array_typing as at

from astar.esdf import robot_radius_per_path
from astar.esdf import sample_batched_esdf_with_valid
from astar.path_sampler import DynamicWaypointPriorConfig
from astar.path_sampler import maybe_normalize_actions
from astar.path_sampler import maybe_unnormalize_actions


@dataclasses.dataclass(frozen=True)
class ObstacleEnergyConfig:
    """Knobs for the ESDF obstacle cost."""

    sample_steps: int = 1
    prior: DynamicWaypointPriorConfig = dataclasses.field(
        default_factory=DynamicWaypointPriorConfig
    )
    prefer_path_generation_esdf: bool = False
    # Retained for checkpoint/config compatibility. Collision is diagnostic
    # only and does not contribute to the trajectory energy.
    collision_weight: float = 0.0
    clearance_weight: float = 1.0
    goal_weight: float = 1.0
    progress_weight: float = 0.1
    early_heading_weight: float = 0.5
    smoothness_weight: float = 0.1
    safety_margin_m: float = 0.0
    clearance_cap_m: float = 0.2
    # None preserves the original objective. Valid samples at/above this raw
    # signed distance contribute no obstacle loss/gradient. Safety metrics
    # and penalties for unobserved/out-of-map geometry remain unchanged.
    esdf_learning_cutoff_m: float | None = None
    esdf_ramp_start_m: float = 0.5
    esdf_min_x_m: float = 2.0
    # Legacy trainers retain their historical energy. Aggregation opts into
    # conservative coverage; evaluation always uses it for safety scoring.
    strict_esdf_coverage: bool = False
    min_step_scale_m: float = 0.1
    segment_samples: int = 6
    max_step_length_m: float = 0.4
    max_increment_correction_m: float = 0.08
    prior_goal_heading_fraction: float = 1.0
    prior_goal_heading_limit_rad: float = 1.0471975512
    required_progress_fraction: float = 0.65
    oracle_step_size: float = 0.1
    energy_temperature: float = 1.0
    train_particles: int = 2
    diversity_direction_weight: float = 0.35


def bias_prior_toward_goal(
    prior_actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> _model.Actions:
    """Rotate a structured local path toward the clipped goal bearing."""
    prior_metric = maybe_unnormalize_actions(prior_actions, metric_tensors)
    goal_xy = jnp.asarray(metric_tensors["goal_xy"], dtype=prior_metric.dtype)
    goal_heading = jnp.arctan2(goal_xy[:, 1], goal_xy[:, 0])
    goal_heading = jnp.clip(
        energy_config.prior_goal_heading_fraction * goal_heading,
        -energy_config.prior_goal_heading_limit_rad,
        energy_config.prior_goal_heading_limit_rad,
    )
    cos_heading = jnp.cos(goal_heading)[:, None]
    sin_heading = jnp.sin(goal_heading)[:, None]
    prior_x = prior_metric[..., 0]
    prior_y = prior_metric[..., 1]
    rotated_prior_metric = jnp.stack(
        [
            cos_heading * prior_x - sin_heading * prior_y,
            sin_heading * prior_x + cos_heading * prior_y,
        ],
        axis=-1,
    )
    rotated_prior_metric = rotated_prior_metric.at[:, 0, :].set(0.0)
    return maybe_normalize_actions(rotated_prior_metric, metric_tensors)


def _safe_norm(x: at.Array, axis: int = -1, eps: float = 1.0e-6) -> at.Array:
    squared_norm = jnp.sum(jnp.square(x), axis=axis)
    eps_array = jnp.asarray(eps, dtype=x.dtype)
    return jnp.sqrt(squared_norm + eps_array) - jnp.sqrt(eps_array)


def _safe_unit(x: at.Array, axis: int = -1, eps: float = 1.0e-6) -> at.Array:
    squared_norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    eps_array = jnp.asarray(eps, dtype=x.dtype)
    return x / jnp.sqrt(squared_norm + eps_array)


def decode_bounded_waypoint_path(
    raw_actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> tuple[at.Array, at.Array]:
    """Project an absolute waypoint candidate to bounded path increments.

    Both the structured prior and the corrected model output are absolute local
    `(x, y)` waypoints. Slot zero is fixed to the robot origin. Each subsequent
    candidate displacement is radially clipped to max_step_length_m, then
    integrated from the already projected predecessor. A valid structured prior
    therefore decodes unchanged; the projection only repairs an excessive
    residual-model jump.
    """
    raw_metric = maybe_unnormalize_actions(raw_actions, metric_tensors)[..., :2]
    finite_waypoints = jnp.all(jnp.isfinite(raw_metric), axis=-1)
    candidate = jnp.nan_to_num(raw_metric, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    candidate = candidate.at[:, 0, :].set(0.0)
    raw_increments = candidate[:, 1:, :] - candidate[:, :-1, :]
    increment_norm = jnp.sqrt(jnp.sum(jnp.square(raw_increments), axis=-1, keepdims=True) + 1.0e-12)
    scale = jnp.minimum(1.0, energy_config.max_step_length_m / increment_norm)
    increments = raw_increments * scale
    future_xy = jnp.cumsum(increments, axis=1)
    origin = jnp.zeros((raw_actions.shape[0], 1, 2), dtype=future_xy.dtype)
    return jnp.concatenate([origin, future_xy], axis=1), finite_waypoints[:, 1:]


def compute_obstacle_energy(
    pred_actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
) -> tuple[at.Array, dict[str, at.Array]]:
    required = {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution", "goal_xy"}
    if not required.issubset(metric_tensors):
        missing = ", ".join(sorted(required - set(metric_tensors)))
        raise ValueError(f"Obstacle-energy training requires metric_tensors keys: {missing}")

    xy_with_start, finite_xy_mask = decode_bounded_waypoint_path(
        pred_actions,
        metric_tensors,
        energy_config,
    )
    batch_size = xy_with_start.shape[0]
    num_segments = xy_with_start.shape[1] - 1
    pred_xy = xy_with_start[:, 1:, :]

    # Check the continuous polyline, not just the H discrete waypoint samples.
    # Alpha excludes each segment start (already covered by the previous
    # segment) and includes its endpoint. The first segment starts at the robot.
    segment_start_xy = xy_with_start[:, :-1, :]
    segment_end_xy = xy_with_start[:, 1:, :]
    segment_vectors = segment_end_xy - segment_start_xy
    alphas = jnp.linspace(
        1.0 / energy_config.segment_samples,
        1.0,
        energy_config.segment_samples,
        dtype=xy_with_start.dtype,
    )
    dense_xy = (
        segment_start_xy[:, :, None, :]
        + alphas[None, None, :, None] * segment_vectors[:, :, None, :]
    )
    pred_xy_for_esdf = dense_xy.reshape(
        (batch_size, num_segments * energy_config.segment_samples, 2)
    )
    finite_xy_with_start = jnp.concatenate(
        [jnp.ones((batch_size, 1), dtype=bool), finite_xy_mask],
        axis=1,
    )
    segment_finite_mask = finite_xy_with_start[:, :-1] & finite_xy_with_start[:, 1:]
    esdf_finite_mask = jnp.repeat(segment_finite_mask, energy_config.segment_samples, axis=1)

    rows, cols = metric_tensors["esdf"].shape[-2:]
    esdf_x_min = jnp.asarray(metric_tensors["esdf_x_min"], dtype=xy_with_start.dtype)
    esdf_y_min = jnp.asarray(metric_tensors["esdf_y_min"], dtype=xy_with_start.dtype)
    esdf_resolution = jnp.asarray(metric_tensors["esdf_resolution"], dtype=xy_with_start.dtype)
    x_low = esdf_x_min + 0.5 * esdf_resolution
    y_low = esdf_y_min + 0.5 * esdf_resolution
    x_high = esdf_x_min + (cols - 1.0) * esdf_resolution
    y_high = esdf_y_min + (rows - 1.0) * esdf_resolution

    pred_esdf, valid = sample_batched_esdf_with_valid(
        metric_tensors["esdf"],
        pred_xy_for_esdf,
        esdf_x_min,
        esdf_y_min,
        esdf_resolution,
    )
    esdf_sample_ok = valid & esdf_finite_mask & jnp.isfinite(pred_esdf)
    invalid_mask = ~esdf_sample_ok

    robot_radius = robot_radius_per_path(metric_tensors, batch_size, xy_with_start.dtype)
    safe_radius = robot_radius + energy_config.safety_margin_m

    closest_xy = jnp.stack(
        [
            jnp.clip(pred_xy_for_esdf[..., 0], x_low[:, None], x_high[:, None]),
            jnp.clip(pred_xy_for_esdf[..., 1], y_low[:, None], y_high[:, None]),
        ],
        axis=-1,
    )
    closest_esdf, closest_valid = sample_batched_esdf_with_valid(
        metric_tensors["esdf"],
        closest_xy,
        esdf_x_min,
        esdf_y_min,
        esdf_resolution,
    )
    outside_dx = jnp.maximum(x_low[:, None] - pred_xy_for_esdf[..., 0], 0.0) + jnp.maximum(
        pred_xy_for_esdf[..., 0] - x_high[:, None], 0.0
    )
    outside_dy = jnp.maximum(y_low[:, None] - pred_xy_for_esdf[..., 1], 0.0) + jnp.maximum(
        pred_xy_for_esdf[..., 1] - y_high[:, None], 0.0
    )
    outside_distance = _safe_norm(jnp.stack([outside_dx, outside_dy], axis=-1), axis=-1)
    outside_distance = jnp.where(esdf_finite_mask, outside_distance, 1.0e3)
    closest_esdf = jnp.where(
        closest_valid & jnp.isfinite(closest_esdf), closest_esdf, -safe_radius[:, None]
    )
    if energy_config.strict_esdf_coverage:
        # Outside the observed grid, positive boundary clearance is not
        # evidence of free space. Preserve an inward recovery gradient while
        # assigning every unsupported sample a positive collision penalty.
        closest_esdf = jnp.minimum(closest_esdf, 0.0)
    extended_esdf = closest_esdf - outside_distance
    esdf_for_energy = jnp.where(esdf_sample_ok, pred_esdf, extended_esdf)

    # The camera-derived ESDF is unreliable directly under/behind the robot.
    # Check every segment sample, but smoothly activate geometry after the
    # known near-field blind region instead of declaring every start invalid.
    ramp_width = jnp.maximum(
        jnp.asarray(
            energy_config.esdf_min_x_m - energy_config.esdf_ramp_start_m,
            dtype=xy_with_start.dtype,
        ),
        1.0e-6,
    )
    ramp_u = jnp.clip(
        (pred_xy_for_esdf[..., 0] - energy_config.esdf_ramp_start_m) / ramp_width,
        0.0,
        1.0,
    )
    smooth_ramp = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
    esdf_weight = esdf_finite_mask.astype(xy_with_start.dtype) * smooth_ramp
    if energy_config.strict_esdf_coverage:
        # Never let the near-field ramp erase an out-of-map/nonfinite sample.
        esdf_weight = jnp.where(esdf_sample_ok, esdf_weight, 1.0)
    esdf_active = esdf_weight > 0.0
    esdf_weight_count = jnp.maximum(jnp.sum(esdf_weight, axis=-1), 1.0)

    learning_weight = esdf_weight
    learning_weight_count = esdf_weight_count
    if energy_config.esdf_learning_cutoff_m is not None:
        cutoff = energy_config.esdf_learning_cutoff_m
        if not 0.0 < cutoff < float("inf"):
            raise ValueError("ESDF learning cutoff must be finite and positive, or None.")
        ignored = esdf_sample_ok & (pred_esdf >= cutoff)
        learning_weight = jnp.where(ignored, 0.0, esdf_weight)
        learning_weight_count = jnp.maximum(jnp.sum(learning_weight, axis=-1), 1.0)

    collision_per_point = jnp.square(jnp.maximum(0.0, safe_radius[:, None] - esdf_for_energy))
    collision_energy = jnp.sum(learning_weight * collision_per_point, axis=-1) / learning_weight_count

    # Maximize raw ESDF only until the configured cap. Do not clamp negative
    # ESDF to zero: clearance is the sole obstacle-training term and must keep
    # a recovery gradient for occupied and unsupported samples.
    clearance_per_point = -jnp.minimum(
        esdf_for_energy, energy_config.clearance_cap_m
    )
    clearance_energy = jnp.sum(learning_weight * clearance_per_point, axis=-1) / learning_weight_count

    goal_xy = jnp.asarray(metric_tensors["goal_xy"], dtype=xy_with_start.dtype)
    clamped_goal_xy = jnp.stack(
        [
            jnp.clip(goal_xy[:, 0], x_low, x_high),
            jnp.clip(goal_xy[:, 1], y_low, y_high),
        ],
        axis=-1,
    )
    final_goal_error = _safe_norm(pred_xy[:, -1, :] - clamped_goal_xy, axis=-1)
    clamped_goal_distance = _safe_norm(clamped_goal_xy, axis=-1)
    motion_budget = jnp.asarray(
        num_segments * energy_config.max_step_length_m,
        dtype=xy_with_start.dtype,
    )
    required_progress = jnp.minimum(
        clamped_goal_distance,
        energy_config.required_progress_fraction * motion_budget,
    )
    achieved_progress = clamped_goal_distance - final_goal_error
    progress_shortfall = jnp.maximum(required_progress - achieved_progress, 0.0)
    required_progress_scale = jnp.maximum(
        required_progress,
        jnp.asarray(energy_config.min_step_scale_m, dtype=xy_with_start.dtype),
    )
    # "goal_energy" is deliberately horizon-aware: zero means the short chunk
    # made the requested local progress, not that it teleported to a far goal.
    goal_energy = jnp.square(progress_shortfall / required_progress_scale)

    distance_to_goal = _safe_norm(xy_with_start - clamped_goal_xy[:, None, :], axis=-1)
    distance_increase = distance_to_goal[:, 1:] - distance_to_goal[:, :-1]
    progress_energy = jnp.mean(
        jnp.square(jnp.maximum(0.0, distance_increase)), axis=-1
    ) / jnp.square(energy_config.max_step_length_m)

    segment_lengths = _safe_norm(segment_vectors, axis=-1)

    goal_direction_vectors = clamped_goal_xy[:, None, :] - segment_start_xy
    segment_unit = _safe_unit(segment_vectors, axis=-1)
    goal_direction_unit = _safe_unit(goal_direction_vectors, axis=-1)
    heading_alignment = jnp.sum(segment_unit * goal_direction_unit, axis=-1)
    heading_alignment = jnp.clip(heading_alignment, -1.0, 1.0)
    early_heading_mask = (
        (segment_start_xy[..., 0] <= energy_config.esdf_min_x_m)
        & finite_xy_with_start[:, :-1]
        & finite_xy_with_start[:, 1:]
    )
    early_heading_mask_f = early_heading_mask.astype(xy_with_start.dtype)
    early_heading_count = jnp.maximum(jnp.sum(early_heading_mask_f, axis=-1), 1.0)
    early_heading_energy = (
        jnp.sum(early_heading_mask_f * (1.0 - heading_alignment), axis=-1) / early_heading_count
    )

    if num_segments > 1:
        second_diff = (
            xy_with_start[:, 2:, :] - 2.0 * xy_with_start[:, 1:-1, :] + xy_with_start[:, :-2, :]
        )
        smoothness_energy = jnp.mean(
            jnp.sum(jnp.square(second_diff), axis=-1), axis=-1
        ) / jnp.square(energy_config.max_step_length_m)
    else:
        smoothness_energy = jnp.zeros((batch_size,), dtype=xy_with_start.dtype)

    total_energy_per_traj = (
        energy_config.clearance_weight * clearance_energy
        + energy_config.goal_weight * goal_energy
        + energy_config.progress_weight * progress_energy
        + energy_config.early_heading_weight * early_heading_energy
        + energy_config.smoothness_weight * smoothness_energy
    )

    esdf_for_metrics = jnp.where(esdf_sample_ok, pred_esdf, jnp.inf)
    esdf_for_safety_metrics = esdf_for_energy
    valid_traj_mask = jnp.any(esdf_sample_ok, axis=-1)
    d_min_metric = jnp.min(esdf_for_metrics, axis=-1)
    finite_d_min_metric = jnp.where(valid_traj_mask, d_min_metric, 0.0)
    valid_traj_count = jnp.maximum(jnp.sum(valid_traj_mask.astype(jnp.float32)), 1.0)
    clearance = esdf_for_safety_metrics - robot_radius[:, None]
    safety_active = (
        jnp.ones_like(esdf_active) if energy_config.strict_esdf_coverage else esdf_active
    )
    collision_indicator = jnp.any(safety_active & (clearance < 0.0), axis=-1).astype(jnp.float32)
    unsafe_indicator = jnp.any(
        safety_active & (esdf_for_safety_metrics < safe_radius[:, None]), axis=-1
    ).astype(jnp.float32)
    valid_rate = jnp.mean(esdf_sample_ok.astype(jnp.float32))
    all_invalid_rate = 1.0 - jnp.mean(valid_traj_mask.astype(jnp.float32))

    goal_x = goal_xy[:, 0]
    goal_y = goal_xy[:, 1]
    goal_inside_esdf = (
        (goal_x >= x_low) & (goal_x <= x_high) & (goal_y >= y_low) & (goal_y <= y_high)
    )
    goal_distance = _safe_norm(goal_xy, axis=-1)
    goal_clamp_distance = _safe_norm(goal_xy - clamped_goal_xy, axis=-1)

    return jnp.mean(total_energy_per_traj), {
        "obstacle_energy": jnp.mean(total_energy_per_traj),
        "collision_energy": jnp.mean(collision_energy),
        "clearance_energy": jnp.mean(clearance_energy),
        "goal_energy": jnp.mean(goal_energy),
        "progress_energy": jnp.mean(progress_energy),
        "early_heading_energy": jnp.mean(early_heading_energy),
        "smoothness_energy": jnp.mean(smoothness_energy),
        # Compatibility metric: collision is no longer part of total energy.
        "weighted_collision_energy": jnp.zeros_like(jnp.mean(collision_energy)),
        "weighted_clearance_energy": energy_config.clearance_weight * jnp.mean(clearance_energy),
        "weighted_goal_energy": energy_config.goal_weight * jnp.mean(goal_energy),
        "weighted_progress_energy": energy_config.progress_weight * jnp.mean(progress_energy),
        "weighted_early_heading_energy": energy_config.early_heading_weight
        * jnp.mean(early_heading_energy),
        "weighted_smoothness_energy": energy_config.smoothness_weight * jnp.mean(smoothness_energy),
        "collision_rate": jnp.mean(collision_indicator),
        "unsafe_rate": jnp.mean(unsafe_indicator),
        "invalid_esdf_rate": jnp.mean(invalid_mask.astype(jnp.float32)),
        "esdf_valid_rate": valid_rate,
        "all_invalid_traj_rate": all_invalid_rate,
        "esdf_active_rate": jnp.mean(esdf_active.astype(jnp.float32)),
        "esdf_weight_mean": jnp.mean(esdf_weight),
        "early_heading_active_rate": jnp.mean(early_heading_mask_f),
        "action_finite_rate": jnp.mean(finite_xy_mask.astype(jnp.float32)),
        "outside_esdf_distance_m": jnp.mean(jnp.where(invalid_mask, outside_distance, 0.0)),
        "pred_xy_abs_max_m": jnp.max(jnp.abs(pred_xy_for_esdf)),
        "min_esdf_m": jnp.sum(finite_d_min_metric) / valid_traj_count,
        "min_clearance_m": jnp.sum(finite_d_min_metric - robot_radius * valid_traj_mask)
        / valid_traj_count,
        "safe_radius_m": jnp.mean(safe_radius),
        "goal_distance_m": jnp.mean(goal_distance),
        "clamped_goal_distance_m": jnp.mean(clamped_goal_distance),
        "motion_budget_m": motion_budget,
        "required_progress_m": jnp.mean(required_progress),
        "achieved_progress_m": jnp.mean(achieved_progress),
        "progress_shortfall_m": jnp.mean(progress_shortfall),
        "achieved_required_progress_ratio": jnp.mean(achieved_progress / required_progress_scale),
        "path_length_m": jnp.mean(jnp.sum(segment_lengths, axis=-1)),
        "endpoint_radius_m": jnp.mean(_safe_norm(pred_xy[:, -1, :], axis=-1)),
        "max_step_length_m": energy_config.max_step_length_m,
        "mean_segment_length_m": jnp.mean(segment_lengths),
        "max_segment_length_m": jnp.max(segment_lengths),
        "step_violation_rate": jnp.mean(
            (segment_lengths > energy_config.max_step_length_m + 1.0e-5).astype(jnp.float32)
        ),
        "goal_clamp_distance_m": jnp.mean(goal_clamp_distance),
        "goal_was_clamped_rate": jnp.mean((goal_clamp_distance > 1.0e-6).astype(jnp.float32)),
        "final_goal_error_m": jnp.mean(final_goal_error),
        "goal_inside_esdf_rate": jnp.mean(goal_inside_esdf.astype(jnp.float32)),
        "goal_x_m": jnp.mean(goal_x),
        "goal_y_m": jnp.mean(goal_y),
    }


def _repeat_batch_tree(tree, batch_size: int, repeats: int):
    """Repeat each conditioning example into adjacent independent particles."""
    return jax.tree.map(
        lambda x: (
            jnp.repeat(x, repeats, axis=0)
            if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == batch_size
            else x
        ),
        tree,
    )


def compute_particle_path_diversity(
    raw_actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    energy_config: ObstacleEnergyConfig,
    *,
    batch_size: int,
    particles: int,
) -> at.Array:
    """Mean physical path-shape distance for equal-condition particles."""
    if particles <= 1:
        return jnp.asarray(0.0, dtype=raw_actions.dtype)
    path_xy, _ = decode_bounded_waypoint_path(raw_actions, metric_tensors, energy_config)
    flat = path_xy[:, 1:, :].reshape((batch_size, particles, -1))
    pairwise_delta = flat[:, :, None, :] - flat[:, None, :, :]
    pairwise_rms_m = jnp.sqrt(jnp.mean(jnp.square(pairwise_delta), axis=-1) + 1.0e-6)
    off_diagonal = 1.0 - jnp.eye(particles, dtype=raw_actions.dtype)
    return jnp.sum(pairwise_rms_m * off_diagonal[None, :, :]) / float(
        batch_size * particles * (particles - 1)
    )
