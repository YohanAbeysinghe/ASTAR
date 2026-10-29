"""Path samplers and projections for waypoint action chunks."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class DynamicWaypointPriorConfig:
    """Unicycle random-walk prior for local navigation waypoint chunks."""

    dt_s: float = 0.2
    min_v_mps: float = 0.0
    max_v_mps: float = 2.0
    max_omega_radps: float = 0.5
    length_scale: float = 1.0
    forward_only: bool = True
    fallback_x_min_m: float = 0.0
    fallback_x_max_m: float = 25.0
    fallback_y_min_m: float = -10.0
    fallback_y_max_m: float = 10.0


def _stat_for_broadcast(stat):
    stat = jnp.asarray(stat)
    if stat.ndim == 2:
        return stat[:, None, :]
    return stat


def maybe_normalize_actions(
    actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
) -> _model.Actions:
    if "action_q01" in metric_tensors and "action_q99" in metric_tensors:
        q01 = _stat_for_broadcast(metric_tensors["action_q01"])
        q99 = _stat_for_broadcast(metric_tensors["action_q99"])
        if q01.shape[-1] != actions.shape[-1] and actions.shape[-1] == 2:
            q01 = q01[..., :2]
            q99 = q99[..., :2]
        return 2.0 * (actions - q01) / (q99 - q01 + 1e-6) - 1.0
    if "action_mean" in metric_tensors and "action_std" in metric_tensors:
        mean = _stat_for_broadcast(metric_tensors["action_mean"])
        std = _stat_for_broadcast(metric_tensors["action_std"])
        if mean.shape[-1] != actions.shape[-1] and actions.shape[-1] == 2:
            mean = mean[..., :2]
            std = std[..., :2]
        return (actions - mean) / (std + 1e-6)
    return actions


def maybe_unnormalize_actions(
    actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
) -> _model.Actions:
    if "action_q01" in metric_tensors and "action_q99" in metric_tensors:
        q01 = _stat_for_broadcast(metric_tensors["action_q01"])
        q99 = _stat_for_broadcast(metric_tensors["action_q99"])
        if q01.shape[-1] != actions.shape[-1] and actions.shape[-1] == 2:
            q01 = q01[..., :2]
            q99 = q99[..., :2]
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if "action_mean" in metric_tensors and "action_std" in metric_tensors:
        mean = _stat_for_broadcast(metric_tensors["action_mean"])
        std = _stat_for_broadcast(metric_tensors["action_std"])
        if mean.shape[-1] != actions.shape[-1] and actions.shape[-1] == 2:
            mean = mean[..., :2]
            std = std[..., :2]
        return actions * (std + 1e-6) + mean
    return actions


def _batch_scalar(value, batch_size: int):
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float32).reshape((-1,))[0], (batch_size,))


def _as_batch_vector(value, batch_size: int):
    value = jnp.asarray(value, dtype=jnp.float32).reshape((-1,))
    if value.shape[0] == 1:
        return jnp.broadcast_to(value[0], (batch_size,))
    return value


def _path_bounds(
    metric_tensors: dict[str, at.Array],
    config: DynamicWaypointPriorConfig,
    batch_size: int,
) -> tuple[at.Array, at.Array, at.Array, at.Array]:
    if {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution"}.issubset(metric_tensors):
        rows, cols = metric_tensors["esdf"].shape[-2:]
        x_min = _as_batch_vector(metric_tensors["esdf_x_min"], batch_size)
        y_min = _as_batch_vector(metric_tensors["esdf_y_min"], batch_size)
        resolution = _as_batch_vector(metric_tensors["esdf_resolution"], batch_size)
        x_max = x_min + cols * resolution
        y_max = y_min + rows * resolution
    else:
        x_min = _batch_scalar(config.fallback_x_min_m, batch_size)
        x_max = _batch_scalar(config.fallback_x_max_m, batch_size)
        y_min = _batch_scalar(config.fallback_y_min_m, batch_size)
        y_max = _batch_scalar(config.fallback_y_max_m, batch_size)

    if config.forward_only:
        x_min = jnp.maximum(x_min, 0.0)
    return x_min, x_max, y_min, y_max


def _project_bounds(
    metric_tensors: dict[str, at.Array],
    config: DynamicWaypointPriorConfig,
    batch_size: int,
) -> tuple[at.Array, at.Array, at.Array, at.Array]:
    """Bounds that stay inside the ESDF bilinear sampling domain."""
    x_min, x_max, y_min, y_max = _path_bounds(metric_tensors, config, batch_size)
    if {"esdf", "esdf_x_min", "esdf_y_min", "esdf_resolution"}.issubset(metric_tensors):
        resolution = _as_batch_vector(metric_tensors["esdf_resolution"], batch_size)
        x_min = x_min + 0.5 * resolution
        y_min = y_min + 0.5 * resolution
        x_max = x_max - resolution
        y_max = y_max - resolution
        if config.forward_only:
            x_min = jnp.maximum(x_min, 0.0)
    return x_min, x_max, y_min, y_max


def start_action(
    batch_size: int,
    action_dim: int,
    metric_tensors: dict[str, at.Array],
) -> at.Float[at.Array, "b d"]:
    if action_dim != 2:
        raise ValueError(f"Dynamic waypoint prior is 2D-only for energy training, got {action_dim}.")
    start_value = jnp.array([0.0, 0.0], dtype=jnp.float32)
    start = jnp.tile(start_value, (batch_size, 1))
    return maybe_normalize_actions(start[:, None, :], metric_tensors)[:, 0, :]


def project_waypoint_path(
    actions: _model.Actions,
    metric_tensors: dict[str, at.Array],
    config: DynamicWaypointPriorConfig,
) -> _model.Actions:
    """Project a model-space waypoint chunk back to path constraints."""
    if actions.shape[-1] != 2:
        raise ValueError(
            f"Dynamic waypoint prior is 2D-only for energy training, got {actions.shape[-1]}."
        )

    batch_size = actions.shape[0]
    physical_actions = maybe_unnormalize_actions(actions, metric_tensors)
    x_min, x_max, y_min, y_max = _project_bounds(metric_tensors, config, batch_size)

    x = jnp.clip(physical_actions[..., 0], x_min[:, None], x_max[:, None])
    y = jnp.clip(physical_actions[..., 1], y_min[:, None], y_max[:, None])
    projected = jnp.stack([x, y], axis=-1)
    projected = projected.at[:, 0, :].set(
        jnp.tile(jnp.array([0.0, 0.0], dtype=jnp.float32), (batch_size, 1))
    )
    return maybe_normalize_actions(projected, metric_tensors)


def sample_dynamic_waypoint_prior(
    rng: at.KeyArrayLike,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    config: DynamicWaypointPriorConfig,
    metric_tensors: dict[str, at.Array],
) -> _model.Actions:
    """Sample bounded 2D waypoint chunks from a unicycle random walk.

    Heading is used only inside the rollout. The returned model action is
    `(x, y)` in meters unless action stats are provided in metric_tensors.
    """
    start = start_action(batch_size, action_dim, metric_tensors)
    if horizon == 1:
        return start[:, None, :]

    x_min, x_max, y_min, y_max = _project_bounds(metric_tensors, config, batch_size)
    control_rng = jax.random.split(rng, 2)
    v = jax.random.uniform(
        control_rng[0],
        (batch_size, horizon - 1),
        minval=config.min_v_mps,
        maxval=config.max_v_mps,
    )
    omega = jax.random.uniform(
        control_rng[1],
        (batch_size, horizon - 1),
        minval=-config.max_omega_radps,
        maxval=config.max_omega_radps,
    )
    controls = jnp.stack([v, omega], axis=-1)

    def rollout_step(pose, control_t):
        x, y, theta = pose
        v_t = control_t[:, 0]
        omega_t = control_t[:, 1]
        next_theta = theta + omega_t * config.dt_s
        step_distance = v_t * config.dt_s * config.length_scale
        next_x = x + step_distance * jnp.cos(theta)
        next_y = y + step_distance * jnp.sin(theta)

        next_x = jnp.clip(next_x, x_min, x_max)
        next_y = jnp.clip(next_y, y_min, y_max)
        next_pose = (next_x, next_y, next_theta)
        return next_pose, jnp.stack(
            [next_x, next_y, jnp.cos(next_theta), jnp.sin(next_theta)],
            axis=-1,
        )

    _, future_actions = jax.lax.scan(
        rollout_step,
        (
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.float32),
        ),
        jnp.swapaxes(controls, 0, 1),
    )
    future_actions = jnp.swapaxes(future_actions, 0, 1)
    start_value = jnp.tile(jnp.array([0.0, 0.0], dtype=jnp.float32), (batch_size, 1))
    physical_actions = jnp.concatenate([start_value[:, None, :], future_actions[..., :2]], axis=1)
    return project_waypoint_path(maybe_normalize_actions(physical_actions, metric_tensors), metric_tensors, config)
