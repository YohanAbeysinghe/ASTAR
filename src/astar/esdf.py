"""Cell-centered ESDF sampling and robot footprint metadata."""

from __future__ import annotations

import jax
import jax.numpy as jnp

DEFAULT_ROBOT_RADIUS_M = 0.5


def robot_radius_per_path(metric_tensors: dict, batch_size: int, dtype) -> jax.Array:
    """Return one radius per path, broadcasting a scalar across the batch.

    Dataset metadata supplies ``robot_radius_m`` in meters as a scalar, [B], or
    [B, 1]. Missing metadata retains the historical 0.5 m energy default.
    """
    radius = jnp.asarray(metric_tensors.get("robot_radius_m", DEFAULT_ROBOT_RADIUS_M), dtype=dtype)
    if radius.ndim > 2 or (radius.ndim == 2 and radius.shape[1] != 1):
        raise ValueError("robot_radius_m must be a scalar, [B], or [B, 1].")
    return jnp.broadcast_to(radius.reshape((-1,)), (batch_size,))


def sample_batched_esdf_with_valid(
    esdf: jax.Array,
    xy: jax.Array,
    x_min: jax.Array,
    y_min: jax.Array,
    resolution: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Bilinearly sample [B, H, W] fields at [B, N, 2] physical XY points.

    Map minima describe cell edges; values live half a cell inward. The
    supported rectangle includes the first and last cell centers. Outside it,
    return the nearest boundary sample with ``valid=False``; callers decide
    how to penalize unsupported space. Metadata may be scalar or per map.
    """
    x_min = jnp.asarray(x_min).reshape((-1, 1))
    y_min = jnp.asarray(y_min).reshape((-1, 1))
    resolution = jnp.asarray(resolution).reshape((-1, 1))
    cols = (xy[..., 0] - x_min) / resolution - 0.5
    rows = (xy[..., 1] - y_min) / resolution - 0.5
    grid_rows, grid_cols = esdf.shape[-2:]
    if grid_rows < 2 or grid_cols < 2:
        raise ValueError("Bilinear ESDF sampling requires at least a 2-by-2 grid.")
    valid = (
        jnp.isfinite(rows)
        & jnp.isfinite(cols)
        & (rows >= 0.0)
        & (cols >= 0.0)
        & (rows <= grid_rows - 1)
        & (cols <= grid_cols - 1)
    )

    # Clamp coordinates to the LAST center, and lower indices to the
    # PENULTIMATE center. Clamping coordinates to the penultimate center
    # would erase interpolation and its gradient throughout the final cell.
    rows_clamped = jnp.clip(rows, 0, grid_rows - 1)
    cols_clamped = jnp.clip(cols, 0, grid_cols - 1)
    r0 = jnp.minimum(jnp.floor(rows_clamped).astype(jnp.int32), grid_rows - 2)
    c0 = jnp.minimum(jnp.floor(cols_clamped).astype(jnp.int32), grid_cols - 2)
    r1 = r0 + 1
    c1 = c0 + 1
    wr = rows_clamped - r0
    wc = cols_clamped - c0
    batch_index = jnp.arange(esdf.shape[0])[:, None]
    v00 = esdf[batch_index, r0, c0]
    v01 = esdf[batch_index, r0, c1]
    v10 = esdf[batch_index, r1, c0]
    v11 = esdf[batch_index, r1, c1]
    sampled = (
        (1.0 - wr) * (1.0 - wc) * v00
        + (1.0 - wr) * wc * v01
        + wr * (1.0 - wc) * v10
        + wr * wc * v11
    )
    return sampled, valid


def sample_batched_esdf(
    esdf: jax.Array,
    xy: jax.Array,
    x_min: jax.Array,
    y_min: jax.Array,
    resolution: jax.Array,
) -> jax.Array:
    """Sample an ESDF, using negative infinity outside its supported rectangle."""
    sampled, valid = sample_batched_esdf_with_valid(esdf, xy, x_min, y_min, resolution)
    return jnp.where(valid, sampled, -jnp.inf)
