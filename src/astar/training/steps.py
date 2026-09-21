"""Waypoint-dependent integration steps shared by training and evaluation."""

import math

import jax.numpy as jnp


def waypoint_step_sizes(paths, step_size, step_size_start=None):
    """Return broadcastable [1, horizon, 1] integration coefficients.

    The origin is fixed. Future slots ramp linearly from ``step_size_start``
    to ``step_size``. Omitting the start preserves historical uniform steps.
    With only one future slot, use the start coefficient.
    These are field multipliers, not bounds on displacement in meters.
    """
    start = step_size if step_size_start is None else step_size_start
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("Final waypoint step size must be finite and positive.")
    if not math.isfinite(start) or not 0 < start <= step_size:
        raise ValueError(
            "Starting waypoint step size must be finite and in (0, final step size]."
        )
    if paths.shape[-2] < 2:
        raise ValueError("A path requires an origin and at least one future waypoint.")
    future = jnp.linspace(start, step_size, paths.shape[-2] - 1, dtype=paths.dtype)
    return jnp.concatenate([jnp.zeros((1,), dtype=paths.dtype), future])[None, :, None]
