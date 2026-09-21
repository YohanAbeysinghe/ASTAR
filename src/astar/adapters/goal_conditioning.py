"""Goal-conditioning adapters for pi0.5 prefix tokens."""

from __future__ import annotations

from collections.abc import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.shared import array_typing as at


class WaypointGoalProjector(nnx.Module):
    """Project continuous waypoint goals into the Gemma/PaliGemma prefix space.

    This follows the same idea as OmniVLA-style proprio projectors:

        continuous goal vector -> MLP -> soft LLM token

    The adapter expects `Observation.goal_waypoints` with shape `[B, G, D]`.
    If a dataloader has a single goal waypoint, it should provide `[B, 1, D]`.
    """

    llm_dim: int
    waypoint_dim: int
    hidden_dim: int
    default_num_tokens: int
    input_scale: tuple[float, ...] | None

    def __init__(
        self,
        llm_dim: int,
        waypoint_dim: int = 2,
        *,
        hidden_dim: int | None = None,
        default_num_tokens: int = 1,
        input_scale: Sequence[float] | None = None,
        rngs: nnx.Rngs,
    ):
        if waypoint_dim <= 0:
            raise ValueError(f"waypoint_dim must be positive, got {waypoint_dim}.")
        if input_scale is not None and len(input_scale) != waypoint_dim:
            raise ValueError(
                f"input_scale must have length waypoint_dim={waypoint_dim}, "
                f"got {len(input_scale)}."
            )

        self.llm_dim = llm_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim or llm_dim
        self.default_num_tokens = default_num_tokens
        self.input_scale = tuple(float(v) for v in input_scale) if input_scale is not None else None
        self.fc1 = nnx.Linear(waypoint_dim, self.hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(self.hidden_dim, llm_dim, rngs=rngs)

    def _normalize(self, waypoints: at.Float[at.Array, "b g d"]) -> at.Float[at.Array, "b g d"]:
        if self.input_scale is None:
            return waypoints
        scale = jnp.asarray(self.input_scale, dtype=waypoints.dtype)
        return waypoints / scale

    @at.typecheck
    def embed(
        self,
        obs: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b g emb"], at.Bool[at.Array, "b g"], at.Bool[at.Array, " g"]]:
        """Return soft prefix tokens, input mask, and prefix-LM AR mask."""
        if obs.goal_waypoints is None:
            batch_size = obs.state.shape[0]
            tokens = jnp.zeros((batch_size, self.default_num_tokens, self.llm_dim), dtype=obs.state.dtype)
            input_mask = jnp.zeros((batch_size, self.default_num_tokens), dtype=jnp.bool_)
            ar_mask = jnp.zeros((self.default_num_tokens,), dtype=jnp.bool_)
            return tokens, input_mask, ar_mask

        waypoints = obs.goal_waypoints
        if waypoints.ndim == 2:
            waypoints = waypoints[:, None, :]
        if waypoints.shape[-1] != self.waypoint_dim:
            raise ValueError(f"Expected waypoint_dim={self.waypoint_dim}, got {waypoints.shape[-1]}.")

        x = self._normalize(waypoints)
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        tokens = self.fc2(x)

        if obs.goal_waypoint_mask is None:
            input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        else:
            input_mask = obs.goal_waypoint_mask
            if input_mask.ndim == 1:
                input_mask = input_mask[:, None]

        ar_mask = jnp.zeros((tokens.shape[1],), dtype=jnp.bool_)
        return tokens, input_mask, ar_mask


def waypoint_goal_projector_factory(prefix_width: int, rngs: nnx.Rngs) -> WaypointGoalProjector:
    """Factory compatible with `Pi0Config.goal_adapter_factory`.

    Defaults to a single `[x, y]` navigation goal in meters. The scale matches the
    current navigation ESDF convention: roughly `0 <= x <= 25` and lateral range
    around `[-10, 10]`.
    """
    return WaypointGoalProjector(
        llm_dim=prefix_width,
        waypoint_dim=2,
        input_scale=(25.0, 10.0),
        rngs=rngs,
    )
