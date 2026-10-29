"""pi0.5-style action adapter for navigation waypoints."""

from __future__ import annotations

import flax.nnx as nnx
from openpi.models import model as _model
from openpi.models.pi0 import posemb_sincos
from openpi.shared import array_typing as at


class Pi05WaypointActionAdapter(nnx.Module):
    """Action adapter for navigation waypoints.

    This preserves the pi0.5 adapter architecture: action projection, timestep
    MLP for adaRMSNorm conditioning, and output projection. The Gemma action
    expert itself remains owned by the parent Pi0 model.

    Supported action layouts:
    - 2D: `(x, y)` waypoints, used by ESDF energy training.
    - 4D: `(x, y, cos(theta), sin(theta))` waypoints, used by flow training.
    """

    action_dim: int
    action_horizon: int
    action_width: int

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        action_width: int,
        *,
        rngs: nnx.Rngs,
    ):
        if action_dim not in (2, 4):
            raise ValueError(
                "Pi05WaypointActionAdapter expects action_dim=2 for `(x, y)` "
                "or action_dim=4 for `(x, y, cos(theta), sin(theta))` waypoints."
            )

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_width = action_width
        self.action_in_proj = nnx.Linear(action_dim, action_width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_width, action_width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_width, action_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_width, action_dim, rngs=rngs)

    @at.typecheck
    def embed(
        self,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b h emb"], at.Float[at.Array, "b emb"]]:
        if noisy_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {noisy_actions.shape[-1]}."
            )
        if noisy_actions.shape[-2] != self.action_horizon:
            raise ValueError(
                f"Expected action_horizon={self.action_horizon}, got {noisy_actions.shape[-2]}."
            )

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        return action_tokens, time_emb

    @at.typecheck
    def decode(self, action_hidden: at.Float[at.Array, "b h emb"]) -> _model.Actions:
        return self.action_out_proj(action_hidden)


def pi05_waypoint_action_adapter_factory(
    action_dim: int,
    action_horizon: int,
    action_width: int,
    pi05: bool,
    rngs: nnx.Rngs,
) -> Pi05WaypointActionAdapter:
    """Factory compatible with `Pi0Config.action_adapter_factory`."""
    if not pi05:
        raise ValueError("Pi05WaypointActionAdapter can only be used with Pi0Config(pi05=True).")
    return Pi05WaypointActionAdapter(action_dim, action_horizon, action_width, rngs=rngs)
