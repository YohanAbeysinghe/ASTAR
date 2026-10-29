"""Goal-conditioning adapters for GEAR-Nav navigation models."""

from gear_nav.adapters.goal_conditioning import WaypointGoalProjector
from gear_nav.adapters.goal_conditioning import waypoint_goal_projector_factory

__all__ = [
    "WaypointGoalProjector",
    "waypoint_goal_projector_factory",
]
