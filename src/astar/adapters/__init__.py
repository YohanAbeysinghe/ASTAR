"""Goal-conditioning adapters for ASTAR navigation models."""

from astar.adapters.goal_conditioning import WaypointGoalProjector
from astar.adapters.goal_conditioning import waypoint_goal_projector_factory

__all__ = [
    "WaypointGoalProjector",
    "waypoint_goal_projector_factory",
]
