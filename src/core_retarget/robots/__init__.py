"""Robot model contracts and registry."""

from core_retarget.robots.registry import get_robot, list_robots
from core_retarget.robots.schema import RobotSpec

__all__ = ["RobotSpec", "get_robot", "list_robots"]
