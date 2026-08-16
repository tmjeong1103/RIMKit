"""Robot model contracts and registry."""

from rimkit.robots.registry import get_robot, list_robots
from rimkit.robots.schema import RobotSpec

__all__ = ["RobotSpec", "get_robot", "list_robots"]
