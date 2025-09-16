"""Base classes for the CUAS domain."""

from enum import Enum

import numpy as np


class Vector3D:
    """Vector3D class.

    Parameters
    ----------
    x: float
        The x coordinate.
    y: float
        The y coordinate.
    z: float
        The z coordinate.
    """

    coords: np.ndarray = np.array([0, 0, 0])

    def __init__(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        self.coords = np.array([x, y, z])


class BoundingBox:
    """BoundingBox class.

    Parameters
    ----------
    min: Vector3D
        The minimum vector.
    max: Vector3D
        The maximum vector.
    """

    min: Vector3D
    max: Vector3D

    def __init__(self, min: Vector3D, max: Vector3D):
        self.min = min
        self.max = max


class Waypoint:
    """Waypoint class.

    Parameters
    ----------
    position: Vector3D
        The position.
    speed_magnitude: float
        The speed magnitude.
    """

    position: Vector3D
    speed_magnitude: float

    def __init__(self, position: Vector3D, speed_magnitude: float):
        self.position = position
        self.speed_magnitude = speed_magnitude


class EffectorKinematicState(Enum):
    """EffectorKinematicState class."""

    CHASING = 0
    TRACKING = 1


class EffectorWeaponState(Enum):
    """EffectorWeaponState class."""

    READY = 0
    SHOOTING = 1
    RECHARGING = 2


class ExplosiveType(Enum):
    """ExplosiveType class."""

    LIGHT = 0
    MEDIUM = 1
    STRONG = 2


class ChassisType(Enum):
    """ChassisType class."""

    SMALL = 0
    MEDIUM = 1
    LARGE = 2


class DroneState(Enum):
    """DroneState class."""

    ACTIVE = 0
    NEUTRALIZED = 1
    IMPACTED = 2
