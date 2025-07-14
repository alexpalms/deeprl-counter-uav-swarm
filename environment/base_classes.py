import numpy as np
from enum import Enum

class Vector3D:
    coords: np.array = np.array([0, 0, 0])

    def __init__(self, x:float=0, y:float=0, z:float=0):
        self.coords = np.array([x, y, z])

class BoundingBox:
    min: Vector3D = None
    max: Vector3D = None

    def __init__(self, min:Vector3D, max:Vector3D):
        self.min = min
        self.max = max

class Waypoint:
    position: Vector3D = None
    speed_magnitude: float = None

    def __init__(self, position:Vector3D, speed_magnitude:float):
        self.position = position
        self.speed_magnitude = speed_magnitude

class EffectorKinematicState(Enum):
    CHASING = 0
    TRACKING = 1

class EffectorWeaponState(Enum):
    READY = 0
    SHOOTING = 1
    RECHARGING = 2

class ExplosiveType(Enum):
    LIGHT = 0
    MEDIUM = 1
    STRONG = 2

class ChassisType(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2

class DroneState(Enum):
    ACTIVE = 0
    NEUTRALIZED = 1
    IMPACTED = 2