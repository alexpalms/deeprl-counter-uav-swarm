"""Utility functions for the CUAS domain."""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from rl_cuas.environment.base_classes import Vector3D, Waypoint

if TYPE_CHECKING:
    from rl_cuas.environment.supporting_classes import Drone, Effector, SensitiveZone


def calculate_spherical_coordinates(
    target: Vector3D, origin: Vector3D
) -> tuple[float, float, float, float]:
    """
    Calculate the spherical coordinates of a target relative to an origin.

    Parameters
    ----------
    target: Vector3D
        The target position.
    origin: Vector3D
        The origin position.
    """
    position_vector = target.coords - origin.coords
    distance = float(np.linalg.norm(position_vector))
    distance_xy = float(np.linalg.norm(position_vector[:2]))
    elevation = math.asin(position_vector[2] / distance)

    if distance_xy < 0.01:
        azimuth = 0.0
    else:
        azimuth = math.atan2(
            position_vector[1] / distance_xy, position_vector[0] / distance_xy
        )

    return azimuth, elevation, distance, distance_xy


def calculate_delta_azimuth_delta_elevation(
    azimuth: float, elevation: float, azimuth_ref: float, elevation_ref: float
) -> tuple[float, float]:
    """
    Calculate the delta azimuth and delta elevation.

    Parameters
    ----------
    azimuth: float
        The azimuth.
    elevation: float
        The elevation.
    azimuth_ref: float
        The reference azimuth.
    elevation_ref: float
        The reference elevation.

    Returns
    -------
    delta_azimuth: float
        The delta azimuth.
    delta_elevation: float
        The delta elevation.
    """
    # Limit Az-El by max angular speeds
    delta_elevation = elevation - elevation_ref
    delta_azimuth = np.abs(azimuth - azimuth_ref)
    delta_azimuth = min(delta_azimuth, 2 * math.pi - delta_azimuth)
    current_x = math.cos(azimuth_ref)
    current_y = math.sin(azimuth_ref)
    target_x = math.cos(azimuth)
    target_y = math.sin(azimuth)
    cross_prod = current_x * target_y - current_y * target_x
    if (
        cross_prod == 0.0 and current_x != target_x
    ):  # singular case when azimuth 180 deg apart
        cross_prod = 1
    delta_azimuth *= np.sign(cross_prod)

    return delta_azimuth, delta_elevation


def swarm_generate_intermediate_drone_steps(
    time_step: float, point_1: Vector3D, point_2: Vector3D, max_speed: float
) -> list[Waypoint]:
    """
    Generate intermediate drone steps.

    Parameters
    ----------
    time_step: float
        The time step.
    point_1: Vector3D
        The first point.
    point_2: Vector3D
        The second point.
    max_speed: float
        The max speed.

    Returns
    -------
    waypoints: list[Waypoint]
        The waypoints.
    """
    direction = point_2.coords - point_1.coords
    distance = np.linalg.norm(direction)
    direction /= distance  # Normalize the direction vector

    num_steps = int(distance / max_speed / time_step)
    delta_pos = max_speed * time_step

    # Generate a list of multipliers for each step
    steps = np.arange(1, num_steps + 1) * delta_pos
    waypoints = [
        Waypoint(
            Vector3D(
                point_1.coords[0] + step * direction[0],
                point_1.coords[1] + step * direction[1],
                point_1.coords[2] + step * direction[2],
            ),
            max_speed,
        )
        for step in steps
    ]

    return waypoints


def swarm_generate_drone_trajectory(
    np_random: np.random.Generator,
    spawn_min: Vector3D,
    spawn_max: Vector3D,
    waypoints_min: Vector3D,
    waypoints_max: Vector3D,
    sensitive_min: Vector3D,
    sensitive_max: Vector3D,
    sensitive_zones: list[SensitiveZone],
    swarm_drones_trajectories_number_of_intermediate_points: int,
    time_step: float,
    max_speed: float,
    zone_idx: int,
) -> list[Waypoint]:
    """
    Generate a drone trajectory.

    Parameters
    ----------
    np_random: np.random.Generator
        The random number generator.
    spawn_min: Vector3D
        The minimum spawn point.
    spawn_max: Vector3D
        The maximum spawn point.
    waypoints_min: Vector3D
        The minimum waypoints point.
    waypoints_max: Vector3D
        The maximum waypoints point.
    sensitive_min: Vector3D
        The minimum sensitive point.
    sensitive_max: Vector3D
        The maximum sensitive point.
    sensitive_zones: list[SensitiveZone]
        The sensitive zones.
    swarm_drones_trajectories_number_of_intermediate_points: int
        The number of intermediate points.
    time_step: float
        The time step.
    max_speed: float
        The max speed.
    zone_idx: int
        The zone index.

    Returns
    -------
    trajectory: list[Waypoint]
        The trajectory.
    """

    def random_point(min_point: Vector3D, max_point: Vector3D) -> Vector3D:
        """
        Generate a random point.

        Parameters
        ----------
        min_point: Vector3D
            The minimum coordinates.
        max_point: Vector3D
            The maximum coordinates.
        """
        return Vector3D(
            np_random.uniform(min_point.coords[0], max_point.coords[0]),  # pyright: ignore[reportUnknownArgumentType]
            np_random.uniform(min_point.coords[1], max_point.coords[1]),  # pyright: ignore[reportUnknownArgumentType]
            np_random.uniform(min_point.coords[2], max_point.coords[2]),  # pyright: ignore[reportUnknownArgumentType]
        )

    start_point = random_point(spawn_min, spawn_max)
    trajectory = [Waypoint(start_point, max_speed)]

    for _ in range(swarm_drones_trajectories_number_of_intermediate_points):
        intermediate_point = random_point(waypoints_min, waypoints_max)
        intermediate_steps = swarm_generate_intermediate_drone_steps(
            time_step, trajectory[-1].position, intermediate_point, max_speed
        )
        trajectory.extend(intermediate_steps)

    def generate_end_point() -> Vector3D:
        """
        Generate an end point.

        Returns
        -------
        end_point: Vector3D
            The end point.
        """
        if zone_idx < len(sensitive_zones):
            return sensitive_zones[zone_idx].random_inner_point()
        else:
            while True:
                end_point = random_point(sensitive_min, sensitive_max)
                if all(not zone.contains(end_point) for zone in sensitive_zones):
                    return end_point

    end_point = generate_end_point()
    intermediate_steps = swarm_generate_intermediate_drone_steps(
        time_step, trajectory[-1].position, end_point, max_speed
    )
    trajectory.extend(intermediate_steps)
    trajectory.append(Waypoint(end_point, max_speed))

    return trajectory


def drone_effector_aiming_distance_calculation(
    effector: Effector, drone: Drone, tick: int
) -> float:
    """
    Calculate the distance between a drone and an effector aiming vector.

    Parameters
    ----------
    effector: Effector
        The effector.
    drone: Drone
        The drone.
    tick: int
        The tick.

    Returns
    -------
    distance: float
        The distance between the drone and the effector aiming vector.
    """
    i_step = min(tick, len(drone.trajectory) - 1)
    effector_aiming_versor = np.array(
        [
            np.cos(effector.aiming[0]) * np.cos(effector.aiming[1]),
            np.sin(effector.aiming[0]) * np.cos(effector.aiming[1]),
            np.sin(effector.aiming[1]),
        ]
    )

    line_to_target = drone.trajectory[i_step].position.coords - effector.location.coords
    projection_length = np.dot(line_to_target, effector_aiming_versor)
    projection_vector = projection_length * effector_aiming_versor
    point_to_projection = line_to_target - projection_vector
    distance = float(np.linalg.norm(point_to_projection))

    return distance


def calculate_neutralization_probability(
    distance: float,
    neutralization_dynamics_distance_buckets: NDArray[np.float32],
    neutralization_dynamics_prob_buckets: NDArray[np.float32],
) -> float:
    """
    Calculate the neutralization probability.

    Parameters
    ----------
    distance: float
        The distance.
    neutralization_dynamics_distance_buckets: np.ndarray
        The neutralization dynamics distance buckets.
    neutralization_dynamics_prob_buckets: np.ndarray
        The neutralization dynamics probability buckets.

    Returns
    -------
    probability: float
        The neutralization probability.
    """
    index = bisect_left(neutralization_dynamics_distance_buckets, distance)
    if index == 0:
        return float(neutralization_dynamics_prob_buckets[0])
    elif index == len(neutralization_dynamics_distance_buckets):
        return float(neutralization_dynamics_prob_buckets[-1])
    else:
        x0, x1 = (
            neutralization_dynamics_distance_buckets[index - 1],
            neutralization_dynamics_distance_buckets[index],
        )
        y0, y1 = (
            neutralization_dynamics_prob_buckets[index - 1],
            neutralization_dynamics_prob_buckets[index],
        )
        return float(y0 + (y1 - y0) * (distance - x0) / (x1 - x0))


def control_effectors(
    effectors_list: list[Effector],
    swarm_drones_list: list[Drone],
    tick: int,
    actions: NDArray[np.int_],
) -> None:
    """
    Control the effectors.

    Parameters
    ----------
    effectors_list: list[Effector]
        The effectors list.
    swarm_drones_list: list[Drone]
        The swarm drones list.
    tick: int
        The tick.
    actions: np.ndarray
        The actions.
    """
    for idx, action in enumerate(actions):
        action_int: int = int(action)
        i_step = min(tick, len(swarm_drones_list[action_int].trajectory) - 1)
        effectors_list[idx].assign_target(
            swarm_drones_list[action_int].detections[i_step].position
        )


def calculate_drones_coordinates(
    tick: int, swarm_drones_list: list[Drone], max_distance_weighted: float
) -> np.ndarray:
    """
    Calculate the drones coordinates.

    Parameters
    ----------
    tick: int
        The tick.
    swarm_drones_list: list[Drone]
        The swarm drones list.
    max_distance_weighted: float
        The max distance weighted.

    Returns
    -------
    drones_coordinates: np.ndarray
        The drones coordinates.
    """
    # Compute the minimum index value once
    min_index = [min(tick, len(drone.trajectory) - 1) for drone in swarm_drones_list]

    drones_coordinates: list[float] = []
    for drone, idx in zip(swarm_drones_list, min_index, strict=False):
        coords = drone.detections[idx].position.coords / max_distance_weighted
        drones_coordinates.extend(coords.tolist())  # Flatten into one list

    # Convert to float32 numpy array
    return np.array(drones_coordinates, dtype=np.float32)


def calculate_drones_zones_distance(
    tick: int,
    swarm_drones_list: list[Drone],
    sensitive_zones: list[SensitiveZone],
    max_distance_weighted: float,
) -> np.ndarray:
    """
    Calculate the drones zones distance.

    Parameters
    ----------
    tick: int
        The tick.
    swarm_drones_list: list[Drone]
        The swarm drones list.
    sensitive_zones: list[SensitiveZone]
        The sensitive zones.
    max_distance_weighted: float
        The max distance weighted.

    Returns
    -------
    drones_zones_distance: np.ndarray
        The drones zones distance.
    """
    # Compute the minimum index value once
    min_index = [min(tick, len(drone.trajectory) - 1) for drone in swarm_drones_list]

    drones_zones_distance: list[float] = []
    for drone, idx in zip(swarm_drones_list, min_index, strict=False):
        weighted_distance: float = 0.0
        for sensitive_zone in sensitive_zones:
            distance = np.linalg.norm(
                sensitive_zone.location.coords - drone.detections[idx].position.coords
            )
            weighted_distance += float(
                distance / (sensitive_zone.value * sensitive_zone.radius)
            )
        weighted_distance /= drone.detections[idx].explosive.value + 1
        weighted_distance += drone.state.value * 1000
        drones_zones_distance.append(
            (
                min(weighted_distance, max_distance_weighted)
                / (max_distance_weighted * 0.5)
            )
            - 1
        )

    # Convert to float32 numpy array
    return np.array(drones_zones_distance, dtype=np.float32)
