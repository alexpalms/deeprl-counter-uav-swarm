import numpy as np
import math
from bisect import bisect_left

from environment.base_classes import Vector3D, Waypoint


def calculate_spherical_coordinates(target, origin):
    position_vector = target - origin
    distance =  np.linalg.norm(position_vector)
    distance_xy = np.linalg.norm(position_vector[:2])
    elevation = math.asin(position_vector[2] / distance)

    if distance_xy < 0.01:
        azimuth = 0
    else:
        azimuth = math.atan2(position_vector[1]/distance_xy, position_vector[0]/distance_xy)

    return azimuth, elevation, distance, distance_xy

def calculate_delta_azimuth_delta_elevation(azimuth, elevation, azimuth_ref, elevation_ref):
    # Limit Az-El by max angular speeds
    delta_elevation = elevation - elevation_ref
    delta_azimuth = np.abs(azimuth - azimuth_ref)
    delta_azimuth = min(delta_azimuth, 2*math.pi - delta_azimuth)
    current_x = math.cos(azimuth_ref)
    current_y = math.sin(azimuth_ref)
    target_x = math.cos(azimuth)
    target_y = math.sin(azimuth)
    cross_prod = current_x * target_y - current_y * target_x
    if (cross_prod == 0.0 and current_x != target_x): # singular case when azimuth 180 deg apart
        cross_prod = 1
    delta_azimuth *= np.sign(cross_prod)

    return delta_azimuth, delta_elevation

def swarm_generate_intermediate_drone_steps(time_step, point_1, point_2, max_speed):
    direction = point_2.coords - point_1.coords
    distance = np.linalg.norm(direction)
    direction /= distance  # Normalize the direction vector

    num_steps = int(distance / max_speed / time_step)
    delta_pos = max_speed * time_step

    # Generate a list of multipliers for each step
    steps = np.arange(1, num_steps + 1) * delta_pos
    waypoints = [
        Waypoint(Vector3D(point_1.coords[0] + step * direction[0],
                          point_1.coords[1] + step * direction[1],
                          point_1.coords[2] + step * direction[2]), max_speed)
        for step in steps
    ]

    return waypoints

def swarm_generate_drone_trajectory(np_random, spawn_min, spawn_max, waypoints_min,
                                    waypoints_max, sensitive_min, sensitive_max,
                                    sensitive_zones, swarm_drones_trajectories_number_of_intermediate_points,
                                    time_step, max_speed, zone_idx):

    def random_point(min_coords, max_coords):
        return Vector3D(
            np_random.uniform(min_coords[0], max_coords[0]),
            np_random.uniform(min_coords[1], max_coords[1]),
            np_random.uniform(min_coords[2], max_coords[2])
        )

    start_point = random_point(spawn_min, spawn_max)
    trajectory = [Waypoint(start_point, max_speed)]

    for _ in range(swarm_drones_trajectories_number_of_intermediate_points):
        intermediate_point = random_point(waypoints_min, waypoints_max)
        intermediate_steps = swarm_generate_intermediate_drone_steps(time_step, trajectory[-1].position, intermediate_point, max_speed)
        trajectory.extend(intermediate_steps)

    def generate_end_point():
        if zone_idx < len(sensitive_zones):
            return sensitive_zones[zone_idx].random_inner_point()
        else:
            while True:
                end_point = random_point(sensitive_min, sensitive_max)
                if all(not zone.contains(end_point.coords) for zone in sensitive_zones):
                    return end_point

    end_point = generate_end_point()
    intermediate_steps = swarm_generate_intermediate_drone_steps(time_step, trajectory[-1].position, end_point, max_speed)
    trajectory.extend(intermediate_steps)
    trajectory.append(Waypoint(end_point, max_speed))

    return trajectory

def drone_effector_aiming_distance_calculation(effector, drone, tick):
    i_step = min(tick, len(drone.trajectory)-1)
    effector_aiming_versor = np.array([np.cos(effector.aiming[0]) * np.cos(effector.aiming[1]),
                                       np.sin(effector.aiming[0]) * np.cos(effector.aiming[1]),
                                       np.sin(effector.aiming[1])])

    line_to_target = drone.trajectory[i_step].position.coords - effector.location.coords
    projection_length = np.dot(line_to_target, effector_aiming_versor)
    projection_vector = projection_length * effector_aiming_versor
    point_to_projection = line_to_target - projection_vector
    distance = np.linalg.norm(point_to_projection)

    return distance

def calculate_neutralization_probability(distance, neutralization_dynamics_distance_buckets, neutralization_dynamics_prob_buckets):
    index = bisect_left(neutralization_dynamics_distance_buckets, distance)
    if index == 0:
        return neutralization_dynamics_prob_buckets[0]
    elif index == len(neutralization_dynamics_distance_buckets):
        return neutralization_dynamics_prob_buckets[-1]
    else:
        x0, x1 = neutralization_dynamics_distance_buckets[index - 1], neutralization_dynamics_distance_buckets[index]
        y0, y1 = neutralization_dynamics_prob_buckets[index - 1], neutralization_dynamics_prob_buckets[index]
        return y0 + (y1 - y0) * (distance - x0) / (x1 - x0)

def control_effectors(effectors_list, swarm_drones_list, tick, actions):
    for idx, action in enumerate(actions):
        i_step = min(tick, len(swarm_drones_list[action].trajectory)-1)
        effectors_list[idx].assign_target(swarm_drones_list[action].detections[i_step].position)

def calculate_drones_zones_distance(tick, swarm_drones_list, sensitive_zones, max_distance_weighted):
    # Compute the minimum index value once
    min_index = [min(tick, len(drone.trajectory)-1) for drone in swarm_drones_list]

    drones_zones_distance = []
    for drone, idx in zip(swarm_drones_list, min_index):
        weighted_distance = 0
        for sensitive_zone in sensitive_zones:
            distance = np.linalg.norm(sensitive_zone.location.coords - drone.detections[idx].position.coords)
            weighted_distance += distance / (sensitive_zone.value * sensitive_zone.radius)
            weighted_distance /= (drone.detections[idx].explosive.value + 1)
        weighted_distance += drone.state.value * 1000
        drones_zones_distance.append((min(weighted_distance, max_distance_weighted) / (max_distance_weighted * 0.5)) - 1)

    # Convert to float32 numpy array
    return np.array(drones_zones_distance, dtype=np.float32)
