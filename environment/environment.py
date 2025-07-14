import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from base_classes import BoundingBox, Vector3D, ExplosiveType, ChassisType, EffectorWeaponState, DroneState
from supporting_classes import SensitiveZone, Drone, Effector, Detection, ScenarioRenderer
from utils import calculate_drones_zones_distance, control_effectors, drone_effector_aiming_distance_calculation, calculate_neutralization_probability, swarm_generate_drone_trajectory

from neodynamics.interface import create_environment_server

class Environment(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        self.time_step = 0.1

        # Domain limits
        self.domain_bb = BoundingBox(Vector3D(x=-100, y=-100, z=0),
                                    Vector3D(x=100, y=100, z=50))

        # Sensitive zones area
        self.sensitive_zones_bb = BoundingBox(Vector3D(x=-100, y=-100, z=0),
                                            Vector3D(x=0, y=100, z=50))

        # Sensitive zones
        # For now we do not support intersecting sensitive zones
        self.sensitive_zones = []
        self.sensitive_zones.append(SensitiveZone(id=0, location=Vector3D(x=-30, y=-50, z=0), radius=10, value=2))
        self.sensitive_zones.append(SensitiveZone(id=1, location=Vector3D(x=-30, y=50, z=0), radius=30, value=5))
        self.sensitive_zones.append(SensitiveZone(id=2, location=Vector3D(x=-60, y=-10, z=0), radius=20, value=10))
        self.sensitive_zones_num = len(self.sensitive_zones)

        # Swarm
        self.swarm_spawning_bb = BoundingBox(Vector3D(x=95, y=-100, z=25),
                                            Vector3D(x=100, y=100, z=50))

        self.swarm_intermediate_waypoints_bb = BoundingBox(Vector3D(x=-20, y=-100, z=10),
                                                        Vector3D(x=80, y=100, z=45))
        self.swarm_drones_trajectories_number_of_intermediate_points = 1

        self.swarm_drones_num = 50
        self.swarm_drones_features = {
            "max_speed": [[10, 20, 30], [0.4, 0.4, 0.2]],
            "explosive": [[ExplosiveType.LIGHT, ExplosiveType.MEDIUM, ExplosiveType.STRONG], [0.6, 0.3, 0.1]],
            "chassis": [[ChassisType.LARGE, ChassisType.MEDIUM, ChassisType.SMALL], [0.3, 0.4, 0.3]],
        }

        # Detections
        self.detections_features = {
            "position_uncertainty": {ChassisType.LARGE: 0.25,
                                    ChassisType.MEDIUM: 0.5,
                                    ChassisType.SMALL: 0.75},
            "chassis_classification": {ChassisType.LARGE: [0.8, 0.1, 0.1],
                                    ChassisType.MEDIUM: [0.1, 0.8, 0.1],
                                    ChassisType.SMALL: [0.1, 0.1, 0.8]},
            "explosive_classification": {ExplosiveType.LIGHT: [0.8, 0.1, 0.1],
                                        ExplosiveType.MEDIUM: [0.3, 0.4, 0.3],
                                        ExplosiveType.STRONG: [0.1, 0.2, 0.7]},
        }

        # Effectors
        self.effectors_list = []
        self.effectors_list.append(Effector(id="E1", location=Vector3D(x=0, y=-60, z=0),
                                            shooting_time=0.5, max_angular_speeds=[0.5, 0.333]))
        self.effectors_list.append(Effector(id="E2", location=Vector3D(x=0, y=-20, z=0),
                                            shooting_time=0.5, max_angular_speeds=[0.5, 0.333]))
        self.effectors_list.append(Effector(id="E3", location=Vector3D(x=0, y=20, z=0),
                                            shooting_time=0.5, max_angular_speeds=[0.5, 0.333]))
        self.effectors_list.append(Effector(id="E4", location=Vector3D(x=0, y=60, z=0),
                                            shooting_time=0.5, max_angular_speeds=[0.5, 0.333]))
        self.effectors_num = len(self.effectors_list)

        # Effector - Target dynamics
        # Piecewise neutralization probability as a function of the distance between target and aiming vector
        self.neutralization_dynamics_distance_buckets = np.array([0.0, 0.5, 0.25, 0.5]) # WARNING: this is wrongly defined, causes a vertical drop in prob after 0.25 m distance from 0.85 to 0.2
        self.neutralization_dynamics_prob_buckets = np.array([0.99, 0.75, 0.20, 0.0])


        max_distance = np.linalg.norm(self.domain_bb.max.coords - self.domain_bb.min.coords)
        self.max_distance_weighted = 0
        for sensitive_zone in self.sensitive_zones:
            self.max_distance_weighted += max_distance / (sensitive_zone.value * sensitive_zone.radius)

        self.n_steps = 3
        self.stacked_obs = deque(maxlen=self.n_steps)

        # Observations are dictionaries
        self.observation_space = spaces.Dict(
            {
                "drones_zones_distance": spaces.Box(low=np.array([-1 for _ in range(self.swarm_drones_num * self.n_steps)]),
                                                    high=np.array([1 for _ in range(self.swarm_drones_num * self.n_steps)]), dtype=np.float32)
            }
        )

        self.action_space = spaces.MultiDiscrete([self.swarm_drones_num for _ in range(self.effectors_num)], dtype=np.int32)

        self.render_mode = render_mode
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = ScenarioRenderer(self.domain_bb, self.sensitive_zones_bb,
                                            self.swarm_spawning_bb, self.swarm_intermediate_waypoints_bb,
                                            self.effectors_list, self.sensitive_zones,
                                            plot_trajectories=False, plot_detections=False, grid=False,
                                            render_mode=render_mode)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super(type(self), self).reset(seed=seed)

        # Time tick
        self.tick = 0

        # Initialize effectors states
        for effector in self.effectors_list:
            effector.reset(self.time_step)

        # Initialize sensitive zones states
        for sensitive_zone in self.sensitive_zones:
            sensitive_zone.reset(self.np_random)

        # Generate swarm
        self.swarm_drones_list = []
        self.max_length = 0
        sampled_parameters = {}

        # Distribute drones equally between sensitive zones + 1
        assert self.swarm_drones_num >= (len(self.sensitive_zones) + 1), "swarm_drones_num needs to be >= len(self.sensitive_zones) + 1"
        self.swarm_drones_per_zone = int(self.swarm_drones_num / (len(self.sensitive_zones) + 1))

        # Sample drone parameters using vectorized operations
        for key in ["max_speed", "explosive", "chassis"]:
            sampled_parameters[key] = self.np_random.choice(self.swarm_drones_features[key][0],
                                                            size=self.swarm_drones_num,
                                                            p=self.swarm_drones_features[key][1])

        # Precompute zone indices
        zone_indices = np.arange(self.swarm_drones_num) // self.swarm_drones_per_zone

        # Instantiate drones and compute their trajectories
        self.swarm_drones_list = [
            Drone(sampled_parameters["max_speed"][i],
                sampled_parameters["explosive"][i],
                sampled_parameters["chassis"][i])
            for i in range(self.swarm_drones_num)
        ]

        for i, drone in enumerate(self.swarm_drones_list):
            zone_idx = zone_indices[i]
            drone.trajectory = swarm_generate_drone_trajectory(self.np_random, self.swarm_spawning_bb.min.coords, self.swarm_spawning_bb.max.coords,
                                                            self.swarm_intermediate_waypoints_bb.min.coords, self.swarm_intermediate_waypoints_bb.max.coords,
                                                            self.sensitive_zones_bb.min.coords, self.sensitive_zones_bb.max.coords,
                                                            self.sensitive_zones, self.swarm_drones_trajectories_number_of_intermediate_points,
                                                            self.time_step, sampled_parameters["max_speed"][i], zone_idx)
            if zone_idx < len(self.sensitive_zones):
                drone.sensitive_zone_targeted = self.sensitive_zones[zone_idx]
            self.max_length = max(self.max_length, len(drone.trajectory))

        # Calculate potential damage and max reward magnitude
        self.max_reward_magnitude = 0
        sensitive_zone_values = [zone.value for zone in self.sensitive_zones]

        for drone in self.swarm_drones_list:
            end_position = drone.trajectory[-1].position.coords
            for sensitive_zone, value in zip(self.sensitive_zones, sensitive_zone_values):
                if sensitive_zone.contains(end_position):
                    drone.potential_damage += value * 3 ** drone.explosive.value
            drone.potential_damage = max(1, drone.potential_damage)
            self.max_reward_magnitude += drone.potential_damage

        # Generate detections
        np_random = self.np_random
        detections_features = self.detections_features
        swarm_drones_features = self.swarm_drones_features

        for drone in self.swarm_drones_list:
            position_uncertainty = detections_features["position_uncertainty"][drone.chassis]
            chassis = np_random.choice(
                swarm_drones_features["chassis"][0],
                p=detections_features["chassis_classification"][drone.chassis]
            )
            explosive = np_random.choice(
                swarm_drones_features["explosive"][0],
                p=detections_features["explosive_classification"][drone.explosive]
            )

            trajectory_coords = np.array([waypoint.position.coords for waypoint in drone.trajectory])
            noisy_positions = trajectory_coords + np_random.normal(0, position_uncertainty, trajectory_coords.shape)

            detections = [
                Detection(
                    Vector3D(pos[0], pos[1], pos[2]),
                    position_uncertainty,
                    chassis,
                    explosive
                ) for pos in noisy_positions
            ]
            drone.detections = detections

        self.stacked_obs.clear()
        self.stacked_obs.extend([calculate_drones_zones_distance(self.tick, self.swarm_drones_list, self.sensitive_zones, self.max_distance_weighted)] * self.n_steps)

        if self.render_mode == "human":
            trajectories = []
            for drone in self.swarm_drones_list:
                trajectories.append(drone.trajectory)
            self.renderer.plot_drones_trajectories(trajectories)
            self._render_frame()

        return self._get_observation(), self._get_info()

    def step(self, actions):
        # Apply control
        control_effectors(self.effectors_list, self.swarm_drones_list, self.tick, actions)

        # Calculate neutralization
        for idx, action in enumerate(actions):
            effector = self.effectors_list[idx]
            drone = self.swarm_drones_list[action]
            if effector.weapon_state.value != EffectorWeaponState.SHOOTING.value or drone.state.value != DroneState.ACTIVE.value:
                continue
            distance = drone_effector_aiming_distance_calculation(effector, drone, self.tick)
            neutralization_probability = calculate_neutralization_probability(distance, self.neutralization_dynamics_distance_buckets, self.neutralization_dynamics_prob_buckets)
            neutralized = self.np_random.choice([True, False], p=[neutralization_probability, 1 - neutralization_probability])
            if neutralized:
                effector.neutralized += 1
                drone.neutralize(self.tick)
                # Update max length
                self.max_length = 0
                for i_drone in self.swarm_drones_list:
                    if i_drone.state.value != DroneState.ACTIVE.value:
                        continue
                    self.max_length = max(self.max_length, len(i_drone.trajectory))

        # Time tick
        self.tick += 1

        self.stacked_obs.append(calculate_drones_zones_distance(self.tick, self.swarm_drones_list, self.sensitive_zones, self.max_distance_weighted))

        return self._get_observation(), self._get_reward(), self._get_episode_termination(), self._get_episode_abortion(), self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            return None

    def close(self):
        pass

    def _get_observation(self):
        return {"drones_zones_distance": np.concatenate(self.stacked_obs, axis=0)}

    def _get_reward(self):
        reward = 0
        for drone in self.swarm_drones_list:
            if drone.state.value != DroneState.ACTIVE.value:
                continue
            if self.tick == len(drone.trajectory) - 1:
                drone.state = DroneState.IMPACTED
                if drone.sensitive_zone_targeted is not None:
                    self.sensitive_zones[drone.sensitive_zone_targeted.id].impacts += 1
                reward -= drone.potential_damage

        return reward

    def _get_episode_termination(self):
        return True if self.tick >= self.max_length else False

    def _get_episode_abortion(self):
        return False

    def _get_info(self):
        # Access frequently used variables once
        swarm_drones_list = self.swarm_drones_list
        effectors_list = self.effectors_list
        sensitive_zones = self.sensitive_zones

        # Process drones using list comprehensions
        drones_state = [drone.state.name for drone in swarm_drones_list]
        active_drones = sum(1 for drone in swarm_drones_list if drone.state.value == DroneState.ACTIVE.value)
        neutralized_drones = sum(1 for drone in swarm_drones_list if drone.state.value == DroneState.NEUTRALIZED.value)
        impacted_drones = sum(1 for drone in swarm_drones_list if drone.state.value == DroneState.IMPACTED.value)

        # Process effectors using list comprehensions
        effectors_kinematic_state = [effector.kinematic_state.name for effector in effectors_list]
        effectors_weapon_state = [effector.weapon_state.name for effector in effectors_list]
        effectors_kinematic_state_count = {
            effector.id: {k.name: v for k, v in effector.kinematic_states_counts.items()}
            for effector in effectors_list
        }

        # Initialize the info dictionary
        info = {
            "max_reward_magnitude": self.max_reward_magnitude,
            "active_drones": active_drones,
            "neutralized_drones": neutralized_drones,
            "impacted_drones": impacted_drones,
            "drones_state": drones_state,
            "effectors_kinematic_state": effectors_kinematic_state,
            "effectors_weapon_state": effectors_weapon_state,
            "effectors_kinematic_state_count": effectors_kinematic_state_count,
        }

        # Calculate impacts in zones
        total_impacts_in_zones = 0
        for zone in sensitive_zones:
            impacts = zone.impacts
            info[f"Sensitive Zone {zone.id} Impacts:"] = impacts
            total_impacts_in_zones += impacts
        info["Base Zone Impacts:"] = impacted_drones - total_impacts_in_zones

        return info

    def _render_frame(self):
        self.renderer.plot_drones_positions(self.swarm_drones_list, self.tick)
        self.renderer.plot_effectors_aiming(self.effectors_list)
        self.renderer.update_drones_data(self.swarm_drones_list)
        self.renderer.update_effectors_data(self.effectors_list)
        if self.render_mode == "human":
            self.renderer.update()
        else:
            return self.renderer.get_rgb_array()

if __name__ == "__main__":
    create_environment_server(Environment)