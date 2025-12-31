"""Environment class for the CUAS domain."""

import math
from collections import deque
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_cuas.environment.base_classes import (
    BoundingBox,
    ChassisType,
    DroneState,
    EffectorWeaponState,
    ExplosiveType,
    Vector3D,
    Waypoint,
)
from rl_cuas.environment.supporting_classes import (
    Detection,
    Drone,
    Effector,
    ScenarioRenderer,
    SensitiveZone,
)
from rl_cuas.environment.utils import (
    calculate_drones_coordinates,
    calculate_drones_zones_distance,
    calculate_neutralization_probability,
    control_effectors,
    drone_effector_aiming_distance_calculation,
    swarm_generate_drone_trajectory,
)


class Environment(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """
    Environment class for the CUAS domain.

    Parameters
    ----------
    render_mode: str
        The mode to render the environment.
    """

    def __init__(self, render_mode: str | None = "rgb_array") -> None:
        self.time_step: float = 0.1
        self.tick: int = 0

        # Domain limits
        self.domain_bb = BoundingBox(
            Vector3D(x=-100, y=-100, z=0), Vector3D(x=100, y=100, z=50)
        )

        # Sensitive zones area
        self.sensitive_zones_bb = BoundingBox(
            Vector3D(x=-100, y=-100, z=0), Vector3D(x=0, y=100, z=0)
        )

        # Sensitive zones
        # For now we do not support intersecting sensitive zones
        self.sensitive_zones: list[SensitiveZone] = []
        self.sensitive_zones.append(
            SensitiveZone(
                identifier=0, location=Vector3D(x=-30, y=-50, z=0), radius=10, value=2
            )
        )
        self.sensitive_zones.append(
            SensitiveZone(
                identifier=1, location=Vector3D(x=-30, y=50, z=0), radius=30, value=5
            )
        )
        self.sensitive_zones.append(
            SensitiveZone(
                identifier=2, location=Vector3D(x=-60, y=-10, z=0), radius=20, value=10
            )
        )
        self.sensitive_zones_num = len(self.sensitive_zones)

        # Swarm
        self.swarm_spawning_bb = BoundingBox(
            Vector3D(x=95, y=-100, z=25), Vector3D(x=100, y=100, z=50)
        )

        self.swarm_intermediate_waypoints_bb = BoundingBox(
            Vector3D(x=-20, y=-100, z=10), Vector3D(x=80, y=100, z=45)
        )
        self.swarm_drones_trajectories_number_of_intermediate_points = 1

        self.swarm_drones_num = 50
        self.swarm_drones_features: dict[
            str, list[list[int | float | ExplosiveType | ChassisType]]
        ] = {
            "max_speed": [[10, 20, 30], [0.4, 0.4, 0.2]],
            "explosive": [
                [ExplosiveType.LIGHT, ExplosiveType.MEDIUM, ExplosiveType.STRONG],
                [0.6, 0.3, 0.1],
            ],
            "chassis": [
                [ChassisType.LARGE, ChassisType.MEDIUM, ChassisType.SMALL],
                [0.3, 0.4, 0.3],
            ],
        }

        # Detections
        self.detections_features: dict[
            str, dict[ChassisType | ExplosiveType, float | list[float]]
        ] = {
            "position_uncertainty": {
                ChassisType.LARGE: 0.25,
                ChassisType.MEDIUM: 0.5,
                ChassisType.SMALL: 0.75,
            },
            "chassis_classification": {
                ChassisType.LARGE: [0.8, 0.1, 0.1],
                ChassisType.MEDIUM: [0.1, 0.8, 0.1],
                ChassisType.SMALL: [0.1, 0.1, 0.8],
            },
            "explosive_classification": {
                ExplosiveType.LIGHT: [0.8, 0.1, 0.1],
                ExplosiveType.MEDIUM: [0.3, 0.4, 0.3],
                ExplosiveType.STRONG: [0.1, 0.2, 0.7],
            },
        }

        # Effectors
        self.effectors_list: list[Effector] = []
        self.effectors_list.append(
            Effector(
                identifier="E1",
                location=Vector3D(x=0, y=-60, z=0),
                shooting_time=0.5,
                max_angular_speeds=[0.5, 0.333],
            )
        )
        self.effectors_list.append(
            Effector(
                identifier="E2",
                location=Vector3D(x=0, y=-20, z=0),
                shooting_time=0.5,
                max_angular_speeds=[0.5, 0.333],
            )
        )
        self.effectors_list.append(
            Effector(
                identifier="E3",
                location=Vector3D(x=0, y=20, z=0),
                shooting_time=0.5,
                max_angular_speeds=[0.5, 0.333],
            )
        )
        self.effectors_list.append(
            Effector(
                identifier="E4",
                location=Vector3D(x=0, y=60, z=0),
                shooting_time=0.5,
                max_angular_speeds=[0.5, 0.333],
            )
        )
        self.effectors_num = len(self.effectors_list)

        # Effector - Target dynamics
        # Piecewise neutralization probability as a function of the distance between target and aiming vector
        self.neutralization_dynamics_distance_buckets = np.array([0.0, 0.5, 0.25, 0.5])
        self.neutralization_dynamics_prob_buckets = np.array([0.99, 0.75, 0.20, 0.0])

        max_distance = np.linalg.norm(
            self.domain_bb.max.coords - self.domain_bb.min.coords
        )
        self.max_distance_weighted: float = 0.0
        for sensitive_zone in self.sensitive_zones:
            self.max_distance_weighted += float(
                max_distance / (sensitive_zone.value * sensitive_zone.radius)
            )

        self.n_steps = 3
        self.stacked_obs: dict[str, deque[np.ndarray]] = {
            "drones_zones_distance": deque(maxlen=self.n_steps),
            "drones_coordinates": deque(maxlen=self.n_steps),
        }

        # Observations are dictionaries
        self.observation_space = spaces.Dict(
            {
                "drones_zones_distance": spaces.Box(
                    low=np.array(
                        [-1 for _ in range(self.swarm_drones_num * self.n_steps)]
                    ),
                    high=np.array(
                        [1 for _ in range(self.swarm_drones_num * self.n_steps)]
                    ),
                    dtype=np.float32,
                ),
                "drones_coordinates": spaces.Box(
                    low=np.array(
                        [-1 for _ in range(self.swarm_drones_num * 3 * self.n_steps)]
                    ),
                    high=np.array(
                        [1 for _ in range(self.swarm_drones_num * 3 * self.n_steps)]
                    ),
                    dtype=np.float32,
                ),
                "effectors_azel": spaces.Box(
                    low=np.array([-1 for _ in range(self.effectors_num * 2)]),
                    high=np.array([1 for _ in range(self.effectors_num * 2)]),
                    dtype=np.float32,
                ),
                "effectors_kinematic_state": spaces.MultiBinary(
                    len(self.effectors_list)
                ),
                "effectors_weapon_state": spaces.MultiDiscrete(
                    [3 for _ in range(len(self.effectors_list))]
                ),
                "drones_state": spaces.MultiDiscrete(
                    [3 for _ in range(self.swarm_drones_num)]
                ),
                "drones_explosive_power": spaces.MultiDiscrete(
                    [3 for _ in range(self.swarm_drones_num)]
                ),
            }
        )

        self.action_space = spaces.MultiDiscrete(
            [self.swarm_drones_num for _ in range(self.effectors_num)], dtype=np.int64
        )

        self.render_mode = render_mode
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = ScenarioRenderer(
                self.domain_bb,
                self.sensitive_zones_bb,
                self.swarm_spawning_bb,
                self.swarm_intermediate_waypoints_bb,
                self.effectors_list,
                self.sensitive_zones,
                plot_trajectories=False,
                plot_detections=False,
                grid=False,
                render_mode=self.render_mode,
            )

    def action_masks(self) -> list[bool]:
        """
        Get the action masks.

        Returns
        -------
        mask: list[bool]
            The action masks.
        """
        mask = [True for _ in range(self.effectors_num * self.swarm_drones_num)]
        for idx, drone in enumerate(self.swarm_drones_list):
            if drone.state.value != DroneState.ACTIVE.value:
                for effector_idx in range(self.effectors_num):
                    mask[effector_idx * self.swarm_drones_num + idx] = False
        return mask

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int
            The seed.
        options: dict
            The options.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The observation.
        info: dict[str, Any]
            The info.
        """
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
        self.swarm_drones_list: list[Drone] = []
        self.max_length = 0
        sampled_parameters: dict[str, list[ExplosiveType | ChassisType | float]] = {}

        # Distribute drones equally between sensitive zones + 1
        if self.swarm_drones_num < (len(self.sensitive_zones) + 1):
            raise ValueError(
                "swarm_drones_num needs to be >= len(self.sensitive_zones) + 1"
            )
        self.swarm_drones_per_zone = int(
            self.swarm_drones_num / (len(self.sensitive_zones) + 1)
        )

        # Sample drone parameters using vectorized operations
        for key in ["max_speed", "explosive", "chassis"]:
            sampled_parameters[key] = self.np_random.choice(  # type: ignore
                self.swarm_drones_features[key][0],  # type: ignore
                size=self.swarm_drones_num,
                p=self.swarm_drones_features[key][1],  # type: ignore
            )

        # Precompute zone indices
        zone_indices = np.arange(self.swarm_drones_num) // self.swarm_drones_per_zone

        # Instantiate drones and compute their trajectories
        self.swarm_drones_list = [
            Drone(
                cast(float, sampled_parameters["max_speed"][i]),
                cast(ExplosiveType, sampled_parameters["explosive"][i]),
                cast(ChassisType, sampled_parameters["chassis"][i]),
            )
            for i in range(self.swarm_drones_num)
        ]

        for i, drone in enumerate(self.swarm_drones_list):
            zone_idx = zone_indices[i]
            drone.trajectory = swarm_generate_drone_trajectory(
                self.np_random,
                self.swarm_spawning_bb.min,
                self.swarm_spawning_bb.max,
                self.swarm_intermediate_waypoints_bb.min,
                self.swarm_intermediate_waypoints_bb.max,
                self.sensitive_zones_bb.min,
                self.sensitive_zones_bb.max,
                self.sensitive_zones,
                self.swarm_drones_trajectories_number_of_intermediate_points,
                self.time_step,
                cast(float, sampled_parameters["max_speed"][i]),
                zone_idx,
            )
            if zone_idx < len(self.sensitive_zones):
                drone.sensitive_zone_targeted = self.sensitive_zones[zone_idx]
            self.max_length = max(self.max_length, len(drone.trajectory))

        # Calculate potential damage and max reward magnitude
        self.max_reward_magnitude = 0.0
        sensitive_zone_values = [zone.value for zone in self.sensitive_zones]

        for drone in self.swarm_drones_list:
            end_position = drone.trajectory[-1].position
            for sensitive_zone, value in zip(
                self.sensitive_zones, sensitive_zone_values, strict=False
            ):
                if sensitive_zone.contains(end_position):
                    drone.potential_damage += value * 3**drone.explosive.value
            drone.potential_damage = max(1, drone.potential_damage)
            self.max_reward_magnitude += drone.potential_damage

        # Generate detections
        np_random = self.np_random
        detections_features = self.detections_features
        swarm_drones_features = self.swarm_drones_features

        for drone in self.swarm_drones_list:
            position_uncertainty = detections_features["position_uncertainty"][
                drone.chassis
            ]
            chassis = cast(
                ChassisType,
                np_random.choice(
                    swarm_drones_features["chassis"][0],  # type: ignore
                    p=detections_features["chassis_classification"][drone.chassis],
                ),
            )
            explosive = cast(
                ExplosiveType,
                np_random.choice(
                    swarm_drones_features["explosive"][0],  # type: ignore
                    p=detections_features["explosive_classification"][drone.explosive],
                ),
            )

            trajectory_coords = np.array(
                [waypoint.position.coords for waypoint in drone.trajectory]
            )
            noisy_positions = trajectory_coords + np_random.normal(
                0, position_uncertainty, trajectory_coords.shape
            )

            detections = [
                Detection(
                    Vector3D(pos[0], pos[1], pos[2]),
                    cast(float, position_uncertainty),
                    chassis,
                    explosive,
                )
                for pos in noisy_positions
            ]
            drone.detections = detections

        self.stacked_obs["drones_zones_distance"].clear()
        self.stacked_obs["drones_coordinates"].clear()
        self.stacked_obs["drones_zones_distance"].extend(
            [
                calculate_drones_zones_distance(
                    self.tick,
                    self.swarm_drones_list,
                    self.sensitive_zones,
                    self.max_distance_weighted,
                )
            ]
            * self.n_steps
        )
        self.stacked_obs["drones_coordinates"].extend(
            [calculate_drones_coordinates(self.tick, self.swarm_drones_list, 100.0)]
            * self.n_steps
        )

        if self.render_mode == "human":
            trajectories: list[list[Waypoint]] = []
            for drone in self.swarm_drones_list:
                trajectories.append(drone.trajectory)
            self.renderer.plot_drones_trajectories(trajectories)
            self._render_frame()

        return self._get_observation(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Step the environment.

        Parameters
        ----------
        actions: np.ndarray
            The actions.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The observation.
        reward: float
            The reward.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated.
        info: dict[str, Any]
            The info.
        """
        # Apply control
        control_effectors(
            self.effectors_list, self.swarm_drones_list, self.tick, action
        )

        # Calculate neutralization
        for idx, action_item in enumerate(action):
            effector = self.effectors_list[idx]
            drone = self.swarm_drones_list[int(action_item)]
            if (
                effector.weapon_state.value != EffectorWeaponState.SHOOTING.value
                or drone.state.value != DroneState.ACTIVE.value
            ):
                continue
            distance = drone_effector_aiming_distance_calculation(
                effector, drone, self.tick
            )
            neutralization_probability = calculate_neutralization_probability(
                distance,
                self.neutralization_dynamics_distance_buckets,
                self.neutralization_dynamics_prob_buckets,
            )
            neutralized = self.np_random.choice(
                [True, False],
                p=[neutralization_probability, 1 - neutralization_probability],
            )
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

        self.stacked_obs["drones_zones_distance"].append(
            calculate_drones_zones_distance(
                self.tick,
                self.swarm_drones_list,
                self.sensitive_zones,
                self.max_distance_weighted,
            )
        )
        self.stacked_obs["drones_coordinates"].append(
            calculate_drones_coordinates(self.tick, self.swarm_drones_list, 100.0)
        )

        return (
            self._get_observation(),
            self._get_reward(),
            self._get_episode_termination(),
            self._get_episode_abortion(),
            self._get_info(),
        )

    def render(self) -> None | Any:
        """
        Render the environment.

        Returns
        -------
        None | np.ndarray:
            The rendered environment.
        """
        if self.render_mode == "human":
            self._render_frame()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            return None

    def close(self) -> None:
        """Close the environment."""
        pass

    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Get the observation.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The observation.
        """
        return {
            "drones_zones_distance": np.concatenate(
                self.stacked_obs["drones_zones_distance"], axis=0
            ),
            "drones_coordinates": np.concatenate(
                self.stacked_obs["drones_coordinates"], axis=0
            ),
            "effectors_azel": (
                np.array([effector.aiming for effector in self.effectors_list])
                / math.pi
            ).flatten(),
            "effectors_kinematic_state": np.array(
                [effector.kinematic_state.value for effector in self.effectors_list]
            ),
            "effectors_weapon_state": np.array(
                [effector.weapon_state.value for effector in self.effectors_list]
            ),
            "drones_state": np.array(
                [drone.state.value for drone in self.swarm_drones_list]
            ),
            "drones_explosive_power": np.array(
                [drone.explosive.value for drone in self.swarm_drones_list]
            ),
        }

    def _get_reward(self) -> float:
        """
        Get the reward.

        Returns
        -------
        reward: float
            The reward.
        """
        reward = 0.0
        for drone in self.swarm_drones_list:
            if drone.state.value != DroneState.ACTIVE.value:
                continue
            if self.tick == len(drone.trajectory) - 1:
                drone.state = DroneState.IMPACTED
                if drone.sensitive_zone_targeted is not None:
                    self.sensitive_zones[drone.sensitive_zone_targeted.id].impacts += 1
                reward -= drone.potential_damage

        return reward

    def _get_episode_termination(self) -> bool:
        """
        Get the episode termination.

        Returns
        -------
        episode_termination: bool
            The episode termination.
        """
        return True if self.tick >= self.max_length else False

    def _get_episode_abortion(self) -> bool:
        """
        Get the episode abortion.

        Returns
        -------
        episode_abortion: bool
            The episode abortion.
        """
        return False

    def _get_info(self) -> dict[str, Any]:
        """
        Get the info.

        Returns
        -------
        info: dict[str, Any]
            The info.
        """
        # Access frequently used variables once
        swarm_drones_list = self.swarm_drones_list
        effectors_list = self.effectors_list
        sensitive_zones = self.sensitive_zones

        # Process drones using list comprehensions
        drones_state = [drone.state.name for drone in swarm_drones_list]
        active_drones = sum(
            1
            for drone in swarm_drones_list
            if drone.state.value == DroneState.ACTIVE.value
        )
        neutralized_drones = sum(
            1
            for drone in swarm_drones_list
            if drone.state.value == DroneState.NEUTRALIZED.value
        )
        impacted_drones = sum(
            1
            for drone in swarm_drones_list
            if drone.state.value == DroneState.IMPACTED.value
        )

        # Process effectors using list comprehensions
        effectors_kinematic_state = [
            effector.kinematic_state.name for effector in effectors_list
        ]
        effectors_weapon_state = [
            effector.weapon_state.name for effector in effectors_list
        ]
        effectors_kinematic_state_count = {
            effector.id: {
                k.name: v for k, v in effector.kinematic_states_counts.items()
            }
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

    def _render_frame(self) -> None | np.ndarray:
        """Render the frame."""
        self.renderer.plot_drones_positions(self.swarm_drones_list, self.tick)
        self.renderer.plot_effectors_aiming(self.effectors_list)
        self.renderer.update_drones_data(self.swarm_drones_list)
        self.renderer.update_effectors_data(self.effectors_list)
        if self.render_mode == "human":
            self.renderer.update()
            return None
        else:
            return self.renderer.get_rgb_array()
