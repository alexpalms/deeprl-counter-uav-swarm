"""A gym wrapper that normalizes rewards by dividing them by a scaling factor."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_cuas.environment.environment import Environment


class CustomWrapper(
    gym.Wrapper[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]
):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.

    Parameters
    ----------
    env: Environment
        The environment.
    """

    env: Environment

    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self.max_reward_magnitude: float = 0.0

        # Drop 'drones_zones_distance' from observation space
        if not isinstance(self.env.observation_space, spaces.Dict):
            raise TypeError(
                "CustomWrapper requires an environment with Dict observation space."
            )

        # Create a modified observation space without 'drones_zones_distance'
        new_spaces = {
            key: space
            for key, space in self.env.observation_space.spaces.items()
            if key != "drones_zones_distance"
        }
        self.observation_space = spaces.Dict(new_spaces)

    def _filter_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Filter the observation.

        Parameters
        ----------
        observation: dict[str, np.ndarray]
            The observation.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The filtered observation.
        """
        # Remove 'drones_zones_distance' from the actual observation
        observation = {
            key: value
            for key, value in observation.items()
            if key != "drones_zones_distance"
        }
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Step the environment.

        Parameters
        ----------
        action: np.ndarray
            The action.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The observation.
        reward: float
            The reward.
        terminated: bool
            Whether the episode is terminated.
        truncated: bool
            Whether the episode is truncated.
        info: dict[str, Any]
            The info.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        normalized_reward = float(reward / self.max_reward_magnitude)
        filtered_observation = self._filter_observation(observation)
        return filtered_observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        **kwargs: Any
            The keyword arguments.

        Returns
        -------
        observation: dict[str, np.ndarray]
            The observation.
        info: dict[str, Any]
            The info.
        """
        observation, info = self.env.reset(**kwargs)
        self.max_reward_magnitude = info["max_reward_magnitude"]
        filtered_observation = self._filter_observation(observation)
        return filtered_observation, info
