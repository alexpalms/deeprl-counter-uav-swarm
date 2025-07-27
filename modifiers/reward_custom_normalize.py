import gymnasium as gym
from gymnasium import spaces

class CustomWrapper(gym.Wrapper):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_reward_magnitude = None

        # Drop 'drones_zones_distance' from observation space
        if not isinstance(self.env.observation_space, spaces.Dict):
            raise TypeError("CustomWrapper requires an environment with Dict observation space.")

        # Create a modified observation space without 'drones_zones_distance'
        new_spaces = {
            key: space for key, space in self.env.observation_space.spaces.items()
            if key != "drones_zones_distance"
        }
        self.observation_space = spaces.Dict(new_spaces)

    def _filter_observation(self, observation):
        # Remove 'drones_zones_distance' from the actual observation
        if isinstance(observation, dict):
            observation = {
                key: value for key, value in observation.items()
                if key != "drones_zones_distance"
            }
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        normalized_reward = float(reward / self.max_reward_magnitude)
        filtered_observation = self._filter_observation(observation)
        return filtered_observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.max_reward_magnitude = info["max_reward_magnitude"]
        filtered_observation = self._filter_observation(observation)
        return filtered_observation, info
