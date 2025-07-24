import gymnasium as gym

class CustomWrapper(gym.Wrapper):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_reward_magnitude = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        normalized_reward = float(reward / self.max_reward_magnitude)
        return observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.max_reward_magnitude = info["max_reward_magnitude"]
        return observation, info
