import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_reward_magnitude = None
        self.cumulative_damage = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        normalized_reward = abs(float(reward / self.max_reward_magnitude)) * 100
        self.cumulative_damage += normalized_reward
        if terminated or truncated:
            info["custom_eval_metrics/damage"] = self.cumulative_damage

        return observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.cumulative_damage = 0
        self.max_reward_magnitude = info["max_reward_magnitude"]
        return observation, info
