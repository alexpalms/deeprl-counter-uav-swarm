from sb3_contrib import MaskablePPO
from gymnasium import spaces
import numpy as np
import os

class Agent:
    def __init__(self, env, model_path="./model.zip", deterministic=True):
        self.env = env
        self.deterministic = deterministic
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot create agent, policy file '{model_path}' not found!")

        self.agent = MaskablePPO.load(model_path, device="cpu")

    def get_action(self, obs):
        obs = {
                key: value for key, value in obs.items()
                if key != "drones_zones_distance"
            }
        actions, _ = self.agent.predict(obs, deterministic=self.deterministic, action_masks=self.env.unwrapped.action_masks())
        return actions