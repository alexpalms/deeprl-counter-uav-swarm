from stable_baselines3 import PPO
from gymnasium import spaces
import numpy as np
import os

class Agent:
    def __init__(self, model_path="./model.zip", deterministic=True):
        self.swarm_drones_num = 50
        self.n_steps = 3
        self.effectors_num = 4
        self.observation_space = spaces.Dict(
            {
                "drones_zones_distance": spaces.Box(low=np.array([-1 for _ in range(self.swarm_drones_num * self.n_steps)]),
                                                    high=np.array([1 for _ in range(self.swarm_drones_num * self.n_steps)]), dtype=np.float32)
            }
        )

        self.action_space = spaces.MultiDiscrete([self.swarm_drones_num for _ in range(self.effectors_num)], dtype=np.int32)

        self.deterministic = deterministic
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot create agent, policy file '{model_path}' not found!")

        self.agent = PPO.load(model_path, device="cpu")

    def get_action(self, obs):
        actions, _ = self.agent.predict(obs, deterministic=self.deterministic)
        return actions