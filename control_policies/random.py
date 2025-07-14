import numpy as np

from gymnasium import spaces

class Agent:
    def __init__(self):
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

    def get_action(self, obs):
        return self.action_space.sample()