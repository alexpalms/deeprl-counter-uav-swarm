"""A simple agent that selects the drones randomly."""

import numpy as np
from gymnasium import spaces

from rl_cuas.control_policies.common import Agent


class RandomAgent(Agent):
    """A simple agent that selects the drones randomly."""

    def __init__(self) -> None:
        self.swarm_drones_num: int = 50
        self.n_steps: int = 3
        self.effectors_num: int = 4
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "drones_zones_distance": spaces.Box(
                    low=np.array(
                        [-1 for _ in range(self.swarm_drones_num * self.n_steps)]
                    ),
                    high=np.array(
                        [1 for _ in range(self.swarm_drones_num * self.n_steps)]
                    ),
                    dtype=np.float32,
                )
            }
        )

        self.action_space = spaces.MultiDiscrete(
            [self.swarm_drones_num for _ in range(self.effectors_num)], dtype=np.int32
        )

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """
        Get the action for the agent.

        Parameters
        ----------
        obs: dict[str, np.ndarray]
            The observation of the environment.

        Returns
        -------
        np.ndarray
            The action for the agent.
        """
        action: np.ndarray = self.action_space.sample()
        return action
