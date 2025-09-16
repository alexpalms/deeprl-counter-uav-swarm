"""A deep reinforcement learning agent that selects the drones based on the observation."""

import os
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO


class Agent:
    """
    A deep reinforcement learning agent that selects the drones based on the observation.

    Parameters
    ----------
        env: The environment.
        model_path: The path to the model.
        deterministic: Whether to use a deterministic policy.
    """

    def __init__(
        self, env: Any, model_path: str = "./model.zip", deterministic: bool = True
    ):
        self.env: Any = env
        self.deterministic = deterministic
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_path
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot create agent, policy file '{model_path}' not found!"
            )

        self.agent = MaskablePPO.load(model_path, device="cpu")

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
        obs = {
            key: value for key, value in obs.items() if key != "drones_zones_distance"
        }
        action_masks: np.ndarray = self.env.unwrapped.action_masks()
        actions, _ = self.agent.predict(
            obs,
            deterministic=self.deterministic,
            action_masks=action_masks,
        )
        return actions
