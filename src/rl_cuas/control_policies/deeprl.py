"""A deep reinforcement learning agent that selects the drones based on the observation."""

import os
from importlib import resources
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from rl_cuas.control_policies.common import Agent
from rl_cuas.environment.environment import Environment
from rl_cuas.modifiers.metric_damage import CustomWrapper


class DeepRlAgent(Agent):
    """
    A deep reinforcement learning agent that selects the drones based on the observation.

    Parameters
    ----------
        env: The environment.
        model_path: The path to the model.
        deterministic: Whether to use a deterministic policy.
    """

    def __init__(
        self,
        env: Environment | CustomWrapper,
        model_name: str = "model.zip",
        deterministic: bool = True,
    ):
        self.env: Any = env
        self.deterministic = deterministic
        with resources.as_file(
            resources.files("rl_cuas.data") / "model.zip"
        ) as model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Cannot create agent, policy file '{model_path}' not found!"
                )

            self.agent = MaskablePPO.load(model_path, device="cpu")  # pyright:ignore[reportUnknownMemberType]

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
