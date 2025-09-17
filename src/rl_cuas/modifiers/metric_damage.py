"""A gym wrapper that normalizes rewards by dividing them by a scaling factor."""

from typing import Any

import gymnasium as gym
import numpy as np

from rl_cuas.environment.base_classes import (
    EffectorKinematicState,
    EffectorWeaponState,
)
from rl_cuas.environment.environment import Environment


class CustomWrapper(
    gym.Wrapper[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]
):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.

    Parameters
    ----------
    env: Any
        The environment.
    """

    env: Environment

    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self.max_reward_magnitude: float = 0.0
        self.cumulative_damage: float = 0.0
        self.effectors_kinematic_states_counters = {
            EffectorKinematicState.CHASING: 0,
            EffectorKinematicState.TRACKING: 0,
        }
        self.effectors_weapon_states_counters = {
            EffectorWeaponState.READY: 0,
            EffectorWeaponState.SHOOTING: 0,
            EffectorWeaponState.RECHARGING: 0,
        }
        self.num_effectors: int = len(self.env.effectors_list)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Step the environment.

        Parameters
        ----------
        action: np.ndarray
            The action to take.

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
        for effector in self.env.effectors_list:
            self.effectors_kinematic_states_counters[effector.kinematic_state] += 1
            self.effectors_weapon_states_counters[effector.weapon_state] += 1
        normalized_reward = abs(float(reward) / self.max_reward_magnitude) * 100.0
        self.cumulative_damage += normalized_reward
        if terminated or truncated:
            info["custom_eval_metrics/damage"] = self.cumulative_damage
            info["custom_eval_metrics/effectors_kinematic_states"] = {
                k.name: v / (self.num_effectors * self.env.tick) * 100
                for k, v in self.effectors_kinematic_states_counters.items()
            }
            info["custom_eval_metrics/effectors_weapon_states"] = {
                k.name: v / (self.num_effectors * self.env.tick) * 100
                for k, v in self.effectors_weapon_states_counters.items()
            }
            info["custom_eval_metrics/effectors_weapon_states"]["UTILIZATION"] = (
                (
                    self.effectors_weapon_states_counters[EffectorWeaponState.SHOOTING]
                    + self.effectors_weapon_states_counters[
                        EffectorWeaponState.RECHARGING
                    ]
                )
                / (self.num_effectors * self.env.tick)
                * 100
            )

        return observation, normalized_reward, terminated, truncated, info

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
        self.cumulative_damage = 0.0
        self.effectors_kinematic_states_counters = {
            EffectorKinematicState.CHASING: 0,
            EffectorKinematicState.TRACKING: 0,
        }
        self.effectors_weapon_states_counters = {
            EffectorWeaponState.READY: 0,
            EffectorWeaponState.SHOOTING: 0,
            EffectorWeaponState.RECHARGING: 0,
        }
        self.max_reward_magnitude = info["max_reward_magnitude"]
        return observation, info
