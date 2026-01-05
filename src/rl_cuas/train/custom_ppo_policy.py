"""Custom PPO policy."""

from collections.abc import Callable
from typing import Any

import numpy as np
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class CustomPPOPolicy(ActorCriticPolicy):
    """
    Custom PPO policy.

    Parameters
    ----------
    observation_space: spaces.Space
        The observation space.
    action_space: spaces.Space
        The action space.
    lr_schedule: Callable[[float], float]
        The learning rate schedule.
    *args: Any
        Additional arguments.
    **kwargs: Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        observation_space: spaces.Space[dict[str, spaces.Space[Any]]],
        action_space: spaces.Space[np.ndarray],
        lr_schedule: Callable[[float], float],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.actor_critic_kwargs = kwargs.pop("actor_critic_kwargs", {})
        super().__init__(  # pyright:ignore[reportUnknownMemberType]
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor."""
        net_arch: dict[str, list[int]] = {
            "pi": self.actor_critic_kwargs.get("actor_layers", []),
            "vf": self.actor_critic_kwargs.get("critic_layers", []),
        }
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch, activation_fn=nn.ReLU, device="auto"
        )


class CustomMaskablePPOPolicy(MaskableActorCriticPolicy):
    """
    Custom Maskable PPO policy.

    Parameters
    ----------
    observation_space: spaces.Space
        The observation space.
    action_space: spaces.Space
        The action space.
    lr_schedule: Callable[[float], float]
        The learning rate schedule.
    *args: Any
        Additional arguments.
    **kwargs: Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        observation_space: spaces.Space[dict[str, spaces.Space[Any]]],
        action_space: spaces.Space[np.ndarray],
        lr_schedule: Callable[[float], float],
        *args: Any,
        **kwargs: Any,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.actor_critic_kwargs = kwargs.pop("actor_critic_kwargs", {})
        super().__init__(  # pyright:ignore[reportUnknownMemberType]
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor."""
        net_arch: dict[str, list[int]] = {
            "pi": self.actor_critic_kwargs.get("actor_layers", []),
            "vf": self.actor_critic_kwargs.get("critic_layers", []),
        }
        self.mlp_extractor = MlpExtractor(
            self.features_dim, net_arch, activation_fn=nn.ReLU, device="auto"
        )
