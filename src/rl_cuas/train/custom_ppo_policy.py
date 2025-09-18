"""Custom PPO policy."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function with just identity module (no layers).

    It receives as input the features extracted by the features extractor.

    Parameters
    ----------
    feature_dim: int
        Dimension of the features extracted with the features_extractor (e.g. features from a CNN).
    actor_layers: list[int]
        List of layer sizes for the actor network.
    critic_layers: list[int]
        List of layer sizes for the critic network.
    """

    def __init__(
        self,
        feature_dim: int,
        actor_layers: list[int],
        critic_layers: list[int],
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        # Initialize the policy network
        self.policy_net = self.build_network(feature_dim, actor_layers)

        # Initialize the value network
        self.value_net = self.build_network(feature_dim, critic_layers)

        # The latent dimensions are the output sizes of the last layers
        self.latent_dim_pi = actor_layers[-1] if actor_layers else feature_dim
        self.latent_dim_vf = critic_layers[-1] if critic_layers else feature_dim

    def build_network(self, input_dim: int, layers: list[int]) -> nn.Module:
        """
        Build a feedforward neural network with specified layer sizes.

        Parameters
        ----------
        input_dim: int
            Size of the input layer.
        layers: list[int]
            List of layer sizes for the network.
        :param layers: list of layer sizes for the network

        Returns
        -------
        nn.Module:
            A sequential model containing the layers.
        """
        network: list[nn.Module] = []
        last_dim = input_dim

        for layer_size in layers:
            network.append(nn.Linear(last_dim, layer_size))
            network.append(nn.ReLU())
            last_dim = layer_size

        # If no layers are provided, the network will act as an Identity module
        if not layers:
            return nn.Identity()

        return nn.Sequential(*network)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        features: th.Tensor
            The features to forward.

        Returns
        -------
        tuple[th.Tensor, th.Tensor]:
            Latent policy and latent value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass of the actor network.

        Parameters
        ----------
        features: th.Tensor
            The features to forward.

        Returns
        -------
        th.Tensor:
            The latent policy of the actor network.
        """
        return cast(th.Tensor, self.policy_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass of the critic network.

        Parameters
        ----------
        features: th.Tensor
            The features to forward.

        Returns
        -------
        th.Tensor:
            The latent value of the critic network.
        """
        return cast(th.Tensor, self.value_net(features))


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
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
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
        # Old solution not necessarily needed
        # self.mlp_extractor = CustomNetwork(
        #    self.features_dim, **self.actor_critic_kwargs
        # )


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
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
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
        # Old solution not necessarily needed
        # self.mlp_extractor = CustomNetwork(
        #    self.features_dim, **self.actor_critic_kwargs
        # )
