from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, List, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function with just identity module (no layers)
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    """

    def __init__(
        self,
        feature_dim: int,
        actor_layers: List[int]=[],
        critic_layers: List[int]=[],
    ):
        super().__init__()

        # Initialize the policy network
        self.policy_net = self.build_network(feature_dim, actor_layers)

        # Initialize the value network
        self.value_net = self.build_network(feature_dim, critic_layers)

        # The latent dimensions are the output sizes of the last layers
        self.latent_dim_pi = actor_layers[-1] if actor_layers else feature_dim
        self.latent_dim_vf = critic_layers[-1] if critic_layers else feature_dim

    def build_network(self, input_dim: int, layers: List[int]) -> nn.Sequential:
        """
        Helper method to build a feedforward neural network with specified layer sizes.
        :param input_dim: size of the input layer
        :param layers: list of layer sizes for the network
        :return: a sequential model containing the layers
        """
        network = []
        last_dim = input_dim

        for layer_size in layers:
            network.append(nn.Linear(last_dim, layer_size))
            network.append(nn.ReLU())
            last_dim = layer_size

        # If no layers are provided, the network will act as an Identity module
        if not layers:
            return nn.Identity()

        return nn.Sequential(*network)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.actor_critic_kwargs = kwargs.pop("actor_critic_kwargs", {})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, **self.actor_critic_kwargs)

class CustomMaskablePPOPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.actor_critic_kwargs = kwargs.pop("actor_critic_kwargs", {})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, **self.actor_critic_kwargs)