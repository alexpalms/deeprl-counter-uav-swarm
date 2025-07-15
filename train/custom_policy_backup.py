from gymnasium import spaces
import torch as th
from torch import nn
from typing import Callable, List, Tuple

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

class CustomFlatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, layers: List[int]=[64, 64], activation: List[str]=["relu", "relu"]):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        assert (isinstance(layers, List) and len(layers) > 0), "`layers` in `CustomFlatExtractor` needs to be a non-empty `List[int]`"
        assert (isinstance(activation, List) and len(activation) > 0), "`activation` in `CustomFlatExtractor` needs to be a non-empty `List[str]`"
        assert (len(layers) == len(activation)), "`layers` and `activation` must have the same length"
        assert (all(act in ["relu", "elu"] for act in activation)), "`activation` must be a list of `relu` or `elu`"

        assert (isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1), \
                            "Only one dimensional `Box` spaces (i.e. flattened) are supported"

        self.flatten = nn.Flatten()

        # Obs FC processing
        obs_extractor_layers = []
        prev_size = observation_space.shape[0]
        for size, act in zip(layers, activation):
            obs_extractor_layers.append(nn.Linear(prev_size, size))
            if act == "relu":
                obs_extractor_layers.append(nn.ReLU())
            else:  # act == "elu"
                obs_extractor_layers.append(nn.ELU())
            prev_size = size

        self.obs_extractor = nn.Sequential(*obs_extractor_layers)

        # Update the features dim manually
        self._features_dim = layers[-1]

    def forward(self, observations) -> th.Tensor:

        observations_tensor = self.flatten(observations)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return self.obs_extractor(observations_tensor)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, layers: List[int]=[64, 64], activation: List[str]=["relu", "relu"]):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        assert (isinstance(layers, List) and len(layers) > 0), "`layers` in `CustomFlatSpaceExtractor` needs to be a non-empty `List[int]`"
        assert (isinstance(activation, List) and len(activation) > 0), "`activation` in `CustomFlatSpaceExtractor` needs to be a non-empty `List[str]`"
        assert (len(layers) == len(activation)), "`layers` and `activation` must have the same length"
        assert (all(act in ["relu", "elu"] for act in activation)), "`activation` must be a list of `relu` or `elu`"

        # Observation flattening
        obs_flatten = {}
        obs_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            obs_flatten[key] = nn.Flatten()
            obs_concat_size += get_flattened_obs_dim(subspace)

        self.obs_flatten = nn.ModuleDict(obs_flatten)

        # Obs FC processing
        obs_extractor_layers = []
        prev_size = obs_concat_size
        for size, act in zip(layers, activation):
            obs_extractor_layers.append(nn.Linear(prev_size, size))
            if act == "relu":
                obs_extractor_layers.append(nn.ReLU())
            else:  # act == "elu"
                obs_extractor_layers.append(nn.ELU())
            prev_size = size

        self.obs_extractor = nn.Sequential(*obs_extractor_layers)

        # Update the features dim manually
        self._features_dim = layers[-1]

    def forward(self, observations) -> th.Tensor:

        embedding_list = []

        # Obs extractor
        obs_tensor_list = []
        for key, obs_flattener in self.obs_flatten.items():
            obs_tensor_list.append(obs_flattener(observations[key]))

        obs_flattened = th.cat(obs_tensor_list, dim=1)

        embedding_list.append(self.obs_extractor(obs_flattened))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(embedding_list, dim=1)

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function with just identity module (no layers)
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    """

    def __init__(
        self,
        feature_dim: int,
        policy_layers: List[int]=[],
        policy_activation: List[str]=[],
        value_layers: List[int]=[],
        value_activation: List[str]=[]
    ):
        super().__init__()

        assert (len(policy_layers) == len(policy_activation)), "`policy_layers` and `policy_activation` must have the same length"
        assert (len(value_layers) == len(value_activation)), "`value_layers` and `value_activation` must have the same length"
        assert (all(act in ["relu", "elu"] for act in policy_activation)), "`policy_activation` must be a list of `relu` or `elu`"
        assert (all(act in ["relu", "elu"] for act in value_activation)), "`value_activation` must be a list of `relu` or `elu`"

        # Initialize the policy network
        self.policy_net = self.build_network(feature_dim, policy_layers, policy_activation)

        # Initialize the value network
        self.value_net = self.build_network(feature_dim, value_layers, value_activation)

        # The latent dimensions are the output sizes of the last layers
        self.latent_dim_pi = policy_layers[-1] if policy_layers else feature_dim
        self.latent_dim_vf = value_layers[-1] if value_layers else feature_dim

    def build_network(self, input_dim: int, layers: List[int], activation: List[str]) -> nn.Sequential:
        """
        Helper method to build a feedforward neural network with specified layer sizes.
        :param input_dim: size of the input layer
        :param layers: list of layer sizes for the network
        :return: a sequential model containing the layers
        """
        network = []
        last_dim = input_dim

        for layer_size, act in zip(layers, activation):
            network.append(nn.Linear(last_dim, layer_size))
            if act == "relu":
                network.append(nn.ReLU())
            else:  # act == "elu"
                network.append(nn.ELU())
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


class CustomActorCriticPolicy(ActorCriticPolicy):
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
        self.policy_value_kwargs = kwargs.pop("policy_value_kwargs", {})
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, **self.policy_value_kwargs)