"""Custom extractors."""

from typing import cast

import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import (
    get_flattened_obs_dim,  # pyright: ignore[reportUnknownVariableType]
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomFlatExtractor(BaseFeaturesExtractor):
    """
    Custom flat extractor.

    Parameters
    ----------
    observation_space: spaces.Box
        The observation space.
    layers: list[int]
        The layers.
    """

    def __init__(self, observation_space: spaces.Box, layers: list[int]):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)  # pyright: ignore[reportUnknownMemberType]

        if len(observation_space.shape) != 1:
            raise ValueError(
                "Only one dimensional `Box` spaces (i.e. flattened) are supported"
            )

        self.flatten = nn.Flatten()

        # Obs FC processing
        obs_extractor_layers: list[nn.Module] = []
        prev_size = observation_space.shape[0]
        for size in layers:
            obs_extractor_layers.append(nn.Linear(prev_size, size))
            obs_extractor_layers.append(nn.ReLU())
            prev_size = size

        self.obs_extractor = nn.Sequential(*obs_extractor_layers)

        # Update the features dim manually
        self._features_dim = layers[-1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the extractor.

        Parameters
        ----------
        observations: th.Tensor
            The observations.

        Returns
        -------
        th.Tensor:
            The extracted features.
        """
        observations_tensor = self.flatten(observations)

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return cast(th.Tensor, self.obs_extractor(observations_tensor))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom combined extractor.

    Parameters
    ----------
    observation_space: spaces.Dict
        The observation space.
    layers: list[int]
        The layers.
    """

    def __init__(self, observation_space: spaces.Dict, layers: list[int]) -> None:
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)  # pyright: ignore[reportUnknownMemberType]

        # Observation flattening
        obs_flatten: dict[str, nn.Module] = {}
        obs_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            obs_flatten[key] = nn.Flatten()
            obs_concat_size += get_flattened_obs_dim(subspace)

        self.obs_flatten = nn.ModuleDict(obs_flatten)

        # Obs FC processing
        obs_extractor_layers: list[nn.Module] = []
        prev_size = obs_concat_size
        for size in layers:
            obs_extractor_layers.append(nn.Linear(prev_size, size))
            obs_extractor_layers.append(nn.ReLU())
            prev_size = size

        self.obs_extractor = nn.Sequential(*obs_extractor_layers)

        # Update the features dim manually
        self._features_dim = layers[-1]

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass of the extractor.

        Parameters
        ----------
        observations: th.Tensor
            The observations.

        Returns
        -------
        th.Tensor:
            The extracted features.
        """
        embedding_list: list[th.Tensor] = []

        # Obs extractor
        obs_tensor_list: list[th.Tensor] = []
        for key, obs_flattener in self.obs_flatten.items():
            obs_tensor_list.append(obs_flattener(observations[key]))

        obs_flattened = th.cat(obs_tensor_list, dim=1)

        embedding_list.append(self.obs_extractor(obs_flattened))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(embedding_list, dim=1)
