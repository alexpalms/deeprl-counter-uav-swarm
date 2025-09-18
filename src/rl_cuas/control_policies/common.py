"""Common classes for all agents."""

from abc import abstractmethod
from typing import Any

import numpy as np


class Agent:
    """Base class for all agents."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Get the action for the agent."""
        pass
