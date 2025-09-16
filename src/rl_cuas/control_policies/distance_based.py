"""
A simple agent that selects the drones with the smallest distances from sensitive zones.

This is a classic agent that does not use reinforcement learning.
"""

import heapq

import numpy as np


class Agent:
    """
    A simple agent that selects the drones with the smallest distances from sensitive zones.

    This is a classic agent that does not use reinforcement learning.
    """

    def __init__(self) -> None:
        self.swarm_drones_num: int = 50
        self.effectors_num: int = 4

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
        # Get the most recent distances for each drone by taking the last drones_num values
        distance_from_sensitive_zones = obs["drones_zones_distance"][
            -self.swarm_drones_num :
        ]
        target_drones = self.k_smallest_distances(
            distance_from_sensitive_zones, self.effectors_num
        )
        return np.array(target_drones, dtype=np.int32)

    def k_smallest_distances(
        self, distance_from_sensitive_zones: np.ndarray, k: int
    ) -> list[int]:
        """
        Get the k smallest distances from sensitive zones.

        Parameters
        ----------
            distance_from_sensitive_zones: The distances from sensitive zones.
            k: The number of smallest distances to return.

        Returns
        -------
            The k smallest distances from sensitive zones.
        """
        heap = [
            (distance, i) for i, distance in enumerate(distance_from_sensitive_zones)
        ]
        heapq.heapify(heap)
        smallest_indices: list[int] = []
        for _ in range(k):
            smallest_indices.append(heapq.heappop(heap)[1])
        return smallest_indices
