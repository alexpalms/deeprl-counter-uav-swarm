import heapq
import numpy as np

from gymnasium import spaces

from neodynamics.interface import create_agent_server

class Agent:
    def __init__(self):
        self.swarm_drones_num = 50
        self.n_steps = 3
        self.effectors_num = 4
        self.observation_space = spaces.Dict(
            {
                "drones_zones_distance": spaces.Box(low=np.array([-1 for _ in range(self.swarm_drones_num * self.n_steps)]),
                                                    high=np.array([1 for _ in range(self.swarm_drones_num * self.n_steps)]), dtype=np.float32)
            }
        )

        self.action_space = spaces.MultiDiscrete([self.swarm_drones_num for _ in range(self.effectors_num)], dtype=np.int32)

    def get_action(self, obs):
        # Get the most recent distances for each drone by taking the last drones_num values
        distance_from_sensitive_zones = obs["drones_zones_distance"][-self.swarm_drones_num:]
        target_drones = self.k_smallest_distances(distance_from_sensitive_zones, self.effectors_num)
        return np.array(target_drones, dtype=np.int32)

    def k_smallest_distances(self, distance_from_sensitive_zones, k):
        heap = [(distance, i) for i, distance in enumerate(distance_from_sensitive_zones)]
        heapq.heapify(heap)
        smallest_indices = []
        for _ in range(k):
            smallest_indices.append(heapq.heappop(heap)[1])
        return smallest_indices


if __name__ == "__main__":
    agent = Agent()
    create_agent_server(agent)