import heapq
import numpy as np

class Agent:
    def __init__(self):
        self.swarm_drones_num = 50
        self.effectors_num = 4

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