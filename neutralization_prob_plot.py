
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left

# Original (malformed) inputs
distance_buckets = np.array([0.0, 0.05, 0.06, 0.35, 0.36, 0.5])
prob_buckets = np.array([0.99, 0.95, 0.25, 0.20, 0.40, 0.35])

# Define the function
def calculate_neutralization_probability(distance, neutralization_dynamics_distance_buckets, neutralization_dynamics_prob_buckets):
    index = bisect_left(neutralization_dynamics_distance_buckets, distance)
    if index == 0:
        return neutralization_dynamics_prob_buckets[0]
    elif index == len(neutralization_dynamics_distance_buckets):
        return neutralization_dynamics_prob_buckets[-1]
    else:
        x0, x1 = neutralization_dynamics_distance_buckets[index - 1], neutralization_dynamics_distance_buckets[index]
        y0, y1 = neutralization_dynamics_prob_buckets[index - 1], neutralization_dynamics_prob_buckets[index]
        return y0 + (y1 - y0) * (distance - x0) / (x1 - x0)

# Plotting
distances = np.linspace(0, 0.5, 500)
probabilities = [calculate_neutralization_probability(d, distance_buckets, prob_buckets) for d in distances]

plt.figure(figsize=(8, 5))
plt.plot(distances, probabilities)
plt.scatter(distance_buckets, prob_buckets, color='red', zorder=5)
plt.title('Neutralization Probability vs Miss Distance')
plt.xlabel('Miss Distance [m]')
plt.ylabel('Neutralization Probability')
plt.grid(True)
plt.tight_layout()
# Save high-res image
plt.savefig("neutralization_probability_plot.svg")
plt.show()