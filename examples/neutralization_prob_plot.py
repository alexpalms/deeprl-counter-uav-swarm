"""Neutralization probability plot."""

from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np


# Define the function
def calculate_neutralization_probability(
    distance: float,
    neutralization_dynamics_distance_buckets: np.ndarray,
    neutralization_dynamics_prob_buckets: np.ndarray,
) -> float:
    """
    Calculate the neutralization probability.

    Parameters
    ----------
    distance: float
        The distance.
    neutralization_dynamics_distance_buckets: np.ndarray
        The neutralization dynamics distance buckets.
    neutralization_dynamics_prob_buckets: np.ndarray
        The neutralization dynamics probability buckets.

    Returns
    -------
    probability: float
        The neutralization probability.
    """
    index = bisect_left(neutralization_dynamics_distance_buckets, distance)
    if index == 0:
        return float(neutralization_dynamics_prob_buckets[0])
    elif index == len(neutralization_dynamics_distance_buckets):
        return float(neutralization_dynamics_prob_buckets[-1])
    else:
        x0, x1 = (
            neutralization_dynamics_distance_buckets[index - 1],
            neutralization_dynamics_distance_buckets[index],
        )
        y0, y1 = (
            neutralization_dynamics_prob_buckets[index - 1],
            neutralization_dynamics_prob_buckets[index],
        )
        return float(y0 + (y1 - y0) * (distance - x0) / (x1 - x0))


if __name__ == "__main__":
    # Original (malformed) inputs
    distance_buckets = np.array([0.0, 0.05, 0.06, 0.35, 0.36, 0.5])
    prob_buckets = np.array([0.99, 0.95, 0.25, 0.20, 0.40, 0.35])

    # Plotting
    distances = np.linspace(0, 0.5, 500)
    probabilities = [
        calculate_neutralization_probability(d, distance_buckets, prob_buckets)
        for d in distances
    ]

    plt.figure(figsize=(8, 5))  # pyright: ignore[reportUnknownMemberType]
    plt.plot(distances, probabilities)  # pyright: ignore[reportUnknownMemberType]
    plt.scatter(distance_buckets, prob_buckets, color="red", zorder=5)  # pyright: ignore[reportUnknownMemberType]
    plt.title("Neutralization Probability vs Miss Distance")  # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("Miss Distance [m]")  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("Neutralization Probability")  # pyright: ignore[reportUnknownMemberType]
    plt.grid(True)  # pyright: ignore[reportUnknownMemberType]
    plt.tight_layout()
    # Save high-res image
    plt.savefig("neutralization_probability_plot.svg")  # pyright: ignore[reportUnknownMemberType]
    plt.show()  # pyright: ignore[reportUnknownMemberType]
