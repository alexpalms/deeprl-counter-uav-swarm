"""Comparative evaluation script."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm  # type: ignore

from rl_cuas.evaluation.evaluation import evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bucket_and_plot(values: dict[str, list[float]], features: dict[str, str]) -> None:
    """
    Bucket and plot the values.

    Parameters
    ----------
    values: dict[str, list[float]]
        The values to plot.
    features: dict[str, str]
        The features to plot.
    """
    grouped_lists = list(values.values())
    group_names = ["DeepRL", "Classic", "Random"]
    base_colors = ["#cf3030", "#aaaaaa", "#30a0cf"]  # Red, Gray, Blue-ish

    # Flatten all data to get global binning range
    all_values = np.concatenate(grouped_lists)
    overall_min, overall_max = float(np.min(all_values)), float(np.max(all_values))
    bins: np.ndarray = np.linspace(overall_min, overall_max, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    width = float(
        (bins[1] - bins[0]) * 0.9 / 3
    )  # Divide by number of groups to avoid overlap

    # Set dark theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))  # pyright: ignore[reportUnknownMemberType]
    fig.patch.set_facecolor("#121212")

    font_size = 14
    ax.set_title(  # pyright: ignore[reportUnknownMemberType]
        features["title"], color="white", fontsize=font_size + 2, pad=20
    )  # Move title up
    ax.set_xlabel(  # pyright: ignore[reportUnknownMemberType]
        features["xlabel"], color="white", fontsize=font_size, labelpad=20
    )  # Move x label down
    ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Frequency []", color="white", fontsize=font_size, labelpad=20
    )  # Move y label left
    ax.grid(True, linestyle="--", alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
    ax.tick_params(colors="white", labelsize=font_size)  # pyright: ignore[reportUnknownMemberType]

    legend_handles = []
    for group_idx, (arr, base_color, group_name) in enumerate(
        zip(grouped_lists, base_colors, group_names, strict=False)
    ):
        arr_np = np.array(arr)
        hist, _ = np.histogram(arr, bins=bins)
        offset = (group_idx - 1) * width  # -1, 0, 1 for left, center, right

        # Plot histogram
        ax.bar(  # pyright: ignore[reportUnknownMemberType]
            centers + offset,
            hist,
            width=width,
            alpha=0.6,
            label=f"{group_name}",
            color=base_color,
            edgecolor=base_color,
            linewidth=1.2,
        )

        # Gaussian fit
        mu, std = arr_np.mean(), arr_np.std()
        x = np.linspace(overall_min, overall_max, 1000)
        pdf = norm.pdf(x, mu, std) * len(arr) * (bins[1] - bins[0])  # pyright: ignore[reportUnknownMemberType]
        (_,) = ax.plot(x, pdf, linestyle="--", linewidth=2.0, color=base_color)  # pyright: ignore[reportUnknownMemberType]

        # Vertical line for the mean
        ax.axvline(mu, color=base_color, linestyle=":", linewidth=2.0)  # pyright: ignore[reportUnknownMemberType]

        # Add legend entry for Gaussian curve with mean and std
        legend_handles.append(  # pyright: ignore[reportUnknownMemberType]
            plt.Line2D(  # type: ignore
                [0],
                [0],
                color=base_color,
                linestyle="--",
                linewidth=2.0,
                label=f"{group_name} Gaussian (μ={mu:.2f}, σ={std:.2f})",
            )
        )

    # Add histogram handles to legend in reversed order
    for bar, group_name in reversed(
        list(zip(ax.containers, group_names, strict=False))
    ):
        legend_handles.insert(  # pyright: ignore[reportUnknownMemberType]
            0,
            plt.Rectangle(  # type: ignore
                (0, 0),
                1,
                1,
                color=bar.patches[0].get_facecolor(),  # type: ignore
                alpha=0.6,
                label=group_name + " Histogram",
            ),
        )

    ax.legend(handles=legend_handles, fontsize=font_size - 2)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    plt.tight_layout()
    plt.savefig(f"{features['image']}.svg", format="svg", dpi=300, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    plt.show(block=False)  # pyright: ignore[reportUnknownMemberType]


def scatter_correlation_plot(
    list_x: np.ndarray,
    list_y: np.ndarray,
    title: str = "Correlation Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    image_name: str = "correlation_scatter",
) -> None:
    """
    Scatter correlation plot.

    Parameters
    ----------
    list_x: np.ndarray
    list_y: np.ndarray
    title: str
    xlabel: str
    ylabel: str
    image_name: str
    """
    corr_coef = np.corrcoef(list_x, list_y)[0, 1]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))  # pyright: ignore[reportUnknownMemberType]
    fig.patch.set_facecolor("#121212")

    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        list_x[: int(len(list_x) / 2)],
        list_y[: int(len(list_x) / 2)],
        alpha=0.7,
        color="#cf3030",
        edgecolor="white",
        s=60,
        label="DeepRL",
    )
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        list_x[int(len(list_x) / 2) :],
        list_y[int(len(list_x) / 2) :],
        alpha=0.7,
        color="#aaaaaa",
        edgecolor="white",
        s=60,
        label="Classic",
    )
    ax.set_title(title, color="white", fontsize=16, pad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel(xlabel, color="white", fontsize=14, labelpad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(ylabel, color="white", fontsize=14, labelpad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.text(  # pyright: ignore[reportUnknownMemberType]
        0.05,
        0.95,
        f"Corr coef: {corr_coef:.2f}",
        transform=ax.transAxes,
        fontsize=14,
        color="white",
        verticalalignment="top",
        bbox={"facecolor": "#222222", "alpha": 0.7, "edgecolor": "none"},
    )
    ax.grid(True, linestyle="--", alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
    ax.tick_params(colors="white", labelsize=12)  # pyright: ignore[reportUnknownMemberType]
    ax.legend(fontsize=12)  # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()
    plt.savefig(f"{image_name}.svg", format="svg", dpi=300, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    plt.show(block=False)  # pyright: ignore[reportUnknownMemberType]


def comparative_evaluation(n_episodes: int, seeds: list[int]) -> None:
    """
    Run the comparative evaluation.

    Parameters
    ----------
    n_episodes: int
        The number of episodes to run.
    seeds: list[int]
        The seeds to use for the evaluation.
    """
    working_dir = Path.cwd()

    results_folder = os.path.join(working_dir, "runs", "comparative_evaluation")
    os.makedirs(results_folder, exist_ok=True)

    comparative_evaluation_results_path = os.path.join(
        results_folder, "comparative_evaluation_results.json"
    )

    if not os.path.exists(comparative_evaluation_results_path):
        logger.info("No results file found. Running comparative evaluation...")

        cumulative_rewards_group: dict[str, list[float]] = {
            "deeprl": [],
            "classic": [],
            "random": [],
        }
        effectors_tracking_group: dict[str, list[float]] = {
            "deeprl": [],
            "classic": [],
            "random": [],
        }
        effectors_weapon_utilization_group: dict[str, list[float]] = {
            "deeprl": [],
            "classic": [],
            "random": [],
        }
        for policy in ["deeprl", "classic", "random"]:
            logger.info(f"Running inference for policy: {policy}")
            for seed in seeds:
                (
                    cumulative_rewards,
                    effectors_tracking_states,
                    effectors_weapon_utilization,
                ) = evaluation(n_episodes, seed, policy)
                cumulative_rewards_group[policy].extend(cumulative_rewards)
                effectors_tracking_group[policy].extend(effectors_tracking_states)
                effectors_weapon_utilization_group[policy].extend(
                    effectors_weapon_utilization
                )

        results: dict[str, Any] = {
            "n_episodes": n_episodes,
            "seeds": seeds,
            "cumulative_rewards": cumulative_rewards_group,
            "effectors_tracking": effectors_tracking_group,
            "effectors_weapon_utilization": effectors_weapon_utilization_group,
        }

        # Dump results to a file
        with open(comparative_evaluation_results_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        logger.info("Results file found. Loading comparative evaluation results...")

        with open(comparative_evaluation_results_path) as f:
            results = json.load(f)
            if results["n_episodes"] != n_episodes:
                raise ValueError(
                    'Number of episodes in the file does not match the requested number. Please remove the "comparative_evaluation_results.json" file to re-run the evaluation.'
                )
            if results["seeds"] != seeds:
                raise ValueError(
                    'Seeds in the file do not match the requested seeds. Please remove the "comparative_evaluation_results.json" file to re-run the evaluation.'
                )
            cumulative_rewards_group = results["cumulative_rewards"]
            effectors_tracking_group = results["effectors_tracking"]
            effectors_weapon_utilization_group = results["effectors_weapon_utilization"]

    scatter_correlation_plot(
        np.concatenate(
            [cumulative_rewards_group["deeprl"], cumulative_rewards_group["classic"]]
        ),
        np.concatenate(
            [effectors_tracking_group["deeprl"], effectors_tracking_group["classic"]]
        ),
        title="Damage vs Tracking",
        xlabel="Cumulative Damage [%]",
        ylabel="In-Tracking Time [%]",
        image_name=os.path.join(results_folder, "damage_vs_tracking"),
    )
    scatter_correlation_plot(
        np.concatenate(
            [cumulative_rewards_group["deeprl"], cumulative_rewards_group["classic"]]
        ),
        np.concatenate(
            [
                effectors_weapon_utilization_group["deeprl"],
                effectors_weapon_utilization_group["classic"],
            ]
        ),
        title="Damage vs Weapon Utilization",
        xlabel="Cumulative Damage [%]",
        ylabel="Weapon Utilization [%]",
        image_name=os.path.join(results_folder, "damage_vs_weapon_utilization"),
    )
    features = [
        {
            "title": "Comparison of Damage Distributions",
            "xlabel": "Cumulative Damage [%]",
            "image": os.path.join(results_folder, "damage_distributions"),
        },
        {
            "title": "Comparison of Effectors Kinematic Performance",
            "xlabel": "In-Tracking Time [%]",
            "image": os.path.join(results_folder, "tracking_performance"),
        },
        {
            "title": "Comparison of Effectors Weapon Utilization",
            "xlabel": "Weapon Utilization [%]",
            "image": os.path.join(results_folder, "weapon_utilization"),
        },
    ]
    bucket_and_plot(cumulative_rewards_group, features[0])
    bucket_and_plot(effectors_tracking_group, features[1])
    bucket_and_plot(effectors_weapon_utilization_group, features[2])
