"""Training plots."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main(ppo_file: str, masked_ppo_file: str) -> None:
    """
    Rims the training plots.

    Parameters
    ----------
    ppo_file: str
    masked_ppo_file: str
    """
    # Read CSV files
    ppo_df = pd.read_csv(ppo_file)  # pyright: ignore[reportUnknownMemberType]
    masked_ppo_df = pd.read_csv(masked_ppo_file)  # pyright: ignore[reportUnknownMemberType]

    # Set dark theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))  # pyright: ignore[reportUnknownMemberType]
    fig.patch.set_facecolor("#121212")

    font_size = 14
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        ppo_df["Step"], ppo_df["Value"], label="PPO", color="#cf3030", linewidth=2.0
    )
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        masked_ppo_df["Step"],
        masked_ppo_df["Value"],
        label="MaskedPPO",
        color="#30a0cf",
        linewidth=2.0,
    )

    ax.set_xlabel("Step", color="white", fontsize=font_size, labelpad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Cumulative Reward", color="white", fontsize=font_size, labelpad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Training Progression", color="white", fontsize=font_size + 2, pad=20)  # pyright: ignore[reportUnknownMemberType]
    ax.legend(fontsize=font_size - 2)  # pyright: ignore[reportUnknownMemberType]
    ax.grid(True, linestyle="--", alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
    ax.tick_params(colors="white", labelsize=font_size)  # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()
    plt.savefig("training_progression.svg", format="svg", dpi=300, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    plt.show(block=False)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training plots")
    parser.add_argument("--ppo_file", type=str, default="./PPO.csv")
    parser.add_argument("--masked_ppo_file", type=str, default="./MaskedPPO.csv")
    args = parser.parse_args()
    main(args.ppo_file, args.masked_ppo_file)
