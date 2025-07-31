from inference import main as inference_main
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors
import json

def bucket_and_plot(values, features):
    grouped_lists = [v for v in values.values()]
    group_names = ["DeepRL", "Classic", "Random"]
    base_colors = ['#cf3030', '#aaaaaa', '#30a0cf']  # Red, Gray, Blue-ish

    # Flatten all data to get global binning range
    all_values = np.concatenate(grouped_lists)
    overall_min, overall_max = np.min(all_values), np.max(all_values)
    bins = np.linspace(overall_min, overall_max, 41)
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.9 / 3  # Divide by number of groups to avoid overlap

    # Set dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#121212')

    font_size = 14
    ax.set_title(features["title"], color='white', fontsize=font_size + 2, pad=20)  # Move title up
    ax.set_xlabel(features["xlabel"], color='white', fontsize=font_size, labelpad=20)  # Move x label down
    ax.set_ylabel("Frequency []", color='white', fontsize=font_size, labelpad=20)  # Move y label left
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(colors='white', labelsize=font_size)

    legend_handles = []
    for group_idx, (arr, base_color, group_name) in enumerate(zip(grouped_lists, base_colors, group_names)):
        arr = np.array(arr)
        hist, _ = np.histogram(arr, bins=bins)
        offset = (group_idx - 1) * width  # -1, 0, 1 for left, center, right

        # Plot histogram
        bar = ax.bar(centers + offset, hist, width=width,
                     alpha=0.6, label=f"{group_name}", color=base_color,
                     edgecolor=base_color, linewidth=1.2)

        # Gaussian fit
        mu, std = arr.mean(), arr.std()
        x = np.linspace(overall_min, overall_max, 1000)
        pdf = norm.pdf(x, mu, std) * len(arr) * (bins[1] - bins[0])
        curve, = ax.plot(x, pdf, linestyle='--', linewidth=2.0, color=base_color)

        # Vertical line for the mean
        ax.axvline(mu, color=base_color, linestyle=":", linewidth=2.0)

        # Add legend entry for Gaussian curve with mean and std
        legend_handles.append(
            plt.Line2D([0], [0], color=base_color, linestyle='--', linewidth=2.0,
                       label=f"{group_name} Gaussian (μ={mu:.2f}, σ={std:.2f})")
        )

    # Add histogram handles to legend in reversed order
    for bar, group_name in reversed(list(zip(ax.containers, group_names))):
        legend_handles.insert(0, plt.Rectangle((0,0),1,1, color=bar.patches[0].get_facecolor(), alpha=0.6, label=group_name + " Histogram"))

    ax.legend(handles=legend_handles, fontsize=font_size - 2)
    plt.tight_layout()
    plt.savefig(features["image"]+".svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show(block=False)

def scatter_correlation_plot(list_x, list_y, title="Correlation Scatter Plot", xlabel="X", ylabel="Y", image_name="correlation_scatter"):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array(list_x)
    y = np.array(list_y)
    corr_coef = np.corrcoef(x, y)[0, 1]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#121212')

    ax.scatter(x[:int(len(x)/2)], y[:int(len(x)/2)], alpha=0.7, color='#cf3030', edgecolor='white', s=60, label=f"DeepRL")
    ax.scatter(x[int(len(x)/2):], y[int(len(x)/2):], alpha=0.7, color='#aaaaaa', edgecolor='white', s=60, label=f"Classic")
    ax.set_title(title, color='white', fontsize=16, pad=20)  # Move title up
    ax.set_xlabel(xlabel, color='white', fontsize=14, labelpad=20)  # Move x label down
    ax.set_ylabel(ylabel, color='white', fontsize=14, labelpad=20)  # Move y label left
    ax.text(0.05, 0.95, f"Corr coef: {corr_coef:.2f}", transform=ax.transAxes,
        fontsize=14, color='white', verticalalignment='top', bbox=dict(facecolor='#222222', alpha=0.7, edgecolor='none'))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(colors='white', labelsize=12)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{image_name}.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show(block=False)

def main(n_episodes, seeds):
    # Check if results file exists, if not, run the evaluation
    import os
    if not os.path.exists("comparative_evaluation_results.json"):
        print("No results file found. Running comparative evaluation...")

        cumulative_rewards_group = { "deeprl": [], "classic": [], "random": [] }
        effectors_tracking_group = { "deeprl": [], "classic": [], "random": [] }
        effectors_weapon_utilization_group = { "deeprl": [], "classic": [], "random": [] }
        for policy in ["deeprl", "classic", "random"]:
            print(f"Running inference for policy: {policy}")
            for seed in seeds:
                cumulative_rewards, effectors_tracking_states, effectors_weapon_utilization = inference_main(n_episodes, seed, policy)
                cumulative_rewards_group[policy].extend(cumulative_rewards)
                effectors_tracking_group[policy].extend(effectors_tracking_states)
                effectors_weapon_utilization_group[policy].extend(effectors_weapon_utilization)

        results = {
            "n_episodes": n_episodes,
            "seeds": seeds,
            "cumulative_rewards": cumulative_rewards_group,
            "effectors_tracking": effectors_tracking_group,
            "effectors_weapon_utilization": effectors_weapon_utilization_group
        }

        # Dump results to a file
        with open("./comparative_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4)
    else:
        print("Results file found. Loading comparative evaluation results...")
        with open("./comparative_evaluation_results.json", "r") as f:
            results = json.load(f)
            assert results["n_episodes"] == n_episodes, "Number of episodes in the file does not match the requested number. Please remove the \"comparative_evaluation_results.json\" file to re-run the evaluation."
            assert results["seeds"] == seeds, "Seeds in the file do not match the requested seeds. Please remove the \"comparative_evaluation_results.json\" file to re-run the evaluation."
            cumulative_rewards_group = results["cumulative_rewards"]
            effectors_tracking_group = results["effectors_tracking"]
            effectors_weapon_utilization_group = results["effectors_weapon_utilization"]

    scatter_correlation_plot(np.concatenate([cumulative_rewards_group["deeprl"], cumulative_rewards_group["classic"]]),
                             np.concatenate([effectors_tracking_group["deeprl"], effectors_tracking_group["classic"]]),
                             title="Damage vs Tracking", xlabel="Cumulative Damage [%]", ylabel="In-Tracking Time [%]",
                             image_name="damage_vs_tracking")
    scatter_correlation_plot(np.concatenate([cumulative_rewards_group["deeprl"], cumulative_rewards_group["classic"]]),
                             np.concatenate([effectors_weapon_utilization_group["deeprl"], effectors_weapon_utilization_group["classic"]]),
                             title="Damage vs Weapon Utilization", xlabel="Cumulative Damage [%]", ylabel="Weapon Utilization [%]",
                             image_name="damage_vs_weapon_utilization")
    features = [
        {"title": "Comparison of Damage Distributions", "xlabel": "Cumulative Damage [%]", "image": "damage_distributions"},
        {"title": "Comparison of Effectors Kinematic Performance", "xlabel": "In-Tracking Time [%]", "image": "tracking_performance"},
        {"title": "Comparison of Effectors Weapon Utilization", "xlabel": "Weapon Utilization [%]", "image": "weapon_utilization"},
    ]
    bucket_and_plot(cumulative_rewards_group, features[0])
    bucket_and_plot(effectors_tracking_group, features[1])
    bucket_and_plot(effectors_weapon_utilization_group, features[2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=100, help='How many episodes to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 20, 30, 42, 50], help='Evaluation seeds')
    opt = parser.parse_args()
    print(opt)

    main(opt.n_episodes, opt.seeds)