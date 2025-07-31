from inference import main as inference_main
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

def bucket_and_plot(values, features):
    grouped_lists = [v for v in values.values()]
    group_names = ["DeepRL", "Classic", "Random"]
    base_colors = ['#cf3030', '#aaaaaa', '#30a0cf']  # Red, Gray, Blue-ish

    # Flatten all data to get global binning range
    all_values = np.concatenate(grouped_lists)
    overall_min, overall_max = np.min(all_values), np.max(all_values)
    bins = np.linspace(overall_min, overall_max, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.9 / 3  # Divide by number of groups to avoid overlap

    # Set dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#121212')

    font_size = 14
    ax.set_title(features["title"], color='white', fontsize=font_size + 2)
    ax.set_xlabel(features["xlabel"], color='white', fontsize=font_size)
    ax.set_ylabel("Frequency []", color='white', fontsize=font_size)
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
                       label=f"{group_name} Gaussian (μ={mu:.1f}, σ={std:.1f})")
        )

    # Add histogram handles to legend in reversed order
    for bar, group_name in reversed(list(zip(ax.containers, group_names))):
        legend_handles.insert(0, plt.Rectangle((0,0),1,1, color=bar.patches[0].get_facecolor(), alpha=0.6, label=group_name + " Histogram"))

    ax.legend(handles=legend_handles, fontsize=font_size - 2)
    plt.tight_layout()
    plt.savefig(features["image"]+".svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show()

def main(n_episodes, seeds):
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

    #cumulative_rewards = {
    #    "deeprl": [10 * np.random.randn(n_episodes) + 100 for _ in seeds],  # Simulated data
    #    "classic": [10 * np.random.randn(n_episodes) + 80 for _ in seeds],  # Simulated data
    #    "random": [10 * np.random.randn(n_episodes) + 50 for _ in seeds]  # Simulated data
    #}

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