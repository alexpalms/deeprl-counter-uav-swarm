import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from inference import main as inference_main
import argparse

def bucket_and_plot(list1, list2):
    # Convert to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Compute overall min and max for shared binning
    overall_min = min(arr1.min(), arr2.min())
    overall_max = max(arr1.max(), arr2.max())
    bins = np.linspace(overall_min, overall_max, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    # Histogram data
    hist1, _ = np.histogram(arr1, bins=bins)
    hist2, _ = np.histogram(arr2, bins=bins)

    # Fit Gaussian distributions
    mu1, std1 = arr1.mean(), arr1.std()
    mu2, std2 = arr2.mean(), arr2.std()

    x = np.linspace(overall_min, overall_max, 1000)
    pdf1 = norm.pdf(x, mu1, std1) * len(arr1) * (bins[1] - bins[0])
    pdf2 = norm.pdf(x, mu2, std2) * len(arr2) * (bins[1] - bins[0])

    # Set dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#121212')

    # Colors
    red_desat = '#cf3030'
    gray_neutral = '#aaaaaa'

    # Plot histograms
    ax.bar(centers - width/2, hist1, width=width, label="DeepRL", alpha=0.7,
           color=red_desat, linewidth=1.5, edgecolor=red_desat)
    ax.bar(centers + width/2, hist2, width=width, label="Classic", alpha=0.7,
           color=gray_neutral, linewidth=1.5, edgecolor=gray_neutral)

    # Plot Gaussian fits with μ and σ in legend
    label1 = f"DeepRL Gaussian (μ={mu1:.2f}, σ={std1:.2f})"
    label2 = f"Classic Gaussian (μ={mu2:.2f}, σ={std2:.2f})"
    ax.plot(x, pdf1, color=red_desat, linestyle="--", label=label1, linewidth=2.5)
    ax.plot(x, pdf2, color=gray_neutral, linestyle="--", label=label2, linewidth=2.5)

    # Vertical mean lines
    ax.axvline(mu1, color=red_desat, linestyle=":", linewidth=2.5)
    ax.axvline(mu2, color=gray_neutral, linestyle=":", linewidth=2.5)

    # Font size adjustments
    font_size = 22
    ax.set_title("Comparison of DeepRL vs Classic Control Damage Distribution", color='white', fontsize=font_size + 2)
    ax.set_xlabel("Value", color='white', fontsize=font_size)
    ax.set_ylabel("Count", color='white', fontsize=font_size)
    ax.legend(fontsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(colors='white', labelsize=font_size)

    plt.tight_layout()
    plt.show()

def main(n_episodes, n_seeds):
    cumulative_rewards = {
        "deep_rl": [],
        "classic": [],
        "random": []
    }
    for policy in ["deeprl", "classic", "random"]:
        print(f"Running inference for policy: {policy}")

        for seed in range(n_seeds):
            cumulative_rewards[policy].append(inference_main(n_episodes, seed, policy))

    bucket_and_plot(cumulative_rewards["deep_rl"], cumulative_rewards["classic"], cumulative_rewards["random"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=100, help='How many episodes to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 20, 30, 42, 50], help='Evaluation seeds')
    opt = parser.parse_args()
    print(opt)

    main(opt.n_episodes, opt.seeds)