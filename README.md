
<p align="center">
    <img src="./results/img/final_state.png" alt="Training Curves" width="75%"/>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2508.00641"><img src="https://img.shields.io/badge/paper-arXiv:2508.00641-B31B1B?logo=arxiv" alt="Paper"/></a>
</p>

<p align="center">
<a href="https://alexpalms.github.io/projects/02-rl_cuas/"><img src="https://img.shields.io/badge/blog-read%20post-blue" alt="Blog Post"/></a>
<a href="https://artificialtwin.com/projects/cuas/"><img src="https://img.shields.io/badge/project-view%20page-gold" alt="Company Project"/></a>
</p>

<p align="center">
<a href="https://github.com/alexpalms/deeprl-counter-uav-swarm/actions/workflows/code-checks.yaml"><img src="https://img.shields.io/github/actions/workflow/status/alexpalms/deeprl-counter-uav-swarm/code-checks.yaml?label=code%20checks%20(ruff%20%26%20pyright)&logo=github" alt="Code Checks"/></a>
<a href="https://github.com/alexpalms/deeprl-counter-uav-swarm/actions/workflows/pytest.yaml"><img src="https://img.shields.io/github/actions/workflow/status/alexpalms/deeprl-counter-uav-swarm/pytest.yaml?label=tests%20(pytest)&logo=github" alt="Pytest"/></a>
<a href="https://codecov.io/github/alexpalms/deeprl-counter-uav-swarm"><img src="https://codecov.io/github/alexpalms/deeprl-counter-uav-swarm/graph/badge.svg?token=4817P3HFDN" alt="PytestCoverage"/></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/supported%20os-linux-blue" alt="Supported OS"/>
<img src="https://img.shields.io/badge/python-%3E%3D3.12-blue?logo=python" alt="Python Version"/>
<img src="https://img.shields.io/github/last-commit/alexpalms/deeprl-counter-uav-swarm/main?label=repo%20latest%20update&logo=readthedocs" alt="Latest Repo Update"/>
</p>
<p align="center">
<img src="https://img.shields.io/github/license/alexpalms/deeprl-counter-uav-swarm?cacheBust=1" alt="Python Version"/>
</p>


# Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense

This repository contains a reinforcement learning (RL) framework for the decision-level interception prioritization of drone swarms. The project is designed to evaluate the performance of RL agents against classical heuristic methods in a simulated environment, focusing on the interception of hostile drones by kinetic effectors to minimize damage to sensitive zones.

The RL agents are trained to prioritize drone targets based on their potential threat levels, with the goal of maximizing the effectiveness of the defense system while minimizing collateral damage.

<!--This codebase integrates and completes the work presented in the paper "Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense", which can be found [here](https://arxiv.org/abs/2401.12345).-->

## Setup

**Everything in this guide will assume you are using Linux OS**

- Install `uv` ([Ref](https://github.com/astral-sh/uv)) (E.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Install package: `uv sync`

## Simulator

The simulator models a defense scenario where a swarm of kamikaze drones autonomously targets high-value zones protected by kinetic effectors (such as interceptors or directed energy weapons). The environment is three-dimensional and includes configurable numbers of hostile drones, static sensitive zones, and effectors with realistic kinematic and weapon dynamics. Each effector can only fire when locked onto a target and ready, and must periodically recharge.

Episodes begin with drones spawned at random locations, each aiming for a zone using pre-defined but unknown policies. The defender receives noisy, partial observations and must prioritize which drones to intercept at every timestep, considering constraints like limited firing rate, angular speed, and line-of-sight. The simulation supports large-scale, multi-agent scenarios and batch evaluation.

Attackers vary in speed, size, explosive power, and flight path, and their coordination is fixed to simulate low-cost adversaries. The defender’s challenge is to minimize total damage by making effective, real-time prioritization decisions under uncertainty and resource limitations. All scenario elements, including zones, drones, effectors, and sensors, are highly configurable for flexible experimentation.

The following figures illustrate key aspects of the simulation environment. The first image shows the scenario simulator in execution, including all relevant infographics such as drone and effector states, and protected zones. The second image presents the drone neutralization probability as a function of miss distance, providing insight into the effectiveness of the defense system under varying engagement conditions.

<table>
  <tr>
    <td width="50%"><img src="./results/img/simulator.png" alt="Simulator" width="100%"/></td>
    <td width="50%"><img src="./results/img/neutralization_probability_plot.png" alt="Neutralization Probability" width="100%"/></td>
  </tr>
  <tr>
    <td align="center">Scenario Simulator In Execution</td>
    <td align="center">Drone Neutralization Probability Plot</td>
  </tr>
</table>

## Training

To train a new reinforcement learning agent, configure your training parameters in a config file like [`examples/config.yaml`](examples/config.yaml) and run the training via the CLI:

```
uv run rl-cuas-cli train --config ./examples/config.yaml
```

The script supports resuming from checkpoints, automatic saving, evaluation during training, and early stopping based on reward thresholds or lack of improvement. Training and evaluation environments, model checkpoints, and logs are managed automatically according to your configuration.

Two versions of the PPO algorithm are available: the original PPO and the MaskablePPO (which supports action masking for invalid actions), both provided via Stable Baselines 3. You can select which algorithm to use by setting the `algo` field in your configuration file.

The following image shows a comparison of training curves between PPO and MaskablePPO:

<table>
  <tr>
    <td align="center" width="100%"><img src="./results/img/training_progression.svg" alt="Training Curves" width="70%"/></td>
  </tr>
  <tr>
    <td align="center">Training performance of PPO vs. MaskedPPO, showing cumulative reward per episode over environment steps. MaskedPPO converges ~10× faster by masking invalid actions (e.g., targeting already-neutralized drones), enabling more efficient and stable learning.</td>
  </tr>
</table>

## Evaluation and Results

To run a single inference episode and visualize or evaluate a specific policy (e.g., DeepRL, Classic, or Random), use the `evaluate` command of the CLI. This allows you to observe the agent's behavior and performance in the environment. For example, to run a single episode with the DeepRL policy and rendering enabled, use:
```
uv run rl-cuas-cli evaluate --policy deeprl --n_episodes 1 --seed 42
```
You can change the `--policy` argument to `classic` or `random` to evaluate other policies. Use the `--no_render` flag to disable visualization and speed up evaluation.

For a comprehensive comparison of all policies and automatic generation of evaluation figures and metrics, use the `compare` command of the CLI. This runs multiple episodes for each policy, aggregates the results, and produces all relevant plots and summary tables for damage, tracking, and weapon utilization:
```
uv run rl-cuas-cli compare --n_episodes 100 --seeds 10 20 30 42 50
```
The script will save the results and figures in the appropriate folders, allowing for easy analysis and reproducibility of the evaluation.

The following table and figures are generated using the default parameters of the `compare` command. They provide a comprehensive summary of the main evaluation metrics and and visual comparisons between the different policies.


| Metric                        | Classical Heuristic | Reinforcement Learning |
|-------------------------------|:------------------:|:---------------------:|
| Total Damage (Avg) [%]        | 52.14              | **40.70**             |
| In-Tracking Time (Avg) [%]    | 53.29              | **66.81**             |
| Weapon Utilization (Avg) [%]  | 54.99              | **63.29**             |

*Table: Evaluation Results. 100 Episodes × 5 Seeds*


<table>
  <tr>
    <td width="50%"><a href="https://youtu.be/GooNFDk42Nw" target="_blank"><img src="https://img.youtube.com/vi/GooNFDk42Nw/0.jpg" alt="Demo Video" width="100%"/></a></td>
    <td width="50%"><img src="./results/img/damage_distributions.svg" alt="Damage Comparison" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><a href="https://youtu.be/GooNFDk42Nw" target="_blank">Demo Video</a></td>
    <td align="center">Distribution of total zone damage percentage for each controller. The RL agent consistently limits damage to critical zones compared to the heuristic baseline and random controller.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%"><img src="./results/img/tracking_performance.svg" alt="Tracking Performance" width="100%"/></td>
    <td width="50%"><img src="./results/img/weapon_utilization.svg" alt="Weapon Utilization" width="100%"/></td>
  </tr>
  <tr>
    <td align="center">a) Tracking Performance</td>
    <td align="center">b) Weapon Utilization</td>
  </tr>
  <tr>
    <td colspan="2" align="center">Comparison of controller performance across two key enabling metrics: (a) target tracking efficiency and (b) weapon utilization. The DeepRL policy consistently achieves superior performance in both categories compared to the classical and random controllers, indicating improved resource allocation and sustained threat engagement over time.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%"><img src="./results/img/damage_vs_tracking.svg" alt="Damage vs Tracking Correlation" width="100%"/></td>
    <td width="50%"><img src="./results/img/damage_vs_weapon_utilization.svg" alt="Damage vs Weapon Utilization Correlation" width="100%"/></td>
  </tr>
  <tr>
    <td align="center">a) Damage vs Tracking Correlation</td>
    <td align="center">b) Damage vs Weapon Utilization Correlation</td>
  </tr>
  <tr>
    <td colspan="2" align="center">Scatter plots showing the relationship between zone damage and: (a) tracking efficiency, and (b) weapon utilization. While both correlations are negative, they are not strongly linear, highlighting that increased engagement opportunities (via better tracking and utilization) generally help reduce damage, but do not fully determine it due to the complex interplay of prioritization and threat behavior.</td>
  </tr>
</table>

## Citation
```latex
@misc{palmas2025reinforcementlearningdecisionlevelinterception,
      title={Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense},
      author={Alessandro Palmas},
      year={2025},
      eprint={2508.00641},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.00641},
}
```
