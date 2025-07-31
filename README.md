# Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense

# Overview


# Setup
```
conda create -n cuas python=3.11
conda activate cuas
```
```
pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu; # For CPU-only systems
pip install -r requirements
```

# Simulator
![Alt text](./results/img/simulator.png)

![Alt text](./results/img/neutralization_probability_plot.png)

# Training

![Alt text](./results/img/training_progression.svg)

# Evaluation and Results

| Metric                        | Classical Heuristic | Reinforcement Learning |
|-------------------------------|:------------------:|:---------------------:|
| Total Damage (Avg) [%]        | 50.34              | **41.30**             |
| In-Tracking Time (Avg) [%]    | 52.87              | **65.59**             |
| Weapon Utilization (Avg) [%]  | 54.35              | **62.79**             |

*Table: Evaluation Results. 100 Episodes × 5 Seeds*

[![Demo Video](https://img.youtube.com/vi/GooNFDk42Nw/0.jpg)](https://youtu.be/GooNFDk42Nw)

![Alt text](./results/img/damage_distributions.svg)

| ![Alt text](./results/img/tracking_performance.svg) | ![Alt text](./results/img/weapon_utilization.svg) |
|:---------------------------------------------------:|:-------------------------------------------------:|
| Tracking Performance                                | Weapon Utilization                                |


| ![Alt text](./results/img/damage_vs_tracking.svg) | ![Alt text](./results/img/damage_vs_weapon_utilization.svg) |
|:-------------------------------------------------:|:-----------------------------------------------------------:|
| Damage vs Tracking Correlation                    | Damage vs Weapon Utilization Correlation                    |

# Citation