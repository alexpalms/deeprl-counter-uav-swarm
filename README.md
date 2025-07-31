# Reinforcement Learning for Decision-Level Interception Prioritization in Drone Swarm Defense

# Overview


# Environment Setup
```
conda create -n cuas python=3.11
conda activate cuas
```
```
pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu; # For CPU-only systems
pip install -r requirements
```

# Training

![Alt text](./results/img/training_progression.svg)

# Evaluation and Results

![Alt text](./results/img/damage_distributions.svg)

| ![Alt text](./results/img/tracking_performance.svg) | ![Alt text](./results/img/weapon_utilization.svg) |
|:---------------------------------------------------:|:-------------------------------------------------:|
| Tracking Performance                                | Weapon Utilization                                |


| ![Alt text](./results/img/damage_vs_tracking.svg) | ![Alt text](./results/img/damage_vs_weapon_utilization.svg) |
|:-------------------------------------------------:|:-----------------------------------------------------------:|
| Damage vs Tracking Correlation                    | Damage vs Weapon Utilization Correlation                    |

# Citation