# dairy-rl-diet-optimisation

Reinforcement learning framework for optimising dairy cow feeding strategies under economic and environmental objectives.

This project applies Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) agents to a dairy cow simulation environment in order to learn feeding policies that balance:

- Feed margin
- Enteric methane emissions

The environment is based on a dynamic dairy cow model representing lactation Mode 3 over an 83-day horizon.

---

# Project Overview

The reinforcement learning agent interacts with a simulated dairy cow environment by selecting daily diet compositions. The learned feeding strategies aim to optimise long-term performance while satisfying nutritional constraints.

The project investigates:
- Economic vs environmental trade-offs
- PPO vs SAC performance
- Stability across multiple random seeds
- Learned diet compositions over time

---

# Environment

The environment is implemented in:

```python
env_dairy_mode3.py
```

The state space includes:
- Feed margin
- DVE requirement/intake
- OEB intake
- Milk production
- Methane emissions

Actions correspond to diet ingredient proportions on a dry matter basis.

---

# Algorithms

Implemented algorithms:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

Main training scripts:
```python
ppo.py
sac.py
```

Additional experiments:
```python
ppo_methane_min_5seeds.py
sac_methane_5seeds.py
```

---

# Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Training

Train PPO:

```bash
python ppo.py
```

Train SAC:

```bash
python sac.py
```

# Repository Structure

```text
├── env_dairy_mode3.py
├── ppo.py
├── sac.py
├── simulateModel.py
├── simulationLibrary.py
├── requirements.txt
├── plot_diet_composition_ppo.py
├── plot_ppo_methane_min.py
└── README.md
```

---

# Author

Eleni Anastasi  
MSc Statistics & Data Science  
Leiden University

---

# Research Context

This repository was developed as part of a Master's thesis on reinforcement learning for sustainable dairy feeding optimisation.
